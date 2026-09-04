"""Reading Wannier90 ``seedname_tb.dat``: lattice, Hamiltonian, positions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unmochan.core.site_projection import detect_layers
from unmochan.io.wannier90 import read_wannier90_tb


def _tb_text(
    lattice: np.ndarray,
    hoppings: dict[tuple[int, int, int], np.ndarray],
    positions: dict[tuple[int, int, int], np.ndarray] | None,
    *,
    num_wann: int,
    degeneracies: dict[tuple[int, int, int], int] | None = None,
) -> str:
    """Render a synthetic tb file in the layout Wannier90 writes."""

    cells = list(hoppings)
    degeneracies = degeneracies or {}
    lines = ["written by the test suite"]
    lines.extend(" ".join(f"{float(value)!r}" for value in row) for row in lattice)
    lines.append(str(num_wann))
    lines.append(str(len(cells)))
    lines.extend(
        " ".join(str(degeneracies.get(cell, 1)) for cell in cells[start : start + 15])
        for start in range(0, len(cells), 15)
    )
    for cell in cells:
        lines.append("")
        lines.append(f"{cell[0]} {cell[1]} {cell[2]}")
        block = hoppings[cell]
        for column in range(num_wann):
            for row in range(num_wann):
                value = complex(block[row, column])
                lines.append(f"{row + 1} {column + 1} {value.real!r} {value.imag!r}")
    if positions is not None:
        for cell in cells:
            lines.append("")
            lines.append(f"{cell[0]} {cell[1]} {cell[2]}")
            block = positions[cell]
            for column in range(num_wann):
                for row in range(num_wann):
                    entries = " ".join(
                        f"{complex(block[axis, row, column]).real!r} "
                        f"{complex(block[axis, row, column]).imag!r}"
                        for axis in range(3)
                    )
                    lines.append(f"{row + 1} {column + 1} {entries}")
    return "\n".join(lines) + "\n"


def _two_orbital_chain(
    tmp_path: Path,
    *,
    with_positions: bool = True,
    degeneracies: dict[tuple[int, int, int], int] | None = None,
) -> Path:
    lattice = np.diag([2.5, 6.0, 8.0])
    hoppings = {
        (0, 0, 0): np.array([[0.3, -0.4], [-0.4, -0.3]]),
        (1, 0, 0): np.array([[-1.0, 0.0], [0.2, -1.0]]),
        (-1, 0, 0): np.array([[-1.0, 0.2], [0.0, -1.0]]),
    }
    positions = None
    if with_positions:
        positions = {cell: np.zeros((3, 2, 2), dtype=complex) for cell in hoppings}
        # Two Wannier centres at different heights: a two-layer model.
        positions[(0, 0, 0)][2, 0, 0] = 0.5
        positions[(0, 0, 0)][2, 1, 1] = 4.5
        positions[(0, 0, 0)][0, 0, 0] = 0.25
        positions[(0, 0, 0)][0, 1, 1] = 1.25
        positions[(1, 0, 0)][0, 0, 1] = 0.1 + 0.2j
    path = tmp_path / "chain_tb.dat"
    path.write_text(
        _tb_text(
            lattice,
            hoppings,
            positions,
            num_wann=2,
            degeneracies=degeneracies,
        )
    )
    return path


def test_reads_lattice_hamiltonian_and_positions(tmp_path: Path) -> None:
    data = read_wannier90_tb(_two_orbital_chain(tmp_path))

    assert data.n_orbitals == 2
    assert data.lattice == pytest.approx(np.diag([2.5, 6.0, 8.0]))
    assert set(data.model.hoppings) == {(0, 0, 0), (1, 0, 0), (-1, 0, 0)}
    assert data.model.hoppings[(1, 0, 0)] == pytest.approx(np.array([[-1.0, 0.0], [0.2, -1.0]]))
    assert data.positions is not None
    assert data.positions[(1, 0, 0)][0, 0, 1] == pytest.approx(0.1 + 0.2j)


def test_wannier_centres_are_the_diagonal_of_the_r_zero_block(tmp_path: Path) -> None:
    data = read_wannier90_tb(_two_orbital_chain(tmp_path))

    assert data.centres == pytest.approx(np.array([[0.25, 0.0, 0.5], [1.25, 0.0, 4.5]]))
    assert data.fractional_centres == pytest.approx(
        np.array([[0.1, 0.0, 0.0625], [0.5, 0.0, 0.5625]])
    )


def test_centre_structure_feeds_the_layer_detector(tmp_path: Path) -> None:
    data = read_wannier90_tb(_two_orbital_chain(tmp_path))

    layers = detect_layers(data.centre_structure(), axis=2, tol=1.0)

    assert layers.n_layers == 2
    assert layers.index[0] != layers.index[1]


def test_degeneracies_divide_both_the_hamiltonian_and_the_positions(
    tmp_path: Path,
) -> None:
    path = _two_orbital_chain(tmp_path, degeneracies={(1, 0, 0): 2})

    divided = read_wannier90_tb(path)
    raw = read_wannier90_tb(path, divide_by_degeneracy=False)

    assert divided.model.hoppings[(1, 0, 0)] == pytest.approx(raw.model.hoppings[(1, 0, 0)] / 2.0)
    assert divided.positions is not None
    assert raw.positions is not None
    assert divided.positions[(1, 0, 0)] == pytest.approx(raw.positions[(1, 0, 0)] / 2.0)
    # The R = 0 block has degeneracy one, so the centres are unaffected.
    assert divided.centres == pytest.approx(raw.centres)


def test_a_file_without_a_position_operator_is_accepted(tmp_path: Path) -> None:
    data = read_wannier90_tb(_two_orbital_chain(tmp_path, with_positions=False))

    assert data.positions is None
    with pytest.raises(ValueError, match="no position operator"):
        _ = data.centres


def test_hermitize_completes_the_model(tmp_path: Path) -> None:
    data = read_wannier90_tb(_two_orbital_chain(tmp_path), hermitize=True)

    for cell, block in data.model.hoppings.items():
        opposite = (-cell[0], -cell[1], -cell[2])
        assert data.model.hoppings[opposite] == pytest.approx(block.conj().T)


def test_bands_from_a_tb_file_match_the_analytic_chain(tmp_path: Path) -> None:
    data = read_wannier90_tb(_two_orbital_chain(tmp_path), hermitize=True)

    # At the zone centre every phase is one, so H(0) is the sum of the blocks.
    total = sum(data.model.hoppings.values())
    assert data.model.bands([[0.0, 0.0, 0.0]])[0] == pytest.approx(np.linalg.eigvalsh(total))


def test_a_truncated_block_is_rejected(tmp_path: Path) -> None:
    path = _two_orbital_chain(tmp_path)
    text = path.read_text().splitlines()
    path.write_text("\n".join(text[:-3]) + "\n")

    with pytest.raises(ValueError, match="truncated"):
        read_wannier90_tb(path)


def test_a_singular_lattice_is_rejected(tmp_path: Path) -> None:
    path = _two_orbital_chain(tmp_path)
    lines = path.read_text().splitlines()
    lines[3] = "0.0 0.0 0.0"
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="singular"):
        read_wannier90_tb(path)
