"""Reading Wannier90 ``seedname_hr.dat`` real-space Hamiltonians."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unfoldlab.io.wannier90 import read_wannier90_hr


def _hr_text(
    entries: dict[tuple[int, int, int], np.ndarray],
    degeneracies: dict[tuple[int, int, int], int] | None = None,
    *,
    num_wann: int,
) -> str:
    """Render a synthetic hr file, one degeneracy per lattice vector."""

    cells = list(entries)
    degeneracies = degeneracies or {}
    lines = ["written by the test suite", str(num_wann), str(len(cells))]
    lines.extend(
        " ".join(str(degeneracies.get(cell, 1)) for cell in cells[start : start + 15])
        for start in range(0, len(cells), 15)
    )
    for cell in cells:
        block = entries[cell]
        for column in range(num_wann):
            for row in range(num_wann):
                value = complex(block[row, column])
                lines.append(
                    f"{cell[0]} {cell[1]} {cell[2]} {row + 1} {column + 1} "
                    f"{value.real!r} {value.imag!r}"
                )
    return "\n".join(lines) + "\n"


def _chain(tmp_path: Path, **kwargs: object) -> Path:
    entries = {
        (0, 0, 0): np.array([[0.2]]),
        (1, 0, 0): np.array([[-1.0]]),
        (-1, 0, 0): np.array([[-1.0]]),
    }
    path = tmp_path / "chain_hr.dat"
    path.write_text(_hr_text(entries, num_wann=1, **kwargs))  # type: ignore[arg-type]
    return path


def test_reads_a_one_orbital_chain(tmp_path: Path) -> None:
    model = read_wannier90_hr(_chain(tmp_path))

    assert model.n_orbitals == 1
    assert set(model.hoppings) == {(0, 0, 0), (1, 0, 0), (-1, 0, 0)}
    assert model.hoppings[(1, 0, 0)] == pytest.approx(np.array([[-1.0]]))
    # E(k) = onsite + 2 t cos(2 pi k) reproduces the analytic chain dispersion.
    energies = model.bands([[0.0, 0.0, 0.0]])
    assert energies[0, 0] == pytest.approx(0.2 - 2.0)


def test_divides_by_the_wigner_seitz_degeneracies(tmp_path: Path) -> None:
    path = _chain(tmp_path, degeneracies={(1, 0, 0): 2, (-1, 0, 0): 2})
    model = read_wannier90_hr(path)

    assert model.hoppings[(0, 0, 0)] == pytest.approx(np.array([[0.2]]))
    assert model.hoppings[(1, 0, 0)] == pytest.approx(np.array([[-0.5]]))
    assert model.hoppings[(-1, 0, 0)] == pytest.approx(np.array([[-0.5]]))


def test_reads_complex_multi_orbital_blocks_and_stays_hermitian(tmp_path: Path) -> None:
    hop = np.array([[0.0, 0.5j], [-0.25, 0.0]])
    entries = {
        (0, 0, 0): np.array([[0.3, 0.1j], [-0.1j, -0.3]]),
        (1, 0, 0): hop,
        (-1, 0, 0): hop.conj().T,
    }
    path = tmp_path / "two_hr.dat"
    path.write_text(_hr_text(entries, num_wann=2))

    model = read_wannier90_hr(path)
    assert model.hoppings[(1, 0, 0)] == pytest.approx(hop)
    matrix = model.bloch_hamiltonian([0.17, 0.0, 0.0])
    assert np.allclose(matrix, matrix.conj().T)


def test_hermitize_repairs_a_one_sided_file(tmp_path: Path) -> None:
    entries = {(0, 0, 0): np.array([[0.0]]), (1, 0, 0): np.array([[-1.0]])}
    path = tmp_path / "half_hr.dat"
    path.write_text(_hr_text(entries, num_wann=1))

    raw = read_wannier90_hr(path)
    assert (-1, 0, 0) not in raw.hoppings
    matrix = raw.bloch_hamiltonian([0.17, 0.0, 0.0])
    assert not np.allclose(matrix, matrix.conj().T)

    fixed = read_wannier90_hr(path, hermitize=True)
    assert fixed.hoppings[(-1, 0, 0)] == pytest.approx(np.array([[-1.0]]))
    matrix = fixed.bloch_hamiltonian([0.17, 0.0, 0.0])
    assert np.allclose(matrix, matrix.conj().T)
    # Hermitizing must not halve the amplitudes: the band edges stay at +-2|t|.
    assert fixed.bands([[0.0, 0.0, 0.0]])[0, 0] == pytest.approx(-2.0)


def test_rejects_malformed_files(tmp_path: Path) -> None:
    short = tmp_path / "short_hr.dat"
    short.write_text("header\n1\n")
    with pytest.raises(ValueError, match="header"):
        read_wannier90_hr(short)

    bad_counts = tmp_path / "bad_hr.dat"
    bad_counts.write_text("header\n0\n3\n")
    with pytest.raises(ValueError, match="positive"):
        read_wannier90_hr(bad_counts)

    truncated = tmp_path / "truncated_hr.dat"
    truncated.write_text("header\n1\n2\n1 1\n0 0 0 1 1 0.2 0.0\n")
    with pytest.raises(ValueError, match="expected 2 hopping lines"):
        read_wannier90_hr(truncated)

    ragged = tmp_path / "ragged_hr.dat"
    ragged.write_text("header\n1\n1\n1\n0 0 0 1 1 0.2\n")
    with pytest.raises(ValueError, match="Rx Ry Rz"):
        read_wannier90_hr(ragged)

    out_of_range = tmp_path / "range_hr.dat"
    out_of_range.write_text("header\n1\n1\n1\n0 0 0 2 1 0.2 0.0\n")
    with pytest.raises(ValueError, match="orbital index"):
        read_wannier90_hr(out_of_range)

    zero_degeneracy = tmp_path / "degen_hr.dat"
    zero_degeneracy.write_text("header\n1\n1\n0\n0 0 0 1 1 0.2 0.0\n")
    with pytest.raises(ValueError, match="degeneracies must be positive"):
        read_wannier90_hr(zero_degeneracy)


def test_a_wannier_file_unfolds_like_the_json_model(tmp_path: Path) -> None:
    """A perfect chain read from hr data unfolds to the primitive dispersion."""

    from unfoldlab.core.tight_binding import unfold_tight_binding_model
    from unfoldlab.core.transformations import TransformationMatrix

    model = read_wannier90_hr(_chain(tmp_path))
    transform = TransformationMatrix(np.diag([3.0, 1.0, 1.0]))
    kpoints, energies, weights = unfold_tight_binding_model(
        model, transform, [np.array([0.05, 0.0, 0.0])]
    )

    assert weights.shape == (3, 3)
    assert np.allclose(weights.sum(axis=0), 1.0)
    for index, point in enumerate(kpoints):
        expected = 0.2 - 2.0 * np.cos(2.0 * np.pi * point[0])
        matched = energies[index][np.argmax(weights[index])]
        assert matched == pytest.approx(expected)
