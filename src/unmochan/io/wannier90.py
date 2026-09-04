"""Reading a Wannier90 ``seedname_hr.dat`` real-space Hamiltonian.

The format is fixed:

* a comment line (the date Wannier90 wrote the file),
* ``num_wann``, the number of Wannier functions per primitive cell,
* ``nrpts``, the number of lattice vectors in the Wigner-Seitz list,
* ``nrpts`` integer degeneracies, fifteen per line,
* then ``nrpts * num_wann**2`` lines ``Rx Ry Rz m n Re Im`` giving
  ``<0 m| H |R n>`` in eV, *multiplied* by the degeneracy of ``R``.

Wannier90 stores the amplitudes multiplied by the degeneracy because the
Wigner-Seitz list repeats a vector that sits on the zone boundary; the physical
hopping is the stored value divided by that degeneracy, and that division is
done here, so the resulting :class:`~unmochan.core.tight_binding.TightBindingModel`
can be used directly.

The file already contains both ``R`` and ``-R``, so no Hermitian completion is
needed; :func:`read_wannier90_hr` can optionally symmetrize anyway, which
repairs the small asymmetry a disentangled calculation can leave behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unmochan.core.structures import Structure
from unmochan.core.tight_binding import TightBindingModel

__all__ = [
    "Wannier90TB",
    "read_wannier90_hr",
    "read_wannier90_tb",
]


def _tokens(lines: list[str], start: int, count: int) -> tuple[list[str], int]:
    """Collect ``count`` whitespace-separated tokens starting at line ``start``."""

    collected: list[str] = []
    index = start
    while len(collected) < count:
        if index >= len(lines):
            raise ValueError("truncated Wannier90 hr file")
        collected.extend(lines[index].split())
        index += 1
    if len(collected) > count:
        raise ValueError("malformed degeneracy block in Wannier90 hr file")
    return collected, index


def read_wannier90_hr(path: str | Path, *, hermitize: bool = False) -> TightBindingModel:
    """Read ``seedname_hr.dat`` into a tight-binding model."""

    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("a Wannier90 hr file needs a header, num_wann and nrpts")
    try:
        num_wann = int(lines[1].split()[0])
        nrpts = int(lines[2].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("could not read num_wann and nrpts") from exc
    if num_wann <= 0 or nrpts <= 0:
        raise ValueError("num_wann and nrpts must be positive")

    raw_degeneracies, cursor = _tokens(lines, 3, nrpts)
    degeneracies = [int(value) for value in raw_degeneracies]
    if any(value <= 0 for value in degeneracies):
        raise ValueError("Wigner-Seitz degeneracies must be positive")

    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    order: list[tuple[int, int, int]] = []
    expected = nrpts * num_wann * num_wann
    body = lines[cursor:]
    if len(body) != expected:
        raise ValueError(f"expected {expected} hopping lines, found {len(body)}")
    for line in body:
        parts = line.split()
        if len(parts) != 7:
            raise ValueError("hopping lines must be: Rx Ry Rz m n Re Im")
        cell = (int(parts[0]), int(parts[1]), int(parts[2]))
        row = int(parts[3]) - 1
        column = int(parts[4]) - 1
        if not (0 <= row < num_wann and 0 <= column < num_wann):
            raise ValueError("orbital index outside 1..num_wann")
        if cell not in blocks:
            blocks[cell] = np.zeros((num_wann, num_wann), dtype=np.complex128)
            order.append(cell)
        blocks[cell][row, column] = complex(float(parts[5]), float(parts[6]))

    if len(order) != nrpts:
        raise ValueError(f"expected {nrpts} lattice vectors, found {len(order)}")
    for cell, degeneracy in zip(order, degeneracies, strict=True):
        blocks[cell] = blocks[cell] / float(degeneracy)

    model = TightBindingModel(num_wann, blocks)
    return model.hermitized() if hermitize else model


@dataclass(frozen=True)
class Wannier90TB:
    """The contents of a Wannier90 ``seedname_tb.dat`` file.

    Unlike ``seedname_hr.dat`` this format carries the *lattice* and the
    *position operator*, which is what makes a Wannier model self-contained:

    * :attr:`lattice` — the real-space lattice (rows, angstrom), so a k-path can
      be built and a Cartesian abscissa accumulated without a separate
      structure file;
    * :attr:`positions` — the matrices ``<0 m| r_alpha |R n>`` in angstrom, one
      ``(3, num_wann, num_wann)`` block per lattice vector ``R``;
    * :attr:`centres` — the Wannier centres, the diagonal of the ``R = 0``
      position block.

    The centres are what let a Wannier model be resolved by layer, sublattice,
    or region: assign each Wannier function to a group by where its centre sits
    and pass the groups to
    :func:`~unmochan.core.tight_binding.unfold_tight_binding_path_projected`.
    The *weights* themselves do not depend on the centres — moving a Wannier
    function inside its cell only multiplies its amplitudes by a phase, and
    ``UnfoldLab.tbWeight_orbital_phase`` proves that leaves every weight
    unchanged — so the position operator is metadata for grouping and plotting,
    not an input to the unfolding kernel.
    """

    model: TightBindingModel
    lattice: NDArray[np.float64]
    positions: dict[tuple[int, int, int], NDArray[np.complex128]] | None = None

    @property
    def n_orbitals(self) -> int:
        return self.model.n_orbitals

    @property
    def centres(self) -> NDArray[np.float64]:
        """Cartesian Wannier centres, shape ``(num_wann, 3)``, in angstrom.

        Raises when the file carried no position operator.
        """

        if self.positions is None:
            raise ValueError("this tb file carried no position operator")
        block = self.positions.get((0, 0, 0))
        if block is None:
            raise ValueError("the position operator has no R = 0 block")
        return np.real(np.einsum("aii->ia", block)).astype(float)

    @property
    def fractional_centres(self) -> NDArray[np.float64]:
        """The Wannier centres in fractional coordinates of :attr:`lattice`."""

        return self.centres @ np.linalg.inv(self.lattice)

    def centre_structure(self, *, species: str = "X") -> Structure:
        """The Wannier centres as a :class:`~unmochan.core.structures.Structure`.

        Handy for reusing the site machinery — ``detect_layers`` in particular —
        to build orbital groups for projected unfolding.
        """

        centres = self.fractional_centres
        return Structure(
            self.lattice,
            tuple(species for _ in range(centres.shape[0])),
            centres,
            name="wannier-centres",
        )


def _split_tb_blocks(
    lines: list[str], start: int, nrpts: int, num_wann: int, width: int, what: str
) -> tuple[dict[tuple[int, int, int], NDArray[np.complex128]], list[tuple[int, int, int]], int]:
    """Read ``nrpts`` blocks of ``num_wann**2`` rows of ``width`` real numbers.

    Each block is headed by the three integer components of ``R``; blank lines
    separate the blocks and are skipped.  Returns the blocks keyed by ``R``
    (shape ``(width // 2, num_wann, num_wann)``), the order they appeared in,
    and the index of the first unconsumed line.
    """

    blocks: dict[tuple[int, int, int], NDArray[np.complex128]] = {}
    order: list[tuple[int, int, int]] = []
    cursor = start
    n_components = width // 2
    for _ in range(nrpts):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            raise ValueError(f"truncated {what} block in Wannier90 tb file")
        header = lines[cursor].split()
        if len(header) != 3:
            raise ValueError(
                f"expected a lattice vector heading a {what} block, got {lines[cursor]!r}"
            )
        try:
            cell = (int(header[0]), int(header[1]), int(header[2]))
        except ValueError as exc:
            raise ValueError("lattice vectors must be integers") from exc
        if cell in blocks:
            raise ValueError(f"lattice vector {cell} appears twice in the {what} blocks")
        cursor += 1
        block = np.zeros((n_components, num_wann, num_wann), dtype=np.complex128)
        for _row in range(num_wann * num_wann):
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            if cursor >= len(lines):
                raise ValueError(f"truncated {what} block in Wannier90 tb file")
            parts = lines[cursor].split()
            if len(parts) != width + 2:
                raise ValueError(
                    f"a {what} row needs two indices and {width} numbers, got {len(parts)} fields"
                )
            row = int(parts[0]) - 1
            column = int(parts[1]) - 1
            if not (0 <= row < num_wann and 0 <= column < num_wann):
                raise ValueError("orbital index outside 1..num_wann")
            values = [float(item) for item in parts[2:]]
            for component in range(n_components):
                block[component, row, column] = complex(
                    values[2 * component], values[2 * component + 1]
                )
            cursor += 1
        blocks[cell] = block
        order.append(cell)
    return blocks, order, cursor


def read_wannier90_tb(
    path: str | Path,
    *,
    hermitize: bool = False,
    divide_by_degeneracy: bool = True,
) -> Wannier90TB:
    """Read ``seedname_tb.dat`` — lattice, Hamiltonian and position operator.

    The layout is the one Wannier90 writes with ``write_tb = .true.``:

    * a comment line,
    * the three real-space lattice vectors in angstrom,
    * ``num_wann`` and ``nrpts``,
    * the ``nrpts`` Wigner-Seitz degeneracies, fifteen per line,
    * ``nrpts`` Hamiltonian blocks, each a line ``Rx Ry Rz`` followed by
      ``num_wann**2`` lines ``m n Re Im`` in eV,
    * ``nrpts`` position blocks in the same layout, each row carrying the three
      Cartesian components ``Re Im`` of ``<0 m| r |R n>`` in angstrom.

    As in ``seedname_hr.dat`` the stored amplitudes are divided by the
    Wigner-Seitz degeneracy of their lattice vector, which is what
    ``divide_by_degeneracy`` does; pass ``False`` for a file that was written
    with the division already applied.  The position blocks are divided the same
    way, so the ``R = 0`` diagonal is the Wannier centre either way when all the
    degeneracies are one, as they are for a Gamma-centred Wigner-Seitz cell that
    puts no vector on the boundary.

    The position operator is optional: a file that stops after the Hamiltonian
    blocks is accepted and gives ``positions = None``.
    """

    lines = Path(path).read_text().splitlines()
    stripped = [line for line in lines if line.strip()]
    if len(stripped) < 6:
        raise ValueError("a Wannier90 tb file needs a header, a lattice, num_wann, nrpts")
    lattice_rows = []
    for row in stripped[1:4]:
        values = [float(item) for item in row.split()]
        if len(values) != 3:
            raise ValueError("each lattice vector needs three Cartesian components")
        lattice_rows.append(values)
    lattice = np.array(lattice_rows, dtype=float)
    if abs(float(np.linalg.det(lattice))) <= 0.0:
        raise ValueError("the lattice in the tb file is singular")
    try:
        num_wann = int(stripped[4].split()[0])
        nrpts = int(stripped[5].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("could not read num_wann and nrpts") from exc
    if num_wann <= 0 or nrpts <= 0:
        raise ValueError("num_wann and nrpts must be positive")

    raw_degeneracies, cursor = _tokens(stripped, 6, nrpts)
    degeneracies = [int(value) for value in raw_degeneracies]
    if any(value <= 0 for value in degeneracies):
        raise ValueError("Wigner-Seitz degeneracies must be positive")

    # Work on the original lines from here: the block structure is positional.
    offset = 0
    seen = 0
    for index, line in enumerate(lines):
        if line.strip():
            seen += 1
            if seen == cursor:
                offset = index + 1
                break

    hoppings, order, offset = _split_tb_blocks(lines, offset, nrpts, num_wann, 2, "Hamiltonian")
    scale = (
        {cell: float(value) for cell, value in zip(order, degeneracies, strict=True)}
        if divide_by_degeneracy
        else {cell: 1.0 for cell in order}
    )
    blocks = {cell: hoppings[cell][0] / scale[cell] for cell in order}

    positions: dict[tuple[int, int, int], NDArray[np.complex128]] | None = None
    if any(line.strip() for line in lines[offset:]):
        raw_positions, position_order, _ = _split_tb_blocks(
            lines, offset, nrpts, num_wann, 6, "position"
        )
        if position_order != order:
            raise ValueError(
                "the position operator lists different lattice vectors than the Hamiltonian"
            )
        positions = {cell: raw_positions[cell] / scale[cell] for cell in order}

    model = TightBindingModel(num_wann, blocks)
    return Wannier90TB(model.hermitized() if hermitize else model, lattice, positions)
