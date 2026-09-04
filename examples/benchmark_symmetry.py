"""Timing of the space-group search on supercells of growing size.

Run with ``python examples/benchmark_symmetry.py``.  A perfect supercell is the
worst case for the search and the case that matters for unfolding: every one of
its ``|det T|`` internal translations is a symmetry, so the operation count
itself grows with the cell.  The numbers printed here are the ones quoted in
``docs/findings.md``.

The script also checks, on the smallest cell, that the operations found agree
with a brute-force search that tests every (rotation, translation) pair on its
own -- the shortcut must not change the answer.
"""

from __future__ import annotations

import time

import numpy as np

from unmochan.core.spacegroup import (
    _candidate_translations,
    _label_codes,
    _reduce,
    _site_assignment,
    lattice_point_group,
    site_labels,
    space_group_operations,
)
from unmochan.core.structures import Structure


def cubic_supercell(n: int, rattle: float = 0.0) -> Structure:
    axis = np.arange(n) / n
    coords = np.array([(i, j, k) for i in axis for j in axis for k in axis])
    if rattle:
        coords = coords + np.random.default_rng(0).normal(scale=rattle, size=coords.shape)
    return Structure(
        lattice=np.eye(3) * n,
        species=("Si",) * len(coords),
        frac_coords=coords,
    )


def brute_force(structure: Structure, symprec: float = 1e-5) -> int:
    """Every (rotation, translation) pair tested from scratch."""

    frac = _reduce(np.asarray(structure.frac_coords, dtype=float))
    labels = site_labels(structure, None)
    codes = _label_codes(labels)
    lattice = np.asarray(structure.lattice, dtype=float)
    tol = float(symprec * max(1.0, np.abs(lattice @ lattice.T).max()))
    total = 0
    for rotation in lattice_point_group(structure.lattice, tol=tol):
        for translation in _candidate_translations(
            frac, list(range(len(labels))), 0, rotation, symprec
        ):
            if _site_assignment(frac, codes, rotation, translation, symprec) is not None:
                total += 1
    return total


def main() -> None:
    smallest = cubic_supercell(2)
    assert len(space_group_operations(smallest)) == brute_force(smallest)
    print("agrees with the brute-force search on the 8-site cell")
    for n in (2, 3, 4, 5, 6):
        structure = cubic_supercell(n)
        start = time.perf_counter()
        operations = space_group_operations(structure)
        elapsed = time.perf_counter() - start
        print(f"perfect {n**3:4d} sites  {len(operations):6d} operations  {elapsed:7.3f} s")
    structure = cubic_supercell(5, rattle=2e-3)
    start = time.perf_counter()
    operations = space_group_operations(structure)
    print(
        f"rattled  125 sites  {len(operations):6d} operations  {time.perf_counter() - start:7.3f} s"
    )


if __name__ == "__main__":
    main()
