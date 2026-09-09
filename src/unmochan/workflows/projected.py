"""Projector-assisted (approximate) projected unfolding.

The plane-wave unfolding weight ``w(k; K, n)`` says how much of supercell band
``(K, n)`` belongs to the primitive k-point ``k``.  A projected-character file
such as VASP's ``PROCAR`` says how much of the same band sits on a chosen set of
sites and orbitals, as a fraction ``f(K, n)`` in ``[0, 1]``.  The projected
weight used here is the product

``w_P(k; K, n) = f(K, n) * w(k; K, n)``.

This is an *approximation*: it assumes the site/orbital character is the same
for every primitive k-point of the fiber over ``K``, because the character
tables are not resolved in ``k``.  What it does preserve exactly is the
bookkeeping (``UnfoldLab.Projection`` in the Lean development):

* ``0 <= w_P <= w`` (``UnfoldLab.projectedWeight_le``);
* the fractions of a partition of the site/orbital table sum to one, so the
  projected weights of that partition sum back to the plain weight
  (``UnfoldLab.sum_projectedWeight_of_partition``);
* consequently the fiber sum rule holds group by group: summing the projected
  weights over the fiber *and* over a full partition returns
  ``1`` per band (``UnfoldLab.sum_fiber_projected_partition``).

A projection is therefore a redistribution of the unfolded weight, never a
creation of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.kpoints import match_kpoints_modulo_lattice
from unmochan.core.projections import ProjectionSelector
from unmochan.core.site_projection import projection_fractions, resolve_site_orbital_mask
from unmochan.core.spectral import EffectiveBandStructure
from unmochan.core.structures import Structure
from unmochan.io.procar import ProcarData, read_procar

__all__ = [
    "apply_projection_fractions",
    "procar_projection_fractions",
]


def procar_projection_fractions(
    procar: ProcarData | str | Path,
    selectors: Iterable[ProjectionSelector],
    supercell_structure: Structure,
    folded_kpoints: ArrayLike,
    *,
    spin: int = 1,
    atol: float = 1e-6,
    site_groups: Mapping[str, Mapping[object, Sequence[int]]] | None = None,
    layer_axis: int = 2,
    layer_tol: float = 0.5,
    defect_center: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Per-band projection fractions for each folded supercell k-point.

    ``folded_kpoints`` are supercell fractional coordinates, one row per row of
    the unfolded band structure.  A PROCAR written for the folded path lists the
    same k-points in the same order, and that correspondence is used whenever it
    holds -- a folded path legitimately repeats the same supercell k-point many
    times, so matching purely by value would be ambiguous.  Otherwise each
    requested k-point is matched by value modulo a reciprocal lattice vector,
    and an ambiguous or missing match is an error.

    ``spin`` is 1-based, as everywhere else in the package.
    """

    data = procar if isinstance(procar, ProcarData) else read_procar(procar)
    if data.n_ions != supercell_structure.n_sites:
        raise ValueError(
            f"PROCAR has {data.n_ions} ions but the supercell structure has "
            f"{supercell_structure.n_sites} sites"
        )
    if not 1 <= spin <= data.n_spin:
        raise ValueError(f"spin channel {spin} is out of range for {data.n_spin} channels")

    mask = resolve_site_orbital_mask(
        selectors,
        supercell_structure,
        data.orbital_labels,
        site_groups=site_groups,
        layer_axis=layer_axis,
        layer_tol=layer_tol,
        defect_center=defect_center,
    )
    if not mask.any():
        raise ValueError("the projection selectors select no site/orbital pair")

    fractions = projection_fractions(data.projections[spin - 1], mask)
    rows = _procar_rows(np.asarray(folded_kpoints, dtype=float), data.kpoints, atol=atol)
    return fractions[rows]


def _procar_rows(
    folded_kpoints: NDArray[np.float64],
    procar_kpoints: NDArray[np.float64],
    *,
    atol: float,
) -> NDArray[np.int64]:
    if folded_kpoints.shape == procar_kpoints.shape:
        delta = folded_kpoints - procar_kpoints
        delta -= np.rint(delta)
        if np.all(np.abs(delta) <= atol):
            return np.arange(len(folded_kpoints), dtype=np.int64)
    return match_kpoints_modulo_lattice(folded_kpoints, procar_kpoints, atol=atol)


def apply_projection_fractions(
    ebs: EffectiveBandStructure,
    fractions: ArrayLike,
    *,
    label: str | None = None,
) -> EffectiveBandStructure:
    """Multiply the weights of ``ebs`` by per-``(k, band)`` projection fractions."""

    values = np.asarray(fractions, dtype=float)
    if values.shape != ebs.energies.shape:
        raise ValueError("fractions must have shape (n_kpoints, n_bands)")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("projection fractions must lie in [0, 1]")
    metadata = dict(ebs.metadata)
    if label is not None:
        metadata["projection"] = label
    return EffectiveBandStructure(
        kpoints=ebs.kpoints,
        energies=ebs.energies,
        weights=ebs.weights * values,
        distances=ebs.distances,
        reference_energy=ebs.reference_energy,
        metadata=metadata,
    )
