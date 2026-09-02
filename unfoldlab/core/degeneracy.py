"""Degeneracy grouping and gauge-invariant unfolded weights.

Why this exists
---------------

A diagonalizer returns *some* orthonormal basis of each eigenspace.  Inside a
degenerate multiplet that basis is arbitrary up to a unitary mixing, and the
unfolding weight of an individual band of the multiplet changes under that
mixing.  Two runs of the same calculation -- different code version, different
machine, different number of MPI ranks -- can therefore report visibly different
weights for the individual members of a degenerate multiplet while describing
exactly the same physics.

What *is* invariant is the weight of the whole degenerate subspace.  This module
groups bands into multiplets by energy and reports the subspace weight, either
summed over the multiplet or (the default, because it keeps the band-indexed
array rectangular) shared equally among its members.

The corresponding formal statements are in
``RequestProject/Unfolding/Degeneracy.lean``:

* ``UnfoldLab.subspaceWeight_mix`` -- the subspace weight is unchanged by any
  unitary mixing of the members of the multiplet;
* ``UnfoldLab.IsFiberRepr.sum_subspaceWeight_eq_one`` -- it still obeys the
  fiber sum rule, so grouping costs no diagnostic power;
* ``UnfoldLab.subspaceWeight_eq_average_of_normalized`` -- for normalized states
  it is the plain average of the member weights, which is what
  :func:`degeneracy_averaged_weights` writes back into each band.

Grouping rule
-------------

Bands are sorted by energy and cut wherever the gap between neighbours exceeds
``tol``: a *single-linkage* rule.  It is the only rule that is invariant under
adding a band in the middle of a cluster, but it does chain, so a dense ladder
of near-degenerate states can end up in one group.  :func:`degeneracy_report`
gives the group sizes and the widest group, so chaining is visible rather than
silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.spectral import EffectiveBandStructure

__all__ = [
    "DegeneracyReport",
    "average_degenerate_weights",
    "degeneracy_averaged_weights",
    "degeneracy_report",
    "group_degenerate_bands",
    "subspace_weights",
]


def _check_tol(tol: float) -> float:
    value = float(tol)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"tol must be a finite non-negative energy, got {tol!r}")
    return value


def group_degenerate_bands(energies: ArrayLike, *, tol: float = 1e-4) -> list[list[int]]:
    """Group band indices of one k-point into degenerate multiplets.

    ``energies`` is a one-dimensional array of band energies, in any order.  Two
    bands land in the same group when they are connected by a chain of
    neighbouring energies no more than ``tol`` apart.  The groups are returned
    in order of increasing energy, and the indices inside each group are sorted
    ascending, so the result is a deterministic partition of
    ``range(len(energies))``.

    ``tol = 0`` groups only exactly equal energies, which is the identity
    grouping for generic floating-point data.
    """

    values = np.asarray(energies, dtype=float)
    if values.ndim != 1:
        raise ValueError("energies must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("energies must be finite")
    width = _check_tol(tol)
    if values.size == 0:
        return []

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    # A new group starts wherever the gap to the previous energy exceeds tol.
    gaps = np.diff(sorted_values)
    cuts = np.nonzero(gaps > width)[0] + 1
    groups = np.split(order, cuts)
    return [sorted(int(index) for index in group) for group in groups]


def _grouped_rows(energies: NDArray[np.float64], tol: float) -> list[list[list[int]]]:
    return [group_degenerate_bands(row, tol=tol) for row in energies]


def _as_band_arrays(
    energies: ArrayLike, weights: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    band_energies = np.asarray(energies, dtype=float)
    band_weights = np.asarray(weights, dtype=float)
    if band_energies.shape != band_weights.shape:
        raise ValueError("energies and weights must have the same shape")
    if band_energies.ndim == 1:
        return band_energies[np.newaxis, :], band_weights[np.newaxis, :], True
    if band_energies.ndim == 2:
        return band_energies, band_weights, False
    raise ValueError("energies must have shape (n_bands,) or (n_kpoints, n_bands)")


def degeneracy_averaged_weights(
    energies: ArrayLike,
    weights: ArrayLike,
    *,
    tol: float = 1e-4,
) -> NDArray[np.float64]:
    """Replace the weights of every degenerate multiplet by their average.

    Accepts ``(n_bands,)`` or ``(n_kpoints, n_bands)`` arrays and returns an
    array of the same shape.  The total weight of each multiplet is preserved
    exactly (up to floating-point summation), so every sum rule that held for
    the input still holds for the output, while the output no longer depends on
    the basis the diagonalizer chose inside the multiplet.
    """

    band_energies, band_weights, was_flat = _as_band_arrays(energies, weights)
    result = np.array(band_weights, dtype=float, copy=True)
    for row, groups in enumerate(_grouped_rows(band_energies, _check_tol(tol))):
        for group in groups:
            if len(group) == 1:
                continue
            index = np.asarray(group, dtype=int)
            result[row, index] = float(np.mean(band_weights[row, index]))
    return result[0] if was_flat else result


def subspace_weights(
    energies: ArrayLike,
    weights: ArrayLike,
    *,
    tol: float = 1e-4,
) -> list[list[tuple[float, int, float]]]:
    """Return the multiplets of each k-point as ``(energy, multiplicity, weight)``.

    The energy is the mean energy of the multiplet, the multiplicity its number
    of bands, and the weight the *sum* of the member weights -- the quantity
    ``UnfoldLab.subspaceWeight`` scales, and the one that is gauge invariant.
    Unlike :func:`degeneracy_averaged_weights` this collapses the band axis, so
    the result is a ragged list of lists, one per k-point.
    """

    band_energies, band_weights, was_flat = _as_band_arrays(energies, weights)
    collapsed: list[list[tuple[float, int, float]]] = []
    for row, groups in enumerate(_grouped_rows(band_energies, _check_tol(tol))):
        collapsed.append(
            [
                (
                    float(np.mean(band_energies[row, np.asarray(group, dtype=int)])),
                    len(group),
                    float(np.sum(band_weights[row, np.asarray(group, dtype=int)])),
                )
                for group in groups
            ]
        )
    return [collapsed[0]] if was_flat else collapsed


@dataclass(frozen=True)
class DegeneracyReport:
    """How much of an unfolded band structure is gauge dependent.

    ``max_spread`` is the largest deviation of an individual band weight from
    the average of its multiplet.  It bounds how much the reported weight of a
    single band could change if the diagonalizer had picked another basis, and
    is therefore the number to quote as the gauge uncertainty of a per-band
    weight.  ``max_group_width`` exposes single-linkage chaining: a value much
    larger than ``tol`` means some group was assembled from a ladder of small
    gaps rather than from a true degeneracy.
    """

    tol: float
    n_kpoints: int
    n_bands: int
    n_groups: int
    n_degenerate_groups: int
    max_multiplicity: int
    max_group_width: float
    max_spread: float

    @property
    def gauge_sensitive(self) -> bool:
        """Whether any per-band weight is basis dependent at all."""

        return self.n_degenerate_groups > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tol": self.tol,
            "n_kpoints": self.n_kpoints,
            "n_bands": self.n_bands,
            "n_groups": self.n_groups,
            "n_degenerate_groups": self.n_degenerate_groups,
            "max_multiplicity": self.max_multiplicity,
            "max_group_width": self.max_group_width,
            "max_spread": self.max_spread,
            "gauge_sensitive": self.gauge_sensitive,
        }


def degeneracy_report(
    energies: ArrayLike,
    weights: ArrayLike,
    *,
    tol: float = 1e-4,
) -> DegeneracyReport:
    """Summarize the degeneracy structure and the gauge sensitivity of weights."""

    band_energies, band_weights, _ = _as_band_arrays(energies, weights)
    width = _check_tol(tol)
    n_groups = 0
    n_degenerate = 0
    max_multiplicity = 1 if band_energies.shape[1] else 0
    max_group_width = 0.0
    max_spread = 0.0
    for row, groups in enumerate(_grouped_rows(band_energies, width)):
        n_groups += len(groups)
        for group in groups:
            index = np.asarray(group, dtype=int)
            max_multiplicity = max(max_multiplicity, len(group))
            if len(group) == 1:
                continue
            n_degenerate += 1
            values = band_energies[row, index]
            max_group_width = max(max_group_width, float(values.max() - values.min()))
            member = band_weights[row, index]
            max_spread = max(max_spread, float(np.max(np.abs(member - member.mean()))))
    return DegeneracyReport(
        tol=width,
        n_kpoints=int(band_energies.shape[0]),
        n_bands=int(band_energies.shape[1]),
        n_groups=n_groups,
        n_degenerate_groups=n_degenerate,
        max_multiplicity=max_multiplicity,
        max_group_width=max_group_width,
        max_spread=max_spread,
    )


def average_degenerate_weights(
    structure: EffectiveBandStructure,
    *,
    tol: float = 1e-4,
) -> EffectiveBandStructure:
    """Return a copy of ``structure`` with gauge-invariant weights.

    The energies, k-points, distances and reference energy are untouched; only
    the weights are replaced by their multiplet averages.  The metadata records
    the tolerance used, so a serialized run says whether it was averaged.
    """

    averaged = degeneracy_averaged_weights(structure.energies, structure.weights, tol=tol)
    metadata = dict(structure.metadata)
    metadata["degeneracy_tol"] = float(tol)
    return EffectiveBandStructure(
        kpoints=structure.kpoints,
        energies=structure.energies,
        weights=averaged,
        distances=structure.distances,
        reference_energy=structure.reference_energy,
        metadata=metadata,
    )
