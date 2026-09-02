"""Tracking spectral peaks into band branches.

:mod:`unfoldlab.core.dispersion` turns ``A(k, E)`` into a table of peaks, one
list per k-point.  Everything one plots as a *band* -- and every velocity or
effective mass read off it -- additionally needs to know which peak at one
k-point continues which peak at the next, and that is a matching problem rather
than a peak-finding one.

The rule used here is the global matching that minimizes the total squared
energy jump, restricted to matchings that preserve the energy order.  Two
statements in ``RequestProject/Unfolding/Tracking.lean`` say why:

``UnfoldLab.matchCost_id_le``
    for two peak lists of equal length in increasing order the order-preserving
    pairing is already optimal among *all* permutations, so there is nothing to
    search: sorting the peaks is the assignment algorithm;
``UnfoldLab.matchCost_greedy_gt``
    the obvious cheap alternative -- walk the peaks in order and give each its
    nearest free partner -- is strictly worse on an explicit example, so
    nearest-neighbour tracking is not merely a different convention.

When the two k-points have different numbers of peaks, some peaks have no
partner: a branch begins or ends.  The alignment is then the order-preserving
one of least cost with a penalty per unmatched peak, computed by a small dynamic
program; with equal counts and no jump limit it reduces to the sorted matching
above.

What no rule can do is resolve a degeneracy.  Where two peaks of one k-point
coincide, two different matchings cost exactly the same
(``UnfoldLab.exists_matchCost_tie``): the branch labels are not determined by
the energies, and a crossing and an anticrossing are indistinguishable from the
peak table alone.  :func:`track_branches` therefore *reports* those k-points as
``crossings`` instead of pretending the continuation is known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.dispersion import SpectralPeak, spectral_peaks
from unfoldlab.core.spectral import BroadeningKind, EffectiveBandStructure

__all__ = [
    "Branch",
    "BranchTracking",
    "Crossing",
    "track_branches",
    "track_spectral_peaks",
]


@dataclass(frozen=True)
class Branch:
    """One tracked band branch.

    ``energies`` and ``intensities`` have one entry per k-point of the run and
    are ``nan`` where the branch is absent -- a branch need not span the whole
    path, and a broadened spectral function genuinely loses peaks where the
    weight vanishes.
    """

    energies: NDArray[np.float64]
    intensities: NDArray[np.float64]

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies, dtype=float)
        intensities = np.asarray(self.intensities, dtype=float)
        if energies.ndim != 1 or intensities.shape != energies.shape:
            raise ValueError("energies and intensities must be 1-D and of equal length")
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "intensities", intensities)

    @property
    def present(self) -> NDArray[np.bool_]:
        """Mask of the k-points at which the branch has a peak."""

        return ~np.isnan(self.energies)

    @property
    def length(self) -> int:
        """Number of k-points the branch covers."""

        return int(self.present.sum())

    @property
    def span(self) -> tuple[int, int]:
        """First and last k-point index of the branch."""

        indices = np.flatnonzero(self.present)
        if indices.size == 0:
            raise ValueError("an empty branch has no span")
        return int(indices[0]), int(indices[-1])

    @property
    def max_jump(self) -> float:
        """Largest energy step between consecutive k-points of the branch."""

        values = self.energies[self.present]
        if values.size < 2:
            return 0.0
        return float(np.abs(np.diff(values)).max())

    def to_dict(self) -> dict[str, Any]:
        return {
            "energies": [None if math.isnan(e) else e for e in self.energies],
            "intensities": [None if math.isnan(v) else v for v in self.intensities],
            "length": self.length,
            "max_jump": self.max_jump,
        }


@dataclass(frozen=True)
class Crossing:
    """A k-point at which the continuation of a branch is not determined.

    ``kind`` is ``"degenerate"`` when two peaks of the k-point lie within the
    degeneracy tolerance -- there the matching has an exact tie
    (``UnfoldLab.exists_matchCost_tie``) and a crossing cannot be told from an
    anticrossing -- or ``"terminus"`` when a branch begins or ends in the
    interior of the path.  A terminus means the peak either merged with another
    one, which is the same ambiguity seen from the side where the two peaks are
    no longer resolved, or lost its spectral weight; the peak table does not say
    which.
    """

    kpoint_index: int
    lower_energy: float
    upper_energy: float
    kind: str = "degenerate"

    @property
    def separation(self) -> float:
        return self.upper_energy - self.lower_energy

    def to_dict(self) -> dict[str, Any]:
        return {
            "kpoint_index": self.kpoint_index,
            "lower_energy": self.lower_energy,
            "upper_energy": self.upper_energy,
            "separation": self.separation,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class BranchTracking:
    """The branches of a run, and where the tracking is ambiguous."""

    branches: list[Branch]
    crossings: list[Crossing]
    distances: NDArray[np.float64]
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    @property
    def max_jump(self) -> float:
        """Largest energy step any branch takes between neighbouring k-points."""

        return max((branch.max_jump for branch in self.branches), default=0.0)

    def energy_table(self) -> NDArray[np.float64]:
        """``(n_branches, n_kpoints)`` array of branch energies, ``nan`` padded."""

        if not self.branches:
            return np.zeros((0, self.distances.size), dtype=float)
        return np.vstack([branch.energies for branch in self.branches])

    def to_dict(self) -> dict[str, Any]:
        return {
            "distances": np.asarray(self.distances, dtype=float).tolist(),
            "branches": [branch.to_dict() for branch in self.branches],
            "crossings": [crossing.to_dict() for crossing in self.crossings],
            "cost": self.cost,
            "max_jump": self.max_jump,
            "n_branches": self.n_branches,
            "metadata": self.metadata,
        }


def _align(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    *,
    max_jump: float,
    gap_cost: float,
) -> tuple[list[tuple[int, int]], float]:
    """Least-cost order-preserving alignment of two increasing energy lists.

    Returns the matched index pairs and the total cost.  A pair costs the
    squared energy difference and is forbidden beyond ``max_jump``; leaving a
    peak unmatched costs ``gap_cost``.  For equal lengths, an unrestricted jump
    and a large gap cost this is the sorted matching, which
    ``UnfoldLab.matchCost_id_le`` shows to be optimal among all permutations and
    not only among order-preserving ones.
    """

    n = previous.size
    m = current.size
    limit = max_jump**2
    total = np.full((n + 1, m + 1), np.inf)
    choice = np.zeros((n + 1, m + 1), dtype=np.int8)
    total[0, 0] = 0.0
    for i in range(1, n + 1):
        total[i, 0] = total[i - 1, 0] + gap_cost
        choice[i, 0] = 1
    for j in range(1, m + 1):
        total[0, j] = total[0, j - 1] + gap_cost
        choice[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            pair = (previous[i - 1] - current[j - 1]) ** 2
            best = total[i - 1, j] + gap_cost
            move = 1
            if total[i, j - 1] + gap_cost < best:
                best = total[i, j - 1] + gap_cost
                move = 2
            if pair <= limit and total[i - 1, j - 1] + pair < best:
                best = total[i - 1, j - 1] + pair
                move = 0
            total[i, j] = best
            choice[i, j] = move

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = choice[i, j]
        if move == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs, float(total[n, m])


def track_branches(
    peaks: list[SpectralPeak],
    n_kpoints: int,
    distances: ArrayLike | None = None,
    *,
    max_jump: float = math.inf,
    degeneracy_tolerance: float = 0.0,
    min_length: int = 1,
) -> BranchTracking:
    """Connect a peak table into branches.

    ``peaks`` is the output of :func:`unfoldlab.core.dispersion.spectral_peaks`,
    in any order; it is grouped by k-point and sorted in energy here.

    ``max_jump`` refuses to continue a branch across an energy step larger than
    itself, which is how a branch ends and a new one begins where the spectral
    weight moves.  The default admits any step, so every k-point keeps as many
    branches as it has peaks.

    ``degeneracy_tolerance`` is the separation below which two peaks of one
    k-point are reported as a :class:`Crossing`: there the matching has a tie
    (``UnfoldLab.exists_matchCost_tie``) and the branch labels either side are
    not determined by the energies.  A sensible value is the broadening width,
    which is also the resolution limit of
    :func:`unfoldlab.core.dispersion.unresolvable_pairs`.  Branches that begin
    or end inside the path are reported too, as crossings of kind
    ``"terminus"``: two peaks that have merged are no longer two peaks, so the
    tolerance alone would miss exactly the k-point where the ambiguity is
    worst.

    ``min_length`` drops branches covering fewer k-points, which are usually
    noise peaks rather than bands.
    """

    if n_kpoints <= 0:
        raise ValueError("n_kpoints must be positive")
    if max_jump <= 0.0:
        raise ValueError("max_jump must be positive")
    if degeneracy_tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be non-negative")
    if min_length < 1:
        raise ValueError("min_length must be at least one")

    abscissa = (
        np.arange(n_kpoints, dtype=float)
        if distances is None
        else np.asarray(distances, dtype=float)
    )
    if abscissa.shape != (n_kpoints,):
        raise ValueError("distances must have one entry per k-point")

    by_kpoint: list[list[SpectralPeak]] = [[] for _ in range(n_kpoints)]
    for peak in peaks:
        if not 0 <= peak.kpoint_index < n_kpoints:
            raise ValueError(f"peak at k-point {peak.kpoint_index} outside the run")
        by_kpoint[peak.kpoint_index].append(peak)
    for group in by_kpoint:
        group.sort(key=lambda peak: peak.energy)

    crossings: list[Crossing] = []
    if degeneracy_tolerance > 0.0:
        for index, group in enumerate(by_kpoint):
            for lower, upper in zip(group, group[1:], strict=False):
                if upper.energy - lower.energy < degeneracy_tolerance:
                    crossings.append(
                        Crossing(
                            kpoint_index=index,
                            lower_energy=lower.energy,
                            upper_energy=upper.energy,
                        )
                    )

    # A gap must be more expensive than any admissible pairing, or the alignment
    # would rather drop a peak than continue a branch across a legitimate step.
    finite_gap = max_jump**2 if math.isfinite(max_jump) else None
    energies_all = [peak.energy for peak in peaks]
    if finite_gap is None:
        spread = (max(energies_all) - min(energies_all)) if energies_all else 0.0
        finite_gap = spread**2 + 1.0

    energy_rows: list[NDArray[np.float64]] = []
    intensity_rows: list[NDArray[np.float64]] = []
    active: list[int] = []  # branch row index of each peak of the previous k-point
    previous = np.zeros(0, dtype=float)
    cost = 0.0

    for index, group in enumerate(by_kpoint):
        current = np.array([peak.energy for peak in group], dtype=float)
        pairs: list[tuple[int, int]] = []
        if previous.size and current.size:
            pairs, step_cost = _align(previous, current, max_jump=max_jump, gap_cost=finite_gap)
            cost += step_cost
        continued = dict(pairs)
        next_active: list[int] = []
        for position, peak in enumerate(group):
            row = None
            for old, new in continued.items():
                if new == position:
                    row = active[old]
                    break
            if row is None:
                row = len(energy_rows)
                energy_rows.append(np.full(n_kpoints, np.nan))
                intensity_rows.append(np.full(n_kpoints, np.nan))
            energy_rows[row][index] = peak.energy
            intensity_rows[row][index] = peak.intensity
            next_active.append(row)
        active = next_active
        previous = current

    branches = [
        Branch(energies=energy_rows[row], intensities=intensity_rows[row])
        for row in range(len(energy_rows))
    ]
    branches = [branch for branch in branches if branch.length >= min_length]
    branches.sort(key=lambda branch: float(np.nanmin(branch.energies)))
    for branch in branches:
        first, last = branch.span
        if first > 0:
            crossings.append(
                Crossing(
                    kpoint_index=first,
                    lower_energy=float(branch.energies[first]),
                    upper_energy=float(branch.energies[first]),
                    kind="terminus",
                )
            )
        if last < n_kpoints - 1:
            crossings.append(
                Crossing(
                    kpoint_index=last,
                    lower_energy=float(branch.energies[last]),
                    upper_energy=float(branch.energies[last]),
                    kind="terminus",
                )
            )
    crossings.sort(key=lambda crossing: (crossing.kpoint_index, crossing.lower_energy))
    return BranchTracking(
        branches=branches,
        crossings=crossings,
        distances=abscissa,
        cost=cost,
        metadata={
            "max_jump": max_jump,
            "degeneracy_tolerance": degeneracy_tolerance,
            "min_length": min_length,
            "n_peaks": len(peaks),
        },
    )


def track_spectral_peaks(
    structure: EffectiveBandStructure,
    energy_grid: ArrayLike,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
    min_intensity: float = 0.0,
    max_jump: float = math.inf,
    degeneracy_tolerance: float | None = None,
    min_length: int = 1,
) -> BranchTracking:
    """Find the peaks of ``A(k, E)`` and track them in one step.

    ``degeneracy_tolerance`` defaults to the broadening width when that is a
    single number: two peaks closer than the width are not resolved as two
    states at all (``gaussPair_merged``), so the continuation through them
    cannot be trusted either.
    """

    peaks = spectral_peaks(
        structure,
        energy_grid,
        broadening=broadening,
        kind=kind,
        min_intensity=min_intensity,
    )
    if degeneracy_tolerance is None:
        widths = np.asarray(broadening, dtype=float)
        degeneracy_tolerance = float(widths) if widths.ndim == 0 else float(widths.max())
    return track_branches(
        peaks,
        structure.n_kpoints,
        structure.distances,
        max_jump=max_jump,
        degeneracy_tolerance=degeneracy_tolerance,
        min_length=min_length,
    )
