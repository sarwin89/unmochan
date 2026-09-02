"""Shadow bands: what a weak superlattice perturbation looks like unfolded.

A charge-density wave, a Peierls dimerization, a Jahn-Teller distortion or a
weak substitutional ordering all do the same thing to a band structure: they
enlarge the cell, so the primitive band acquires *shadow* copies displaced by
the ordering wavevector ``Q``.  Unfolded back to the primitive zone the shadow
appears as a faint replica of the main band, and its faintness -- not its
position -- is what measures the strength of the distortion.

Away from accidental degeneracies this is a two-level problem: the perturbation
mixes the primitive states ``|k>`` and ``|k - Q>``, whose unperturbed energies
are ``mean +/- detuning``, through a single matrix element ``coupling``.  This
module solves that problem exactly, in both directions:

* forward -- :func:`two_level_solution` gives the perturbed energies and the
  unfolded weights, and :func:`shadow_band_structure` assembles them into an
  :class:`~unfoldlab.core.spectral.EffectiveBandStructure` for a whole path, so
  a model prediction can be plotted next to a calculation;
* backward -- :func:`coupling_from_peaks` reads ``|coupling|`` and the
  unperturbed levels back off a measured pair of peaks, and
  :func:`diagnose_shadow_bands` does that at every k-point of an unfolded
  calculation, reporting how well the two-level description holds.

The statements behind the module are proved in
``RequestProject/Unfolding/Perturbation.lean``:

``UnfoldLab.Shadow.mulVec_upperVector`` / ``mulVec_lowerVector``
    the energies and weights used here really are the eigenpairs of the 2x2
    Hamiltonian, not a fitted formula;
``UnfoldLab.Shadow.upperWeight_add_lowerWeight``
    the main band and its shadow share the primitive character of ``|k>`` and
    exhaust it -- the fiber sum rule in its smallest nontrivial case;
``UnfoldLab.Shadow.lowerWeight_le_quadratic`` and ``lowerWeight_ge``
    the shadow intensity is second order in the coupling, between
    ``v^2 / (2 (d + |v|) (2 d + |v|))`` and ``v^2 / (4 d^2)``;
``UnfoldLab.Shadow.splitting_ge``
    level repulsion: the splitting never falls below ``2 |coupling|``;
``UnfoldLab.Shadow.abs_coupling_eq`` and ``detuning_eq``
    the inversion :func:`coupling_from_peaks` performs is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.spectral import EffectiveBandStructure, default_distances

__all__ = [
    "ShadowDiagnosis",
    "TwoLevelSolution",
    "coupling_from_peaks",
    "diagnose_shadow_bands",
    "shadow_band_structure",
    "shadow_weight_bounds",
    "two_level_hamiltonian",
    "two_level_solution",
]


@dataclass(frozen=True)
class TwoLevelSolution:
    """Exact solution of the two-level shadow-band problem.

    ``upper_weight`` and ``lower_weight`` are the unfolded weights the two
    perturbed states carry on the *same* primitive character ``|k>``; they sum
    to one.  With ``detuning > 0`` the upper state is the main band and the
    lower one is its shadow.
    """

    upper_energy: NDArray[np.float64]
    lower_energy: NDArray[np.float64]
    upper_weight: NDArray[np.float64]
    lower_weight: NDArray[np.float64]

    @property
    def splitting(self) -> NDArray[np.float64]:
        """The perturbed level splitting, never below ``2 |coupling|``."""

        return self.upper_energy - self.lower_energy


def two_level_hamiltonian(
    mean: float, detuning: float, coupling: complex
) -> NDArray[np.complex128]:
    """The 2x2 Hamiltonian whose eigenpairs the module uses.

    Provided so that the closed forms can be checked against a dense
    diagonalization rather than trusted.
    """

    return np.array(
        [[mean + detuning, coupling], [np.conjugate(coupling), mean - detuning]],
        dtype=complex,
    )


def two_level_solution(
    mean: ArrayLike, detuning: ArrayLike, coupling: ArrayLike
) -> TwoLevelSolution:
    """Diagonalize the two-level problem elementwise.

    ``mean`` is the average and ``detuning`` half the difference of the two
    unperturbed energies, so the unperturbed levels are ``mean +/- detuning``.
    ``coupling`` may be complex; only its modulus enters, which is the statement
    ``UnfoldLab.Shadow.upperWeight_neg`` (a phase on the matrix element is a
    gauge choice).

    At the fully degenerate, uncoupled point ``detuning = coupling = 0`` the
    weights are not determined by the physics -- any pair of orthogonal
    combinations diagonalizes the Hamiltonian.  The even split ``1/2`` is
    returned there, the only choice independent of that arbitrary basis.
    """

    mean_a = np.asarray(mean, dtype=float)
    detuning_a = np.asarray(detuning, dtype=float)
    coupling_a = np.abs(np.asarray(coupling, dtype=complex))
    mean_a, detuning_a, coupling_a = np.broadcast_arrays(mean_a, detuning_a, coupling_a)

    radius = np.hypot(detuning_a, coupling_a)
    ratio = np.divide(detuning_a, radius, out=np.zeros_like(radius), where=radius > 0.0)
    return TwoLevelSolution(
        upper_energy=np.array(mean_a + radius, dtype=float),
        lower_energy=np.array(mean_a - radius, dtype=float),
        upper_weight=np.array(0.5 * (1.0 + ratio), dtype=float),
        lower_weight=np.array(0.5 * (1.0 - ratio), dtype=float),
    )


def shadow_weight_bounds(
    detuning: ArrayLike, coupling: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Lower and upper bound on the shadow weight, as proved in Lean.

    For ``detuning >= 0`` the shadow weight lies between

    ``v^2 / (2 (d + |v|) (2 d + |v|))``  and  ``v^2 / (4 d^2)``,

    both of order ``(v/d)^2``.  The upper bound is infinite at ``d = 0``, where
    the true weight is ``1/2``; it is returned as ``inf`` rather than clipped,
    so that a caller comparing against it is never silently given a bound that
    does not follow from the theorem.
    """

    detuning_a = np.asarray(detuning, dtype=float)
    coupling_a = np.abs(np.asarray(coupling, dtype=complex))
    detuning_a, coupling_a = np.broadcast_arrays(detuning_a, coupling_a)
    if np.any(detuning_a < 0.0):
        raise ValueError("the bounds are stated for a non-negative detuning")

    denominator = 2.0 * (detuning_a + coupling_a) * (2.0 * detuning_a + coupling_a)
    lower = np.divide(
        coupling_a**2,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    upper = np.divide(
        coupling_a**2,
        4.0 * detuning_a**2,
        out=np.full_like(detuning_a, np.inf),
        where=detuning_a > 0.0,
    )
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def shadow_band_structure(
    kpoints: ArrayLike,
    main_energies: ArrayLike,
    partner_energies: ArrayLike,
    coupling: ArrayLike,
    *,
    distances: ArrayLike | None = None,
    reference_energy: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> EffectiveBandStructure:
    """Unfolded band structure of a weakly perturbed superlattice.

    ``main_energies[k]`` is the unperturbed primitive energy at ``k`` and
    ``partner_energies[k]`` the energy of the state at ``k - Q`` that the
    perturbation mixes it with; ``coupling`` is the matrix element, a scalar or
    one value per k-point.  The returned structure has two states per k-point --
    the main band and its shadow -- with the exact unfolded weights, so its sum
    rule is one at every k-point by construction.

    This is the model an unfolded calculation is compared against; it is not a
    substitute for one, because it assumes exactly two states participate.
    """

    kpoints_a = np.asarray(kpoints, dtype=float)
    if kpoints_a.ndim != 2 or kpoints_a.shape[1] != 3:
        raise ValueError("kpoints must have shape (n_kpoints, 3)")
    main_a = np.asarray(main_energies, dtype=float)
    partner_a = np.asarray(partner_energies, dtype=float)
    if main_a.shape != (kpoints_a.shape[0],) or partner_a.shape != (kpoints_a.shape[0],):
        raise ValueError("the two energy arrays must have shape (n_kpoints,)")

    solution = two_level_solution(0.5 * (main_a + partner_a), 0.5 * (main_a - partner_a), coupling)
    energies = np.stack([solution.lower_energy, solution.upper_energy], axis=1)
    weights = np.stack([solution.lower_weight, solution.upper_weight], axis=1)
    return EffectiveBandStructure(
        kpoints=kpoints_a,
        energies=energies,
        weights=weights,
        distances=(
            default_distances(kpoints_a.shape[0])
            if distances is None
            else np.asarray(distances, dtype=float)
        ),
        reference_energy=float(reference_energy),
        metadata=dict(metadata or {}),
    )


def coupling_from_peaks(
    upper_energy: ArrayLike,
    lower_energy: ArrayLike,
    upper_weight: ArrayLike,
    lower_weight: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Invert the two-level problem: read the perturbation off the picture.

    Given the two peak energies and their unfolded weights, returns
    ``(abs_coupling, main_energy, partner_energy)`` -- the modulus of the matrix
    element and the two *unperturbed* primitive energies it mixed.  The
    inversion is exact,

    ``|v| = dE * sqrt(w+ w-)``  and  ``d = (dE / 2) (w+ - w-)``,

    with ``dE`` the splitting (``UnfoldLab.Shadow.abs_coupling_eq``,
    ``UnfoldLab.Shadow.detuning_eq``).  The weights are renormalized to sum to
    one first, so that a pair carrying only part of the k-point's weight is
    still interpretable -- :func:`diagnose_shadow_bands` reports how much was
    left out.
    """

    upper_e = np.asarray(upper_energy, dtype=float)
    lower_e = np.asarray(lower_energy, dtype=float)
    upper_w = np.asarray(upper_weight, dtype=float)
    lower_w = np.asarray(lower_weight, dtype=float)
    upper_e, lower_e, upper_w, lower_w = np.broadcast_arrays(upper_e, lower_e, upper_w, lower_w)
    if np.any(upper_e < lower_e):
        raise ValueError("the upper peak must not lie below the lower one")
    if np.any(upper_w < 0.0) or np.any(lower_w < 0.0):
        raise ValueError("weights must be non-negative")

    total = upper_w + lower_w
    if np.any(total <= 0.0):
        raise ValueError("a pair of peaks must carry some weight")
    fraction_upper = upper_w / total
    fraction_lower = lower_w / total

    splitting = upper_e - lower_e
    abs_coupling = splitting * np.sqrt(fraction_upper * fraction_lower)
    mean = 0.5 * (upper_e + lower_e)
    detuning = 0.5 * splitting * (fraction_upper - fraction_lower)
    return (
        np.array(abs_coupling, dtype=float),
        np.array(mean + detuning, dtype=float),
        np.array(mean - detuning, dtype=float),
    )


@dataclass(frozen=True)
class ShadowDiagnosis:
    """Two-level inversion of the strongest pair of peaks at each k-point."""

    distances: NDArray[np.float64]
    main_energies: NDArray[np.float64]
    shadow_energies: NDArray[np.float64]
    main_weights: NDArray[np.float64]
    shadow_weights: NDArray[np.float64]
    couplings: NDArray[np.float64]
    residual_weights: NDArray[np.float64]

    @property
    def pair_fraction(self) -> NDArray[np.float64]:
        """Fraction of each k-point's weight held by the pair."""

        pair = self.main_weights + self.shadow_weights
        total = pair + self.residual_weights
        return np.divide(pair, total, out=np.ones_like(pair), where=total > 0.0)

    def coupling_estimate(self, *, minimum_pair_fraction: float = 0.9) -> float:
        """Weight-averaged ``|coupling|`` over the k-points the model fits.

        k-points where the two strongest states hold less than
        ``minimum_pair_fraction`` of the weight are excluded: there the
        two-level picture is not the whole story and its inversion means
        nothing.  ``nan`` is returned if no k-point qualifies.
        """

        keep = self.pair_fraction >= float(minimum_pair_fraction)
        if not np.any(keep):
            return float("nan")
        return float(np.mean(self.couplings[keep]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "distances": self.distances.tolist(),
            "main_energies": self.main_energies.tolist(),
            "shadow_energies": self.shadow_energies.tolist(),
            "main_weights": self.main_weights.tolist(),
            "shadow_weights": self.shadow_weights.tolist(),
            "couplings": self.couplings.tolist(),
            "residual_weights": self.residual_weights.tolist(),
            "pair_fraction": self.pair_fraction.tolist(),
        }


def diagnose_shadow_bands(
    structure: EffectiveBandStructure,
    *,
    energy_window: float | None = None,
) -> ShadowDiagnosis:
    """Estimate the superlattice coupling from an unfolded band structure.

    At each k-point the two states with the largest unfolded weight are taken to
    be a main band and its shadow, and :func:`coupling_from_peaks` is applied to
    them.  The weight left in every other state is reported as
    ``residual_weights``: the two-level inversion is only meaningful where that
    residual is small, which :meth:`ShadowDiagnosis.coupling_estimate` enforces.

    ``energy_window`` restricts the search to states within that distance of the
    reference energy, which is how one isolates the pair straddling a
    charge-density-wave gap at the Fermi level.

    The pair is chosen by weight, which is a heuristic and not a theorem: a
    bright state that is *not* the shadow partner will displace the real one,
    and the residual will not reveal it because the intruder is inside the pair.
    An energy window around the gap of interest is the defence.
    """

    energies = structure.shifted_energies()
    weights = np.asarray(structure.weights, dtype=float)
    if structure.n_bands < 2:
        raise ValueError("a shadow-band pair needs at least two states per k-point")

    usable = np.ones(energies.shape, dtype=bool)
    if energy_window is not None:
        if energy_window <= 0.0:
            raise ValueError("energy_window must be positive")
        usable = np.abs(energies) <= float(energy_window)
        if np.any(usable.sum(axis=1) < 2):
            raise ValueError("some k-point has fewer than two states in the energy window")

    masked = np.where(usable, weights, -np.inf)
    order = np.argsort(masked, axis=1)[:, ::-1]
    first = order[:, 0]
    second = order[:, 1]
    rows = np.arange(energies.shape[0])

    e_a = energies[rows, first]
    e_b = energies[rows, second]
    w_a = weights[rows, first]
    w_b = weights[rows, second]
    swap = e_a < e_b
    upper_e = np.where(swap, e_b, e_a)
    lower_e = np.where(swap, e_a, e_b)
    upper_w = np.where(swap, w_b, w_a)
    lower_w = np.where(swap, w_a, w_b)

    couplings, main_e, partner_e = coupling_from_peaks(upper_e, lower_e, upper_w, lower_w)
    del main_e, partner_e

    total = weights.sum(axis=1)
    residual = total - (upper_w + lower_w)

    brighter_is_upper = upper_w >= lower_w
    main_energies = np.where(brighter_is_upper, upper_e, lower_e)
    shadow_energies = np.where(brighter_is_upper, lower_e, upper_e)
    main_weights = np.where(brighter_is_upper, upper_w, lower_w)
    shadow_weights = np.where(brighter_is_upper, lower_w, upper_w)

    return ShadowDiagnosis(
        distances=np.asarray(structure.distances, dtype=float),
        main_energies=np.array(main_energies + structure.reference_energy, dtype=float),
        shadow_energies=np.array(shadow_energies + structure.reference_energy, dtype=float),
        main_weights=np.array(main_weights, dtype=float),
        shadow_weights=np.array(shadow_weights, dtype=float),
        couplings=np.array(couplings, dtype=float),
        residual_weights=np.array(np.maximum(residual, 0.0), dtype=float),
    )
