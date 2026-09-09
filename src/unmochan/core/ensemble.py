"""Configurational averaging: disorder, alloy and defect-ensemble unfolding.

A disordered supercell has no single band structure.  One unfolds several
configurations -- snapshots of an alloy, inequivalent defect placements, the
members of a special quasirandom structure -- and averages the results with the
probability of each configuration,

``<A>(k, E) = sum_i p_i A_i(k, E)``, with ``p_i >= 0`` and ``sum_i p_i = 1``.

This module implements that average and the two numbers that make it readable:
the *centre* of the averaged spectrum at each k-point and its *width*, split by
the law of total variance into the width a typical configuration already has
and the extra width caused by the disorder itself.

The corresponding statements are proved in ``RequestProject/Unfolding/Ensemble.lean``:

* ``UnfoldLab.integral_mixture_spectralFunction`` -- averaging conserves
  spectral weight, so the average is again a spectral function;
* ``UnfoldLab.mixture_nonneg`` -- and it is still non-negative;
* ``UnfoldLab.ensembleSpread_decomposition`` -- the law of total variance;
* ``UnfoldLab.le_ensembleSpread`` -- disorder never sharpens the average;
* ``UnfoldLab.ensembleSpread_const`` -- an ensemble of identical configurations
  has zero disorder broadening.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.spectral import BroadeningKind, EffectiveBandStructure

__all__ = [
    "BandMoments",
    "DisorderBroadening",
    "EnsembleSpectralFunction",
    "band_moments",
    "configuration_weights",
    "disorder_broadening",
    "ensemble_spectral_function",
    "stack_configurations",
]


def configuration_weights(weights: ArrayLike | None, n_configurations: int) -> NDArray[np.float64]:
    """Return normalized, non-negative configuration probabilities.

    ``None`` means "equally likely".  Otherwise the values must be finite and
    non-negative with a positive sum; they are divided by that sum, so relative
    multiplicities (``[2, 1, 1]`` for a configuration that occurs twice) are
    accepted directly.
    """

    if n_configurations <= 0:
        raise ValueError("an ensemble needs at least one configuration")
    if weights is None:
        return np.full(n_configurations, 1.0 / n_configurations, dtype=float)

    values = np.asarray(weights, dtype=float)
    if values.shape != (n_configurations,):
        raise ValueError(
            f"expected {n_configurations} configuration weights, got shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("configuration weights must be finite")
    if np.any(values < 0.0):
        raise ValueError("configuration weights must be non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("configuration weights must not all be zero")
    return values / total


def _check_common_kpoints(
    structures: Sequence[EffectiveBandStructure], *, kpoint_tol: float
) -> None:
    reference = structures[0]
    for index, other in enumerate(structures[1:], start=1):
        if other.n_kpoints != reference.n_kpoints:
            raise ValueError(
                "configurations must share the k-point list: configuration "
                f"{index} has {other.n_kpoints} k-points, configuration 0 has "
                f"{reference.n_kpoints}"
            )
        if not np.allclose(other.kpoints, reference.kpoints, atol=kpoint_tol, rtol=0.0):
            raise ValueError(f"configuration {index} has different k-points from configuration 0")
        if not np.allclose(other.distances, reference.distances, atol=kpoint_tol, rtol=0.0):
            raise ValueError(
                f"configuration {index} has a different path abscissa from configuration 0"
            )


def stack_configurations(
    structures: Sequence[EffectiveBandStructure],
    weights: ArrayLike | None = None,
    *,
    kpoint_tol: float = 1e-8,
    metadata: dict[str, Any] | None = None,
) -> EffectiveBandStructure:
    """Merge an ensemble into a single effective band structure.

    The states of all configurations are concatenated along the band axis and
    each configuration's weights are scaled by its probability.  The result is
    the mixture: its spectral function is exactly ``sum_i p_i A_i`` for any
    broadening, and its sum rule is the average of the configurations' sum
    rules.

    Energies are stacked *relative to each configuration's own reference*
    (``shifted_energies()``), because an alloy snapshot has its own Fermi level;
    the merged structure therefore has ``reference_energy = 0``.

    All configurations must share the k-point list and the path abscissa; the
    number of bands may differ between them.
    """

    if not structures:
        raise ValueError("an ensemble needs at least one configuration")
    probabilities = configuration_weights(weights, len(structures))
    _check_common_kpoints(structures, kpoint_tol=kpoint_tol)

    energies = np.concatenate([ebs.shifted_energies() for ebs in structures], axis=1)
    scaled = np.concatenate(
        [p * ebs.weights for p, ebs in zip(probabilities, structures, strict=True)], axis=1
    )
    merged: dict[str, Any] = {
        "ensemble": {
            "n_configurations": len(structures),
            "configuration_weights": probabilities.tolist(),
            "bands_per_configuration": [ebs.n_bands for ebs in structures],
        }
    }
    if metadata:
        merged.update(metadata)
    return EffectiveBandStructure(
        kpoints=structures[0].kpoints,
        energies=energies,
        weights=scaled,
        distances=structures[0].distances,
        reference_energy=0.0,
        metadata=merged,
    )


@dataclass(frozen=True)
class EnsembleSpectralFunction:
    """The configurational average of ``A(k, E)`` and its scatter."""

    energy_grid: NDArray[np.float64]
    mean: NDArray[np.float64]
    variance: NDArray[np.float64]
    probabilities: NDArray[np.float64]

    @property
    def standard_deviation(self) -> NDArray[np.float64]:
        """Configuration-to-configuration scatter of each pixel of the map."""

        return np.sqrt(np.maximum(self.variance, 0.0))


def ensemble_spectral_function(
    structures: Sequence[EffectiveBandStructure],
    energy_grid: ArrayLike,
    *,
    broadening: float | ArrayLike | Sequence[float | ArrayLike],
    kind: BroadeningKind = "gaussian",
    weights: ArrayLike | None = None,
    kpoint_tol: float = 1e-8,
) -> EnsembleSpectralFunction:
    """Average the spectral functions of an ensemble of configurations.

    ``broadening`` is either one width (or per-state table) used for every
    configuration, or one entry per configuration.  Energies are measured from
    each configuration's own reference energy, exactly as in
    :func:`stack_configurations`.

    Alongside the mean the scatter ``sum_i p_i (A_i - <A>)^2`` is returned, so a
    heat map can be accompanied by an honest statement of how much the picture
    depends on the particular configurations that were sampled.
    """

    if not structures:
        raise ValueError("an ensemble needs at least one configuration")
    probabilities = configuration_weights(weights, len(structures))
    _check_common_kpoints(structures, kpoint_tol=kpoint_tol)

    grid = np.asarray(energy_grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("energy_grid must be one-dimensional")

    widths: list[float | ArrayLike]
    if isinstance(broadening, Sequence) and not isinstance(broadening, (str, bytes)):
        widths = list(broadening)
        if len(widths) != len(structures):
            raise ValueError("broadening must be a single width or one entry per configuration")
    else:
        widths = [broadening] * len(structures)

    intensities = np.stack(
        [
            ebs.spectral_function(grid, broadening=width, kind=kind)
            for ebs, width in zip(structures, widths, strict=True)
        ]
    )
    mean = np.tensordot(probabilities, intensities, axes=(0, 0))
    deviation = intensities - mean[None, :, :]
    variance = np.tensordot(probabilities, deviation**2, axes=(0, 0))
    return EnsembleSpectralFunction(
        energy_grid=grid,
        mean=mean,
        variance=variance,
        probabilities=probabilities,
    )


@dataclass(frozen=True)
class BandMoments:
    """Per-k-point moments of the weighted spectrum of one configuration.

    ``total`` is the unfolded spectral weight of the k-point, ``centre`` its
    weighted mean energy and ``spread`` the weighted variance about that mean;
    ``width`` is the square root of the spread.  A k-point with no weight has
    ``total = 0`` and, by the same ``0/0 -> 0`` convention the weights
    themselves use, ``centre = spread = 0``.
    """

    total: NDArray[np.float64]
    centre: NDArray[np.float64]
    spread: NDArray[np.float64]

    @property
    def width(self) -> NDArray[np.float64]:
        return np.sqrt(np.maximum(self.spread, 0.0))


def band_moments(structure: EffectiveBandStructure) -> BandMoments:
    """Zeroth, first and second moments of ``A(k, .)`` at each k-point.

    The moments are taken of the *discrete* weighted spectrum, not of a
    broadened map, so they do not depend on a choice of kernel or energy grid.
    Broadening with a normalized kernel of width ``w`` would add ``w^2`` to the
    spread (and nothing to the centre), which is precisely the amount of
    structure a plot invents.
    """

    energies = structure.shifted_energies()
    weights = np.asarray(structure.weights, dtype=float)
    total = weights.sum(axis=1)
    safe = np.where(total > 0.0, total, 1.0)
    centre = np.where(total > 0.0, (weights * energies).sum(axis=1) / safe, 0.0)
    deviation = energies - centre[:, None]
    spread = np.where(total > 0.0, (weights * deviation**2).sum(axis=1) / safe, 0.0)
    return BandMoments(
        total=total.astype(float, copy=False),
        centre=centre.astype(float, copy=False),
        spread=np.maximum(spread, 0.0).astype(float, copy=False),
    )


@dataclass(frozen=True)
class DisorderBroadening:
    """The law-of-total-variance split of an ensemble's width, per k-point.

    ``intrinsic`` is the probability-weighted mean of the configurations' own
    squared widths, ``disorder`` the variance of their centres, and ``total``
    the squared width of the averaged spectrum.  ``UnfoldLab.
    ensembleSpread_decomposition`` proves ``total = intrinsic + disorder``
    exactly; ``residual`` reports how far the computed arrays are from that
    identity and is a rounding-level number for any correct input.
    """

    centre: NDArray[np.float64]
    intrinsic: NDArray[np.float64]
    disorder: NDArray[np.float64]
    total: NDArray[np.float64]
    probabilities: NDArray[np.float64]

    @property
    def disorder_width(self) -> NDArray[np.float64]:
        """The disorder-induced broadening itself, in energy units."""

        return np.sqrt(np.maximum(self.disorder, 0.0))

    @property
    def residual(self) -> float:
        return float(np.max(np.abs(self.total - self.intrinsic - self.disorder)))


def disorder_broadening(
    structures: Sequence[EffectiveBandStructure],
    weights: ArrayLike | None = None,
    *,
    kpoint_tol: float = 1e-8,
) -> DisorderBroadening:
    """Split the width of a disorder-averaged band into its two causes.

    At each k-point the squared width of the averaged spectrum is

    ``total = sum_i p_i spread_i + sum_i p_i (centre_i - centre)^2``,

    the first term being the width a typical configuration already has and the
    second the spread of the configurations' band centres -- the disorder
    broadening.  The identity holds only when each configuration carries the
    same total weight at that k-point, which the unfolding sum rule guarantees;
    it is checked here and reported through :attr:`DisorderBroadening.residual`.
    """

    if not structures:
        raise ValueError("an ensemble needs at least one configuration")
    probabilities = configuration_weights(weights, len(structures))
    _check_common_kpoints(structures, kpoint_tol=kpoint_tol)

    moments = [band_moments(ebs) for ebs in structures]
    centres = np.stack([m.centre for m in moments])
    spreads = np.stack([m.spread for m in moments])

    centre = probabilities @ centres
    intrinsic = probabilities @ spreads
    disorder = probabilities @ (centres - centre[None, :]) ** 2

    merged = stack_configurations(structures, probabilities, kpoint_tol=kpoint_tol)
    merged_moments = band_moments(merged)
    return DisorderBroadening(
        centre=centre,
        intrinsic=intrinsic,
        disorder=disorder,
        total=merged_moments.spread,
        probabilities=probabilities,
    )
