"""Reading a dispersion off an unfolded spectral function.

An unfolded band structure is a spectral function ``A(k, E)``, not a set of
bands: the weight of a k-point is spread over many supercell eigenvalues and
then broadened.  Everything one wants to *quote* -- a band energy, a gap, an
effective mass -- comes from locating peaks of ``A(k, .)`` and differentiating
their positions, and both steps have systematic errors.

The statements behind this module are in
``RequestProject/Unfolding/Dispersion.lean``:

``parabolic_vertex_exact``
    the three-point parabolic refinement of a grid peak is exact when the peak
    is locally quadratic;
``central_second_difference_cubic``
    the symmetric second difference reproduces the second derivative of any
    cubic exactly, so the leading error of an effective mass is quartic in the
    k-spacing -- a one-sided difference has no such property;
``gaussPair_merged`` / ``gaussPair_resolved``
    two states closer than the broadening width ``sigma`` produce a *single*
    peak, and it sits between them where no state is; states separated by
    ``2 sigma`` or more are genuinely resolved.  The width is therefore a hard
    resolution limit, and :func:`unresolvable_pairs` reports where a peak table
    is about to claim more than the broadening allows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.spectral import BroadeningKind, EffectiveBandStructure

#: ``hbar^2 / m_e`` in eV * Angstrom^2.  With energies in eV and a path
#: abscissa in inverse Angstrom, ``m* / m_e = HBAR_SQ_OVER_ME / (d2E/dk2)``.
HBAR_SQ_OVER_ME = 7.6199682464


def parabolic_vertex(x0: float, h: float, yl: float, y0: float, yr: float) -> tuple[float, float]:
    """Refine a grid extremum by the parabola through three samples.

    ``yl``, ``y0``, ``yr`` are the values at ``x0 - h``, ``x0`` and ``x0 + h``.
    Returns the position and the value of the vertex.  The result is exact for
    a quadratic (``parabolic_vertex_exact``); for a flat triple, where no
    parabola is determined, the central sample is returned unchanged.
    """

    if h == 0.0:
        raise ValueError("h must be nonzero")
    curvature = yl - 2.0 * y0 + yr
    if curvature == 0.0:
        return float(x0), float(y0)
    offset = 0.5 * (yl - yr) / curvature
    return float(x0 + h * offset), float(y0 - 0.125 * (yl - yr) ** 2 / curvature)


@dataclass(frozen=True)
class SpectralPeak:
    """One maximum of ``A(k, .)`` at a single k-point."""

    kpoint_index: int
    distance: float
    energy: float
    intensity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kpoint_index": self.kpoint_index,
            "distance": self.distance,
            "energy": self.energy,
            "intensity": self.intensity,
        }


def spectral_peaks(
    structure: EffectiveBandStructure,
    energy_grid: ArrayLike,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
    min_intensity: float = 0.0,
    refine: bool = True,
) -> list[SpectralPeak]:
    """Local maxima of the broadened spectral function, k-point by k-point.

    A sample is a peak when it is strictly greater than its left neighbour and
    not less than its right one, which picks exactly one representative of a
    plateau.  End points are never peaks: an extremum at the edge of the energy
    window is not resolved by the window.  With ``refine`` the position is
    improved by :func:`parabolic_vertex`, which is exact for a locally
    quadratic peak and always stays inside the sampled interval.
    """

    grid = np.asarray(energy_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("energy_grid must be one-dimensional with at least three points")
    spacing = np.diff(grid)
    if not np.all(spacing > 0.0):
        raise ValueError("energy_grid must be strictly increasing")
    uniform = bool(np.allclose(spacing, spacing[0]))

    intensity = structure.spectral_function(grid, broadening=broadening, kind=kind)
    peaks: list[SpectralPeak] = []
    for index in range(structure.n_kpoints):
        row = intensity[index]
        interior = row[1:-1]
        is_peak = (interior > row[:-2]) & (interior >= row[2:])
        is_peak &= interior > min_intensity
        for offset in np.flatnonzero(is_peak):
            position = int(offset) + 1
            energy = float(grid[position])
            value = float(row[position])
            if refine and uniform:
                energy, value = parabolic_vertex(
                    energy,
                    float(spacing[0]),
                    float(row[position - 1]),
                    value,
                    float(row[position + 1]),
                )
            peaks.append(
                SpectralPeak(
                    kpoint_index=index,
                    distance=float(structure.distances[index]),
                    energy=energy,
                    intensity=value,
                )
            )
    return peaks


@dataclass(frozen=True)
class UnresolvedPair:
    """Two states of one k-point that the broadening merges into one peak."""

    kpoint_index: int
    lower_band: int
    upper_band: int
    separation: float
    width: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kpoint_index": self.kpoint_index,
            "lower_band": self.lower_band,
            "upper_band": self.upper_band,
            "separation": self.separation,
            "width": self.width,
        }


def unresolvable_pairs(
    structure: EffectiveBandStructure,
    *,
    broadening: float | ArrayLike,
    weight_threshold: float = 1e-6,
) -> list[UnresolvedPair]:
    """States a peak table cannot separate at the given broadening.

    Two states carrying weight and lying within one broadening width of each
    other appear as a single peak, at an energy where neither of them is
    (``gaussPair_merged``); at twice the width they are resolved
    (``gaussPair_resolved``).  Only adjacent-in-energy pairs are reported --
    if a chain of states is unresolvable, every link of it appears.
    """

    if weight_threshold < 0.0:
        raise ValueError("weight_threshold must be non-negative")
    energies = structure.shifted_energies()
    widths = np.asarray(broadening, dtype=float)
    widths = np.broadcast_to(widths, energies.shape)
    if not np.all(widths > 0.0):
        raise ValueError("broadening must be positive")

    pairs: list[UnresolvedPair] = []
    for index in range(structure.n_kpoints):
        carried = np.flatnonzero(structure.weights[index] > weight_threshold)
        if carried.size < 2:
            continue
        order = carried[np.argsort(energies[index, carried], kind="stable")]
        for lower, upper in zip(order[:-1], order[1:], strict=False):
            separation = float(energies[index, upper] - energies[index, lower])
            width = float(max(widths[index, lower], widths[index, upper]))
            if separation <= width:
                pairs.append(
                    UnresolvedPair(
                        kpoint_index=index,
                        lower_band=int(lower),
                        upper_band=int(upper),
                        separation=separation,
                        width=width,
                    )
                )
    return pairs


def second_derivative(
    abscissa: ArrayLike,
    values: ArrayLike,
    index: int,
) -> float:
    """Three-point second derivative at an interior sample.

    Equal spacing gives the usual symmetric second difference, which is exact
    for any cubic (``central_second_difference_cubic``).  Unequal spacing uses
    the corresponding non-uniform stencil, which is exact for a quadratic.
    """

    x = np.asarray(abscissa, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("abscissa and values must be one-dimensional of the same length")
    if not 0 < index < x.size - 1:
        raise ValueError("index must be an interior sample")
    back = float(x[index] - x[index - 1])
    forward = float(x[index + 1] - x[index])
    if back <= 0.0 or forward <= 0.0:
        raise ValueError("abscissa must be strictly increasing")
    numerator = (
        float(y[index - 1]) * forward
        - float(y[index]) * (back + forward)
        + float(y[index + 1]) * back
    )
    return 2.0 * numerator / (back * forward * (back + forward))


def effective_mass(
    abscissa: ArrayLike,
    energies: ArrayLike,
    index: int,
) -> float:
    """Band mass ``m* / m_e`` at an interior point of a band.

    ``abscissa`` is the path coordinate in inverse Angstrom -- a *Cartesian*
    one, which is what :func:`unmochan.cartesian_path_distances` produces;
    fractional abscissae give a meaningless mass for any non-cubic cell.
    ``energies`` are in eV.  The sign is kept: a band maximum gives a negative
    mass, i.e. a hole.  A point of zero curvature has no finite mass and
    returns ``inf`` with the sign of the curvature's limit, here ``+inf``.
    """

    curvature = second_derivative(abscissa, energies, index)
    if curvature == 0.0:
        return float("inf")
    return HBAR_SQ_OVER_ME / curvature


@dataclass(frozen=True)
class BandExtremum:
    """An extremum of a band along the path, with its curvature mass."""

    index: int
    distance: float
    energy: float
    effective_mass: float
    is_maximum: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "distance": self.distance,
            "energy": self.energy,
            "effective_mass": self.effective_mass,
            "is_maximum": self.is_maximum,
        }


def band_extremum(
    abscissa: ArrayLike,
    energies: ArrayLike,
    *,
    maximum: bool = False,
) -> BandExtremum:
    """Locate the extremum of a sampled band and its effective mass.

    The extremal *sample* is found first and then refined by the parabola
    through it and its neighbours, which is exact for a parabolic band.  An
    extremum at either end of the path is reported at that sample, without
    refinement and without a mass: the band is not sampled on both sides there.
    """

    x = np.asarray(abscissa, dtype=float)
    y = np.asarray(energies, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("abscissa and energies must be one-dimensional of the same length")
    if x.size < 3:
        raise ValueError("at least three samples are needed")
    index = int(np.argmax(y) if maximum else np.argmin(y))
    if index == 0 or index == x.size - 1:
        return BandExtremum(
            index=index,
            distance=float(x[index]),
            energy=float(y[index]),
            effective_mass=float("nan"),
            is_maximum=bool(maximum),
        )
    spacing = float(x[index + 1] - x[index])
    back = float(x[index] - x[index - 1])
    if np.isclose(spacing, back):
        position, value = parabolic_vertex(
            float(x[index]), spacing, float(y[index - 1]), float(y[index]), float(y[index + 1])
        )
    else:
        position, value = float(x[index]), float(y[index])
    return BandExtremum(
        index=index,
        distance=position,
        energy=value,
        effective_mass=effective_mass(x, y, index),
        is_maximum=bool(maximum),
    )


def peaks_to_arrays(peaks: list[SpectralPeak]) -> dict[str, NDArray[np.float64]]:
    """Columnar view of a peak list, for plotting or writing to a table."""

    return {
        "kpoint_index": np.array([peak.kpoint_index for peak in peaks], dtype=float),
        "distance": np.array([peak.distance for peak in peaks], dtype=float),
        "energy": np.array([peak.energy for peak in peaks], dtype=float),
        "intensity": np.array([peak.intensity for peak in peaks], dtype=float),
    }
