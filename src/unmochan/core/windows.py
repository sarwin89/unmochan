"""Energy windows, constant-energy cuts and band filling.

A heat map of ``A(k, E)`` is rarely the last question asked of an unfolded band
structure.  One also wants

* a *constant-energy cut*: the intensity ``A(k, E0)`` as a function of k, which
  at the Fermi level is the unfolded Fermi surface and at any other energy is
  the ARPES-like cut;
* the weight inside an *energy window*, for instance the occupied weight below
  the Fermi level or the weight that leaks into a gap.

Both are computed here in closed form from the discrete weighted spectrum, so
neither depends on the resolution of an energy grid: the window weight of a
Gaussian- or Lorentzian-broadened state is an error function or an arctangent.
Infinite bounds are allowed, so ``(-inf, E_F)`` gives the band filling directly.

The properties that make these numbers meaningful are proved in
``RequestProject/Unfolding/Windows.lean``:

* ``UnfoldLab.windowWeight_add_adjacent`` -- adjacent windows add, so occupied
  and empty parts add up to the sum rule;
* ``UnfoldLab.windowWeight_nonneg`` and ``UnfoldLab.windowWeight_le_total`` -- a
  window holds between zero and all of the k-point's weight;
* ``UnfoldLab.windowWeight_mono`` -- widening a window cannot lose weight.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.spectral import BroadeningKind, EffectiveBandStructure

__all__ = [
    "band_filling",
    "broadening_table",
    "constant_energy_cut",
    "cumulative_kernel",
    "energy_window_weight",
]


_ERF = np.vectorize(math.erf, otypes=[float])


def broadening_table(
    structure: EffectiveBandStructure, broadening: float | ArrayLike
) -> NDArray[np.float64]:
    widths = np.asarray(broadening, dtype=float)
    try:
        widths = np.broadcast_to(widths, structure.energies.shape)
    except ValueError as error:
        raise ValueError(
            "broadening must be a scalar or broadcastable to "
            f"(n_kpoints, n_bands) = {structure.energies.shape}"
        ) from error
    if not np.all(widths > 0.0):
        raise ValueError("broadening must be positive")
    return np.array(widths, dtype=float, copy=True)


def constant_energy_cut(
    structure: EffectiveBandStructure,
    energy: float,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
) -> NDArray[np.float64]:
    """Return ``A(k, energy)`` for every k-point.

    This is one column of :meth:`EffectiveBandStructure.spectral_function`,
    evaluated directly rather than interpolated off a grid, so a Fermi surface
    (``energy = 0`` with the reference energy set to the Fermi level) is exact
    at the energy asked for.  ``energy`` is measured from the structure's
    reference energy, as everywhere else in the package.
    """

    if kind not in ("gaussian", "lorentzian"):
        raise ValueError(f"unsupported broadening kind: {kind}")
    widths = broadening_table(structure, broadening)
    delta = float(energy) - structure.shifted_energies()
    if kind == "gaussian":
        kernel = np.exp(-0.5 * (delta / widths) ** 2) / (widths * math.sqrt(2.0 * math.pi))
    else:
        kernel = widths / math.pi / (delta**2 + widths**2)
    return (structure.weights * kernel).sum(axis=1)


def cumulative_kernel(
    delta: NDArray[np.float64], widths: NDArray[np.float64], kind: str
) -> NDArray[np.float64]:
    """``P(E < bound)`` of a single normalized kernel, for ``delta = bound - e``."""

    if kind == "gaussian":
        return 0.5 * (1.0 + _ERF(delta / (widths * math.sqrt(2.0))))
    return 0.5 + np.arctan2(delta, widths) / math.pi


def energy_window_weight(
    structure: EffectiveBandStructure,
    lower: float,
    upper: float,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
) -> NDArray[np.float64]:
    """Unfolded weight of each k-point inside ``[lower, upper]``.

    The integral of the broadened spectral function over the window is done in
    closed form -- an error function for the Gaussian kernel, an arctangent for
    the Lorentzian -- so the result is exact and independent of any grid.  The
    bounds are measured from the reference energy and may be ``-inf``/``inf``,
    in which case the total weight of the k-point is returned.

    Applied to a nominally empty window this is the leakage a broadened plot
    invents: with a Gaussian of width ``w`` a band lying ``d`` below the gap
    edge still puts about ``exp(-d^2/2w^2)`` of its weight inside the gap, and a
    Lorentzian puts in far more.
    """

    if kind not in ("gaussian", "lorentzian"):
        raise ValueError(f"unsupported broadening kind: {kind}")
    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        raise ValueError("the upper bound of an energy window must not be below the lower")
    widths = broadening_table(structure, broadening)
    energies = structure.shifted_energies()
    below = cumulative_kernel(upper - energies, widths, kind)
    above = cumulative_kernel(lower - energies, widths, kind)
    return (structure.weights * (below - above)).sum(axis=1)


def band_filling(
    structure: EffectiveBandStructure,
    fermi: float = 0.0,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
) -> NDArray[np.float64]:
    """Unfolded weight below ``fermi`` at each k-point.

    With the reference energy set to the Fermi level the default ``fermi = 0``
    is the occupied weight, and the empty weight is the total minus this --
    ``UnfoldLab.windowWeight_add_adjacent`` is the statement that the two really
    do add up.
    """

    return energy_window_weight(structure, -np.inf, fermi, broadening=broadening, kind=kind)
