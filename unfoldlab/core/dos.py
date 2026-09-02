"""The unfolded density of states, and the check it makes possible.

A heat map of ``A(k, E)`` is a *k-resolved* object, and almost every diagnostic
the package offers is local to one k-point: the fiber sum rule, the stored-state
norms, the weight bounds.  Integrating over the Brillouin zone gives the density
of states, which is the one curve an unfolded run can be compared against
*without* referring to the unfolding at all -- the supercell calculation already
has a density of states, computed from its own eigenvalues, and unfolding must
reproduce it.

Concretely, for a complete fiber the weights of a band sum to one, so summing
the unfolded spectral functions over the fiber returns the supercell spectrum of
that k-point.  An unfolded band structure is indexed by *primitive* k-points --
the ``|det T|`` members of each fiber share the zone weight of the supercell
k-point they came from -- so averaging it over its own k-points already gives
the density of states **per primitive cell**, which is the supercell one divided
by ``|det T|``.  That is the same multiplicity the electron count is divided by
(:mod:`unfoldlab.core.fermi`), and it is why ``multiplicity`` enters here only
as the number the result is *checked against*, never as a rescaling:
:func:`diagnose_dos_conservation` compares ``|det T|`` times the unfolded curve
with the unweighted one, and the unfolded state count with ``n_bands /
|det T|``.

``RequestProject/Unfolding/Dos.lean`` proves the statements used here:

``UnfoldLab.IsFiberRepr.fiberDos_eq_bandDos``
    summing over a complete fiber returns the supercell density of states --
    unfolding conserves it;
``UnfoldLab.IsFiberRepr.fiberDos_div_card``
    averaged over the fiber -- per primitive cell -- it is that curve divided by
    ``|det T|``;
``UnfoldLab.integral_bandDos`` / ``UnfoldLab.IsFiberRepr.integral_fiberDos``
    both integrate to the number of bands, so the unfolded curve integrates to
    the number of bands per primitive cell, whatever the broadening;
``UnfoldLab.IsFiberRepr.fiberDos_subset_le`` / ``UnfoldLab.meshDos_le``
    an incomplete fiber -- what a band path samples -- can only undercount, so
    an unfolded density of states *above* the supercell one is a genuine
    violation and is reported as ``max_excess``.

The integrated density of states is evaluated in closed form (an error function
for the Gaussian kernel, an arctangent for the Lorentzian), like the energy
windows of :mod:`unfoldlab.core.windows`, so the state count it reports does not
depend on the resolution of the energy grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.spectral import (
    BroadeningKind,
    EffectiveBandStructure,
    normalized_kpoint_weights,
)
from unfoldlab.core.windows import broadening_table, cumulative_kernel

__all__ = [
    "DensityOfStates",
    "DosConservation",
    "default_energy_grid",
    "diagnose_dos_conservation",
    "supercell_dos",
    "unfolded_dos",
]


@dataclass(frozen=True)
class DensityOfStates:
    """A density of states and its running state count.

    ``dos`` is ``g(E)`` on the ``energies`` grid and ``integrated`` is the
    number of states below each grid energy, evaluated in closed form rather
    than by quadrature on the grid.  ``multiplicity`` records the ``|det T|``
    the curve is expected to be consistent with; it does not scale anything.

    ``states`` is the exact total, ``sum_k w_k sum_n weight_kn``,
    computed from the weights and not from the grid, so it is what the curve
    *should* integrate to; ``tail_loss`` is how much of it falls outside the
    grid.  For a Gaussian that is a rounding error at the default padding, for a
    Lorentzian it is not: the algebraic tail keeps a percent-level fraction of
    every state arbitrarily far from the peak.
    """

    energies: NDArray[np.float64]
    dos: NDArray[np.float64]
    integrated: NDArray[np.float64]
    states: float = 0.0
    multiplicity: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies, dtype=float)
        dos = np.asarray(self.dos, dtype=float)
        integrated = np.asarray(self.integrated, dtype=float)
        if energies.ndim != 1:
            raise ValueError("energies must be one-dimensional")
        if dos.shape != energies.shape or integrated.shape != energies.shape:
            raise ValueError("dos and integrated must have the same shape as energies")
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "dos", dos)
        object.__setattr__(self, "integrated", integrated)
        object.__setattr__(self, "states", float(self.states))
        object.__setattr__(self, "multiplicity", int(self.multiplicity))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def tail_loss(self) -> float:
        """States lying outside the energy grid, ``states - integrated[-1]``."""

        counted = float(self.integrated[-1]) if self.integrated.size else 0.0
        return self.states - counted

    def to_dict(self) -> dict[str, Any]:
        return {
            "energies": self.energies.tolist(),
            "dos": self.dos.tolist(),
            "integrated": self.integrated.tolist(),
            "states": self.states,
            "tail_loss": self.tail_loss,
            "multiplicity": self.multiplicity,
            "metadata": self.metadata,
        }


def default_energy_grid(
    structure: EffectiveBandStructure,
    *,
    broadening: float | ArrayLike,
    n_points: int = 601,
    padding: float = 5.0,
) -> NDArray[np.float64]:
    """A grid covering the spectrum plus ``padding`` broadening widths either side.

    A density of states truncated at the extreme eigenvalues loses the tails the
    broadening puts outside them, and with it the state count; five widths of
    Gaussian padding leaves less than ``3e-7`` of a state outside.  A Lorentzian
    has no such margin -- its tail is algebraic -- which is why the state count
    reported below is computed in closed form and not from this grid.
    """

    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if padding < 0.0:
        raise ValueError("padding must be non-negative")
    widths = broadening_table(structure, broadening)
    energies = structure.shifted_energies()
    if energies.size == 0:
        raise ValueError("an empty band structure has no density of states")
    margin = padding * float(widths.max())
    lo = float(energies.min()) - margin
    hi = float(energies.max()) + margin
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, int(n_points), dtype=float)


def _dos_arrays(
    structure: EffectiveBandStructure,
    grid: NDArray[np.float64],
    state_weights: NDArray[np.float64],
    kweights: NDArray[np.float64],
    widths: NDArray[np.float64],
    kind: BroadeningKind,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(dos, integrated)`` on ``grid`` for the given per-state weights.

    The kernel table is built one k-point at a time: the full
    ``(n_kpoints, n_bands, n_energies)`` array is the product of three numbers
    that are each routinely in the hundreds, and there is no reason to hold it.
    """

    energies = structure.shifted_energies()
    scale = kweights[:, None] * state_weights
    dos = np.zeros(grid.size, dtype=float)
    integrated = np.zeros(grid.size, dtype=float)
    for ik in range(structure.n_kpoints):
        width = widths[ik][:, None]
        delta = grid[None, :] - energies[ik, :, None]
        if kind == "gaussian":
            kernel = np.exp(-0.5 * (delta / width) ** 2) / (width * math.sqrt(2.0 * math.pi))
        else:
            kernel = width / math.pi / (delta**2 + width**2)
        dos += scale[ik] @ kernel
        integrated += scale[ik] @ cumulative_kernel(delta, width, kind)
    return dos, integrated


def _dos(
    structure: EffectiveBandStructure,
    energy_grid: ArrayLike | None,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind,
    kpoint_weights: ArrayLike | None,
    multiplicity: int,
    state_weights: NDArray[np.float64],
    label: str,
    n_points: int,
    padding: float,
) -> DensityOfStates:
    if kind not in ("gaussian", "lorentzian"):
        raise ValueError(f"unsupported broadening kind: {kind}")
    if multiplicity <= 0:
        raise ValueError("multiplicity must be a positive integer")
    widths = broadening_table(structure, broadening)
    kweights = normalized_kpoint_weights(structure, kpoint_weights)
    if energy_grid is None:
        grid = default_energy_grid(structure, broadening=widths, n_points=n_points, padding=padding)
    else:
        grid = np.asarray(energy_grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError("energy_grid must be one-dimensional")
        if grid.size == 0:
            raise ValueError("energy_grid must not be empty")
    dos, integrated = _dos_arrays(structure, grid, state_weights, kweights, widths, kind)
    total = float(kweights @ state_weights.sum(axis=1))
    return DensityOfStates(
        energies=grid,
        dos=dos,
        integrated=integrated,
        states=total,
        multiplicity=multiplicity,
        metadata={
            "kind": label,
            "broadening": kind,
            "n_kpoints": structure.n_kpoints,
            "n_bands": structure.n_bands,
            "reference_energy": structure.reference_energy,
        },
    )


def unfolded_dos(
    structure: EffectiveBandStructure,
    energy_grid: ArrayLike | None = None,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
    kpoint_weights: ArrayLike | None = None,
    multiplicity: int = 1,
    n_points: int = 601,
    padding: float = 5.0,
) -> DensityOfStates:
    """The unfolded density of states, per primitive cell.

    ``kpoint_weights`` are the Brillouin-zone weights of the k-points, defaulting
    to uniform and normalized to sum to one.  A high-symmetry *path* is not a
    zone sample, and the curve computed from one is a path average, not a
    density of states -- the same caveat as for the electron count in
    :mod:`unfoldlab.core.fermi`.

    The result is per *primitive* cell, because the k-points of an unfolded
    band structure are primitive ones: for a complete, defect-free unfolding on
    a mesh of full fibers it integrates to ``n_bands / |det T|``, the band count
    a primitive calculation would have had.  ``multiplicity`` is ``|det T|``,
    carried along so that :func:`diagnose_dos_conservation` and any reader of
    the metadata know what that expectation is; it does not rescale the curve.
    """

    return _dos(
        structure,
        energy_grid,
        broadening=broadening,
        kind=kind,
        kpoint_weights=kpoint_weights,
        multiplicity=multiplicity,
        state_weights=structure.weights,
        label="unfolded",
        n_points=n_points,
        padding=padding,
    )


def supercell_dos(
    structure: EffectiveBandStructure,
    energy_grid: ArrayLike | None = None,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
    kpoint_weights: ArrayLike | None = None,
    n_points: int = 601,
    padding: float = 5.0,
) -> DensityOfStates:
    """The density of states of the same eigenvalues *without* the weights.

    This is what the supercell calculation itself would report, per supercell
    cell: every band counts once.  It is the reference curve the unfolded one is checked against,
    and it never involves the unfolding, so a disagreement is a statement about
    the unfolding and not about the eigenvalues.
    """

    return _dos(
        structure,
        energy_grid,
        broadening=broadening,
        kind=kind,
        kpoint_weights=kpoint_weights,
        multiplicity=1,
        state_weights=np.ones_like(structure.energies),
        label="supercell",
        n_points=n_points,
        padding=padding,
    )


@dataclass(frozen=True)
class DosConservation:
    """How well the unfolded density of states reproduces the supercell one."""

    unfolded_states: float
    supercell_states: float
    expected_states: float
    max_excess: float
    max_fiber_excess: float
    max_fiber_deficit: float
    multiplicity: int
    n_bands: int

    @property
    def states_error(self) -> float:
        """Signed error of the unfolded state count against ``n_bands / |det T|``."""

        return self.unfolded_states - self.expected_states

    def is_bounded(self, *, atol: float = 1e-6) -> bool:
        """True when no weight exceeds one, the check that holds on any sample."""

        return self.max_excess <= atol

    def is_conserved(self, *, atol: float = 1e-6) -> bool:
        """True when the state count matches too -- for a complete-fiber mesh."""

        return self.is_bounded(atol=atol) and abs(self.states_error) <= atol

    def to_dict(self) -> dict[str, Any]:
        return {
            "unfolded_states": self.unfolded_states,
            "supercell_states": self.supercell_states,
            "expected_states": self.expected_states,
            "states_error": self.states_error,
            "max_excess": self.max_excess,
            "max_fiber_excess": self.max_fiber_excess,
            "max_fiber_deficit": self.max_fiber_deficit,
            "multiplicity": self.multiplicity,
            "n_bands": self.n_bands,
        }


def diagnose_dos_conservation(
    structure: EffectiveBandStructure,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
    kpoint_weights: ArrayLike | None = None,
    multiplicity: int = 1,
    energy_grid: ArrayLike | None = None,
    n_points: int = 601,
    padding: float = 5.0,
) -> DosConservation:
    """Compare the unfolded density of states with the supercell one.

    Three numbers, of two different strengths.

    ``max_excess``
        the largest amount by which the unfolded curve exceeds the *unweighted*
        one anywhere on the grid.  By ``UnfoldLab.meshDos_le`` this cannot be
        positive for weights in ``[0, 1]``, whatever the k-point sample, so a
        positive value is a bug and not an incomplete fiber.  This is the check
        that is meaningful on a band path.
    ``states_error``
        the unfolded curve integrates to ``sum_k w_k sum_n weight_kn``, which
        for a mesh of complete fibers and normalized states is
        ``n_bands / |det T|``.  A deficit means weight is missing -- an
        incomplete fiber, a truncated plane-wave set, a pseudo-wavefunction
        norm below one (see ``unfoldlab norms``).
    ``max_fiber_excess`` / ``max_fiber_deficit``
        the two-sided comparison of ``|det T|`` times the unfolded curve with
        the supercell one, which is an *equality* only when every fiber in the
        sample is complete (``UnfoldLab.IsFiberRepr.fiberDos_eq_bandDos``).  On
        an incomplete sample both are positive and neither is a fault; they are
        reported because on a full mesh they localize in energy whatever
        ``states_error`` reports as a total.
    """

    unfolded = unfolded_dos(
        structure,
        energy_grid,
        broadening=broadening,
        kind=kind,
        kpoint_weights=kpoint_weights,
        multiplicity=multiplicity,
        n_points=n_points,
        padding=padding,
    )
    reference = supercell_dos(
        structure,
        unfolded.energies,
        broadening=broadening,
        kind=kind,
        kpoint_weights=kpoint_weights,
    )
    excess = unfolded.dos - reference.dos
    fiber = unfolded.dos * float(multiplicity) - reference.dos
    return DosConservation(
        unfolded_states=unfolded.states,
        supercell_states=reference.states,
        expected_states=structure.n_bands / float(multiplicity),
        max_excess=float(excess.max()) if excess.size else 0.0,
        max_fiber_excess=float(fiber.max()) if fiber.size else 0.0,
        max_fiber_deficit=float((-fiber).max()) if fiber.size else 0.0,
        multiplicity=int(multiplicity),
        n_bands=structure.n_bands,
    )
