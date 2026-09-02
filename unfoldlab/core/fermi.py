"""The Fermi level of an unfolded band structure, and the energy reference.

Every number one quotes from an unfolded band structure -- a gap, an occupied
weight, a Fermi surface -- is measured from an energy reference, and the
reference is fixed by the electron count.  Getting it wrong shifts the whole
picture rigidly, and nothing in the plot reveals it.

Two traps are specific to unfolding:

*The electron count is per primitive cell, the calculation is per supercell.*
A supercell of multiplicity ``|det T|`` holds ``|det T|`` times as many
electrons, so the Fermi level of the *unfolded* structure must be sought against
``n_supercell / |det T|`` electrons.  :func:`electrons_per_primitive_cell` does
that division and refuses a count that is not consistent with it.

*A band path is not a Brillouin-zone sample.*  An electron count is an integral
over the zone; summing over the k-points of a high-symmetry path with equal
weights is not that integral.  :func:`find_fermi_level` therefore takes explicit
``kpoint_weights`` and its docstring says what they have to be.

``RequestProject/Unfolding/Fermi.lean`` proves the statements this module relies
on:

``existsUnique_fermiLevel``
    for an electron count strictly between zero and the total weight there is
    *exactly one* Fermi level -- so the bisection below converges to a
    well-defined number, and any two implementations must agree;
``filling_strictMono``, ``tendsto_filling_atBot`` / ``atTop``
    strict monotonicity and the two limits, which are what make the bracket
    expansion terminate;
``filling_shift`` / ``fermiLevel_shift``
    changing the energy reference shifts the Fermi level by the same amount and
    changes no occupation -- the misreferenced run is wrong by exactly that
    shift, and by nothing else;
``filling_add_empty``
    occupied and empty weights add to the total;
``abs_filling_sub_le``
    the filling is Lipschitz with constant ``(sum w) / (4 T)``, so a colder
    smearing makes the root sharper but the bracket more delicate.

Only Fermi--Dirac smearing is offered, because that is the occupation function
the formal development covers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.spectral import EffectiveBandStructure, normalized_kpoint_weights

#: Absolute tolerance on the electron count at which the search stops.
DEFAULT_ELECTRON_TOLERANCE = 1e-10

#: Hard cap on the bisection iterations; 200 halvings exhaust double precision.
MAX_BISECTION_ITERATIONS = 200


def fermi_dirac_occupation(reduced: ArrayLike) -> NDArray[np.float64]:
    """Fermi--Dirac occupation ``1 / (1 + exp(x))`` of a reduced energy ``x``.

    Evaluated as ``exp(-x) / (1 + exp(-x))`` where ``x >= 0``, which is the same
    number without the overflow: the naive form returns ``0`` from an ``inf``
    for ``x`` above about 709, and warns while doing it.
    """

    array = np.asarray(reduced, dtype=float)
    out = np.empty_like(array)
    positive = array >= 0.0
    exp_neg = np.exp(-array[positive])
    out[positive] = exp_neg / (1.0 + exp_neg)
    out[~positive] = 1.0 / (1.0 + np.exp(array[~positive]))
    return out


def state_occupations(
    structure: EffectiveBandStructure,
    fermi: float,
    *,
    temperature: float,
) -> NDArray[np.float64]:
    """Occupation of every state, shape ``(n_kpoints, n_bands)``.

    ``temperature`` is the smearing width in the energy unit of the structure
    (``kB T``, not a temperature in kelvin).
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive; a zero-width step is not smooth")
    return fermi_dirac_occupation((structure.shifted_energies() - fermi) / temperature)


def total_weight(
    structure: EffectiveBandStructure,
    *,
    kpoint_weights: ArrayLike | None = None,
    spin_degeneracy: float = 1.0,
) -> float:
    """The electron count at infinite chemical potential: every state occupied.

    This is the upper bound in ``existsUnique_fermiLevel``.  For a complete
    unfolded run it equals ``spin_degeneracy`` times the number of primitive
    bands the supercell calculation covers, by the fiber sum rule.
    """

    if spin_degeneracy <= 0.0:
        raise ValueError("spin_degeneracy must be positive")
    kweights = normalized_kpoint_weights(structure, kpoint_weights)
    return float(spin_degeneracy * (kweights @ structure.weights.sum(axis=1)))


def electron_count(
    structure: EffectiveBandStructure,
    fermi: float,
    *,
    temperature: float,
    kpoint_weights: ArrayLike | None = None,
    spin_degeneracy: float = 1.0,
) -> float:
    """Occupied unfolded weight per primitive cell at chemical potential ``fermi``.

    The k-points must sample the Brillouin zone -- ``kpoint_weights`` are the
    zone weights, normalized here, and default to uniform.  A high-symmetry
    *path* is not such a sample, and the number this returns for one is not an
    electron count.
    """

    if spin_degeneracy <= 0.0:
        raise ValueError("spin_degeneracy must be positive")
    kweights = normalized_kpoint_weights(structure, kpoint_weights)
    occupations = state_occupations(structure, fermi, temperature=temperature)
    per_kpoint = (structure.weights * occupations).sum(axis=1)
    return float(spin_degeneracy * (kweights @ per_kpoint))


def electrons_per_primitive_cell(
    supercell_electrons: float,
    multiplicity: int,
    *,
    atol: float = 1e-9,
) -> float:
    """Divide a supercell electron count by ``|det T|``.

    An unfolded band structure lives in the primitive zone, so the Fermi level
    must be sought against the primitive-cell electron count.  A supercell count
    that is not a multiple of the multiplicity means the two cells do not
    describe the same material -- a vacancy, a substitution, a charge state --
    and the caller has to decide what the reference should be, so it is an
    error here rather than a silent fraction.
    """

    if multiplicity <= 0:
        raise ValueError("multiplicity must be a positive integer")
    quotient = supercell_electrons / multiplicity
    if abs(quotient - round(quotient)) > atol:
        raise ValueError(
            f"{supercell_electrons:g} electrons in a supercell of multiplicity "
            f"{multiplicity} is {quotient:g} per primitive cell, not an integer: the "
            "supercell is not a defect-free repetition of the primitive cell, and the "
            "electron count to unfold against has to be supplied explicitly"
        )
    return float(quotient)


@dataclass(frozen=True)
class FermiLevelReport:
    """Where the Fermi level is, and how well it is determined."""

    fermi: float
    electrons: float
    residual: float
    iterations: int
    bracket: tuple[float, float]
    total_weight: float
    temperature: float

    @property
    def converged(self) -> bool:
        return abs(self.residual) <= DEFAULT_ELECTRON_TOLERANCE

    @property
    def sensitivity(self) -> float:
        """Lipschitz constant ``(sum w) / (4 T)`` of the electron count.

        The reciprocal is the energy error a one-electron miscount corresponds
        to at worst, so a small value means a badly determined Fermi level.
        """

        return self.total_weight / (4.0 * self.temperature)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fermi": self.fermi,
            "electrons": self.electrons,
            "residual": self.residual,
            "iterations": self.iterations,
            "bracket": list(self.bracket),
            "total_weight": self.total_weight,
            "temperature": self.temperature,
            "sensitivity": self.sensitivity,
            "converged": self.converged,
        }

    def summary(self) -> str:
        return (
            f"Fermi level {self.fermi:.6f} for {self.electrons:g} electrons "
            f"(residual {self.residual:+.2e} after {self.iterations} bisections, "
            f"bracket [{self.bracket[0]:.6f}, {self.bracket[1]:.6f}], "
            f"d(count)/dE at most {self.sensitivity:.3e})"
        )


def find_fermi_level(
    structure: EffectiveBandStructure,
    electrons: float,
    *,
    temperature: float,
    kpoint_weights: ArrayLike | None = None,
    spin_degeneracy: float = 1.0,
    tolerance: float = DEFAULT_ELECTRON_TOLERANCE,
) -> FermiLevelReport:
    """Solve ``electron_count(structure, E) = electrons`` for ``E``.

    The count runs strictly monotonically from ``0`` to
    :func:`total_weight` (`filling_strictMono`, `tendsto_filling_atBot`,
    `tendsto_filling_atTop`), so a bracket always exists, bisection always
    converges, and the root is unique (`existsUnique_fermiLevel`).  The initial
    bracket is the energy range of the structure widened by the smearing, and it
    is doubled outwards until it straddles the target.

    An ``electrons`` outside ``(0, total_weight)`` is rejected: no chemical
    potential produces it, and returning ``+-inf`` would be worse than failing.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    capacity = total_weight(
        structure, kpoint_weights=kpoint_weights, spin_degeneracy=spin_degeneracy
    )
    if not 0.0 < electrons < capacity:
        raise ValueError(
            f"an electron count of {electrons:g} is not attainable: the states carry a "
            f"total weight of {capacity:g}, and the count runs strictly between 0 and "
            "that as the chemical potential sweeps the real line"
        )

    def count(energy: float) -> float:
        return electron_count(
            structure,
            energy,
            temperature=temperature,
            kpoint_weights=kpoint_weights,
            spin_degeneracy=spin_degeneracy,
        )

    energies = structure.shifted_energies()
    lower = float(energies.min()) - 10.0 * temperature
    upper = float(energies.max()) + 10.0 * temperature
    span = max(upper - lower, 10.0 * temperature)
    for _ in range(MAX_BISECTION_ITERATIONS):
        if count(lower) < electrons:
            break
        lower -= span
        span *= 2.0
    span = max(upper - lower, 10.0 * temperature)
    for _ in range(MAX_BISECTION_ITERATIONS):
        if count(upper) > electrons:
            break
        upper += span
        span *= 2.0

    bracket = (lower, upper)
    iterations = 0
    middle = 0.5 * (lower + upper)
    residual = count(middle) - electrons
    while iterations < MAX_BISECTION_ITERATIONS and abs(residual) > tolerance:
        if residual > 0.0:
            upper = middle
        else:
            lower = middle
        new_middle = 0.5 * (lower + upper)
        if new_middle == middle:
            break
        middle = new_middle
        residual = count(middle) - electrons
        iterations += 1

    return FermiLevelReport(
        fermi=middle,
        electrons=float(electrons),
        residual=float(residual),
        iterations=iterations,
        bracket=bracket,
        total_weight=capacity,
        temperature=float(temperature),
    )


def align_reference(
    structure: EffectiveBandStructure,
    fermi: float,
) -> EffectiveBandStructure:
    """Return the same structure with the reference energy moved to ``fermi``.

    ``fermi`` is measured from the *current* reference, so this composes: the
    energies do not move, only the number they are quoted against.  Occupations
    are unchanged, which is `filling_shift`; the Fermi level of the result is
    zero, which is `fermiLevel_shift`.
    """

    return replace(structure, reference_energy=structure.reference_energy + float(fermi))
