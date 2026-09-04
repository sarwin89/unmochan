"""Bloch-sum conventions for tight-binding and LCAO supercell states.

Why this exists
---------------

A tight-binding or LCAO supercell state reaches this package as a table of
amplitudes ``c[a, R]``, one per (orbital, primitive cell).  What that table
*means* depends on the Bloch-sum convention of the code that produced it, and
there are three in circulation:

``"cell"``
    the convention this package assumes: the supercell Bloch basis is
    ``|q; R, a> = sum_L exp(2 pi i q . L) |R + L, a>``, so the phase is attached
    to the *cell* and the amplitudes are those of the supercell-periodic part of
    the state.  This is the basis
    :func:`unmochan.core.tight_binding.supercell_bloch_hamiltonian` builds, so
    its eigenvectors are already in this convention.

``"atomic"``
    the basis carries the orbital position too,
    ``|q; R, a>_atomic = exp(2 pi i q . tau_a) |q; R, a>``.  Wannier90 and
    several LCAO codes use it.  Relative to ``"cell"`` this is a *per-orbital*
    phase, constant over the cells.

``"wavefunction"``
    the amplitudes of the full state rather than of its periodic part, i.e. the
    extra factor ``exp(2 pi i q . R)``.  This is not really a convention so much
    as a common mistake -- it is what one gets by sampling a real-space
    wavefunction on the sites and forgetting to divide the Bloch factor out.
    Relative to ``"cell"`` it is a *per-cell* phase, constant over the orbitals.

Which of them matter
--------------------

Formalized in ``RequestProject/Unfolding/Convention.lean``:

* ``UnfoldLab.tbWeight_gaugeOrbital`` and
  ``UnfoldLab.tbOrbitalWeight_gaugeOrbital`` — a **per-orbital** phase leaves
  the unfolding weight, and every fat band, *exactly* unchanged.  The
  ``"cell"``/``"atomic"`` distinction is therefore invisible to an
  orthonormal-basis unfolding, and needs no user action.
* ``UnfoldLab.tbWeight_gaugeShift`` — a **per-cell** phase ``exp(-2 pi i x . R)``
  does not change the weights either, it *translates the whole distribution in
  k*: the weight computed at ``k`` is the true weight at ``k + x``.  The fiber
  sum rule survives the translation, so no runtime check catches it; the
  spectral weight simply lands on the wrong primitive k-point.
  :func:`diagnose_bloch_convention` measures how much that matters for a given
  state, and :func:`convert_coefficients` repairs it.
  ``UnfoldLab.tbWeight_gaugeCell`` adds that when the shift is a difference of
  two fiber members the translation restricts to a permutation of the fiber,
  which :func:`fiber_shift_permutation` computes.
* ``UnfoldLab.ovWeight_gauge`` — in a *non-orthogonal* basis the per-orbital
  phase is a gauge only if the overlap kernel is transformed with the state,
  ``S -> diag(u) S diag(u)*``.  :func:`gauge_overlap_kernel` does that, and
  ``UnfoldLab.ovNumer_gaugeOrbital_untransformed`` is the exact expression for
  what forgetting it produces.

All k-points and orbital positions below are in **primitive fractional
coordinates**, as everywhere else in :mod:`unmochan.core.tight_binding` and
:mod:`unmochan.core.lcao`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.kpoints import fiber_kpoints
from unmochan.core.numerics import as_int_array, wrap_fractional
from unmochan.core.tight_binding import tight_binding_weights, validate_supercell_cells
from unmochan.core.transformations import TransformationMatrix

BlochConvention = Literal["cell", "atomic", "wavefunction"]

_CONVENTIONS: tuple[BlochConvention, ...] = ("cell", "atomic", "wavefunction")

__all__ = [
    "BlochConvention",
    "ConventionDiagnosis",
    "apply_cell_phase",
    "apply_orbital_phase",
    "cell_phase_factors",
    "convert_coefficients",
    "diagnose_bloch_convention",
    "fiber_shift_permutation",
    "gauge_overlap_kernel",
    "orbital_phase_factors",
]


def _check_convention(name: str, value: str) -> BlochConvention:
    if value not in _CONVENTIONS:
        raise ValueError(f"{name} must be one of {_CONVENTIONS}, got {value!r}")
    return value  # type: ignore[return-value]


def _as_kpoint(kpoint: ArrayLike, *, name: str = "kpoint") -> NDArray[np.float64]:
    point = np.asarray(kpoint, dtype=float).reshape(-1)
    if point.shape != (3,):
        raise ValueError(f"{name} must have three components")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be finite")
    return point


def _as_coefficients(coefficients: ArrayLike) -> NDArray[np.complex128]:
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_states, n_orbitals, n_cells)")
    return coeffs


def orbital_phase_factors(
    orbital_positions: ArrayLike, kpoint: ArrayLike
) -> NDArray[np.complex128]:
    """``exp(-2 pi i k . tau_a)`` for each orbital position ``tau_a``.

    This is the diagonal unitary relating the ``"cell"`` and ``"atomic"``
    conventions.  It has unit modulus entry by entry, which is the hypothesis of
    ``UnfoldLab.tbWeight_gaugeOrbital``.
    """

    positions = np.asarray(orbital_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("orbital_positions must have shape (n_orbitals, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("orbital_positions must be finite")
    point = _as_kpoint(kpoint)
    return np.exp(-2j * np.pi * (positions @ point))


def cell_phase_factors(cells: ArrayLike, kpoint: ArrayLike) -> NDArray[np.complex128]:
    """``exp(2 pi i k . R)`` for each primitive cell ``R`` of the supercell."""

    cell_array = as_int_array(np.asarray(cells), name="cells")
    if cell_array.ndim != 2 or cell_array.shape[1] != 3:
        raise ValueError("cells must have shape (n_cells, 3)")
    point = _as_kpoint(kpoint)
    return np.exp(2j * np.pi * (cell_array.astype(float) @ point))


def apply_orbital_phase(coefficients: ArrayLike, factors: ArrayLike) -> NDArray[np.complex128]:
    """Multiply a ``(n_states, n_orbitals, n_cells)`` table along the orbital axis."""

    coeffs = _as_coefficients(coefficients)
    phases = np.asarray(factors, dtype=np.complex128).reshape(-1)
    if phases.shape[0] != coeffs.shape[1]:
        raise ValueError("factors must have one entry per orbital")
    return coeffs * phases[np.newaxis, :, np.newaxis]


def apply_cell_phase(coefficients: ArrayLike, factors: ArrayLike) -> NDArray[np.complex128]:
    """Multiply a ``(n_states, n_orbitals, n_cells)`` table along the cell axis."""

    coeffs = _as_coefficients(coefficients)
    phases = np.asarray(factors, dtype=np.complex128).reshape(-1)
    if phases.shape[0] != coeffs.shape[2]:
        raise ValueError("factors must have one entry per cell")
    return coeffs * phases[np.newaxis, np.newaxis, :]


def _to_cell_convention(
    coefficients: NDArray[np.complex128],
    source: BlochConvention,
    *,
    cells: ArrayLike | None,
    kpoint: NDArray[np.float64],
    orbital_positions: ArrayLike | None,
) -> NDArray[np.complex128]:
    if source == "cell":
        return coefficients
    if source == "atomic":
        if orbital_positions is None:
            raise ValueError("the 'atomic' convention needs orbital_positions")
        # c_cell = exp(+2 pi i q . tau_a) c_atomic
        return apply_orbital_phase(
            coefficients, np.conjugate(orbital_phase_factors(orbital_positions, kpoint))
        )
    if cells is None:
        raise ValueError("the 'wavefunction' convention needs cells")
    # c_cell = exp(-2 pi i q . R) c_wavefunction
    return apply_cell_phase(coefficients, np.conjugate(cell_phase_factors(cells, kpoint)))


def _from_cell_convention(
    coefficients: NDArray[np.complex128],
    target: BlochConvention,
    *,
    cells: ArrayLike | None,
    kpoint: NDArray[np.float64],
    orbital_positions: ArrayLike | None,
) -> NDArray[np.complex128]:
    if target == "cell":
        return coefficients
    if target == "atomic":
        if orbital_positions is None:
            raise ValueError("the 'atomic' convention needs orbital_positions")
        return apply_orbital_phase(coefficients, orbital_phase_factors(orbital_positions, kpoint))
    if cells is None:
        raise ValueError("the 'wavefunction' convention needs cells")
    return apply_cell_phase(coefficients, cell_phase_factors(cells, kpoint))


def convert_coefficients(
    coefficients: ArrayLike,
    *,
    source: str,
    target: str,
    supercell_kpoint: ArrayLike,
    cells: ArrayLike | None = None,
    orbital_positions: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Rewrite a coefficient table from one Bloch-sum convention into another.

    ``supercell_kpoint`` is the k-point of the supercell state in **primitive
    fractional coordinates** (the same ``q`` that
    :func:`unmochan.core.tight_binding.supercell_bloch_hamiltonian` takes).
    ``cells`` is needed whenever ``"wavefunction"`` is involved and
    ``orbital_positions`` whenever ``"atomic"`` is.

    Converting to ``"cell"`` is the repair step for coefficients that arrived in
    another convention; converting away from it and back is the identity, which
    the tests check.
    """

    src = _check_convention("source", source)
    dst = _check_convention("target", target)
    coeffs = _as_coefficients(coefficients)
    point = _as_kpoint(supercell_kpoint, name="supercell_kpoint")
    if src == dst:
        return coeffs.copy()
    intermediate = _to_cell_convention(
        coeffs, src, cells=cells, kpoint=point, orbital_positions=orbital_positions
    )
    return _from_cell_convention(
        intermediate, dst, cells=cells, kpoint=point, orbital_positions=orbital_positions
    )


def gauge_overlap_kernel(kernel: ArrayLike, factors: ArrayLike) -> NDArray[np.complex128]:
    """``S -> diag(u) S diag(u)*`` on a ``(n_orbitals, n_orbitals, n_cells)`` kernel.

    This is the transformation the overlap kernel must undergo alongside a
    per-orbital regauging of the state; with it the LCAO weight is unchanged
    (``UnfoldLab.ovWeight_gauge``), without it the off-diagonal — the part that
    only exists because the basis is non-orthogonal — is rotated.
    """

    array = np.asarray(kernel, dtype=np.complex128)
    if array.ndim != 3 or array.shape[0] != array.shape[1]:
        raise ValueError("kernel must have shape (n_orbitals, n_orbitals, n_cells)")
    phases = np.asarray(factors, dtype=np.complex128).reshape(-1)
    if phases.shape[0] != array.shape[0]:
        raise ValueError("factors must have one entry per orbital")
    return (
        phases[:, np.newaxis, np.newaxis] * np.conjugate(phases)[np.newaxis, :, np.newaxis] * array
    )


def fiber_shift_permutation(
    points: ArrayLike, shift: ArrayLike, *, atol: float = 1e-8
) -> NDArray[np.int64]:
    """Index of ``k + shift`` inside the fiber, for every ``k`` of the fiber.

    ``points`` is a fiber as returned by
    :func:`unmochan.core.kpoints.fiber_kpoints`, and the shift is taken modulo
    one primitive reciprocal lattice vector.  Raises if the shift does not map
    the fiber onto itself, which happens exactly when it is not a difference of
    two of its members.
    """

    table = np.asarray(points, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3:
        raise ValueError("points must have shape (n_kpoints, 3)")
    delta = _as_kpoint(shift, name="shift")
    shifted = wrap_fractional(table + delta[np.newaxis, :])
    wrapped = wrap_fractional(table)
    # |dk| measured modulo 1 in each component.
    difference = np.abs(shifted[:, np.newaxis, :] - wrapped[np.newaxis, :, :])
    distance = np.max(np.minimum(difference, 1.0 - difference), axis=2)
    order = np.argmin(distance, axis=1)
    if table.shape[0] and float(np.max(distance[np.arange(table.shape[0]), order])) > atol:
        raise ValueError("the shift does not map the fiber onto itself")
    if np.unique(order).size != order.size:
        raise ValueError("the shift does not permute the fiber")
    return order.astype(np.int64)


@dataclass(frozen=True)
class ConventionDiagnosis:
    """How much the Bloch-sum convention matters for one supercell k-point.

    ``max_total_variation`` is the largest, over the states, of
    ``0.5 * sum_k |W_cell(k) - W_wavefunction(k)|``: the fraction of a state's
    spectral weight that the wrong convention puts on the wrong primitive
    k-point.  It is zero exactly when the unfolded distribution happens to be
    invariant under the fiber permutation below, which for a perfectly periodic
    state at a fiber-symmetric k-point can happen by accident.

    ``permutation`` is the fiber permutation the mistake amounts to, when it is
    one: ``permutation[i]`` is the index of the fiber member at which the weight
    belonging to member ``i`` is reported (``UnfoldLab.tbWeight_gaugeCell``).
    That only happens when the shift is a difference of two fiber members, i.e.
    at the supercell Gamma point; otherwise the whole distribution is translated
    off the fiber (``UnfoldLab.tbWeight_gaugeShift``) and the field is ``None``.

    ``orbital_gauge_matters`` is always ``False``: a per-orbital phase provably
    cannot change an orthonormal-basis weight, and the field is kept so that a
    report can state it explicitly.
    """

    max_total_variation: float
    permutation: tuple[int, ...] | None
    n_states: int
    n_kpoints: int
    orbital_gauge_matters: bool = False

    def matters(self, atol: float = 1e-8) -> bool:
        """Whether the convention changes the unfolded weights at all here."""

        return self.max_total_variation > atol

    def to_dict(self) -> dict[str, object]:
        return {
            "max_total_variation": self.max_total_variation,
            "permutation": None if self.permutation is None else list(self.permutation),
            "n_states": self.n_states,
            "n_kpoints": self.n_kpoints,
            "orbital_gauge_matters": self.orbital_gauge_matters,
        }


def diagnose_bloch_convention(
    cells: ArrayLike,
    coefficients: ArrayLike,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
) -> ConventionDiagnosis:
    """Quantify the cost of getting the Bloch-sum convention wrong.

    ``coefficients`` is read in the package's ``"cell"`` convention; the
    comparison is against the same table read as ``"wavefunction"`` amplitudes,
    i.e. with the supercell Bloch factor ``exp(2 pi i q . R)`` left in.  Both
    obey the fiber sum rule, so this difference is the only thing that
    distinguishes them, and no runtime sum-rule check can.
    """

    validate_supercell_cells(cells, transform)
    point = _as_kpoint(supercell_kpoint, name="supercell_kpoint")
    coeffs = _as_coefficients(coefficients)
    points = fiber_kpoints(point, transform)

    reference = tight_binding_weights(cells, coeffs, points, multiplicity=transform.multiplicity)
    mistaken = tight_binding_weights(
        cells,
        convert_coefficients(
            coeffs,
            source="cell",
            target="wavefunction",
            supercell_kpoint=point,
            cells=cells,
        ),
        points,
        multiplicity=transform.multiplicity,
    )
    variation = (
        float(np.max(0.5 * np.abs(reference - mistaken).sum(axis=0))) if reference.size else 0.0
    )
    try:
        permutation: tuple[int, ...] | None = tuple(
            int(index) for index in fiber_shift_permutation(points, point)
        )
    except ValueError:
        permutation = None
    return ConventionDiagnosis(
        max_total_variation=variation,
        permutation=permutation,
        n_states=int(coeffs.shape[0]),
        n_kpoints=int(points.shape[0]),
    )
