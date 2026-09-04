"""PAW / ultrasoft augmentation: what the plane-wave unfolding weight misses.

Why this exists
---------------

The very first audit of this package recorded a caveat and left it standing for
thirty passes:

    *Plane-wave-only normalization (PAW/USPP).*  Weights are computed from the
    plane-wave coefficients alone.  For PAW or ultrasoft calculations the
    augmentation charge inside the spheres is not represented, so the weights
    are those of the pseudo wavefunction.

Almost every production VASP or Quantum ESPRESSO calculation is a PAW or
ultrasoft one, so this is not a corner case: it is the common case.  This
module says how big the error is, and it turns the caveat into a number the
user can read off their own run.

The mathematics is settled in ``RequestProject/Unfolding/Augmentation.lean``.

The setting
-----------

A PAW pseudo wavefunction is normalized in the *augmented* inner product,

    <psi|psi> = <psi~| S |psi~>,   S = 1 + sum_a sum_ij q^a_ij |p^a_i><p^a_j|,

so the plane-wave norm ``N = sum_G |c_G|^2`` stored in the ``WAVECAR`` or the
QE ``wfc`` file is only *part* of the norm.  Write ``q = 1 - N`` for the
missing part, the **augmentation fraction**.  The unfolding weight the package
computes is ``a / N`` with ``a`` the matched plane-wave norm; the true weight
is ``(a + p) / (N + q)`` with ``p`` the matched augmentation.

Four things follow, and all four are proved:

*the augmentation respects the fiber for an ideal supercell*
    ``UnfoldLab.structureFactor_eq`` and
    ``UnfoldLab.augMatrix_isFiberBlockDiagonal``: the projectors are localized
    at atoms, so the augmentation matrix carries a structure factor
    ``sum_R exp(-2 pi i <m, R>)`` over the primitive cells.  Summed over a full
    set of primitive translations this vanishes unless the reciprocal transfer
    is a *primitive* reciprocal-lattice vector -- exactly the condition that
    two plane waves belong to the same primitive k-point.  So for a perfect
    supercell the augmentation never mixes different members of the fiber;

*and it stops doing so as soon as the supercell has a defect*
    ``UnfoldLab.exists_structureFactor_ne_zero_of_not_mem``: an unreplicated
    site has structure factor ``1`` everywhere.  A defect or relaxed supercell
    -- the whole point of unfolding -- couples the fiber through the
    augmentation;

*the sum rule cannot see any of this*
    ``UnfoldLab.IsFiberRepr.sum_augWeight_eq_one``: the *true* weights add up
    to one over a fiber, and so do the pseudo weights.  The package's headline
    diagnostic is therefore blind to the error, which is why it has to be
    reported separately;

*but the error is bounded by the augmentation fraction*
    ``UnfoldLab.abs_augWeight_sub_weight_le``: ``|w_true - w_pseudo| <= q``
    for an ``S``-normalized state.  And ``q = 1 - N`` is exactly the norm
    deficit that :func:`unmochan.core.plane_waves.state_norms_from_coefficients`
    already computes and ``unmochan norms`` already prints.  A run whose
    stored norms are ``0.98`` has weights good to ``0.02``; one whose norms are
    ``0.7`` has weights that may be wrong by ``0.3``, which is the difference
    between a bright band and a faint one.

``UnfoldLab.augWeight_eq_weight_iff`` completes the picture: the pseudo weight
is exactly right only when the augmentation happens to be distributed over the
fiber in the same proportion as the plane-wave norm.  Nothing enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.numerics import (
    TransformLike,
    as_int_array,
    as_matrix3,
    check_atol,
    integer_adjugate3,
    integer_det3,
)

__all__ = [
    "AugmentationReport",
    "augmentation_fraction",
    "augmented_weights",
    "diagnose_augmentation",
    "is_fiber_block_diagonal",
    "structure_factor",
    "weight_error_bound",
]

#: Below this augmentation fraction the pseudo weights are treated as reliable.
NEGLIGIBLE_FRACTION = 1e-3

#: Above this augmentation fraction a weight can be wrong by more than a tenth.
SEVERE_FRACTION = 0.1


def augmentation_fraction(
    norms: ArrayLike, *, reference: float = 1.0, clip: bool = True
) -> NDArray[np.float64]:
    """The augmentation fraction ``q = reference - N`` of stored states.

    ``norms`` is the table of stored plane-wave norms ``N = sum_G |c_G|^2``
    (:func:`unmochan.core.plane_waves.state_norms_from_coefficients`), and
    ``reference`` the norm the states are supposed to have -- ``1`` for an
    ``S``-normalized PAW state, which is what both readers produce.

    A *negative* value would mean the stored plane-wave norm already exceeds
    the full norm, which no augmentation can explain; with ``clip`` those are
    reported as zero rather than as a spurious error estimate.
    """

    arr = np.asarray(norms, dtype=float)
    fraction = float(reference) - arr
    if clip:
        fraction = np.maximum(fraction, 0.0)
    return np.asarray(fraction, dtype=np.float64)


def weight_error_bound(norms: ArrayLike, *, reference: float = 1.0) -> NDArray[np.float64]:
    """The worst-case error of a pseudo unfolding weight, state by state.

    ``UnfoldLab.abs_augWeight_sub_weight_le``: the difference between the
    plane-wave weight and the augmentation-aware one is at most the
    augmentation fraction.  The bound is on the *absolute* weight, so it is
    directly comparable with the weights themselves, and it is attained in the
    worst case rather than being an order-of-magnitude estimate.
    """

    return augmentation_fraction(norms, reference=reference)


def structure_factor(
    transform: TransformLike,
    cells: ArrayLike,
    transfer: ArrayLike,
    *,
    atol: float = 1e-6,
) -> NDArray[np.complex128]:
    """``sum_R exp(-2 pi i <m, R>)`` over the primitive cells of a supercell.

    ``cells`` are the primitive cells ``R`` (integer, shape ``(n_cells, 3)``)
    and ``transfer`` the reciprocal transfers ``m`` (integer, shape ``(n, 3)``
    or ``(3,)``).  The pairing is ``<m, R> = (T^-1 m) . R``, matching
    ``UnfoldLab.dualPairing``.

    For a full set of primitive cells this is ``|det T|`` on the primitive
    reciprocal lattice and zero elsewhere (``UnfoldLab.structureFactor_eq``);
    for anything else -- a vacancy, a partial occupation -- it is not, which is
    what makes the augmentation of a defect supercell couple the fiber.
    """

    check_atol(atol)
    matrix = as_int_array(as_matrix3(transform, name="transform"), name="transform")
    determinant = integer_det3(matrix)
    if determinant == 0:
        raise ValueError("transform must be invertible")
    cell_array = as_int_array(np.atleast_2d(np.asarray(cells)), name="cells", atol=atol)
    if cell_array.ndim != 2 or cell_array.shape[1] != 3:
        raise ValueError("cells must have shape (n_cells, 3)")
    transfers = as_int_array(np.atleast_2d(np.asarray(transfer)), name="transfer", atol=atol)
    if transfers.ndim != 2 or transfers.shape[1] != 3:
        raise ValueError("transfer must have shape (3,) or (n, 3)")

    # (T^-1 m) . R = (adj(T) m) . R / det T, exact up to the final division.
    projected = transfers @ integer_adjugate3(matrix).T
    phases = projected.astype(float) @ cell_array.astype(float).T / float(determinant)
    return np.asarray(np.exp(-2j * np.pi * phases).sum(axis=1), dtype=np.complex128)


def is_fiber_block_diagonal(
    augmentation: ArrayLike,
    gvectors: ArrayLike,
    transform: TransformLike,
    *,
    atol: float = 1e-8,
) -> bool:
    """Whether an augmentation matrix couples only plane waves of one fiber member.

    ``UnfoldLab.IsFiberBlockDiagonal``: ``A[i, j]`` must vanish whenever
    ``g_i - g_j`` is not a primitive reciprocal-lattice vector, i.e. not in
    ``T Z^3``.  The membership test is the exact integer one of
    :func:`unmochan.core.plane_waves.matching_g_mask` (adjugate and
    divisibility), so only the size of ``A`` is compared against ``atol``.

    This is the hypothesis under which the augmented weights still obey the
    fiber sum rule.  It holds for an ideal supercell and fails for a defect
    one.
    """

    tol = check_atol(atol)
    matrix = as_int_array(as_matrix3(transform, name="transform"), name="transform")
    determinant = integer_det3(matrix)
    if determinant == 0:
        raise ValueError("transform must be invertible")
    g_int = as_int_array(np.atleast_2d(np.asarray(gvectors)), name="gvectors")
    if g_int.ndim != 2 or g_int.shape[1] != 3:
        raise ValueError("gvectors must have shape (n_g, 3)")
    matrix_a = np.asarray(augmentation, dtype=np.complex128)
    if matrix_a.shape != (g_int.shape[0], g_int.shape[0]):
        raise ValueError("augmentation must be square with one row per G-vector")

    differences = g_int[:, None, :] - g_int[None, :, :]
    projected = differences @ integer_adjugate3(matrix).T
    same_class = np.all(projected % determinant == 0, axis=-1)
    return bool(np.all(np.abs(matrix_a[~same_class]) <= tol))


def augmented_weights(
    coefficients: ArrayLike,
    mask: ArrayLike,
    augmentation: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Unfolding weights that include the augmentation, when it is available.

    ``coefficients`` has shape ``(n_states, n_g)``, ``mask`` is the boolean
    plane-wave selection of one primitive k-point
    (:func:`unmochan.core.plane_waves.matching_g_mask`), and ``augmentation``
    is the Hermitian matrix ``A[i, j] = <K+G_i| S - 1 |K+G_j>``.  The weight is
    ``(a + p) / (N + q)`` exactly as in ``UnfoldLab.augWeight``: the matched
    quadratic form restricts *both* indices to the mask.

    With ``augmentation=None`` this is the plane-wave weight, so the two can be
    compared directly.  Most codes do not export ``q_ij`` and the projectors;
    when they do not, use :func:`diagnose_augmentation`, which bounds the error
    from the stored norms alone.
    """

    coeffs = np.atleast_2d(np.asarray(coefficients, dtype=np.complex128))
    if coeffs.ndim != 2:
        raise ValueError("coefficients must have shape (n_states, n_g)")
    selection = np.asarray(mask, dtype=bool).reshape(-1)
    if selection.shape[0] != coeffs.shape[1]:
        raise ValueError("mask must have one entry per plane wave")

    plane_wave = coeffs.real**2 + coeffs.imag**2
    total = plane_wave.sum(axis=1)
    matched = plane_wave[:, selection].sum(axis=1)

    if augmentation is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(np.where(total > 0.0, matched / total, 0.0), dtype=np.float64)

    matrix_a = np.asarray(augmentation, dtype=np.complex128)
    if matrix_a.shape != (coeffs.shape[1], coeffs.shape[1]):
        raise ValueError("augmentation must be square with one row per plane wave")
    full = np.einsum("si,ij,sj->s", coeffs.conj(), matrix_a, coeffs).real
    restricted_coeffs = np.where(selection[None, :], coeffs, 0.0)
    part = np.einsum("si,ij,sj->s", restricted_coeffs.conj(), matrix_a, restricted_coeffs).real
    denominator = total + full
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(denominator != 0.0, (matched + part) / denominator, 0.0)
    return np.asarray(weights, dtype=np.float64)


@dataclass(frozen=True)
class AugmentationReport:
    """How much of the unfolding weight the plane-wave formula cannot see."""

    n_states: int
    min_norm: float
    mean_norm: float
    max_fraction: float
    mean_fraction: float
    n_severe: int
    negative_deficit: bool

    @property
    def negligible(self) -> bool:
        """Whether the augmentation can be ignored at the printing precision."""

        return self.max_fraction <= NEGLIGIBLE_FRACTION

    @property
    def weight_error_bound(self) -> float:
        """Worst-case absolute error of any unfolding weight in the run."""

        return self.max_fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "min_norm": self.min_norm,
            "mean_norm": self.mean_norm,
            "max_fraction": self.max_fraction,
            "mean_fraction": self.mean_fraction,
            "n_severe": self.n_severe,
            "negative_deficit": self.negative_deficit,
            "negligible": self.negligible,
            "weight_error_bound": self.weight_error_bound,
        }

    #: Emitted whenever a stored norm exceeds the reference, in *both* branches
    #: of :meth:`summary`: a norm above one is a normalization-convention
    #: problem, and it must not be hidden behind a negligible augmentation.
    _NEGATIVE_DEFICIT_NOTE = (
        "  some stored norms exceed the reference, which no augmentation "
        "explains: check the normalization convention of the reader"
    )

    def summary(self) -> str:
        if self.negligible:
            lines = [
                "augmentation negligible: stored plane-wave norms are within "
                f"{self.max_fraction:.2e} of one, so the weights are the true ones "
                "to that accuracy (norm-conserving, or a PAW run whose states "
                "carry almost no augmentation charge)"
            ]
            if self.negative_deficit:
                lines.append(self._NEGATIVE_DEFICIT_NOTE)
            return "\n".join(lines)
        lines = [
            f"augmentation fraction: up to {self.max_fraction:.3g} "
            f"(mean {self.mean_fraction:.3g}) over {self.n_states} states",
            f"  every unfolding weight may be wrong by up to {self.max_fraction:.3g} "
            "in absolute terms (UnfoldLab.abs_augWeight_sub_weight_le)",
            f"  {self.n_severe} states exceed {SEVERE_FRACTION:.2g}; the augmentation "
            "charge is largest for localized d and f states",
            "  the fiber sum rule does NOT detect this: the true weights add to one "
            "as well (UnfoldLab.IsFiberRepr.sum_augWeight_eq_one)",
        ]
        if self.negative_deficit:
            lines.append(self._NEGATIVE_DEFICIT_NOTE)
        return "\n".join(lines)


def diagnose_augmentation(norms: ArrayLike, *, reference: float = 1.0) -> AugmentationReport:
    """Bound the PAW error of a whole run from its stored plane-wave norms.

    ``norms`` is the ``(n_kpoints, n_bands)`` table that
    :func:`unmochan.core.unfolding.diagnose_state_norms` summarizes and
    ``unmochan norms`` prints.  Nothing else is needed: the augmentation
    fraction *is* the norm deficit, and the deficit bounds every weight error.
    """

    arr = np.asarray(norms, dtype=float)
    if arr.size == 0:
        raise ValueError("norms must contain at least one state")
    raw = float(reference) - arr
    fraction = np.maximum(raw, 0.0)
    return AugmentationReport(
        n_states=int(arr.size),
        min_norm=float(np.min(arr)),
        mean_norm=float(np.mean(arr)),
        max_fraction=float(np.max(fraction)),
        mean_fraction=float(np.mean(fraction)),
        n_severe=int(np.count_nonzero(fraction > SEVERE_FRACTION)),
        negative_deficit=bool(np.any(raw < -1e-9)),
    )
