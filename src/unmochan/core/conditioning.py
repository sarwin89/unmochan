"""Conditioning of a non-orthogonal overlap matrix.

Unfolding from an LCAO basis solves ``H c = E S c`` and measures every state
with ``c* S c``.  Both steps assume ``S`` is positive definite, and a realistic
basis makes that assumption fragile: diffuse functions on neighbouring atoms
are nearly linearly dependent, ``S`` acquires eigenvalues near zero, and a
Cholesky factorization either fails or returns a state whose norm is rounding
error.

Quantum-chemistry codes answer with *canonical orthogonalization*: build an
``X`` with ``X* S X = 1`` from the eigenvectors of ``S``, dropping the
directions whose eigenvalues lie below a threshold, and solve the standard
problem ``(X* H X) y = E y`` in the surviving space.

``RequestProject/Unfolding/Conditioning.lean`` proves what that does:

``standard_of_generalized`` / ``generalized_of_standard``
    the standard problem is equivalent to the generalized one for *any* ``X``
    with ``X* S X = 1`` -- Cholesky and canonical orthogonalization are two
    choices of the same construction;
``isHermitian_congr``
    the transformed Hamiltonian stays Hermitian, so the energies stay real;
``sNorm_transformed``
    a normalized solution of the standard problem pulls back to an
    ``S``-normalized state, which is what makes an unfolding weight's
    denominator one;
``sNorm_eq_zero_of_mulVec_eq_zero``
    a null direction of ``S`` has zero ``S``-norm: it is not a state, and its
    weight is ``0/0`` -- it must be projected out, not normalized;
``sNorm_ge_of_diagonally_dominant``
    ``c* S c >= (1 - r) |c|^2`` when the off-diagonal row sums of a
    unit-diagonal Hermitian ``S`` are at most ``r``.  That is an ``O(n^2)``
    certificate of positive definiteness, and a lower bound on the smallest
    eigenvalue, with no factorization at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Default cutoff on the eigenvalues of the overlap, the value most LCAO codes
#: use for canonical orthogonalization.
DEFAULT_OVERLAP_THRESHOLD = 1e-6


def _as_square(matrix: ArrayLike, *, name: str) -> NDArray[np.complex128]:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return array


def hermitian_part(matrix: ArrayLike, *, name: str = "matrix") -> NDArray[np.complex128]:
    """Return ``(A + A*)/2``, the Hermitian part of ``A``.

    A Bloch-summed overlap is Hermitian in exact arithmetic; symmetrizing
    removes the rounding asymmetry before any eigenvalue is taken.
    """

    array = _as_square(matrix, name=name)
    return np.asarray(0.5 * (array + array.conj().T), dtype=np.complex128)


def diagonal_dominance_bound(overlap: ArrayLike) -> float:
    """Lower bound ``1 - r`` on the smallest eigenvalue of a scaled overlap.

    The matrix is first scaled to unit diagonal, ``D^-1/2 S D^-1/2`` -- which is
    a congruence, so it preserves positive definiteness -- and ``r`` is the
    largest off-diagonal row sum of the result.  A positive return value
    certifies that the basis is linearly independent
    (``sNorm_ge_of_diagonally_dominant``); a non-positive one certifies nothing
    either way, and the eigenvalues must be computed.

    Costs ``O(n^2)``, against ``O(n^3)`` for a factorization.
    """

    array = hermitian_part(overlap, name="overlap")
    diagonal = np.real(np.diagonal(array))
    if np.any(diagonal <= 0.0):
        raise ValueError("overlap has a non-positive diagonal entry; it is not a Gram matrix")
    scale = 1.0 / np.sqrt(diagonal)
    scaled = np.abs(array) * scale[:, None] * scale[None, :]
    np.fill_diagonal(scaled, 0.0)
    return float(1.0 - scaled.sum(axis=1).max()) if scaled.size else 1.0


@dataclass(frozen=True)
class OverlapConditioning:
    """How close an overlap matrix is to singular."""

    smallest: float
    largest: float
    dominance_bound: float
    n_below_threshold: int
    threshold: float

    @property
    def positive_definite(self) -> bool:
        return self.smallest > 0.0

    @property
    def condition_number(self) -> float:
        if self.smallest <= 0.0:
            return float("inf")
        return self.largest / self.smallest

    @property
    def certified_by_dominance(self) -> bool:
        """Whether the cheap ``O(n^2)`` criterion already settles the question."""

        return self.dominance_bound > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "smallest": self.smallest,
            "largest": self.largest,
            "condition_number": self.condition_number,
            "dominance_bound": self.dominance_bound,
            "certified_by_dominance": self.certified_by_dominance,
            "n_below_threshold": self.n_below_threshold,
            "threshold": self.threshold,
            "positive_definite": self.positive_definite,
        }

    def summary(self) -> str:
        if self.n_below_threshold:
            state = (
                f"{self.n_below_threshold} directions below {self.threshold:g}: the basis is "
                "effectively linearly dependent, use canonical orthogonalization"
            )
        elif self.positive_definite:
            state = "positive definite"
        else:
            state = "NOT positive definite: the basis is linearly dependent"
        return (
            f"overlap eigenvalues in [{self.smallest:.3e}, {self.largest:.3e}], "
            f"condition number {self.condition_number:.3e} -- {state}"
        )


def diagnose_overlap_conditioning(
    overlap: ArrayLike,
    *,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> OverlapConditioning:
    """Eigenvalue range, condition number and near-null direction count."""

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    array = hermitian_part(overlap, name="overlap")
    eigenvalues = np.linalg.eigvalsh(array)
    if np.any(np.real(np.diagonal(array)) <= 0.0):
        # A basis function of zero norm: not a Gram matrix, and the dominance
        # criterion does not apply.  Report that it certifies nothing rather
        # than failing -- the eigenvalues below are still the honest answer.
        bound = float("-inf")
    else:
        bound = diagonal_dominance_bound(array)
    return OverlapConditioning(
        smallest=float(eigenvalues[0]),
        largest=float(eigenvalues[-1]),
        dominance_bound=bound,
        n_below_threshold=int(np.count_nonzero(eigenvalues < threshold)),
        threshold=float(threshold),
    )


def canonical_orthogonalization(
    overlap: ArrayLike,
    *,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Return ``X`` with ``X* S X = 1`` on the well-conditioned subspace.

    ``S = U diag(s) U*`` and ``X = U[:, s >= threshold] diag(s^-1/2)``, so ``X``
    has one column per retained direction and exactly the property the
    equivalence theorems assume.  The retained eigenvalues are returned
    alongside.

    Directions below the threshold are discarded rather than inverted: they are
    the ones whose ``S``-norm is dominated by rounding error, and in the limit
    of an exactly singular overlap they are not states at all
    (``sNorm_eq_zero_of_mulVec_eq_zero``).
    """

    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    array = hermitian_part(overlap, name="overlap")
    eigenvalues, vectors = np.linalg.eigh(array)
    keep = eigenvalues >= threshold
    if not np.any(keep):
        raise ValueError(
            f"every eigenvalue of the overlap is below threshold={threshold:g}; "
            "the basis carries no well-conditioned direction"
        )
    retained = eigenvalues[keep]
    transform = vectors[:, keep] / np.sqrt(retained)[None, :]
    return np.asarray(transform, dtype=np.complex128), np.asarray(retained, dtype=np.float64)


def solve_generalized_eigenproblem_truncated(
    hamiltonian: ArrayLike,
    overlap: ArrayLike,
    *,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Solve ``H c = E S c`` through canonical orthogonalization.

    Unlike :func:`unmochan.core.lcao.solve_generalized_eigenproblem`, which
    needs a Cholesky factorization and therefore a strictly positive definite
    ``S``, this discards the near-null directions and returns *fewer* states
    than the dimension of the basis when the overlap is rank deficient.  The
    states it does return are ``S``-orthonormal (``sNorm_transformed``) and
    solve the generalized problem restricted to the retained subspace
    (``generalized_of_standard``).

    Returns ``(energies, vectors)`` with ``vectors[:, s]`` the state ``s``.
    """

    matrix = hermitian_part(hamiltonian, name="hamiltonian")
    metric = _as_square(overlap, name="overlap")
    if metric.shape != matrix.shape:
        raise ValueError("overlap must have the same shape as the hamiltonian")
    transform, _ = canonical_orthogonalization(metric, threshold=threshold)
    reduced = transform.conj().T @ matrix @ transform
    energies, vectors = np.linalg.eigh(0.5 * (reduced + reduced.conj().T))
    return np.asarray(energies, dtype=np.float64), np.asarray(
        transform @ vectors, dtype=np.complex128
    )
