"""Unfolding from a symmetry-reduced (irreducible-wedge) k-point mesh.

A self-consistent run usually stores wavefunctions only for one representative
of each star of supercell k-points.  A band-unfolding k-path, on the other
hand, asks for arbitrary primitive k-points, whose folded images are generally
*not* the stored representatives.

No reconstruction of the wavefunction is needed to bridge the two.  Writing
``S_pc`` for a point-group operation in primitive fractional reciprocal
coordinates and ``S_sc = T S_pc T^-1`` for the same operation in supercell
coordinates, the unfolding weights satisfy

``w(S_pc k ; S_sc K) = w(k ; K)``

with the *stored*, unrotated plane-wave list
(``UnfoldLab.weight_symmetry`` in ``RequestProject/Unfolding/Symmetry.lean``).
So to obtain the weight of a primitive k-point ``k`` whose folded image is
``S_sc K_file``, it is enough to unfold ``S_pc^-1 k`` against the file exactly
as it is stored.  :func:`map_kpoints_to_stored` performs that bookkeeping.

The physical input to this argument is the usual one: the state at ``S_sc K_f``
is the symmetry image of the stored state, so its plane-wave coefficients are
the stored ones re-indexed by ``G -> S_sc G``, possibly up to a phase.  Weights
depend only on ``|c|^2``, so the phase is irrelevant, but a mesh whose stored
representatives are *not* related to the requested k-points by symmetries of
the supercell cannot be used this way -- which is exactly what
:func:`map_kpoints_to_stored` checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.numerics import (
    as_int_array,
    check_atol,
    integer_adjugate3,
    integer_det3,
)

__all__ = [
    "StoredKPointMatch",
    "map_kpoints_to_stored",
    "primitive_operation",
    "supercell_operation",
]


def _as_integer_matrix(matrix: ArrayLike, *, name: str) -> NDArray[np.int64]:
    arr = as_int_array(np.asarray(matrix), name=name)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}")
    return arr


def _integer_inverse(matrix: NDArray[np.int64], *, name: str) -> NDArray[np.int64]:
    determinant = integer_det3(matrix)
    if abs(determinant) != 1:
        raise ValueError(
            f"{name} must be unimodular (|det| = 1) to be a lattice symmetry, "
            f"got det = {determinant}"
        )
    return integer_adjugate3(matrix) // determinant


def supercell_operation(
    operation: ArrayLike,
    transform: ArrayLike,
) -> NDArray[np.int64]:
    """Write a primitive-basis symmetry operation in supercell coordinates.

    ``operation`` is ``S_pc`` acting on primitive fractional reciprocal
    coordinates and ``transform`` is the supercell matrix ``T``.  The result is
    ``S_sc = T S_pc T^-1``; a :class:`ValueError` is raised when it is not
    integral, which means the operation is not a symmetry of the supercell.
    """

    s_pc = _as_integer_matrix(operation, name="operation")
    t = _as_integer_matrix(transform, name="transform")
    determinant = integer_det3(t)
    if determinant == 0:
        raise ValueError("transform must be invertible")
    product = t @ s_pc @ integer_adjugate3(t)
    if np.any(product % determinant != 0):
        raise ValueError(
            "operation is not a symmetry of the supercell: T S_pc T^-1 is not integral"
        )
    return (product // determinant).astype(np.int64)


def primitive_operation(
    operation: ArrayLike,
    transform: ArrayLike,
) -> NDArray[np.int64]:
    """Inverse of :func:`supercell_operation`: ``S_pc = T^-1 S_sc T``."""

    s_sc = _as_integer_matrix(operation, name="operation")
    t = _as_integer_matrix(transform, name="transform")
    determinant = integer_det3(t)
    if determinant == 0:
        raise ValueError("transform must be invertible")
    product = integer_adjugate3(t) @ s_sc @ t
    if np.any(product % determinant != 0):
        raise ValueError(
            "operation is not a symmetry of the supercell: T^-1 S_sc T is not integral"
        )
    return (product // determinant).astype(np.int64)


@dataclass(frozen=True)
class StoredKPointMatch:
    """How one requested primitive k-point is served by a stored k-point.

    ``index`` is the position of the stored supercell k-point (i.e. of the
    wavefunction file), ``operation`` the primitive-basis symmetry ``S_pc``
    used, and ``effective_primitive_kpoint`` the primitive k-point that must be
    passed to the unfolding kernels together with that file, namely
    ``S_pc^-1 k``.
    """

    index: int
    operation: NDArray[np.int64]
    effective_primitive_kpoint: NDArray[np.float64]


def map_kpoints_to_stored(
    primitive_kpoints: ArrayLike,
    stored_supercell_kpoints: ArrayLike,
    transform: ArrayLike,
    operations: ArrayLike | None = None,
    *,
    atol: float = 1e-6,
) -> list[StoredKPointMatch]:
    """Match requested primitive k-points against a symmetry-reduced file set.

    ``operations`` is a sequence of integer ``3 x 3`` matrices acting on
    *primitive* fractional reciprocal coordinates (the identity is always
    tried, so it may be omitted for an unreduced mesh).  For each primitive
    k-point ``k`` the routine folds it to ``K = k T^T`` and looks for a stored
    k-point ``K_f`` and an operation ``S_pc`` with ``K = S_sc K_f`` modulo a
    reciprocal-lattice vector, where ``S_sc = T S_pc T^-1``.

    It then returns ``S_pc^-1 k``, which is the primitive k-point to unfold
    against that file; by ``UnfoldLab.weight_symmetry`` the resulting weight is
    the weight of ``k``.  A :class:`ValueError` is raised for a k-point that no
    stored k-point and operation can reach.
    """

    kpoints = np.asarray(primitive_kpoints, dtype=float)
    stored = np.asarray(stored_supercell_kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError("primitive_kpoints must have shape (n_kpoints, 3)")
    if stored.ndim != 2 or stored.shape[1] != 3:
        raise ValueError("stored_supercell_kpoints must have shape (n_stored, 3)")

    t = _as_integer_matrix(transform, name="transform")
    identity = np.eye(3, dtype=np.int64)
    ops = [identity]
    if operations is not None:
        for raw in np.asarray(operations, dtype=float).reshape(-1, 3, 3):
            candidate = _as_integer_matrix(raw, name="operations")
            if not any(np.array_equal(candidate, known) for known in ops):
                ops.append(candidate)

    prepared = [
        (op, supercell_operation(op, t), _integer_inverse(op, name="operations")) for op in ops
    ]
    check_atol(atol)

    # The images of every stored k-point under every operation, computed once:
    # shape (n_operations, n_stored, 3).  Comparing a requested k-point against
    # all of them at once turns a triple Python loop -- which cost seconds for a
    # realistic mesh and a 48-element point group -- into one array operation
    # per k-point.
    images = np.stack([stored @ op_sc.T for _, op_sc, _ in prepared])

    matches: list[StoredKPointMatch] = []
    for kpoint in kpoints:
        folded = kpoint @ t.T
        delta = folded[np.newaxis, np.newaxis, :] - images
        delta -= np.rint(delta)
        hit = np.all(np.abs(delta) <= atol, axis=2)  # (n_operations, n_stored)
        if not hit.any():
            raise ValueError(
                f"primitive k-point {kpoint.tolist()} folds to {folded.tolist()}, which is "
                "not the image of any stored supercell k-point under the given symmetry "
                "operations; the wavefunction set does not cover this k-path"
            )
        # Prefer the earliest stored k-point, and for it the earliest operation,
        # which is the order the equivalent nested loops would have found.
        op_position, index = np.unravel_index(
            int(np.argmax(hit.T.reshape(-1))), (len(stored), len(prepared))
        )[::-1]
        op, _, op_inv = prepared[int(op_position)]
        matches.append(
            StoredKPointMatch(
                index=int(index),
                operation=op,
                effective_primitive_kpoint=kpoint @ op_inv.T,
            )
        )
    return matches
