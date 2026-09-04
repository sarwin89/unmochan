"""Plane-wave unfolding kernels shared by electronic-structure backends.

Conventions
-----------

Real-space lattices are row-vector based and ``A_sc = T @ A_pc`` with ``T``
integer.  Fractional reciprocal coordinates then satisfy ``K_sc = k_pc @ T.T``,
and a supercell plane wave with reciprocal vector ``K_sc + G_sc`` contributes to
primitive k-point ``k_pc`` exactly when

``(K_sc + G_sc) @ inv(T).T - k_pc``

is an integer vector.

Exactness
---------

Writing ``m0 = K_sc - k_pc @ T.T`` (an integer vector, because ``K_sc`` is a
Brillouin-zone representative of the folded k-point), the condition above is
equivalent to the purely integral statement

``G_sc + m0 ∈ ℤ³ @ T.T``,

which by Cramer's rule holds exactly when ``det(T)`` divides every component of
``(G_sc + m0) @ adjugate(T).T``.  The implementation below uses that exact
integer test instead of a floating-point comparison, so the result no longer
depends on a tolerance.  The formal statements are
``UnfoldLab.pwMatches_iff_mem_image`` and
``UnfoldLab.mem_range_mulVec_iff_adjugate`` in
``RequestProject/Unfolding/Matching.lean``.

Representative dependence
-------------------------

The set of matching plane waves depends only on the physical vectors
``K_sc + G_sc`` (``UnfoldLab.pwMatches_of_add_eq``).  The ``g_supercell`` list
must therefore be the Miller-index list *of the very same k-point
representative* that is passed as ``folded_supercell_kpoint``.  Passing a
wrapped k-point together with G-vectors written down for an unwrapped one
silently selects the wrong plane waves; :func:`matching_g_mask` guards against
the inconsistent case by requiring ``K_sc - k_pc @ T.T`` to be integral.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unmochan.core.numerics import (
    TransformLike,
    as_int_array,
    check_atol,
    integer_adjugate3,
    integer_det3,
)

__all__ = [
    "compute_weights_from_coefficient_table",
    "matching_g_mask",
    "shared_weights_from_coefficients",
    "state_norms_from_coefficients",
    "spin_expectation_from_coefficients",
    "spin_texture_from_coefficients",
    "weights_from_coefficients",
]


def _integer_transform(transform: NDArray[np.floating] | NDArray[np.integer], *, atol: float):
    matrix = as_int_array(transform, name="transform", atol=atol)
    if matrix.shape != (3, 3):
        raise ValueError(f"transform must have shape (3, 3), got {matrix.shape}")
    determinant = integer_det3(matrix)
    if determinant == 0:
        raise ValueError("transform must be invertible")
    return matrix, determinant, integer_adjugate3(matrix)


def _folding_offset(
    primitive_kpoint: NDArray[np.float64],
    folded_supercell_kpoint: NDArray[np.float64],
    matrix: NDArray[np.int64],
    *,
    atol: float,
) -> NDArray[np.int64]:
    """Return the integer vector ``m0 = K_sc - k_pc @ T.T``.

    A non-integral value means ``folded_supercell_kpoint`` is not a
    Brillouin-zone image of ``primitive_kpoint`` under the transform, i.e. the
    k-point map and the wavefunction file disagree.
    """

    residual = folded_supercell_kpoint - primitive_kpoint @ matrix.T
    rounded = np.rint(residual)
    if float(np.max(np.abs(residual - rounded))) > check_atol(atol):
        raise ValueError(
            "folded_supercell_kpoint is not the fold of primitive_kpoint: "
            f"K_sc - k_pc @ T.T = {residual.tolist()} is not an integer vector "
            f"within atol={atol:g}"
        )
    return rounded.astype(np.int64)


def _validated_g_vectors(
    g_supercell: NDArray[np.integer] | NDArray[np.floating], *, tol: float
) -> NDArray[np.int64]:
    g_arr = np.asarray(g_supercell)
    if g_arr.ndim != 2 or g_arr.shape[1] != 3:
        raise ValueError("g_supercell must have shape (n_g, 3)")
    return as_int_array(g_arr, name="g_supercell", atol=tol)


_MAX_PACKED_MODULUS = 2_000_000
"""Largest ``|det T|`` whose residues still pack into one ``int64`` label.

A class is a triple of residues modulo ``|det T|``, so packing them into a
single integer needs ``|det T|**3`` distinct values; beyond this bound the
labels are obtained by sorting the triples instead.
"""


def _class_labels(
    g_residues: NDArray[np.int64],
    kpoint_residues: NDArray[np.int64],
    modulus: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Label G-vector classes and k-point classes with one shared encoding.

    Two rows get the same label exactly when they are the same residue triple,
    and G-vector labels are comparable with k-point labels, which is what makes
    the matching test a single integer comparison.
    """

    if modulus <= _MAX_PACKED_MODULUS:

        def pack(rows: NDArray[np.int64]) -> NDArray[np.int64]:
            return (rows[:, 0] * modulus + rows[:, 1]) * modulus + rows[:, 2]

        return pack(g_residues), pack(kpoint_residues)
    rows = np.concatenate((g_residues, kpoint_residues), axis=0)
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    flat = np.asarray(inverse, dtype=np.int64).reshape(-1)
    split = g_residues.shape[0]
    return flat[:split], flat[split:]


def _g_class_residues(
    g_int: NDArray[np.int64], adjugate: NDArray[np.int64], modulus: int
) -> NDArray[np.int64]:
    """``G @ adj(T).T mod |det T|``, the class of ``G`` in the fiber quotient.

    ``G`` matches the primitive k-point whose offset is ``m0`` exactly when
    ``(G + m0) @ adj(T).T ≡ 0``, so the k-point enters only through the target
    residue ``-m0 @ adj(T).T`` and the ``G``-side work is k-independent.
    """

    return (g_int @ adjugate.T) % modulus


def _kpoint_class_residues(
    offsets: NDArray[np.int64], adjugate: NDArray[np.int64], modulus: int
) -> NDArray[np.int64]:
    return (-(offsets @ adjugate.T)) % modulus


def _class_partition(
    labels: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Sort G-vectors by class and return ``(order, unique labels, starts)``."""

    order = np.argsort(labels, kind="stable")
    unique, starts = np.unique(labels[order], return_index=True)
    return order, unique, starts


def _class_sums(
    values: NDArray[np.float64], order: NDArray[np.int64], starts: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Sum ``values[..., g]`` over each class, one pass over the G axis.

    The classes partition the G-vectors -- each ``G`` matches *at most one*
    primitive k-point of a fiber -- so the sums for all the k-points cost one
    traversal together rather than one traversal each.
    """

    return np.add.reduceat(values[..., order], starts, axis=-1)


def _select_classes(
    sums: NDArray[np.float64],
    unique: NDArray[np.int64],
    targets: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Pick each target class out of :func:`_class_sums`, zero if unoccupied."""

    if unique.size == 0:
        return np.zeros(sums.shape[:-1] + (targets.shape[0],), dtype=float)
    position = np.clip(np.searchsorted(unique, targets), 0, unique.size - 1)
    return np.where(unique[position] == targets, sums[..., position], 0.0)


def matching_g_mask(
    g_supercell: NDArray[np.integer] | NDArray[np.floating],
    primitive_kpoint: NDArray[np.float64],
    folded_supercell_kpoint: NDArray[np.float64],
    transform: TransformLike,
    *,
    tol: float = 1e-6,
) -> NDArray[np.bool_]:
    """Return the supercell G-vectors compatible with a primitive k-point.

    The test is exact: ``tol`` is only used to validate that the inputs really
    are integral (the G-vectors, the transform, and the offset between the
    primitive k-point and its folded image).
    """

    g_int = _validated_g_vectors(g_supercell, tol=tol)
    primitive = np.asarray(primitive_kpoint, dtype=float)
    folded = np.asarray(folded_supercell_kpoint, dtype=float)
    if primitive.shape != (3,) or folded.shape != (3,):
        raise ValueError("primitive and folded k-points must have shape (3,)")

    matrix, determinant, adjugate = _integer_transform(transform, atol=tol)
    offset = _folding_offset(primitive, folded, matrix, atol=tol)

    shifted = g_int + offset[np.newaxis, :]
    projected = shifted @ adjugate.T
    return np.all(projected % determinant == 0, axis=1)


def weights_from_coefficients(
    g_supercell: NDArray[np.integer] | NDArray[np.floating],
    coefficients: NDArray[np.complexfloating],
    primitive_kpoint: NDArray[np.float64],
    folded_supercell_kpoint: NDArray[np.float64],
    transform: TransformLike,
    *,
    tol: float = 1e-6,
    component_resolved: bool = False,
) -> NDArray[np.float64]:
    """Compute unfolded weights for all bands in one k-point wavefunction.

    ``coefficients`` must have shape ``(n_bands, n_components, n_g)``. Spinor or
    polarization components are summed in the norm, which preserves the usual
    plane-wave unfolding sum rule for noncollinear files.

    The returned weights lie in ``[0, 1]``, and summing them over the
    ``|det T|`` primitive k-points that fold onto the same supercell k-point
    gives one for every band (formalized as
    ``UnfoldLab.IsFiberRepr.sum_weight_eq_one``).

    With ``component_resolved=True`` the result has shape
    ``(n_bands, n_components)`` and splits each weight into the contributions of
    the individual spinor/polarization components, still normalized by the total
    norm of the state.  Summing over the component axis gives the ordinary
    weight exactly, so the sum rule is unaffected; the split is what a
    spin-resolved unfolded band structure needs.
    """

    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
    if coeffs.shape[2] != len(g_supercell):
        raise ValueError("coefficient G dimension must match number of G-vectors")

    primitive = np.asarray(primitive_kpoint, dtype=float)
    if primitive.shape != (3,):
        raise ValueError("primitive and folded k-points must have shape (3,)")

    # One k-point is the ``n_kpoints == 1`` case of the batched kernel; keeping
    # a single implementation of the arithmetic keeps the two from drifting.
    batched = shared_weights_from_coefficients(
        g_supercell,
        coeffs,
        primitive.reshape(1, 3),
        folded_supercell_kpoint,
        transform,
        tol=tol,
        component_resolved=component_resolved,
    )
    return np.asarray(batched[0], dtype=float)


def shared_weights_from_coefficients(
    g_supercell: NDArray[np.integer] | NDArray[np.floating],
    coefficients: NDArray[np.complexfloating],
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoint: NDArray[np.float64],
    transform: TransformLike,
    *,
    tol: float = 1e-6,
    component_resolved: bool = False,
) -> NDArray[np.float64]:
    """Weights of *several* primitive k-points against one wavefunction.

    ``primitive_kpoints`` has shape ``(n_kpoints, 3)`` and the result has shape
    ``(n_kpoints, n_bands)``, or ``(n_kpoints, n_bands, n_components)`` with
    ``component_resolved=True``.  Each row is exactly what
    :func:`weights_from_coefficients` returns for that k-point.

    This is the shape of the work whenever one supercell wavefunction serves
    more than one primitive k-point: a complete fiber, or a band path that
    revisits an irreducible k-point.  Done one k-point at a time, the squared
    modulus of the coefficient array -- the dominant cost, ``n_bands *
    n_components * n_g`` multiply-adds -- is recomputed for every one of them,
    and so is the integer matching test on the ``(n_g, 3)`` G-vector array.

    Here both are done once.  Every G-vector carries a *class* label
    ``G @ adj(T).T mod |det T|`` that does not depend on the k-point, and a
    k-point matches exactly the G-vectors of one class, so the k-points select
    among the class sums instead of each demanding its own pass over the
    coefficients.  The cost is one traversal of the coefficient array plus one
    sort of the labels, rather than ``n_kpoints`` traversals; for a complete
    fiber that removes a factor of ``|det T|``.
    """

    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
    if coeffs.shape[2] != len(g_supercell):
        raise ValueError("coefficient G dimension must match number of G-vectors")
    kpoints = np.asarray(primitive_kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError("primitive_kpoints must have shape (n_kpoints, 3)")

    g_int = _validated_g_vectors(g_supercell, tol=tol)
    folded = np.asarray(folded_supercell_kpoint, dtype=float)
    if folded.shape != (3,):
        raise ValueError("primitive and folded k-points must have shape (3,)")
    matrix, determinant, adjugate = _integer_transform(transform, atol=tol)
    modulus = abs(determinant)

    offsets = np.zeros((kpoints.shape[0], 3), dtype=np.int64)
    for index, kpoint in enumerate(kpoints):
        offsets[index] = _folding_offset(kpoint, folded, matrix, atol=tol)
    g_labels, kpoint_labels = _class_labels(
        _g_class_residues(g_int, adjugate, modulus),
        _kpoint_class_residues(offsets, adjugate, modulus),
        modulus,
    )
    order, unique, starts = _class_partition(g_labels)

    norms = coeffs.real**2 + coeffs.imag**2
    per_g = norms.sum(axis=1)
    total = per_g.sum(axis=1)
    safe_total = np.where(total > 0.0, total, 1.0)
    if component_resolved:
        partial = _select_classes(_class_sums(norms, order, starts), unique, kpoint_labels)
        weights = np.where(
            total[np.newaxis, :, np.newaxis] > 0.0,
            np.moveaxis(partial, 2, 0) / safe_total[np.newaxis, :, np.newaxis],
            0.0,
        )
        return np.asarray(weights, dtype=float)
    partial_total = _select_classes(_class_sums(per_g, order, starts), unique, kpoint_labels)
    return np.asarray(
        np.where(
            total[np.newaxis, :] > 0.0,
            partial_total.T / safe_total[np.newaxis, :],
            0.0,
        ),
        dtype=float,
    )


def state_norms_from_coefficients(
    coefficients: NDArray[np.complexfloating],
) -> NDArray[np.float64]:
    """Squared norm of the *stored* plane-wave expansion of each band.

    ``coefficients`` has shape ``(n_bands, n_components, n_g)`` and the result
    has shape ``(n_bands,)``.

    The unfolding weight is a ratio and therefore does not depend on this
    number at all (``UnfoldLab.weight_smul``), so a state stored with any
    normalization convention gives the same weights.  What the number *does*
    diagnose is how much of the state the file contains: a norm of ``1 - delta``
    for a wavefunction that ought to be normalized means a fraction ``delta`` of
    the expansion is missing -- a plane-wave cutoff applied by the reader, a
    truncated file, or the augmentation part of a PAW/ultrasoft pseudo
    wavefunction, which is simply not in the plane-wave coefficients.  The
    weights are then only accurate to ``delta / (1 - delta)``
    (``UnfoldLab.weight_truncation_bound``, exposed as
    :func:`unmochan.core.unfolding.truncation_weight_error_bound`).
    """

    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
    norms = coeffs.real**2 + coeffs.imag**2
    return norms.sum(axis=(1, 2)).astype(float, copy=False)


def _spinor_bloch_vectors(
    coefficients: NDArray[np.complexfloating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-plane-wave Bloch vectors and squared norms of a spinor state.

    ``coefficients`` has shape ``(n_bands, 2, n_g)``.  The returned Bloch
    vectors have shape ``(n_bands, n_g, 3)`` and the norms ``(n_bands, n_g)``;
    for every single plane wave the two are equal in length, because a
    two-component spinor is a *pure* state (``UnfoldLab.norm_spinVec``).
    """

    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
    if coeffs.shape[1] != 2:
        raise ValueError(
            "a spin texture needs exactly two spinor components; got "
            f"{coeffs.shape[1]}.  Collinear files carry no transverse spin."
        )
    up = coeffs[:, 0, :]
    down = coeffs[:, 1, :]
    cross = np.conjugate(up) * down
    norm_up = up.real**2 + up.imag**2
    norm_down = down.real**2 + down.imag**2
    bloch = np.stack(
        (2.0 * cross.real, 2.0 * cross.imag, norm_up - norm_down),
        axis=-1,
    )
    return bloch.astype(float, copy=False), (norm_up + norm_down).astype(float, copy=False)


def spin_expectation_from_coefficients(
    coefficients: NDArray[np.complexfloating],
) -> NDArray[np.float64]:
    """Return ``⟨σ⟩`` of each band of a noncollinear state, shape ``(n_bands, 3)``.

    This is the whole-state spin polarization, normalized by the norm of the
    state, and it is what the fiber of unfolded textures must add up to
    (``UnfoldLab.IsFiberRepr.sum_spinTexture``).
    """

    bloch, norms = _spinor_bloch_vectors(coefficients)
    total = norms.sum(axis=1)
    safe_total = np.where(total > 0.0, total, 1.0)
    return np.where(
        total[:, np.newaxis] > 0.0,
        bloch.sum(axis=1) / safe_total[:, np.newaxis],
        0.0,
    ).astype(float)


def spin_texture_from_coefficients(
    g_supercell: NDArray[np.integer] | NDArray[np.floating],
    coefficients: NDArray[np.complexfloating],
    primitive_kpoint: NDArray[np.float64],
    folded_supercell_kpoint: NDArray[np.float64],
    transform: TransformLike,
    *,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Unfolded spin texture of a noncollinear state, shape ``(n_bands, 3)``.

    ``coefficients`` must have shape ``(n_bands, 2, n_g)`` with the two spinor
    components on the middle axis.  The result is

    ``S(k) = Σ_{G matching k} ⟨c_G|σ|c_G⟩ / Σ_G ⟨c_G|c_G⟩``,

    the spin expectation value restricted to the plane waves that unfold onto
    the primitive k-point.  Component-resolved weights give only the ``z``
    component: the transverse components come from the off-diagonal product
    ``conj(c_up) c_down`` and are lost when the components are squared
    separately.

    Two properties are proved in ``RequestProject/Unfolding/Spin.lean`` and
    checked by the test suite:

    * ``‖S(k)‖ ≤ w(k) ≤ 1`` (``UnfoldLab.norm_spinTexture_le_pairWeight``), with
      equality when every matching plane wave carries the same spin direction;
    * summing ``S(k)`` over the ``|det T|`` primitive k-points of a fiber gives
      the spin expectation value of the supercell state itself
      (``UnfoldLab.IsFiberRepr.sum_spinTexture``), a vector identity that also
      constrains the transverse components.
    """

    bloch, norms = _spinor_bloch_vectors(coefficients)
    if bloch.shape[1] != len(g_supercell):
        raise ValueError("coefficient G dimension must match number of G-vectors")
    mask = matching_g_mask(
        g_supercell,
        primitive_kpoint,
        folded_supercell_kpoint,
        transform,
        tol=tol,
    ).astype(float)
    total = norms.sum(axis=1)
    safe_total = np.where(total > 0.0, total, 1.0)
    partial = np.einsum("bgc,g->bc", bloch, mask)
    return np.where(
        total[:, np.newaxis] > 0.0,
        partial / safe_total[:, np.newaxis],
        0.0,
    ).astype(float)


def compute_weights_from_coefficient_table(
    coeff_path: str | Path,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: TransformLike,
    n_bands: int,
    *,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute unfolding weights from rows ``ik ib G1 G2 G3 Re Im``.

    The table is parsed once into arrays and the matching test is then applied
    to *all* rows at once: each row's G-vector carries the k-independent class
    label of :func:`_g_class_residues`, each k-point carries the label it
    selects, and a row contributes exactly when the two agree.  Neither the
    matching test nor the accumulation loops over the k-points.
    """

    primitive = np.asarray(primitive_kpoints, dtype=float)
    folded = np.asarray(folded_supercell_kpoints, dtype=float)
    if primitive.shape != folded.shape or primitive.ndim != 2 or primitive.shape[1] != 3:
        raise ValueError("primitive and folded k-points must both have shape (n_kpoints, 3)")
    if n_bands <= 0:
        raise ValueError("n_bands must be positive")

    n_kpoints = primitive.shape[0]
    rows: list[tuple[int, int, int, int, int, float, float]] = []
    for raw in Path(coeff_path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 7:
            raise ValueError("coefficient rows must be: ik ib G1 G2 G3 Re Im")
        rows.append(
            (
                int(parts[0]) - 1,
                int(parts[1]) - 1,
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
                float(parts[5]),
                float(parts[6]),
            )
        )

    if not rows:
        return np.zeros((n_kpoints, n_bands), dtype=float)

    table = np.array(rows, dtype=float)
    ik = table[:, 0].astype(np.int64)
    ib = table[:, 1].astype(np.int64)
    if np.any(ik < 0) or np.any(ik >= n_kpoints) or np.any(ib < 0) or np.any(ib >= n_bands):
        raise ValueError("coefficient index outside band grid")
    g_vectors = table[:, 2:5].astype(np.int64)
    norms = table[:, 5] ** 2 + table[:, 6] ** 2

    matrix, determinant, adjugate = _integer_transform(transform, atol=tol)
    modulus = abs(determinant)
    offsets = np.zeros((n_kpoints, 3), dtype=np.int64)
    for index in np.unique(ik):
        offsets[index] = _folding_offset(primitive[index], folded[index], matrix, atol=tol)
    row_labels, kpoint_labels = _class_labels(
        _g_class_residues(g_vectors, adjugate, modulus),
        _kpoint_class_residues(offsets, adjugate, modulus),
        modulus,
    )
    matched = row_labels == kpoint_labels[ik]

    cell = ik * n_bands + ib
    size = n_kpoints * n_bands
    total = np.bincount(cell, weights=norms, minlength=size).reshape(n_kpoints, n_bands)
    partial = np.bincount(cell[matched], weights=norms[matched], minlength=size).reshape(
        n_kpoints, n_bands
    )

    return np.where(total > 0.0, partial / np.where(total > 0.0, total, 1.0), 0.0)
