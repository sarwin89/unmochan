"""Residual strain and approximate commensurability of a supercell.

Unfolding assumes an exact integer relation ``A_sc = T @ A_pc`` between the
supercell and the reference primitive lattice.  Supercells taken from real
calculations often miss it by a little:

* a defect supercell whose cell parameters were relaxed;
* a heterostructure in which one layer is strained onto the other;
* a twisted stack replaced by a commensurate approximant.

:func:`unfoldlab.core.transformations.detect_transformation` refuses such a
pair, which is safe but uninformative.  The functions here quantify the
mismatch instead, so that a user can decide whether to re-relax the reference,
accept the distortion, or pick a different reference.

The mathematics is in ``RequestProject/Unfolding/Strain.lean``:

``latticeSpan_subset_iff_intMatrix``
    the supercell is made of whole primitive cells *iff* ``M = A_sc A_pc^-1``
    is an integer matrix -- there is no approximate commensurability, only a
    quantified failure of it;
``strain_kshift``
    unfolding with the nearest integer ``T`` while the true cell is deformed by
    ``D`` mislabels the Cartesian k-vector by ``q @ (I - inv(D))`` -- an error
    *linear in k*: zero at Gamma, largest at the zone boundary;
``strain_kshift_eq_zero_iff``
    that error vanishes for every k-point only when ``D`` is the identity, so
    no choice of integer transform repairs a strained reference;
``round_matrix_optimal``
    entrywise rounding is the best integer approximation available, hence a
    residual close to ``1/2`` means the transform is genuinely ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import as_matrix3, check_atol
from unfoldlab.core.structures import Structure
from unfoldlab.core.transformations import TransformationMatrix

#: Residual above which the nearest-integer transform is treated as ambiguous.
#: The rounding residual can never exceed ``1/2``; a value close to it means a
#: neighbouring integer is essentially as good, so the transform is a guess.
AMBIGUOUS_RESIDUAL = 0.25


def _lattice_of(value: Structure | ArrayLike, *, name: str) -> NDArray[np.float64]:
    lattice = value.lattice if isinstance(value, Structure) else value
    return as_matrix3(lattice, name=name)


def nearest_integer_transformation(
    primitive: Structure | ArrayLike,
    supercell: Structure | ArrayLike,
) -> tuple[NDArray[np.int64], float]:
    """Return the entrywise-nearest integer transform and its residual.

    ``M = A_sc @ inv(A_pc)`` is rounded entrywise; the residual is
    ``max |M - T|``.  Rounding is optimal (``round_matrix_optimal``), so no
    other integer matrix has a smaller residual, and the residual never exceeds
    ``1/2``.  Unlike :func:`detect_transformation` this never raises on a
    mismatch -- reporting the mismatch is the point.
    """

    primitive_arr = _lattice_of(primitive, name="primitive_lattice")
    supercell_arr = _lattice_of(supercell, name="supercell_lattice")
    if abs(float(np.linalg.det(primitive_arr))) == 0.0:
        raise ValueError("primitive lattice is singular")
    raw = supercell_arr @ np.linalg.inv(primitive_arr)
    rounded = np.rint(raw).astype(np.int64)
    residual = float(np.max(np.abs(raw - rounded)))
    return rounded, residual


def deformation_gradient(
    primitive: Structure | ArrayLike,
    supercell: Structure | ArrayLike,
    transform: TransformationMatrix | ArrayLike,
) -> NDArray[np.float64]:
    """Return the Cartesian deformation gradient ``D`` of the supercell.

    ``D`` is defined by ``A_sc = (T @ A_pc) @ D.T``, i.e. every ideal lattice
    vector ``a`` of the unstrained supercell sits at ``D @ a`` in the real one.
    It is the identity exactly for a commensurate pair.
    """

    primitive_arr = _lattice_of(primitive, name="primitive_lattice")
    supercell_arr = _lattice_of(supercell, name="supercell_lattice")
    matrix = (
        transform.matrix
        if isinstance(transform, TransformationMatrix)
        else as_matrix3(transform, name="transform")
    )
    ideal = np.asarray(matrix, dtype=float) @ primitive_arr
    if abs(float(np.linalg.det(ideal))) == 0.0:
        raise ValueError("ideal supercell lattice is singular")
    return np.asarray(np.linalg.solve(ideal, supercell_arr).T, dtype=np.float64)


def strain_tensor(
    gradient: ArrayLike,
    *,
    kind: str = "infinitesimal",
) -> NDArray[np.float64]:
    """Return a symmetric strain tensor from a deformation gradient.

    ``kind="infinitesimal"`` gives ``(D + D.T)/2 - I``, which is the usual
    engineering strain and is accurate to first order.  ``kind="green-lagrange"``
    gives ``(D.T @ D - I)/2``, which is exact and, unlike the infinitesimal
    strain, invariant under a rigid rotation of the supercell.
    """

    matrix = as_matrix3(gradient, name="gradient")
    identity = np.eye(3)
    if kind == "infinitesimal":
        return np.asarray(0.5 * (matrix + matrix.T) - identity, dtype=np.float64)
    if kind == "green-lagrange":
        return np.asarray(0.5 * (matrix.T @ matrix - identity), dtype=np.float64)
    raise ValueError(f"unknown strain kind {kind!r}; use 'infinitesimal' or 'green-lagrange'")


def rotation_angle(gradient: ArrayLike) -> float:
    """Return the rigid rotation hidden in ``D``, in degrees.

    From the polar decomposition ``D = R U`` with ``R`` orthogonal and ``U``
    symmetric positive definite.  A nonzero angle means the supercell is rotated
    relative to the reference: harmless if the reference is only a lattice, but
    it rotates every reported k-direction, so a band path expressed in the
    reference frame no longer points where it did.
    """

    matrix = as_matrix3(gradient, name="gradient")
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def kpoint_shift(
    gradient: ArrayLike,
    kpoints_cart: ArrayLike,
) -> NDArray[np.float64]:
    """Cartesian error made in each reported k-vector, ``q @ (I - inv(D))``.

    ``strain_kshift``: the reported Cartesian k is ``q`` and the point the
    strained cell actually describes is ``q @ inv(D)``, so the error is linear
    in ``q`` -- exactly zero at Gamma.
    """

    matrix = as_matrix3(gradient, name="gradient")
    points = np.atleast_2d(np.asarray(kpoints_cart, dtype=float))
    if points.shape[-1] != 3:
        raise ValueError("kpoints_cart must have shape (3,) or (n, 3)")
    return np.asarray(points @ (np.eye(3) - np.linalg.inv(matrix)), dtype=np.float64)


def _default_probe_kpoints() -> NDArray[np.float64]:
    """Corners of the fractional cube ``[-1/2, 1/2]^3``.

    The error grows with ``|k|``, so the worst case over the Brillouin zone is
    attained on the boundary of this box, which contains the first zone.
    """

    signs = np.array([-0.5, 0.5])
    grid = np.stack(np.meshgrid(signs, signs, signs, indexing="ij"), axis=-1)
    return np.asarray(grid.reshape(-1, 3), dtype=np.float64)


@dataclass(frozen=True)
class CommensurabilityReport:
    """Diagnosis of how far a supercell is from its primitive reference."""

    transform: NDArray[np.int64]
    residual: float
    commensurate: bool
    ambiguous: bool
    deformation_gradient: NDArray[np.float64]
    strain: NDArray[np.float64]
    max_abs_strain: float
    rotation_deg: float
    max_kpoint_shift: float
    reference_kpoint_norm: float

    @property
    def relative_kpoint_shift(self) -> float:
        """Worst-case k error as a fraction of the probe k-vector length."""

        if self.reference_kpoint_norm == 0.0:
            return 0.0
        return self.max_kpoint_shift / self.reference_kpoint_norm

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform.tolist(),
            "residual": self.residual,
            "commensurate": self.commensurate,
            "ambiguous": self.ambiguous,
            "deformation_gradient": self.deformation_gradient.tolist(),
            "strain": self.strain.tolist(),
            "max_abs_strain": self.max_abs_strain,
            "rotation_deg": self.rotation_deg,
            "max_kpoint_shift": self.max_kpoint_shift,
            "reference_kpoint_norm": self.reference_kpoint_norm,
            "relative_kpoint_shift": self.relative_kpoint_shift,
        }

    def summary(self) -> str:
        if self.commensurate:
            return (
                "commensurate: the supercell is an exact integer multiple of the "
                f"primitive cell (residual {self.residual:.2e})"
            )
        lines = [
            f"NOT commensurate: nearest integer transform leaves a residual of {self.residual:.3g}",
            f"  residual strain (max |eps_ij|): {self.max_abs_strain:.3g}",
            f"  rigid rotation:                 {self.rotation_deg:.3g} deg",
            f"  worst-case k error:             {self.max_kpoint_shift:.3g} "
            f"({100.0 * self.relative_kpoint_shift:.2g}% of |k| at the zone corner)",
        ]
        if self.ambiguous:
            lines.append(
                "  the residual is close to 1/2, so the nearest integer transform "
                "is a guess: no integer transform describes this pair"
            )
        lines.append(
            "  the k error is linear in k -- zero at Gamma, largest at the zone "
            "boundary -- and no choice of transform removes it"
        )
        return "\n".join(lines)


def diagnose_commensurability(
    primitive: Structure | ArrayLike,
    supercell: Structure | ArrayLike,
    *,
    atol: float = 1e-6,
    kpoints: ArrayLike | None = None,
) -> CommensurabilityReport:
    """Quantify the mismatch between a supercell and a primitive reference.

    ``kpoints`` are fractional *primitive* coordinates at which to measure the
    k-space error -- the unfolded band structure lives in the primitive zone --
    and default to the eight corners of ``[-1/2, 1/2]^3``.  The error is linear
    in k, so its maximum over any box is attained at a corner.
    """

    check_atol(atol)
    primitive_arr = _lattice_of(primitive, name="primitive_lattice")
    supercell_arr = _lattice_of(supercell, name="supercell_lattice")
    matrix, residual = nearest_integer_transformation(primitive_arr, supercell_arr)
    commensurate = residual <= atol

    gradient = deformation_gradient(primitive_arr, supercell_arr, matrix)
    strain = strain_tensor(gradient)
    probes = (
        _default_probe_kpoints()
        if kpoints is None
        else np.atleast_2d(np.asarray(kpoints, dtype=float))
    )
    if probes.shape[-1] != 3:
        raise ValueError("kpoints must have shape (3,) or (n, 3)")
    reciprocal = 2.0 * np.pi * np.linalg.inv(primitive_arr).T
    cart = probes @ reciprocal
    shifts = kpoint_shift(gradient, cart)
    norms = np.linalg.norm(shifts, axis=-1)
    index = int(np.argmax(norms)) if norms.size else 0

    return CommensurabilityReport(
        transform=matrix,
        residual=residual,
        commensurate=bool(commensurate),
        ambiguous=bool(residual >= AMBIGUOUS_RESIDUAL),
        deformation_gradient=gradient,
        strain=strain,
        max_abs_strain=float(np.max(np.abs(strain))) if strain.size else 0.0,
        rotation_deg=rotation_angle(gradient),
        max_kpoint_shift=float(norms[index]) if norms.size else 0.0,
        reference_kpoint_norm=float(np.linalg.norm(cart[index])) if norms.size else 0.0,
    )
