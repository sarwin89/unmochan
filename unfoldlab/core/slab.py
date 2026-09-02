"""Slabs, wires and other partially periodic cells.

A slab calculation is periodic in two directions only: the third holds vacuum,
and its cell length is a convergence parameter, not a physical period.  Wires
are periodic in one direction and a molecule in a box in none.  Unfolding such a
cell needs three things that a bulk unfolding does not:

*the transform must be trivial along the non-periodic axes*
    :func:`is_slab_transform` and :func:`diagnose_slab_transform` check it.  A
    transform that folds along the vacuum direction produces an artificial
    dispersion that depends on the vacuum thickness alone -- the two Lean
    theorems ``UnfoldLab.weight_vacuum_fold_half`` and
    ``UnfoldLab.weight_vacuum_fold_zero`` exhibit the smallest instance, where a
    single vacuum-direction plane wave is given *all* of its weight at
    ``k_perp = 1/2``;

*the unfolding path must keep the perpendicular k-component fixed*
    ``UnfoldLab.IsSlabTransform.foldsTo_perp``: for a slab transform every
    primitive k-point of a fiber shares its perpendicular component with the
    supercell k-point, modulo an integer.  There is nothing to unfold along the
    vacuum direction.  :func:`flat_kpoint_deviation` measures the violation;

*the perpendicular plane-wave index can be summed over first*
    ``UnfoldLab.IsSlabTransform.pwMatches_perp_free`` and
    ``UnfoldLab.IsSlabTransform.weight_perp_gauge``: the matching test never
    reads the perpendicular component of ``G``, so collapsing the plane-wave
    list over that index changes no weight.  :func:`collapse_perpendicular` does
    it, and :func:`slab_weights_from_coefficients` uses the collapsed list.  A
    slab with a long vacuum direction has many perpendicular shells, so this is
    where most of the matching work goes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import TransformLike, as_int_array, check_atol
from unfoldlab.core.plane_waves import weights_from_coefficients
from unfoldlab.core.structures import Structure

__all__ = [
    "SlabReport",
    "collapse_perpendicular",
    "detect_vacuum_axes",
    "diagnose_slab_transform",
    "embed_parallel_transform",
    "flat_kpoint_deviation",
    "is_slab_transform",
    "slab_weights_from_coefficients",
    "vacuum_gap",
]

#: Gap along an axis, in the length unit of the lattice (normally angstrom),
#: above which the axis is treated as non-periodic.  Five angstrom is the usual
#: minimum vacuum thickness of a converged slab calculation.
VACUUM_THRESHOLD = 5.0


def _axes_tuple(axes: int | ArrayLike) -> tuple[int, ...]:
    values = (axes,) if isinstance(axes, (int, np.integer)) else tuple(np.ravel(np.asarray(axes)))
    out = tuple(int(value) for value in values)
    for axis in out:
        if axis not in (0, 1, 2):
            raise ValueError(f"non-periodic axes must be 0, 1 or 2, got {axis}")
    if len(set(out)) != len(out):
        raise ValueError("non-periodic axes must be distinct")
    return out


def vacuum_gap(structure: Structure, axis: int) -> float:
    """Largest empty stretch along ``axis``, measured along the outward normal.

    The distance is taken along the normal of the plane spanned by the other two
    lattice vectors, which is the physical thickness even when the stacking
    vector is not orthogonal to that plane.  The gap is measured cyclically, so
    a slab centred at the cell boundary is handled like any other.  An empty
    structure has no sites and hence a gap of one full period.
    """

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    lattice = structure.lattice
    other = [i for i in (0, 1, 2) if i != axis]
    normal = np.cross(lattice[other[0]], lattice[other[1]])
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("the two lattice vectors transverse to axis are collinear")
    period = float(abs(np.dot(lattice[axis], normal / norm)))
    if period <= 0.0:
        raise ValueError("the stacking vector lies in the plane of the other two")
    if structure.n_sites == 0:
        return period

    heights = np.sort(np.mod(np.asarray(structure.cart_coords @ (normal / norm)), period))
    gaps = np.diff(heights, append=heights[0] + period)
    return float(np.max(gaps))


def detect_vacuum_axes(
    structure: Structure,
    *,
    threshold: float = VACUUM_THRESHOLD,
) -> tuple[int, ...]:
    """Axes of ``structure`` whose largest gap exceeds ``threshold``.

    This is a heuristic on the geometry, not a proof of anything: a genuinely
    porous bulk crystal can show a large gap along a periodic axis.  It exists
    so that :func:`diagnose_slab_transform` can be run without the user having
    to state which direction the vacuum is in.
    """

    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    return tuple(axis for axis in (0, 1, 2) if vacuum_gap(structure, axis) > threshold)


def is_slab_transform(
    transform: TransformLike,
    non_periodic_axes: int | ArrayLike,
    *,
    atol: float = 1e-6,
) -> bool:
    """Is ``transform`` trivial along every non-periodic axis?

    That means, for each such axis ``p``, that row ``p`` and column ``p`` of the
    transform are the unit vector ``e_p``: the cell is not multiplied along
    ``p`` and ``p`` is not mixed with the periodic directions.  This is
    ``UnfoldLab.IsSlabTransform``.
    """

    matrix = as_int_array(transform, name="transform", atol=check_atol(atol))
    if matrix.shape != (3, 3):
        raise ValueError(f"transform must have shape (3, 3), got {matrix.shape}")
    unit = np.eye(3, dtype=np.int64)
    return all(
        bool(np.array_equal(matrix[axis, :], unit[axis]))
        and bool(np.array_equal(matrix[:, axis], unit[axis]))
        for axis in _axes_tuple(non_periodic_axes)
    )


def embed_parallel_transform(
    parallel: ArrayLike,
    non_periodic_axis: int,
) -> NDArray[np.int64]:
    """Build a 3x3 slab transform from a 2x2 transform of the periodic plane.

    The rows and columns of ``parallel`` are taken in the order of the two
    periodic axes.  The result satisfies :func:`is_slab_transform` by
    construction, which is the easiest way to obtain a valid slab transform.
    """

    if non_periodic_axis not in (0, 1, 2):
        raise ValueError("non_periodic_axis must be 0, 1 or 2")
    block = as_int_array(parallel, name="parallel")
    if block.shape != (2, 2):
        raise ValueError(f"parallel must have shape (2, 2), got {block.shape}")
    axes = [i for i in (0, 1, 2) if i != non_periodic_axis]
    matrix = np.eye(3, dtype=np.int64)
    for i, row in enumerate(axes):
        for j, column in enumerate(axes):
            matrix[row, column] = block[i, j]
    return matrix


def flat_kpoint_deviation(
    primitive_kpoints: ArrayLike,
    folded_supercell_kpoint: ArrayLike,
    non_periodic_axes: int | ArrayLike,
) -> float:
    """How far a set of primitive k-points strays from the flat slab fiber.

    For a slab transform every primitive k-point folding onto ``K`` has the same
    perpendicular component as ``K`` modulo an integer
    (``UnfoldLab.IsSlabTransform.foldsTo_perp``), so an unfolding path that
    disperses along the vacuum direction is asking for weights that are all
    zero.  The return value is the largest deviation from an integer, in
    fractional units; it is zero for a correct path and at most ``1/2``.
    """

    kpoints = np.atleast_2d(np.asarray(primitive_kpoints, dtype=float))
    folded = np.asarray(folded_supercell_kpoint, dtype=float)
    if kpoints.shape[-1] != 3 or folded.shape != (3,):
        raise ValueError("k-points must have shape (3,) or (n, 3)")
    axes = list(_axes_tuple(non_periodic_axes))
    if not axes:
        return 0.0
    residual = kpoints[:, axes] - folded[axes][np.newaxis, :]
    return float(np.max(np.abs(residual - np.round(residual)))) if residual.size else 0.0


@dataclass(frozen=True)
class SlabReport:
    """Whether a transform is a legitimate transform of a partially periodic cell."""

    transform: NDArray[np.int64]
    non_periodic_axes: tuple[int, ...]
    folding_factors: tuple[int, ...]
    mixed_axes: tuple[int, ...]
    spurious_kpoints: int

    @property
    def valid(self) -> bool:
        """Is the transform trivial along every non-periodic axis?"""

        return self.spurious_kpoints == 1 and not self.mixed_axes

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform.tolist(),
            "non_periodic_axes": list(self.non_periodic_axes),
            "folding_factors": list(self.folding_factors),
            "mixed_axes": list(self.mixed_axes),
            "spurious_kpoints": self.spurious_kpoints,
            "valid": self.valid,
        }

    def summary(self) -> str:
        if not self.non_periodic_axes:
            return "fully periodic: every axis is a real period, nothing to check"
        axes = ", ".join(str(axis) for axis in self.non_periodic_axes)
        if self.valid:
            return (
                f"valid slab transform: trivial along the non-periodic axes ({axes}), "
                "so the fiber is flat and no vacuum-direction dispersion is invented"
            )
        lines = [f"INVALID slab transform along the non-periodic axes ({axes}):"]
        for axis, factor in zip(self.non_periodic_axes, self.folding_factors, strict=True):
            if factor != 1:
                lines.append(
                    f"  axis {axis} is folded {factor}-fold: the reference cell has "
                    "less vacuum than the supercell, which is not a physical statement"
                )
        for axis in self.mixed_axes:
            lines.append(
                f"  axis {axis} is mixed with a periodic direction: the vacuum "
                "thickness then enters the parallel k-vectors as well"
            )
        lines.append(
            f"  the fiber carries {self.spurious_kpoints} distinct perpendicular "
            "k-points, all of them artifacts of the chosen vacuum thickness"
        )
        return "\n".join(lines)


def diagnose_slab_transform(
    transform: TransformLike,
    non_periodic_axes: int | ArrayLike,
    *,
    atol: float = 1e-6,
) -> SlabReport:
    """Report whether ``transform`` unfolds a partially periodic cell correctly.

    ``spurious_kpoints`` is the number of distinct perpendicular k-points the
    fiber would contain, that is the product of the diagonal folding factors
    along the non-periodic axes.  Anything above one is an artificial dispersion
    along the vacuum direction.
    """

    matrix = as_int_array(transform, name="transform", atol=check_atol(atol))
    if matrix.shape != (3, 3):
        raise ValueError(f"transform must have shape (3, 3), got {matrix.shape}")
    axes = _axes_tuple(non_periodic_axes)
    factors = tuple(int(matrix[axis, axis]) for axis in axes)
    unit = np.eye(3, dtype=np.int64)
    mixed = tuple(
        axis
        for axis in axes
        if not (
            np.array_equal(np.delete(matrix[axis, :], axis), np.delete(unit[axis], axis))
            and np.array_equal(np.delete(matrix[:, axis], axis), np.delete(unit[axis], axis))
        )
    )
    spurious = 1
    for factor in factors:
        spurious *= abs(factor)
    return SlabReport(
        transform=matrix,
        non_periodic_axes=axes,
        folding_factors=factors,
        mixed_axes=mixed,
        spurious_kpoints=spurious,
    )


def collapse_perpendicular(
    g_supercell: ArrayLike,
    coefficients: NDArray[np.complexfloating],
    non_periodic_axes: int | ArrayLike,
    *,
    atol: float = 1e-6,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Sum the squared coefficients over the non-periodic plane-wave indices.

    ``coefficients`` has the usual shape ``(n_bands, n_components, n_g)``.  The
    result is a reduced G list, with the components along ``non_periodic_axes``
    set to zero and duplicates removed, together with an array of shape
    ``(n_bands, 1, n_reduced)`` holding the *amplitudes* whose squares are the
    summed norms.

    This is exact, not an approximation: the matching test for a slab transform
    never reads those components (``UnfoldLab.IsSlabTransform.pwMatches_perp_free``),
    so the weights computed from the reduced list are the weights of the full
    one (``UnfoldLab.IsSlabTransform.weight_perp_gauge``).  Components are summed
    at the same time, which the ordinary kernel does anyway.
    """

    g_int = as_int_array(g_supercell, name="g_supercell", atol=check_atol(atol))
    if g_int.ndim != 2 or g_int.shape[1] != 3:
        raise ValueError("g_supercell must have shape (n_g, 3)")
    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
    if coeffs.shape[2] != g_int.shape[0]:
        raise ValueError("coefficient G dimension must match number of G-vectors")

    projected = g_int.copy()
    projected[:, list(_axes_tuple(non_periodic_axes))] = 0
    reduced, inverse = np.unique(projected, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()

    norms = (coeffs.real**2 + coeffs.imag**2).sum(axis=1)
    summed = (
        np.stack(
            [
                np.bincount(inverse, weights=row, minlength=reduced.shape[0])
                for row in np.asarray(norms, dtype=np.float64)
            ]
        )
        if norms.shape[0]
        else np.zeros((0, reduced.shape[0]), dtype=np.float64)
    )
    return reduced.astype(np.int64), np.sqrt(summed)[:, np.newaxis, :]


def slab_weights_from_coefficients(
    g_supercell: ArrayLike,
    coefficients: NDArray[np.complexfloating],
    primitive_kpoint: ArrayLike,
    folded_supercell_kpoint: ArrayLike,
    transform: TransformLike,
    non_periodic_axes: int | ArrayLike,
    *,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Unfolded weights of a slab, computed on the collapsed plane-wave list.

    Identical in value to
    :func:`unfoldlab.core.plane_waves.weights_from_coefficients` whenever the
    transform is a slab transform for the given axes -- and cheaper, because the
    exact integer matching test runs once per *parallel* G-vector instead of
    once per stored plane wave.  The saving is the number of perpendicular
    shells, which for a converged slab is the largest factor in the list.

    A transform that is not trivial along the non-periodic axes is rejected:
    collapsing would then change the answer, and the answer it would change is
    itself an artifact (see :func:`diagnose_slab_transform`).
    """

    if not is_slab_transform(transform, non_periodic_axes, atol=tol):
        raise ValueError(
            "transform is not trivial along the non-periodic axes; "
            "run diagnose_slab_transform for the details"
        )
    reduced_g, reduced_coeffs = collapse_perpendicular(
        g_supercell, coefficients, non_periodic_axes, atol=tol
    )
    return weights_from_coefficients(
        reduced_g,
        reduced_coeffs.astype(np.complex128),
        np.asarray(primitive_kpoint, dtype=float),
        np.asarray(folded_supercell_kpoint, dtype=float),
        transform,
        tol=tol,
    )
