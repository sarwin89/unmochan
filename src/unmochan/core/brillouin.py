"""First-Brillouin-zone (Wigner--Seitz) representatives of a k-point.

Why this exists
---------------

A fractional reciprocal coordinate names a k-point only modulo the reciprocal
lattice, so anything that *reports* a k-point -- an unfolded band path, a
constant-energy cut, a valley assignment, an unfolded mesh -- has to choose a
representative.  Two choices are in circulation:

*the parallelepiped representative*
    ``f - rint(f)``, i.e. each fractional component reduced into
    ``[-1/2, 1/2]``.  This is what :func:`unmochan.core.numerics.wrap_fractional`
    and every ``rint`` call in the package produce.  It is the right answer for
    the *matching* arithmetic, which only cares about the coset;

*the first Brillouin zone representative*
    the member of the coset ``f + Z^3`` of smallest **Cartesian** length: the
    Wigner--Seitz cell of the reciprocal lattice.  This is the right answer for
    anything geometric -- a distance, a radius, a plot, a zone-boundary
    statement.

They coincide for an orthogonal reciprocal basis and differ for a skewed one.
``RequestProject/Unfolding/BrillouinZone.lean`` settles the mathematics:

*the zone is an intersection of half spaces*
    ``UnfoldLab.isBZRepr_iff_bragg``: ``k`` is in the zone exactly when
    ``|2 <k, g>| <= ||g||^2`` for every reciprocal-lattice vector ``g``, so
    membership is a scan over lattice vectors and needs no minimization.
    :func:`zone_boundary_distance` returns the smallest slack of those
    inequalities, in Cartesian units, and :func:`is_in_first_bz` its sign;

*a small enough k-point is in the zone for free*
    ``UnfoldLab.isBZRepr_of_two_mul_lt``: if ``2||k||`` is less than the length
    of every nonzero lattice vector then ``k`` is in the zone.
    :func:`inscribed_radius` computes that radius;

*the representative is unique up to the boundary*
    ``UnfoldLab.bragg_eq_of_isBZRepr`` and
    ``UnfoldLab.eq_of_isBZRepr_of_strict``: two zone representatives of one
    coset have equal length and differ by a vector whose Bragg plane they both
    lie on, so with all inequalities strict the representative is unique;

*rounding is exact for an orthogonal cell and wrong otherwise*
    ``UnfoldLab.isBZRepr_of_rounded_of_orthogonal`` versus
    ``UnfoldLab.exists_rounded_not_isBZRepr``, which exhibits a rounded
    representative five times longer (in squared length) than the zone one.
    :func:`diagnose_bz_reduction` measures that discrepancy for a real k-point
    list;

*the finite search is certified*
    ``UnfoldLab.isBZRepr_of_box_certificate``: the growing-box search of
    :func:`unmochan.core.valleys.certified_minimum_image_shift` stops only when
    no translation outside the box can win, so :func:`reduce_to_first_bz`
    returns a genuine zone representative rather than a nearest-shell guess;

*and reduction is harmless for the unfolding itself*
    ``UnfoldLab.foldsTo_add_ivec``: moving a k-point by a reciprocal-lattice
    vector does not change which supercell k-point it folds to, so weights and
    fiber sum rules are untouched by the choice of representative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.numerics import as_array3, as_matrix3, check_atol
from unmochan.core.valleys import (
    MAX_IMAGE_SEARCH_RADIUS,
    box_shifts,
    certified_minimum_image_shift,
    lattice_growth_constant,
)

#: Relative slack subtracted from a certificate so that floating-point error
#: cannot turn a failing certificate into a passing one.
_CERTIFICATE_SLACK = 1e-12


def _reciprocal_matrix(reciprocal_lattice: ArrayLike | None) -> NDArray[np.float64]:
    """Rows = Cartesian reciprocal-lattice vectors; the identity when unknown."""

    if reciprocal_lattice is None:
        return np.eye(3)
    return as_matrix3(reciprocal_lattice, name="reciprocal_lattice")


@dataclass(frozen=True)
class BZPoint:
    """A k-point reduced into the first Brillouin zone."""

    #: Fractional coordinates of the zone representative.
    fractional: NDArray[np.float64]
    #: Integer translation applied to the input to reach it.
    shift: NDArray[np.int64]
    #: Cartesian coordinates of the zone representative.
    cartesian: NDArray[np.float64]
    #: Cartesian length of the zone representative.
    length: float
    #: Cartesian length of the componentwise-rounded representative.
    rounded_length: float

    @property
    def rounding_suffices(self) -> bool:
        """Whether componentwise rounding already gave the zone representative."""

        tolerance = 1e-10 * max(1.0, self.rounded_length)
        return self.rounded_length <= self.length + tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "fractional": [float(value) for value in self.fractional],
            "shift": [int(value) for value in self.shift],
            "cartesian": [float(value) for value in self.cartesian],
            "length": self.length,
            "rounded_length": self.rounded_length,
            "rounding_suffices": self.rounding_suffices,
        }


def reduce_to_first_bz(
    kpoint_frac: ArrayLike,
    reciprocal_lattice: ArrayLike | None = None,
    *,
    max_radius: int = MAX_IMAGE_SEARCH_RADIUS,
) -> BZPoint:
    """Move a k-point to the shortest member of its reciprocal-lattice coset.

    ``reciprocal_lattice`` holds the Cartesian reciprocal-lattice vectors as
    rows (``2*pi*inv(A).T`` for a real lattice ``A`` with rows as vectors).
    Without it the fractional Euclidean norm is minimized instead, which is the
    same thing only for a cubic cell.

    The search is the certified one of
    :func:`unmochan.core.valleys.certified_minimum_image_shift`, so the result
    is the global minimum over ``Z^3``; ``UnfoldLab.isBZRepr_of_box_certificate``
    turns that certificate into zone membership.
    """

    fractional = as_array3(kpoint_frac, name="kpoint_frac")
    matrix = _reciprocal_matrix(reciprocal_lattice)
    shift, length = certified_minimum_image_shift(fractional, matrix, max_radius=max_radius)
    reduced = fractional + shift.astype(float)
    rounded = fractional - np.rint(fractional)
    return BZPoint(
        fractional=reduced,
        shift=shift,
        cartesian=reduced @ matrix,
        length=length,
        rounded_length=float(np.linalg.norm(rounded @ matrix)),
    )


def reduce_kpoints_to_first_bz(
    kpoints_frac: ArrayLike,
    reciprocal_lattice: ArrayLike | None = None,
    *,
    max_radius: int = MAX_IMAGE_SEARCH_RADIUS,
) -> NDArray[np.float64]:
    """Zone representatives of a list of k-points, as an ``(n, 3)`` array."""

    points = np.atleast_2d(np.asarray(kpoints_frac, dtype=float))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("kpoints_frac must have shape (n, 3)")
    reduced = [
        reduce_to_first_bz(point, reciprocal_lattice, max_radius=max_radius).fractional
        for point in points
    ]
    return np.asarray(reduced, dtype=float).reshape(points.shape)


def shortest_lattice_vector_length(
    reciprocal_lattice: ArrayLike | None = None,
    *,
    max_radius: int = MAX_IMAGE_SEARCH_RADIUS,
) -> float:
    """Length of the shortest nonzero reciprocal-lattice vector.

    Certified in the same way as the minimum-image search: a translation outside
    the box of radius ``r`` has coefficient norm at least ``r + 1`` and hence
    Cartesian length at least ``sigma * (r + 1)``, with ``sigma`` the growth
    constant of :func:`unmochan.core.valleys.lattice_growth_constant`
    (``UnfoldLab.abs_det_mul_euclidLen_le``).  The box grows until the best
    length found beats that bound.
    """

    matrix = _reciprocal_matrix(reciprocal_lattice)
    sigma = lattice_growth_constant(matrix, name="reciprocal_lattice")
    for radius in range(1, max_radius + 1):
        shifts = box_shifts(radius)
        nonzero = shifts[np.any(shifts != 0.0, axis=1)]
        lengths = np.linalg.norm(nonzero @ matrix, axis=1)
        best = float(np.min(lengths))
        if best <= sigma * (radius + 1) * (1.0 - _CERTIFICATE_SLACK):
            return best
    raise RuntimeError(
        "shortest-vector search did not certify a global minimum within "
        f"max_radius={max_radius}; increase max_radius."
    )


def inscribed_radius(
    reciprocal_lattice: ArrayLike | None = None,
    *,
    max_radius: int = MAX_IMAGE_SEARCH_RADIUS,
) -> float:
    """Radius of the largest sphere around Gamma inside the first zone.

    Half the shortest nonzero reciprocal-lattice vector: a k-point strictly
    inside this sphere is in the zone without any search
    (``UnfoldLab.isBZRepr_of_two_mul_lt``).
    """

    return 0.5 * shortest_lattice_vector_length(reciprocal_lattice, max_radius=max_radius)


def zone_boundary_distance(
    kpoint_frac: ArrayLike,
    reciprocal_lattice: ArrayLike | None = None,
    *,
    max_radius: int = MAX_IMAGE_SEARCH_RADIUS,
) -> float:
    """Signed Cartesian distance from a k-point to the nearest Bragg plane.

    The Bragg plane of a reciprocal-lattice vector ``g`` is
    ``{x : 2 <x, g> = ||g||^2}``, at signed distance
    ``(||g||^2 - 2 <k, g>) / (2 ||g||)`` from ``k``.  The zone is the
    intersection of the half spaces where all of these are non-negative
    (``UnfoldLab.isBZRepr_iff_bragg``), so the minimum over ``g`` is positive in
    the interior, zero on the boundary and negative outside.

    Only finitely many ``g`` can attain the minimum: the distance to the plane
    of ``g`` is at least ``||g||/2 - ||k||``, so once ``sigma (r + 1) / 2 -
    ||k||`` exceeds the best value found in the box of radius ``r``, no
    translation outside it can improve on it.  The box grows until that holds.
    """

    kpoint = as_array3(kpoint_frac, name="kpoint_frac")
    matrix = _reciprocal_matrix(reciprocal_lattice)
    sigma = lattice_growth_constant(matrix, name="reciprocal_lattice")
    cartesian = kpoint @ matrix
    norm_k = float(np.linalg.norm(cartesian))

    for radius in range(1, max_radius + 1):
        shifts = box_shifts(radius)
        vectors = shifts[np.any(shifts != 0.0, axis=1)] @ matrix
        lengths = np.linalg.norm(vectors, axis=1)
        best = float(np.min((lengths**2 - 2.0 * (vectors @ cartesian)) / (2.0 * lengths)))
        if best <= sigma * (radius + 1) / 2.0 - norm_k:
            return best

    raise RuntimeError(
        "Bragg-plane search did not certify a global minimum within "
        f"max_radius={max_radius}; increase max_radius."
    )


def is_in_first_bz(
    kpoint_frac: ArrayLike,
    reciprocal_lattice: ArrayLike | None = None,
    *,
    atol: float = 1e-8,
    max_radius: int = MAX_IMAGE_SEARCH_RADIUS,
) -> bool:
    """Whether a k-point is in the first Brillouin zone, boundary included."""

    tolerance = check_atol(atol)
    return (
        zone_boundary_distance(kpoint_frac, reciprocal_lattice, max_radius=max_radius) >= -tolerance
    )


def has_orthogonal_reciprocal_basis(
    reciprocal_lattice: ArrayLike | None = None, *, atol: float = 1e-8
) -> bool:
    """Whether the reciprocal basis vectors are mutually orthogonal.

    Exactly the case in which componentwise rounding is already the zone
    reduction (``UnfoldLab.isBZRepr_of_rounded_of_orthogonal``).  The test is
    scale-free: the off-diagonal Gram entries are compared with the product of
    the two row lengths.
    """

    tolerance = check_atol(atol)
    matrix = _reciprocal_matrix(reciprocal_lattice)
    gram = matrix @ matrix.T
    lengths = np.sqrt(np.diag(gram))
    if np.any(lengths <= 0.0):
        raise ValueError("reciprocal_lattice must have nonzero rows")
    scaled = gram / np.outer(lengths, lengths)
    off_diagonal = scaled - np.diag(np.diag(scaled))
    return bool(np.all(np.abs(off_diagonal) <= tolerance))


@dataclass(frozen=True)
class BZReductionReport:
    """How much the choice of representative matters for a k-point list."""

    #: Number of k-points examined.
    n_points: int
    #: How many of them the rounded representative places outside the zone.
    n_moved: int
    #: Largest ``rounded_length / zone_length`` over the list (1.0 if none moved).
    max_length_ratio: float
    #: Whether the reciprocal basis is orthogonal, so that rounding is exact.
    orthogonal_basis: bool
    #: Radius of the sphere inscribed in the zone.
    inscribed_radius: float
    #: The zone representatives, ``(n, 3)`` fractional.
    reduced: NDArray[np.float64]

    @property
    def rounding_is_exact(self) -> bool:
        """Whether rounding produced the zone representative for every point."""

        return self.n_moved == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "n_moved": self.n_moved,
            "max_length_ratio": self.max_length_ratio,
            "orthogonal_basis": self.orthogonal_basis,
            "inscribed_radius": self.inscribed_radius,
            "rounding_is_exact": self.rounding_is_exact,
            "reduced": [[float(value) for value in row] for row in self.reduced],
        }

    def summary(self) -> str:
        if self.rounding_is_exact:
            reason = (
                "the reciprocal basis is orthogonal, so componentwise rounding is "
                "the zone reduction"
                if self.orthogonal_basis
                else "no k-point of this list left the zone under rounding"
            )
            return f"All {self.n_points} k-points are already in the first zone: {reason}."
        return (
            f"{self.n_moved} of {self.n_points} k-points are outside the first "
            f"Brillouin zone when reduced by componentwise rounding; the worst is "
            f"{self.max_length_ratio:.3f} times too long.  Report the reduced "
            "coordinates instead for any distance, radius or plot."
        )


def diagnose_bz_reduction(
    kpoints_frac: ArrayLike,
    reciprocal_lattice: ArrayLike | None = None,
    *,
    max_radius: int = MAX_IMAGE_SEARCH_RADIUS,
) -> BZReductionReport:
    """Compare rounding with zone reduction over a list of k-points."""

    points = np.atleast_2d(np.asarray(kpoints_frac, dtype=float))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("kpoints_frac must have shape (n, 3)")

    reduced_points: list[NDArray[np.float64]] = []
    moved = 0
    ratio = 1.0
    for point in points:
        result = reduce_to_first_bz(point, reciprocal_lattice, max_radius=max_radius)
        reduced_points.append(result.fractional)
        if not result.rounding_suffices:
            moved += 1
            if result.length > 0.0:
                ratio = max(ratio, result.rounded_length / result.length)

    return BZReductionReport(
        n_points=int(points.shape[0]),
        n_moved=moved,
        max_length_ratio=ratio,
        orthogonal_basis=has_orthogonal_reciprocal_basis(reciprocal_lattice),
        inscribed_radius=inscribed_radius(reciprocal_lattice, max_radius=max_radius),
        reduced=np.asarray(reduced_points, dtype=float).reshape(points.shape),
    )
