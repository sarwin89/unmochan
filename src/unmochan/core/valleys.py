"""Generic valley definitions.

Valley labels are convention- and material-dependent. The core library stores
explicit user or symmetry-derived definitions; it does not assume any built-in
valley names or coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.numerics import as_array3, as_matrix3

#: Largest search box the certified minimum-image search will try before giving
#: up.  A box of radius ``r`` contains ``(2r+1)**3`` translations, so this is a
#: safety net for pathologically skewed cells rather than an expected limit.
MAX_IMAGE_SEARCH_RADIUS = 12

#: Relative slack subtracted from the certificate so that floating-point error
#: can never turn a failing certificate into a passing one.
_CERTIFICATE_SLACK = 1e-12


@cache
def box_shifts(radius: int) -> NDArray[np.float64]:
    """All integer translations with ``max(|m_i|) <= radius``."""

    span = range(-radius, radius + 1)
    return np.array([(i, j, k) for i in span for j in span for k in span], dtype=float)


def lattice_growth_constant(matrix: ArrayLike, *, name: str = "matrix") -> float:
    """Certified lower bound for the length of a lattice translation.

    Returns a ``sigma > 0`` with ``||m @ B|| >= sigma * ||m||`` for every real
    coefficient vector ``m``, where ``B`` is the row-vector lattice ``matrix``.
    The bound used is ``sigma = |det B| / ||adj(B)||_F``, which follows from
    ``adj(B) B = det(B) I`` and the Frobenius bound ``||A x|| <= ||A||_F ||x||``
    (formalized as ``UnfoldLab.abs_det_mul_euclidLen_le``).  It is a lower bound
    for the smallest singular value of ``B`` and needs no eigenvalue solver.
    """

    lattice = as_matrix3(matrix, name=name)
    determinant = float(np.linalg.det(lattice))
    if not np.isfinite(determinant) or determinant == 0.0:
        raise ValueError(f"{name} must be invertible")
    adjugate = determinant * np.linalg.inv(lattice)
    frobenius = float(np.linalg.norm(adjugate))
    if not np.isfinite(frobenius) or frobenius <= 0.0:
        raise ValueError(f"{name} must be invertible")
    return abs(determinant) / frobenius


def certified_minimum_image_shift(
    delta_frac: ArrayLike,
    reciprocal_lattice: ArrayLike | None = None,
    *,
    max_radius: int = MAX_IMAGE_SEARCH_RADIUS,
) -> tuple[NDArray[np.int64], float]:
    """Certified minimum image: the winning translation and its distance.

    Returns ``(shift, distance)`` where ``shift`` is the integer translation
    ``m`` for which ``||(delta_frac + m) @ B||`` is smallest -- the minimum over
    *all* of ``Z^3``, not merely over a fixed neighbour shell -- and
    ``distance`` is that Cartesian length.  See :func:`minimum_image_distance`
    for the certificate that makes the finite search exact.

    The shift is what :mod:`unmochan.core.brillouin` needs in order to move a
    k-point into the first Brillouin zone; the distance alone is what a valley
    radius test needs.
    """

    if max_radius < 1:
        raise ValueError("max_radius must be at least 1")

    delta = as_array3(delta_frac, name="delta_frac")
    rounding = np.rint(delta)
    reduced = delta - rounding
    if reciprocal_lattice is None:
        matrix = np.eye(3)
        sigma = 1.0
    else:
        matrix = as_matrix3(reciprocal_lattice, name="reciprocal_lattice")
        sigma = lattice_growth_constant(matrix, name="reciprocal_lattice")

    reduced_norm = float(np.linalg.norm(reduced))

    for radius in range(1, max_radius + 1):
        shifts = box_shifts(radius)
        candidates = (reduced + shifts) @ matrix
        norms = np.linalg.norm(candidates, axis=1)
        index = int(np.argmin(norms))
        best = float(norms[index])
        bound = sigma * (radius + 1 - reduced_norm)
        if best <= bound * (1.0 - _CERTIFICATE_SLACK):
            shift = np.rint(shifts[index] - rounding).astype(np.int64)
            return shift, best

    raise RuntimeError(
        "minimum-image search did not certify a global minimum within "
        f"max_radius={max_radius}; the reciprocal lattice is extremely skewed "
        f"(growth constant sigma={sigma:g}).  Increase max_radius."
    )


def minimum_image_distance(
    delta_frac: ArrayLike,
    reciprocal_lattice: ArrayLike | None = None,
    *,
    max_radius: int = MAX_IMAGE_SEARCH_RADIUS,
) -> float:
    """Shortest distance between two k-points differing by ``delta_frac``.

    ``delta_frac`` is a difference of fractional reciprocal coordinates.  When
    ``reciprocal_lattice`` is given (rows = reciprocal lattice vectors in
    Cartesian coordinates, e.g. ``2*pi*inv(A).T``) the returned distance is the
    Cartesian one, which is the only meaningful notion of "radius" in
    reciprocal space.  Without it the Euclidean norm of the fractional
    difference is returned; that quantity depends on the cell shape and is only
    correct for a cubic reciprocal lattice.

    The minimum is taken over reciprocal-lattice translations, so the result is
    the distance between the two k-points as points of the Brillouin-zone
    torus.

    The search is *certified*, not heuristic.  Componentwise ``rint`` reduction
    is not the minimum image for a non-orthogonal cell
    (``UnfoldLab.exists_shorter_image_than_reduced``), and no fixed number of
    neighbour shells is correct for every cell shape.  This implementation
    therefore searches a box of translations of growing radius ``r`` and stops
    only when the best distance ``v`` found so far satisfies

    ``v <= sigma * (r + 1 - ||delta_reduced||)``,

    with ``sigma`` from :func:`lattice_growth_constant` and ``delta_reduced``
    the fractional difference reduced into ``[-1/2, 1/2]``.  A translation
    outside the box has coefficient norm at least ``r + 1``, so its Cartesian
    distance to the k-point is at least ``sigma * (r + 1 - ||delta_reduced||)``
    and it cannot beat ``v``.  This is
    ``UnfoldLab.minimum_image_box_exact_frac`` in
    ``RequestProject/Unfolding/MinimumImage.lean``.
    """

    _, distance = certified_minimum_image_shift(
        delta_frac, reciprocal_lattice, max_radius=max_radius
    )
    return distance


@dataclass(frozen=True)
class ValleyDefinition:
    """A named reciprocal-space region around a user-defined center."""

    label: str
    center_frac: NDArray[np.float64]
    radius: float
    reference_bz: str | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("valley label must not be empty")
        if self.radius <= 0:
            raise ValueError("valley radius must be positive")
        object.__setattr__(self, "center_frac", as_array3(self.center_frac, name="center_frac"))
        object.__setattr__(self, "radius", float(self.radius))

    @classmethod
    def from_config(cls, label: str, config: dict[str, Any]) -> ValleyDefinition:
        return cls(
            label=label,
            center_frac=config["center_frac"],
            radius=config["radius"],
            reference_bz=config.get("reference_bz"),
        )

    def distance_to(
        self,
        kpoint_frac: ArrayLike,
        reciprocal_lattice: ArrayLike | None = None,
    ) -> float:
        """Minimum-image distance from ``kpoint_frac`` to this valley center."""

        delta = as_array3(kpoint_frac, name="kpoint_frac") - self.center_frac
        return minimum_image_distance(delta, reciprocal_lattice)

    def contains(
        self,
        kpoint_frac: ArrayLike,
        reciprocal_lattice: ArrayLike | None = None,
    ) -> bool:
        """Return whether a fractional k-point lies inside this valley region.

        ``reciprocal_lattice`` (rows = Cartesian reciprocal lattice vectors)
        makes the test a genuine sphere of radius :attr:`radius` in reciprocal
        space.  If it is omitted the radius is measured with the Euclidean norm
        of the *fractional* difference, which distorts the region for any
        non-cubic cell -- for a hexagonal lattice, for instance, the fractional
        norm is not proportional to the physical distance in any direction.
        Pass the reciprocal lattice whenever the valley radius is quoted in
        inverse length units.
        """

        return self.distance_to(kpoint_frac, reciprocal_lattice) <= self.radius

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "center_frac": self.center_frac.tolist(),
            "radius": self.radius,
            "reference_bz": self.reference_bz,
        }
