"""K-point mapping and path primitives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.lattice_quotient import coset_representatives
from unfoldlab.core.numerics import as_array3, as_matrix3, wrap_fractional
from unfoldlab.core.transformations import TransformationMatrix


@dataclass(frozen=True)
class KPoint:
    """Fractional reciprocal-coordinate k-point."""

    fractional: NDArray[np.float64]
    weight: float = 1.0
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fractional", as_array3(self.fractional, name="fractional"))
        object.__setattr__(self, "weight", float(self.weight))

    def wrapped(self) -> KPoint:
        return KPoint(wrap_fractional(self.fractional), weight=self.weight, label=self.label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fractional": self.fractional.tolist(),
            "weight": self.weight,
            "label": self.label,
        }


@dataclass(frozen=True)
class KPointMapping:
    """One primitive k-point and its folded supercell representative."""

    primitive: KPoint
    supercell: KPoint

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive.to_dict(),
            "supercell": self.supercell.to_dict(),
        }


def fold_kpoint_to_supercell(
    primitive_kpoint: KPoint | ArrayLike,
    transform: TransformationMatrix,
    *,
    label: str | None = None,
) -> KPoint:
    """Fold a primitive fractional k-point into supercell fractional coordinates.

    With row-vector real-space lattices and ``A_s = T @ A_p``, reciprocal
    fractional coordinates transform as ``k_s = k_p @ T.T`` modulo reciprocal
    lattice vectors.
    """

    if isinstance(primitive_kpoint, KPoint):
        fractional = primitive_kpoint.fractional
        weight = primitive_kpoint.weight
        out_label = primitive_kpoint.label if label is None else label
    else:
        fractional = as_array3(primitive_kpoint, name="primitive_kpoint")
        weight = 1.0
        out_label = label

    folded = wrap_fractional(fractional @ transform.matrix.T)
    return KPoint(folded, weight=weight, label=out_label)


def fold_kpoints_to_supercell(
    primitive_kpoints: Iterable[KPoint | ArrayLike],
    transform: TransformationMatrix,
) -> list[KPointMapping]:
    """Fold primitive k-points and retain the primitive/supercell relationship."""

    mappings: list[KPointMapping] = []
    for item in primitive_kpoints:
        primitive = item if isinstance(item, KPoint) else KPoint(as_array3(item, name="kpoint"))
        mappings.append(
            KPointMapping(
                primitive=primitive,
                supercell=fold_kpoint_to_supercell(primitive, transform),
            )
        )
    return mappings


def cartesian_path_distances(
    kpoints: ArrayLike,
    reciprocal_lattice: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Cumulative band-path abscissa of a list of fractional k-points.

    With ``reciprocal_lattice`` (rows = Cartesian reciprocal lattice vectors,
    e.g. ``2*pi*inv(A).T``) consecutive k-points are separated by their physical
    distance ``|Delta k_frac @ B|``.  Without it the Euclidean norm of the
    fractional difference is used, which misrepresents the relative length of
    path segments for every non-cubic reciprocal lattice -- two displacements
    can have equal fractional norm and different physical length
    (``UnfoldLab.exists_equal_fractional_distinct_cartesian``).
    """

    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("kpoints must have shape (n_kpoints, 3)")
    if points.shape[0] == 0:
        return np.zeros(0, dtype=float)
    deltas = np.diff(points, axis=0)
    if reciprocal_lattice is not None:
        deltas = deltas @ as_matrix3(reciprocal_lattice, name="reciprocal_lattice")
    steps = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def match_kpoints_modulo_lattice(
    requested: ArrayLike,
    available: ArrayLike,
    *,
    atol: float = 1e-6,
) -> NDArray[np.int64]:
    """Index in ``available`` of each ``requested`` k-point, modulo integers.

    Both arrays hold fractional reciprocal coordinates in the *same* basis.  Two
    k-points match when they differ by a reciprocal lattice vector, i.e. when
    every component of their difference is an integer to within ``atol``; that
    is the right notion because a reciprocal lattice translation is a relabeling
    of the same k-point (``UnfoldLab.cartesian_fold_eq``).

    Raises when a requested k-point has no match, or more than one, since
    silently taking the first of several would hide a duplicated or degenerate
    k-point list.
    """

    wanted = np.asarray(requested, dtype=float)
    have = np.asarray(available, dtype=float)
    if wanted.ndim != 2 or wanted.shape[1] != 3:
        raise ValueError("requested must have shape (n_requested, 3)")
    if have.ndim != 2 or have.shape[1] != 3:
        raise ValueError("available must have shape (n_available, 3)")
    # The pairwise difference is (n_requested, n_available, 3), which for the
    # k-point lists a mesh unfolding produces is far too large to materialize
    # at once, so it is built in row blocks of bounded size and reduced as it
    # goes.  Only the match count and the first match survive each block.
    n_requested = wanted.shape[0]
    n_available = have.shape[0]
    block = max(1, (1 << 20) // max(1, n_available))
    counts = np.zeros(n_requested, dtype=np.int64)
    first = np.zeros(n_requested, dtype=np.int64)
    for start in range(0, n_requested, block):
        stop = min(start + block, n_requested)
        delta = wanted[start:stop, None, :] - have[None, :, :]
        delta -= np.rint(delta)
        hits = np.all(np.abs(delta) <= atol, axis=2)
        counts[start:stop] = hits.sum(axis=1)
        first[start:stop] = np.argmax(hits, axis=1)
    if np.any(counts == 0):
        missing = int(np.flatnonzero(counts == 0)[0])
        raise ValueError(
            f"k-point {wanted[missing].tolist()} is not present in the available list (atol={atol})"
        )
    if np.any(counts > 1):
        ambiguous = int(np.flatnonzero(counts > 1)[0])
        raise ValueError(
            f"k-point {wanted[ambiguous].tolist()} matches {int(counts[ambiguous])} "
            "entries of the available list"
        )
    return first


def interpolate_segment(
    start: ArrayLike,
    end: ArrayLike,
    *,
    n_points: int,
    start_label: str | None = None,
    end_label: str | None = None,
) -> list[KPoint]:
    """Interpolate a straight fractional segment including both endpoints."""

    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    start_arr = as_array3(start, name="start")
    end_arr = as_array3(end, name="end")
    out: list[KPoint] = []
    for idx, t in enumerate(np.linspace(0.0, 1.0, n_points)):
        label = start_label if idx == 0 else end_label if idx == n_points - 1 else None
        out.append(KPoint((1.0 - t) * start_arr + t * end_arr, label=label))
    return out


def fiber_kpoints(
    supercell_kpoint: ArrayLike,
    transform: TransformationMatrix,
    *,
    wrap: bool = True,
) -> NDArray[np.float64]:
    """The primitive k-points that fold onto ``supercell_kpoint``.

    Returns a ``(|det T|, 3)`` array of fractional primitive coordinates, sorted
    lexicographically so the output is reproducible.  Folding any row with
    :func:`fold_kpoint_to_supercell` returns ``supercell_kpoint`` modulo a
    reciprocal lattice vector, and the rows are pairwise inequivalent.

    This is the inverse of folding, and it is what an unfolded band structure on
    a mesh needs: a supercell calculation samples the small Brillouin zone, and
    each of its k-points carries the spectral weight of exactly ``|det T|``
    primitive k-points (``UnfoldLab.IsFiberRepr.card_eq_natAbs_det``).  The
    representatives come from the Smith normal form of ``T``
    (:func:`unfoldlab.core.lattice_quotient.coset_representatives`), and the
    construction is certified by ``UnfoldLab.IsCosetRepr.isFiberRepr``.

    With ``wrap=False`` the raw solutions of ``T k = K + m`` are returned
    instead of their images in ``[0, 1)``.
    """

    folded = as_array3(supercell_kpoint, name="supercell_kpoint")
    offsets = coset_representatives(transform.matrix)
    shifted = folded[np.newaxis, :] + offsets.astype(float)
    # ``k = inv(T) (K + m)`` via the exact integer adjugate, which avoids the
    # extra conditioning of a general matrix inverse.
    points = shifted @ transform.adjugate.astype(float).T / float(transform.determinant)
    if wrap:
        points = wrap_fractional(points)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return np.ascontiguousarray(points[order])


def fiber_kpoint_mappings(
    supercell_kpoints: Iterable[KPoint | ArrayLike],
    transform: TransformationMatrix,
) -> list[KPointMapping]:
    """Expand supercell k-points into their complete primitive fibers.

    For each supercell k-point this emits the ``|det T|`` mappings whose
    primitive members fold onto it, in the order given by :func:`fiber_kpoints`.
    Feeding the result to an unfolding run makes every fiber complete, which is
    exactly the situation in which the sum rule
    (``UnfoldLab.IsFiberRepr.sum_weight_eq_one``) is an equality and the
    diagnostic can verify the calculation rather than merely bound it.

    The primitive weight is the supercell weight divided by ``|det T|``, so the
    total weight of the expanded list equals that of the input.
    """

    mappings: list[KPointMapping] = []
    for item in supercell_kpoints:
        supercell = item if isinstance(item, KPoint) else KPoint(as_array3(item, name="kpoint"))
        members = fiber_kpoints(supercell.fractional, transform)
        share = supercell.weight / float(transform.multiplicity)
        for point in members:
            mappings.append(
                KPointMapping(
                    primitive=KPoint(point, weight=share, label=supercell.label),
                    supercell=supercell,
                )
            )
    return mappings
