"""K-point mapping and path primitives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import as_array3, wrap_fractional
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
        primitive = item if isinstance(item, KPoint) else KPoint(item)
        mappings.append(
            KPointMapping(
                primitive=primitive,
                supercell=fold_kpoint_to_supercell(primitive, transform),
            )
        )
    return mappings


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
