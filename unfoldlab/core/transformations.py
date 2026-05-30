"""Primitive-to-supercell transformation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import as_matrix3
from unfoldlab.core.structures import Structure


@dataclass(frozen=True)
class TransformationMatrix:
    """Integer primitive-to-supercell transform using row-vector lattices.

    If ``A_p`` is the primitive lattice and ``A_s`` is the supercell lattice,
    then ``A_s = T @ A_p``.
    """

    matrix: NDArray[np.int64]
    residual: float = 0.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=int)
        if matrix.shape != (3, 3):
            raise ValueError(f"matrix must have shape (3, 3), got {matrix.shape}")
        det = round(float(np.linalg.det(matrix)))
        if det == 0:
            raise ValueError("transformation matrix must be invertible")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "residual", float(self.residual))

    @property
    def determinant(self) -> int:
        return int(round(float(np.linalg.det(self.matrix))))

    @property
    def multiplicity(self) -> int:
        return abs(self.determinant)

    @property
    def inverse(self) -> NDArray[np.float64]:
        return np.linalg.inv(self.matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.tolist(),
            "determinant": self.determinant,
            "multiplicity": self.multiplicity,
            "residual": self.residual,
        }


def detect_transformation(
    primitive: Structure | ArrayLike,
    supercell: Structure | ArrayLike,
    *,
    atol: float = 1e-6,
) -> TransformationMatrix:
    """Detect an integer transform from primitive to supercell lattices."""

    primitive_lattice = primitive.lattice if isinstance(primitive, Structure) else primitive
    supercell_lattice = supercell.lattice if isinstance(supercell, Structure) else supercell
    primitive_arr = as_matrix3(primitive_lattice, name="primitive_lattice")
    supercell_arr = as_matrix3(supercell_lattice, name="supercell_lattice")

    raw = supercell_arr @ np.linalg.inv(primitive_arr)
    rounded = np.rint(raw).astype(int)
    residual = float(np.max(np.abs(raw - rounded)))
    if residual > atol:
        raise ValueError(
            "primitive and supercell lattices are not related by an integer "
            f"transformation within atol={atol}; max residual={residual:g}"
        )
    return TransformationMatrix(rounded, residual=residual)


def apply_transformation(
    primitive_lattice: ArrayLike,
    transform: TransformationMatrix,
) -> NDArray[np.float64]:
    """Construct a supercell lattice from a primitive lattice and transform."""

    return transform.matrix @ as_matrix3(primitive_lattice, name="primitive_lattice")
