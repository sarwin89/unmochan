"""Primitive-to-supercell transformation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import (
    as_int_array,
    as_matrix3,
    check_atol,
    integer_adjugate3,
    integer_det3,
)
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
        matrix = as_int_array(self.matrix, name="matrix")
        if matrix.shape != (3, 3):
            raise ValueError(f"matrix must have shape (3, 3), got {matrix.shape}")
        if integer_det3(matrix) == 0:
            raise ValueError("transformation matrix must be invertible")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "residual", float(self.residual))

    @classmethod
    def from_values(cls, values: ArrayLike, *, residual: float = 0.0) -> TransformationMatrix:
        """Build a transform from any array-like of (near-)integer entries.

        ``TransformationMatrix`` stores an exact integer matrix, so its field is
        typed as one.  Callers that only have floats -- a parsed command-line
        string, a detected transform -- should go through this constructor,
        which rounds and validates via :func:`as_int_array`.
        """

        return cls(as_int_array(values, name="matrix"), residual=residual)

    @property
    def determinant(self) -> int:
        """Exact integer determinant (computed by cofactor expansion)."""

        return integer_det3(self.matrix)

    @property
    def multiplicity(self) -> int:
        """Number of primitive cells in the supercell, ``|det T|``.

        This is also the number of primitive k-points that fold onto each
        supercell k-point.
        """

        return abs(self.determinant)

    @property
    def adjugate(self) -> NDArray[np.int64]:
        """Exact integer adjugate, with ``adjugate @ matrix == det * I``."""

        return integer_adjugate3(self.matrix)

    @property
    def inverse(self) -> NDArray[np.float64]:
        """Floating-point inverse.

        Prefer :attr:`adjugate` together with :attr:`determinant` whenever an
        exact answer is required.
        """

        return self.adjugate.astype(float) / float(self.determinant)

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

    check_atol(atol)
    raw = supercell_arr @ np.linalg.inv(primitive_arr)
    rounded = np.rint(raw).astype(int)
    residual = float(np.max(np.abs(raw - rounded)))
    if residual > atol:
        raise ValueError(
            "primitive and supercell lattices are not related by an integer "
            f"transformation within atol={atol}; max residual={residual:g}. "
            "Entrywise rounding is already the best integer approximation, so "
            "the pair is genuinely incommensurate: the supercell is not built "
            "out of whole primitive cells. Call "
            "unfoldlab.core.strain.diagnose_commensurability (or the CLI's "
            "`strain` command) to see the residual strain and the k-space error "
            "it would cause."
        )
    return TransformationMatrix(rounded, residual=residual)


def apply_transformation(
    primitive_lattice: ArrayLike,
    transform: TransformationMatrix,
) -> NDArray[np.float64]:
    """Construct a supercell lattice from a primitive lattice and transform."""

    return transform.matrix @ as_matrix3(primitive_lattice, name="primitive_lattice")


@dataclass(frozen=True)
class SupercellConsistency:
    """Result of cross-checking a detected transform against cell contents."""

    multiplicity: int
    volume_ratio: float
    site_ratio: float | None
    volume_consistent: bool
    sites_consistent: bool | None

    @property
    def consistent(self) -> bool:
        return self.volume_consistent and self.sites_consistent is not False

    def to_dict(self) -> dict[str, Any]:
        return {
            "multiplicity": self.multiplicity,
            "volume_ratio": self.volume_ratio,
            "site_ratio": self.site_ratio,
            "volume_consistent": self.volume_consistent,
            "sites_consistent": self.sites_consistent,
            "consistent": self.consistent,
        }


def check_supercell_consistency(
    primitive: Structure | ArrayLike,
    supercell: Structure | ArrayLike,
    transform: TransformationMatrix,
    *,
    rtol: float = 1e-6,
) -> SupercellConsistency:
    """Cross-check ``|det T|`` against cell volumes and site counts.

    ``det(T * A_pc) == det(T) * det(A_pc)``, so the supercell volume must be
    ``|det T|`` times the primitive volume, and — for a defect-free supercell —
    the number of sites must scale by the same factor.  A mismatch means the
    primitive/supercell pair does not describe the same crystal (a wrong
    primitive cell, a missing or extra atom, or an unintended reconstruction),
    which would silently corrupt every unfolding weight.
    """

    primitive_lattice = primitive.lattice if isinstance(primitive, Structure) else primitive
    supercell_lattice = supercell.lattice if isinstance(supercell, Structure) else supercell
    primitive_volume = abs(float(np.linalg.det(as_matrix3(primitive_lattice, name="primitive"))))
    supercell_volume = abs(float(np.linalg.det(as_matrix3(supercell_lattice, name="supercell"))))
    if primitive_volume == 0.0:
        raise ValueError("primitive lattice is singular")

    multiplicity = transform.multiplicity
    volume_ratio = supercell_volume / primitive_volume
    volume_consistent = bool(abs(volume_ratio - multiplicity) <= rtol * max(1.0, multiplicity))

    site_ratio: float | None = None
    sites_consistent: bool | None = None
    if isinstance(primitive, Structure) and isinstance(supercell, Structure):
        if primitive.n_sites > 0:
            site_ratio = supercell.n_sites / primitive.n_sites
            sites_consistent = supercell.n_sites == multiplicity * primitive.n_sites

    return SupercellConsistency(
        multiplicity=multiplicity,
        volume_ratio=volume_ratio,
        site_ratio=site_ratio,
        volume_consistent=volume_consistent,
        sites_consistent=sites_consistent,
    )
