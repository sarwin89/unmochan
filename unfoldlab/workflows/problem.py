"""High-level unfolding problem orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unfoldlab.core.kpoints import KPoint, KPointMapping, fold_kpoints_to_supercell
from unfoldlab.core.spectral import EffectiveBandStructure
from unfoldlab.core.structures import Structure
from unfoldlab.core.transformations import TransformationMatrix, detect_transformation
from unfoldlab.io.vasp import read_eigenval, read_poscar


@dataclass
class UnfoldingProblem:
    """State for a primitive/supercell unfolding workflow."""

    primitive_structure: Structure
    supercell_structure: Structure
    primitive_kpath: list[KPoint] = field(default_factory=list)
    supercell_outputs: Path | None = None
    code: str = "vasp"
    transformation: TransformationMatrix | None = None

    @classmethod
    def from_vasp(
        cls,
        *,
        primitive: str | Path,
        supercell: str | Path,
        outputs: str | Path | None = None,
        primitive_kpath: Iterable[KPoint] | None = None,
    ) -> UnfoldingProblem:
        return cls(
            primitive_structure=read_poscar(primitive),
            supercell_structure=read_poscar(supercell),
            primitive_kpath=list(primitive_kpath or []),
            supercell_outputs=Path(outputs) if outputs is not None else None,
            code="vasp",
        )

    def find_transformation(self, *, atol: float = 1e-6) -> TransformationMatrix:
        self.transformation = detect_transformation(
            self.primitive_structure,
            self.supercell_structure,
            atol=atol,
        )
        return self.transformation

    def require_transformation(self) -> TransformationMatrix:
        if self.transformation is None:
            return self.find_transformation()
        return self.transformation

    def generate_supercell_kpoints(
        self,
        primitive_kpoints: Iterable[KPoint] | None = None,
    ) -> list[KPointMapping]:
        points = list(self.primitive_kpath if primitive_kpoints is None else primitive_kpoints)
        if not points:
            raise ValueError("no primitive k-points were provided")
        return fold_kpoints_to_supercell(points, self.require_transformation())

    def validate_eigenval_kpoints(
        self,
        eigenval: str | Path,
        primitive_kpoints: Iterable[KPoint] | None = None,
        *,
        atol: float = 1e-8,
    ) -> list[KPointMapping]:
        """Return required mappings missing from an EIGENVAL file."""

        required = self.generate_supercell_kpoints(primitive_kpoints)
        parsed = read_eigenval(eigenval)
        missing: list[KPointMapping] = []
        for mapping in required:
            delta = parsed.kpoints - mapping.supercell.fractional[None, :]
            delta -= np.rint(delta)
            if not np.any(np.all(np.abs(delta) <= atol, axis=1)):
                missing.append(mapping)
        return missing

    def ebs_from_eigenval(
        self,
        eigenval: str | Path,
        *,
        weights: np.ndarray | None = None,
        reference_energy: float = 0.0,
    ) -> EffectiveBandStructure:
        """Create an EBS container from EIGENVAL energies.

        This is a data-model bridge, not a full unfolding backend. Passing
        explicit weights lets early projector or analytic tests use the common
        spectral-function and plotting path.
        """

        parsed = read_eigenval(eigenval)
        if parsed.energies.ndim != 2:
            raise ValueError("spin-channel EIGENVAL data must be selected before EBS construction")
        return EffectiveBandStructure(
            kpoints=parsed.kpoints,
            energies=parsed.energies,
            weights=weights,
            reference_energy=reference_energy,
            metadata={
                "source": str(eigenval),
                "code": self.code,
                "note": "EIGENVAL energies with caller-provided or unit weights",
            },
        )
