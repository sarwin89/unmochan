"""Generic reference-resolved workflow scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.structures import Structure
from unfoldlab.io.vasp import read_poscar
from unfoldlab.twist.geometry import assign_layers_by_axis, in_plane_twist_angle


@dataclass
class TwistedUnfoldingProblem:
    """Workflow state for arbitrary rotated, layered, or reference-resolved systems."""

    references: dict[str, Structure]
    supercell_structure: Structure
    outputs: Path | None = None
    layer_assignments: NDArray[np.int64] | None = None
    relative_rotations: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_structures(
        cls,
        *,
        references: dict[str, str | Path | Structure],
        supercell: str | Path | Structure,
        outputs: str | Path | None = None,
    ) -> TwistedUnfoldingProblem:
        """Build from any number of named reference structures."""

        if not references:
            raise ValueError("at least one reference structure is required")
        loaded_references = {
            label: _load_structure(structure) for label, structure in references.items()
        }
        return cls(
            references=loaded_references,
            supercell_structure=_load_structure(supercell),
            outputs=Path(outputs) if outputs is not None else None,
        )

    def detect_layers(self, *, axis: int = 2, n_layers: int | None = None) -> NDArray[np.int64]:
        """Assign supercell sites to layer-like groups along a Cartesian axis."""

        target_layers = n_layers if n_layers is not None else max(2, len(self.references))
        self.layer_assignments = assign_layers_by_axis(
            self.supercell_structure,
            axis=axis,
            n_layers=target_layers,
        )
        return self.layer_assignments

    def detect_relative_rotations(self, *, vector_index: int = 0) -> dict[str, float]:
        """Estimate in-plane rotations between each reference and the supercell."""

        self.relative_rotations = {
            label: in_plane_twist_angle(
                reference.lattice,
                self.supercell_structure.lattice,
                vector_index=vector_index,
            )
            for label, reference in self.references.items()
        }
        return dict(self.relative_rotations)

    def build_moire_reciprocal_lattice(
        self,
        reference_a: str | None = None,
        reference_b: str | None = None,
    ) -> NDArray[np.float64]:
        """Return the reciprocal-lattice difference between two references."""

        labels = list(self.references)
        if reference_a is None or reference_b is None:
            if len(labels) < 2:
                raise ValueError("at least two references are required for a moire lattice")
            reference_a = labels[0] if reference_a is None else reference_a
            reference_b = labels[1] if reference_b is None else reference_b
        return self.references[reference_a].reciprocal_lattice - self.references[
            reference_b
        ].reciprocal_lattice

    def get_reference(self, label: str) -> Structure:
        try:
            return self.references[label]
        except KeyError as exc:
            available = ", ".join(sorted(self.references))
            raise KeyError(
                f"unknown reference {label!r}; available references: {available}"
            ) from exc

    def unfold_to_reference(self, label: str) -> None:
        """Placeholder for future reference-resolved unfolding backends."""

        self.get_reference(label)
        raise NotImplementedError(
            "reference-resolved unfolding backends are not implemented yet; "
            "this API reserves a material-agnostic target reference"
        )

    def unfold_to_moire_bz(self) -> None:
        """Placeholder for future moire-mini-zone unfolding backends."""

        raise NotImplementedError("moire Brillouin-zone unfolding is not implemented yet")


def _load_structure(structure: str | Path | Structure) -> Structure:
    if isinstance(structure, Structure):
        return structure
    return read_poscar(structure)
