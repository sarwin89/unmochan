"""Generic reference-resolved workflow scaffolding."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unmochan.core.kpoints import KPoint
from unmochan.core.spectral import EffectiveBandStructure
from unmochan.core.structures import Structure
from unmochan.io.qe import read_pw_input_structure
from unmochan.io.vasp import read_poscar
from unmochan.twist.geometry import assign_layers_by_axis, in_plane_twist_angle
from unmochan.workflows.problem import unfold_supercell_to_lattice


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
        code: str = "auto",
    ) -> TwistedUnfoldingProblem:
        """Build from named references using VASP, QE, or already-loaded structures."""

        if not references:
            raise ValueError("at least one reference structure is required")
        loaded_references = {
            label: _load_structure(structure, code=code) for label, structure in references.items()
        }
        return cls(
            references=loaded_references,
            supercell_structure=_load_structure(supercell, code=code),
            outputs=Path(outputs) if outputs is not None else None,
        )

    @classmethod
    def from_qe(
        cls,
        *,
        references: dict[str, str | Path | Structure],
        supercell: str | Path | Structure,
        outputs: str | Path | None = None,
    ) -> TwistedUnfoldingProblem:
        """Build a twist problem from QE ``pw.x`` input files or structures."""

        return cls.from_structures(
            references=references,
            supercell=supercell,
            outputs=outputs,
            code="qe",
        )

    @classmethod
    def from_vasp(
        cls,
        *,
        references: dict[str, str | Path | Structure],
        supercell: str | Path | Structure,
        outputs: str | Path | None = None,
    ) -> TwistedUnfoldingProblem:
        """Build a twist problem from VASP POSCAR/CONTCAR files or structures."""

        return cls.from_structures(
            references=references,
            supercell=supercell,
            outputs=outputs,
            code="vasp",
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
        """Return the reciprocal-lattice difference ``B_a - B_b``.

        Note that for a layered stack this matrix is singular -- the shared
        stacking direction contributes a zero row -- so it cannot be inverted to
        obtain the real-space moire cell.  Use :meth:`moire_lattice` for that.
        """

        labels = list(self.references)
        if reference_a is None or reference_b is None:
            if len(labels) < 2:
                raise ValueError("at least two references are required for a moire lattice")
            reference_a = labels[0] if reference_a is None else reference_a
            reference_b = labels[1] if reference_b is None else reference_b
        return (
            self.references[reference_a].reciprocal_lattice
            - self.references[reference_b].reciprocal_lattice
        )

    def get_reference(self, label: str) -> Structure:
        try:
            return self.references[label]
        except KeyError as exc:
            available = ", ".join(sorted(self.references))
            raise KeyError(
                f"unknown reference {label!r}; available references: {available}"
            ) from exc

    def moire_lattice(
        self,
        reference_a: str | None = None,
        reference_b: str | None = None,
        *,
        axis: int = 2,
        atol: float = 1e-8,
    ) -> NDArray[np.float64]:
        """Real-space moire lattice of two references.

        The moire reciprocal lattice is the difference of the two reciprocal
        lattices, ``B_m = B_a - B_b``.  For a layered system that difference is
        *singular as a 3x3 matrix*: the two layers share their stacking axis, so
        the third row of ``B_m`` vanishes and
        :meth:`build_moire_reciprocal_lattice` cannot simply be inverted.  The
        moire pattern lives in the plane perpendicular to ``axis``, so the
        in-plane 2x2 block is inverted instead and the stacking direction is
        taken from the supercell, which is the cell the wavefunctions live in.

        Two references whose in-plane reciprocal lattices agree produce no moire
        pattern and are rejected, as is a difference with a component along the
        stacking axis (the two references are then not a layered stack).
        """

        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        moire_reciprocal = self.build_moire_reciprocal_lattice(reference_a, reference_b)
        in_plane = [index for index in range(3) if index != axis]
        if float(np.max(np.abs(moire_reciprocal[in_plane, axis]))) > atol:
            raise ValueError(
                "the difference of the two reciprocal lattices has a component along "
                f"axis {axis}; the references do not share a stacking direction"
            )
        block = moire_reciprocal[np.ix_(in_plane, in_plane)]
        determinant = float(np.linalg.det(block))
        if not np.isfinite(determinant) or abs(determinant) < atol:
            raise ValueError(
                "the two references have (nearly) equal in-plane reciprocal lattices, "
                "so they define no moire cell"
            )
        lattice = np.zeros((3, 3), dtype=float)
        lattice[np.ix_(in_plane, in_plane)] = 2.0 * np.pi * np.linalg.inv(block).T
        lattice[axis] = np.asarray(self.supercell_structure.lattice, dtype=float)[axis]
        return lattice

    def unfold_to_reference(
        self,
        label: str,
        *,
        kpath: Iterable[KPoint],
        code: str = "vasp",
        wavecar: str | Path | None = None,
        qe_save_dir: str | Path | None = None,
        bands: str | Path | None = None,
        spin: int = 1,
        tol: float = 1e-6,
        atol: float = 1e-6,
        operations: NDArray[np.integer] | None = None,
        reference_energy: float = 0.0,
    ) -> EffectiveBandStructure:
        """Unfold the supercell wavefunction onto one reference's Brillouin zone.

        Each reference of a commensurate stack is an integer sublattice of the
        supercell -- that is what commensurability means -- so the ordinary
        plane-wave unfolding kernel applies once the transform ``A_sc = T A_ref``
        of *that* reference is used.  For a twisted bilayer this gives the band
        structure resolved onto the chosen layer's own (rotated) Brillouin zone.

        Both backends are available: ``code="vasp"`` reads a ``WAVECAR`` and
        ``code="qe"`` a ``.save`` directory, whose XML also supplies the
        eigenvalues unless a ``bands`` file is given.
        """

        reference = self.get_reference(label)
        return unfold_supercell_to_lattice(
            code,
            wavecar=self._resolve_wavecar(wavecar) if code == "vasp" else None,
            qe_save_dir=self._resolve_qe_save_dir(qe_save_dir) if code == "qe" else None,
            bands=bands if code == "qe" else None,
            supercell_lattice=self.supercell_structure.lattice,
            target_lattice=reference.lattice,
            kpath=kpath,
            spin=spin,
            tol=tol,
            atol=atol,
            operations=operations,
            reference_energy=reference_energy,
            metadata={"reference": label},
        )

    def unfold_to_moire_bz(
        self,
        *,
        kpath: Iterable[KPoint],
        reference_a: str | None = None,
        reference_b: str | None = None,
        code: str = "vasp",
        wavecar: str | Path | None = None,
        qe_save_dir: str | Path | None = None,
        bands: str | Path | None = None,
        spin: int = 1,
        tol: float = 1e-6,
        atol: float = 1e-6,
        operations: NDArray[np.integer] | None = None,
        reference_energy: float = 0.0,
    ) -> EffectiveBandStructure:
        """Unfold onto the moire mini Brillouin zone.

        This is meaningful when the calculation cell is a repetition of the
        moire cell; if the two coincide the transform is the identity and every
        weight is one, which is the correct -- if uninformative -- answer.  A
        supercell that is not an integer multiple of the moire cell is rejected
        by the transform detection.
        """

        return unfold_supercell_to_lattice(
            code,
            wavecar=self._resolve_wavecar(wavecar) if code == "vasp" else None,
            qe_save_dir=self._resolve_qe_save_dir(qe_save_dir) if code == "qe" else None,
            bands=bands if code == "qe" else None,
            supercell_lattice=self.supercell_structure.lattice,
            target_lattice=self.moire_lattice(reference_a, reference_b),
            kpath=kpath,
            spin=spin,
            tol=tol,
            atol=atol,
            operations=operations,
            reference_energy=reference_energy,
            metadata={"target": "moire"},
        )

    def _resolve_wavecar(self, wavecar: str | Path | None) -> Path:
        if wavecar is not None:
            return Path(wavecar)
        if self.outputs is None:
            raise ValueError("provide wavecar, or set outputs on the problem")
        outputs = Path(self.outputs)
        candidate = outputs / "WAVECAR" if outputs.is_dir() else outputs
        if not candidate.is_file():
            raise FileNotFoundError(f"no WAVECAR found at {candidate}")
        return candidate

    def _resolve_qe_save_dir(self, qe_save_dir: str | Path | None) -> Path:
        if qe_save_dir is not None:
            return Path(qe_save_dir)
        if self.outputs is None:
            raise ValueError("provide qe_save_dir, or set outputs on the problem")
        outputs = Path(self.outputs)
        if outputs.is_dir():
            if (outputs / "data-file-schema.xml").is_file():
                return outputs
            candidates = sorted(outputs.glob("*.save"))
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                names = ", ".join(candidate.name for candidate in candidates)
                raise ValueError(
                    f"{outputs} holds several save directories ({names}); pass qe_save_dir"
                )
        raise FileNotFoundError(f"no QE .save directory found at {outputs}")


def _load_structure(structure: str | Path | Structure, *, code: str = "auto") -> Structure:
    if isinstance(structure, Structure):
        return structure
    code_normalized = code.lower()
    if code_normalized == "vasp":
        return read_poscar(structure)
    if code_normalized == "qe":
        return read_pw_input_structure(structure)
    if code_normalized != "auto":
        raise ValueError("code must be auto, vasp, or qe")
    try:
        return read_poscar(structure)
    except Exception:
        try:
            return read_pw_input_structure(structure)
        except Exception as qe_error:
            raise ValueError(
                f"could not read structure {structure!r} as VASP POSCAR or QE pw.x input"
            ) from qe_error
