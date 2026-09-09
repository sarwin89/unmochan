"""High-level unfolding problem orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from unmochan.core.kpoints import (
    KPoint,
    KPointMapping,
    cartesian_path_distances,
    fold_kpoint_to_supercell,
    fold_kpoints_to_supercell,
)
from unmochan.core.projections import ProjectionSelector, parse_projection_selectors
from unmochan.core.spectral import EffectiveBandStructure
from unmochan.core.structures import Structure
from unmochan.core.transformations import TransformationMatrix, detect_transformation
from unmochan.core.unfolding import BandUnfoldingData
from unmochan.io.qe import read_gnu_blocks, read_pw_input_structure
from unmochan.io.qe_wfc import compute_weights_from_qe_save
from unmochan.io.qe_xml import read_qe_xml_eigenvalues
from unmochan.io.vasp import read_eigenval, read_poscar
from unmochan.io.vasp_wfc import compute_weights_from_wavecar
from unmochan.workflows.projected import apply_projection_fractions, procar_projection_fractions


def _lattice_unfolding_kpoints(
    kpath: Iterable[KPoint],
    supercell_lattice: np.ndarray,
    target_lattice: np.ndarray,
    atol: float,
) -> tuple[TransformationMatrix, np.ndarray, np.ndarray]:
    """Transform, primitive k-points and their folded images, for either backend."""

    points = list(kpath)
    if not points:
        raise ValueError("no primitive k-path was provided")
    transform = detect_transformation(target_lattice, supercell_lattice, atol=atol)
    primitive = np.array([point.fractional for point in points], dtype=float)
    folded = np.array(
        [fold_kpoint_to_supercell(point, transform).fractional for point in points],
        dtype=float,
    )
    return transform, primitive, folded


def unfold_wavecar_to_lattice(
    *,
    wavecar: str | Path,
    supercell_lattice: np.ndarray,
    target_lattice: np.ndarray,
    kpath: Iterable[KPoint],
    spin: int = 1,
    tol: float = 1e-6,
    atol: float = 1e-6,
    operations: np.ndarray | None = None,
    reference_energy: float = 0.0,
    source_code: str = "vasp",
    metadata: dict[str, object] | None = None,
) -> EffectiveBandStructure:
    """Unfold a supercell WAVECAR onto the Brillouin zone of any target lattice.

    The only thing the unfolding kernel needs is the integer transform ``T``
    with ``A_sc = T @ A_target``, so the target does not have to be a structure:
    it may be a reference layer of a twisted stack or a moire cell.  A target
    lattice that is not an integer sublattice of the supercell is rejected,
    because the k-point folding map would then not be defined.

    The band-path abscissa is accumulated in the Cartesian reciprocal space of
    the target lattice.
    """

    transform, primitive, folded = _lattice_unfolding_kpoints(
        kpath, supercell_lattice, target_lattice, atol
    )
    _kpoints, energies, weights = compute_weights_from_wavecar(
        wavecar,
        primitive,
        folded,
        transform.matrix,
        spin=spin,
        tol=tol,
        operations=operations,
    )
    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(target_lattice, dtype=float)).T
    return BandUnfoldingData(
        kpoints=primitive,
        energies=energies,
        weights=weights,
        distances=cartesian_path_distances(primitive, reciprocal),
        reference_energy=reference_energy,
        source_code=source_code,
        metadata={
            "wavecar": str(wavecar),
            "spin_channel": spin,
            "symmetry_reduced": operations is not None,
            "transformation": transform.to_dict(),
            **(metadata or {}),
        },
    ).to_effective_band_structure()


def unfold_qe_save_to_lattice(
    *,
    qe_save_dir: str | Path,
    supercell_lattice: np.ndarray,
    target_lattice: np.ndarray,
    kpath: Iterable[KPoint],
    bands: str | Path | None = None,
    spin: int | None = None,
    tol: float = 1e-6,
    atol: float = 1e-6,
    operations: np.ndarray | None = None,
    reference_energy: float = 0.0,
    wfc_pattern: str | None = None,
    wfc_format: str = "auto",
    lattice_alat: np.ndarray | None = None,
    band_chunk: int | None = None,
    source_code: str = "qe",
    metadata: dict[str, object] | None = None,
) -> EffectiveBandStructure:
    """Unfold a QE ``.save`` directory onto the Brillouin zone of any lattice.

    The Quantum ESPRESSO counterpart of :func:`unfold_wavecar_to_lattice`, and
    the same mathematics: only the integer transform ``T`` with
    ``A_sc = T @ A_target`` enters, so the target may be a primitive cell, a
    reference layer of a twisted stack or a moire cell.

    The eigenvalues come from ``bands`` (a ``.gnu`` file) when it is given and
    otherwise from the ``data-file-schema.xml`` inside the save directory, which
    removes one way for the two to disagree.
    """

    transform, primitive, folded = _lattice_unfolding_kpoints(
        kpath, supercell_lattice, target_lattice, atol
    )
    if bands is not None:
        _, energies = read_gnu_blocks(bands)
        energy_source = str(bands)
    else:
        energies = read_qe_xml_eigenvalues(qe_save_dir, spin=spin)
        energy_source = f"{qe_save_dir} (data-file-schema.xml)"
    if energies.shape[0] != primitive.shape[0]:
        raise ValueError(
            f"the k-path has {primitive.shape[0]} points but the eigenvalues cover "
            f"{energies.shape[0]}"
        )

    weights = compute_weights_from_qe_save(
        qe_save_dir,
        primitive,
        folded,
        transform.matrix,
        energies.shape[1],
        tol=tol,
        spin=spin,
        pattern=wfc_pattern,
        file_format=wfc_format,
        lattice_alat=lattice_alat,
        operations=operations,
        band_chunk=band_chunk,
    )
    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(target_lattice, dtype=float)).T
    return BandUnfoldingData(
        kpoints=primitive,
        energies=energies,
        weights=weights,
        distances=cartesian_path_distances(primitive, reciprocal),
        reference_energy=reference_energy,
        source_code=source_code,
        metadata={
            "qe_save_dir": str(qe_save_dir),
            "energy_source": energy_source,
            "spin_channel": spin,
            "symmetry_reduced": operations is not None,
            "transformation": transform.to_dict(),
            **(metadata or {}),
        },
    ).to_effective_band_structure()


def unfold_supercell_to_lattice(
    code: str,
    *,
    supercell_lattice: np.ndarray,
    target_lattice: np.ndarray,
    kpath: Iterable[KPoint],
    wavecar: str | Path | None = None,
    qe_save_dir: str | Path | None = None,
    bands: str | Path | None = None,
    spin: int | None = None,
    tol: float = 1e-6,
    atol: float = 1e-6,
    operations: np.ndarray | None = None,
    reference_energy: float = 0.0,
    metadata: dict[str, object] | None = None,
    **backend_options: object,
) -> EffectiveBandStructure:
    """Backend-neutral unfolding onto the Brillouin zone of any target lattice.

    Dispatches to :func:`unfold_wavecar_to_lattice` for ``code="vasp"`` and to
    :func:`unfold_qe_save_to_lattice` for ``code="qe"``, so a twist or moire
    workflow does not have to know which code produced the wavefunctions.
    """

    if code == "vasp":
        if wavecar is None:
            raise ValueError("the VASP backend needs a WAVECAR")
        if qe_save_dir is not None or bands is not None:
            raise ValueError("qe_save_dir and bands belong to the QE backend")
        return unfold_wavecar_to_lattice(
            wavecar=wavecar,
            supercell_lattice=supercell_lattice,
            target_lattice=target_lattice,
            kpath=kpath,
            spin=1 if spin is None else int(spin),
            tol=tol,
            atol=atol,
            operations=operations,
            reference_energy=reference_energy,
            metadata=metadata,
            **backend_options,  # type: ignore[arg-type]
        )
    if code == "qe":
        if qe_save_dir is None:
            raise ValueError("the QE backend needs a .save directory")
        if wavecar is not None:
            raise ValueError("a WAVECAR belongs to the VASP backend")
        return unfold_qe_save_to_lattice(
            qe_save_dir=qe_save_dir,
            supercell_lattice=supercell_lattice,
            target_lattice=target_lattice,
            kpath=kpath,
            bands=bands,
            spin=spin,
            tol=tol,
            atol=atol,
            operations=operations,
            reference_energy=reference_energy,
            metadata=metadata,
            **backend_options,  # type: ignore[arg-type]
        )
    raise ValueError(f"code must be vasp or qe, got {code!r}")


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

    @classmethod
    def from_qe(
        cls,
        *,
        primitive: str | Path,
        supercell: str | Path,
        outputs: str | Path | None = None,
        primitive_kpath: Iterable[KPoint] | None = None,
    ) -> UnfoldingProblem:
        """Build a problem from two ``pw.x`` input files.

        The counterpart of :meth:`from_vasp`; ``outputs`` is the directory that
        holds the ``.save`` directory of the supercell run.
        """

        return cls(
            primitive_structure=read_pw_input_structure(primitive),
            supercell_structure=read_pw_input_structure(supercell),
            primitive_kpath=list(primitive_kpath or []),
            supercell_outputs=Path(outputs) if outputs is not None else None,
            code="qe",
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

    def generate_kpoint_mapping(
        self,
        primitive_kpoints: Iterable[KPoint] | None = None,
    ) -> list[KPointMapping]:
        """Alias for the material-agnostic public API."""

        return self.generate_supercell_kpoints(primitive_kpoints)

    def validate(self) -> dict[str, object]:
        """Run lightweight structural validation for the current workflow."""

        transform = self.require_transformation()
        return {
            "code": self.code,
            "transformation": transform.to_dict(),
            "primitive_sites": self.primitive_structure.n_sites,
            "supercell_sites": self.supercell_structure.n_sites,
        }

    def unfold(
        self,
        *,
        backend: str = "auto",
        projections: Iterable[str | ProjectionSelector] | None = None,
        wavecar: str | Path | None = None,
        qe_save_dir: str | Path | None = None,
        bands: str | Path | None = None,
        procar: str | Path | None = None,
        spin: int = 1,
        tol: float = 1e-6,
        operations: np.ndarray | None = None,
        reference_energy: float = 0.0,
        site_groups: Mapping[str, Mapping[object, Sequence[int]]] | None = None,
        layer_axis: int = 2,
        layer_tol: float = 0.5,
        defect_center: np.ndarray | None = None,
    ) -> EffectiveBandStructure:
        """Unfold the stored primitive k-path against a supercell wavefunction.

        The transformation matrix is detected from the two structures if it has
        not been set, the k-path is folded into the supercell Brillouin zone,
        and the plane-wave weights are computed with the common kernel.  The
        band-path abscissa is accumulated in Cartesian reciprocal space of the
        primitive cell.

        For the VASP backend ``wavecar`` defaults to the WAVECAR inside
        :attr:`supercell_outputs`; for the Quantum ESPRESSO backend
        ``qe_save_dir`` defaults to the ``.save`` directory found there, and the
        eigenvalues are read from its XML unless a ``bands`` file is given.
        ``operations`` enables unfolding from a symmetry-reduced wavefunction
        set; see :func:`unmochan.io.vasp_wfc.compute_weights_from_wavecar`.

        Passing ``projections`` additionally requires ``procar`` (or a PROCAR
        next to :attr:`supercell_outputs`): the plane-wave weights are then
        multiplied by the site/orbital fractions of that file, which is the
        approximate projector-assisted mode described in
        :mod:`unmochan.workflows.projected`.  ``site_groups`` supplies the
        site lists of user conventions such as ``region:`` or ``sublattice:``.
        """

        parsed_projections = _normalize_projections(projections or [])

        code = self.code if backend == "auto" else backend
        if code not in ("vasp", "qe"):
            raise ValueError(f"backend must be vasp or qe, got {code!r}")
        if not self.primitive_kpath:
            raise ValueError("no primitive k-path was provided")

        transform = self.require_transformation()
        if code == "vasp":
            ebs = unfold_wavecar_to_lattice(
                wavecar=self._resolve_wavecar(wavecar),
                supercell_lattice=self.supercell_structure.lattice,
                target_lattice=self.primitive_structure.lattice,
                kpath=self.primitive_kpath,
                spin=spin,
                tol=tol,
                operations=operations,
                reference_energy=reference_energy,
                source_code=code,
            )
        else:
            if parsed_projections:
                raise ValueError(
                    "projector-assisted projections read a VASP PROCAR; the QE "
                    "backend has no equivalent file"
                )
            ebs = unfold_qe_save_to_lattice(
                qe_save_dir=self._resolve_qe_save_dir(qe_save_dir),
                supercell_lattice=self.supercell_structure.lattice,
                target_lattice=self.primitive_structure.lattice,
                kpath=self.primitive_kpath,
                bands=bands,
                spin=None if spin == 1 else spin,
                tol=tol,
                operations=operations,
                reference_energy=reference_energy,
                source_code=code,
            )
        if not parsed_projections:
            return ebs

        folded = np.array(
            [
                fold_kpoint_to_supercell(point, transform).fractional
                for point in self.primitive_kpath
            ],
            dtype=float,
        )
        fractions = procar_projection_fractions(
            self._resolve_procar(procar),
            parsed_projections,
            self.supercell_structure,
            folded,
            spin=spin,
            site_groups=site_groups,
            layer_axis=layer_axis,
            layer_tol=layer_tol,
            defect_center=defect_center,
        )
        if fractions.shape != ebs.energies.shape:
            raise ValueError(
                f"PROCAR provides {fractions.shape[1]} bands but the wavefunction "
                f"unfolding produced {ebs.energies.shape[1]}"
            )
        return apply_projection_fractions(
            ebs,
            fractions,
            label=" ".join(item.to_string() for item in parsed_projections),
        )

    def _resolve_wavecar(self, wavecar: str | Path | None) -> Path:
        if wavecar is not None:
            return Path(wavecar)
        if self.supercell_outputs is None:
            raise ValueError("provide wavecar, or set supercell_outputs on the problem")
        outputs = Path(self.supercell_outputs)
        candidate = outputs / "WAVECAR" if outputs.is_dir() else outputs
        if not candidate.is_file():
            raise FileNotFoundError(f"no WAVECAR found at {candidate}")
        return candidate

    def _resolve_qe_save_dir(self, qe_save_dir: str | Path | None) -> Path:
        if qe_save_dir is not None:
            return Path(qe_save_dir)
        if self.supercell_outputs is None:
            raise ValueError("provide qe_save_dir, or set supercell_outputs on the problem")
        outputs = Path(self.supercell_outputs)
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

    def _resolve_procar(self, procar: str | Path | None) -> Path:
        if procar is not None:
            return Path(procar)
        if self.supercell_outputs is None:
            raise ValueError(
                "projected unfolding needs a PROCAR: pass procar=, or set "
                "supercell_outputs on the problem"
            )
        outputs = Path(self.supercell_outputs)
        candidate = outputs / "PROCAR" if outputs.is_dir() else outputs.parent / "PROCAR"
        if not candidate.is_file():
            raise FileNotFoundError(f"no PROCAR found at {candidate}")
        return candidate

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
        return BandUnfoldingData(
            kpoints=parsed.kpoints,
            energies=parsed.energies,
            weights=weights,
            reference_energy=reference_energy,
            source_code=self.code,
            metadata={
                "source": str(eigenval),
                "note": "EIGENVAL energies with caller-provided or unit weights",
            },
        ).to_effective_band_structure()


def _normalize_projections(
    projections: Iterable[str | ProjectionSelector],
) -> tuple[ProjectionSelector, ...]:
    raw: list[str] = []
    parsed: list[ProjectionSelector] = []
    for projection in projections:
        if isinstance(projection, ProjectionSelector):
            parsed.append(projection)
        else:
            raw.append(projection)
    return tuple(parsed) + parse_projection_selectors(raw)
