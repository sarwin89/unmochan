"""Unfolding a QE ``.save`` directory onto the Brillouin zone of any lattice.

The VASP path (``unfold_wavecar_to_lattice``) had no Quantum ESPRESSO
counterpart, so the high-level problem API and the whole twist/moire workflow
were VASP-only.  These tests cover the QE path and the backend-neutral
dispatcher that both now go through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic_qe_xml import write_data_file_schema
from test_qe_wfc_kpoints import COEFFS, MILLER, write_synthetic_wfc_dat

from unfoldlab.core.kpoints import KPoint
from unfoldlab.core.structures import Structure
from unfoldlab.io.qe_xml import HARTREE_EV
from unfoldlab.twist.problem import TwistedUnfoldingProblem
from unfoldlab.workflows.problem import (
    UnfoldingProblem,
    unfold_qe_save_to_lattice,
    unfold_supercell_to_lattice,
)

# The supercell doubles the primitive cell along x; its ``at`` matrix in units
# of alat is therefore diag(1, 1/2, 1/2) for a cubic primitive cell.
SUPERCELL_LATTICE = np.diag([2.0, 1.0, 1.0])
TARGET_LATTICE = np.eye(3)
EIGENVALUES_HA = np.array([[-0.2, 0.1]])
KPATH = [KPoint([0.25, 0.0, 0.0])]
# G = (0,0,0) and (2,0,0) belong to the primitive k-point, (1,0,0) does not.
EXPECTED_WEIGHTS = np.array([[5.0 / 6.0, 0.0]])


def _save_dir(tmp_path: Path, name: str = "pwscf.save") -> Path:
    save_dir = tmp_path / name
    save_dir.mkdir(parents=True, exist_ok=True)
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)
    write_data_file_schema(
        save_dir / "data-file-schema.xml",
        lattice_bohr=np.diag([8.0, 4.0, 4.0]),
        alat_bohr=8.0,
        kpoints_cart_alat=np.array([[0.5, 0.0, 0.0]]),
        eigenvalues_hartree=EIGENVALUES_HA,
    )
    return save_dir


def _structures() -> tuple[Structure, Structure]:
    primitive = Structure(lattice=TARGET_LATTICE, species=("X",), frac_coords=np.zeros((1, 3)))
    supercell = Structure(
        lattice=SUPERCELL_LATTICE,
        species=("X", "X"),
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    return primitive, supercell


class TestUnfoldQeSaveToLattice:
    def test_weights_energies_and_metadata(self, tmp_path: Path) -> None:
        ebs = unfold_qe_save_to_lattice(
            qe_save_dir=_save_dir(tmp_path),
            supercell_lattice=SUPERCELL_LATTICE,
            target_lattice=TARGET_LATTICE,
            kpath=KPATH,
        )

        assert ebs.weights == pytest.approx(EXPECTED_WEIGHTS)
        assert ebs.energies == pytest.approx(EIGENVALUES_HA * HARTREE_EV)
        assert ebs.metadata["transformation"]["matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert "data-file-schema.xml" in str(ebs.metadata["energy_source"])
        assert ebs.metadata["symmetry_reduced"] is False

    def test_path_abscissa_is_cartesian_in_the_target_reciprocal_lattice(
        self, tmp_path: Path
    ) -> None:
        ebs = unfold_qe_save_to_lattice(
            qe_save_dir=_save_dir(tmp_path),
            supercell_lattice=SUPERCELL_LATTICE,
            target_lattice=TARGET_LATTICE,
            kpath=KPATH,
        )
        assert ebs.distances == pytest.approx([0.0])

    def test_mismatched_band_path_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="eigenvalues cover"):
            unfold_qe_save_to_lattice(
                qe_save_dir=_save_dir(tmp_path),
                supercell_lattice=SUPERCELL_LATTICE,
                target_lattice=TARGET_LATTICE,
                kpath=[KPoint([0.25, 0.0, 0.0]), KPoint([0.75, 0.0, 0.0])],
            )

    def test_empty_kpath_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no primitive k-path"):
            unfold_qe_save_to_lattice(
                qe_save_dir=_save_dir(tmp_path),
                supercell_lattice=SUPERCELL_LATTICE,
                target_lattice=TARGET_LATTICE,
                kpath=[],
            )

    def test_a_target_that_is_not_a_sublattice_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            unfold_qe_save_to_lattice(
                qe_save_dir=_save_dir(tmp_path),
                supercell_lattice=SUPERCELL_LATTICE,
                target_lattice=np.diag([0.7, 1.0, 1.0]),
                kpath=KPATH,
            )


class TestBackendNeutralDispatcher:
    def test_qe_dispatch_matches_the_direct_call(self, tmp_path: Path) -> None:
        save_dir = _save_dir(tmp_path)
        direct = unfold_qe_save_to_lattice(
            qe_save_dir=save_dir,
            supercell_lattice=SUPERCELL_LATTICE,
            target_lattice=TARGET_LATTICE,
            kpath=KPATH,
        )
        dispatched = unfold_supercell_to_lattice(
            "qe",
            qe_save_dir=save_dir,
            supercell_lattice=SUPERCELL_LATTICE,
            target_lattice=TARGET_LATTICE,
            kpath=KPATH,
        )
        assert dispatched.weights == pytest.approx(direct.weights)
        assert dispatched.energies == pytest.approx(direct.energies)

    def test_backend_specific_arguments_are_not_mixed(self, tmp_path: Path) -> None:
        save_dir = _save_dir(tmp_path)
        with pytest.raises(ValueError, match="belongs to the VASP backend"):
            unfold_supercell_to_lattice(
                "qe",
                qe_save_dir=save_dir,
                wavecar=tmp_path / "WAVECAR",
                supercell_lattice=SUPERCELL_LATTICE,
                target_lattice=TARGET_LATTICE,
                kpath=KPATH,
            )
        with pytest.raises(ValueError, match="belong to the QE backend"):
            unfold_supercell_to_lattice(
                "vasp",
                wavecar=tmp_path / "WAVECAR",
                qe_save_dir=save_dir,
                supercell_lattice=SUPERCELL_LATTICE,
                target_lattice=TARGET_LATTICE,
                kpath=KPATH,
            )

    def test_missing_source_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="needs a .save directory"):
            unfold_supercell_to_lattice(
                "qe",
                supercell_lattice=SUPERCELL_LATTICE,
                target_lattice=TARGET_LATTICE,
                kpath=KPATH,
            )

    def test_unknown_backend_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="code must be vasp or qe"):
            unfold_supercell_to_lattice(
                "abinit",
                supercell_lattice=SUPERCELL_LATTICE,
                target_lattice=TARGET_LATTICE,
                kpath=KPATH,
            )


class TestUnfoldingProblemQeBackend:
    def _problem(self, tmp_path: Path, outputs: Path) -> UnfoldingProblem:
        primitive, supercell = _structures()
        return UnfoldingProblem(
            primitive_structure=primitive,
            supercell_structure=supercell,
            primitive_kpath=list(KPATH),
            supercell_outputs=outputs,
            code="qe",
        )

    def test_problem_unfolds_a_qe_save(self, tmp_path: Path) -> None:
        save_dir = _save_dir(tmp_path)
        problem = self._problem(tmp_path, tmp_path)

        ebs = problem.unfold()

        assert ebs.weights == pytest.approx(EXPECTED_WEIGHTS)
        assert ebs.metadata["qe_save_dir"] == str(save_dir)
        assert problem.transformation is not None

    def test_save_directory_is_found_inside_the_outputs(self, tmp_path: Path) -> None:
        save_dir = _save_dir(tmp_path, "other.save")
        problem = self._problem(tmp_path, tmp_path)
        assert problem._resolve_qe_save_dir(None) == save_dir

    def test_several_save_directories_are_ambiguous(self, tmp_path: Path) -> None:
        _save_dir(tmp_path, "a.save")
        _save_dir(tmp_path, "b.save")
        problem = self._problem(tmp_path, tmp_path)
        with pytest.raises(ValueError, match="several save directories"):
            problem._resolve_qe_save_dir(None)

    def test_missing_save_directory_is_reported(self, tmp_path: Path) -> None:
        problem = self._problem(tmp_path, tmp_path / "empty")
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no QE .save directory"):
            problem._resolve_qe_save_dir(None)

    def test_procar_projections_are_rejected_for_qe(self, tmp_path: Path) -> None:
        _save_dir(tmp_path)
        problem = self._problem(tmp_path, tmp_path)
        with pytest.raises(ValueError, match="VASP PROCAR"):
            problem.unfold(projections=["species:X"])

    def test_unknown_backend_is_rejected(self, tmp_path: Path) -> None:
        problem = self._problem(tmp_path, tmp_path)
        with pytest.raises(ValueError, match="backend must be vasp or qe"):
            problem.unfold(backend="abinit")


class TestTwistWorkflowQeBackend:
    def test_unfold_to_reference_reads_a_qe_save(self, tmp_path: Path) -> None:
        save_dir = _save_dir(tmp_path)
        primitive, supercell = _structures()
        problem = TwistedUnfoldingProblem(
            references={"layer": primitive},
            supercell_structure=supercell,
            outputs=tmp_path,
        )

        ebs = problem.unfold_to_reference("layer", kpath=KPATH, code="qe")

        assert ebs.weights == pytest.approx(EXPECTED_WEIGHTS)
        assert ebs.metadata["reference"] == "layer"
        assert ebs.metadata["qe_save_dir"] == str(save_dir)

    def test_explicit_save_directory_is_honoured(self, tmp_path: Path) -> None:
        save_dir = _save_dir(tmp_path, "explicit.save")
        primitive, supercell = _structures()
        problem = TwistedUnfoldingProblem(
            references={"layer": primitive}, supercell_structure=supercell
        )

        ebs = problem.unfold_to_reference("layer", kpath=KPATH, code="qe", qe_save_dir=save_dir)

        assert ebs.weights == pytest.approx(EXPECTED_WEIGHTS)
