"""A QE ``.save`` XML can stand in for the bands file and for ``--nbnd``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.synthetic_qe_xml import write_data_file_schema
from tests.test_qe_wfc_kpoints import COEFFS, MILLER, write_synthetic_wfc_dat
from unfoldlab.io.qe_xml import HARTREE_EV, eigenvalues_for_spin, read_qe_xml
from unfoldlab.workflows.backend import compute_backend_weights, unfold_backend_bands

TRANSFORM = np.diag([2.0, 1.0, 1.0])
EIGENVALUES_HA = np.array([[-0.2, 0.1]])


def _kmap(tmp_path: Path) -> Path:
    kmap = tmp_path / "kmap.tsv"
    header = (
        "ik\ts_pc\tkpc_1\tkpc_2\tkpc_3\t"
        "Ksc_unfold_1\tKsc_unfold_2\tKsc_unfold_3\t"
        "Ksc_fold_1\tKsc_fold_2\tKsc_fold_3\tlabel"
    )
    row = "\t".join(["1", "0.0", "0.25", "0.0", "0.0"] + ["0.5", "0.0", "0.0"] * 2 + ["X"])
    kmap.write_text(f"{header}\n{row}\n")
    return kmap


def _save(tmp_path: Path, **kwargs) -> Path:
    save_dir = tmp_path / "pwscf.save"
    save_dir.mkdir(exist_ok=True)
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)
    kwargs.setdefault("eigenvalues_hartree", EIGENVALUES_HA)
    write_data_file_schema(
        save_dir / "data-file-schema.xml",
        lattice_bohr=np.eye(3) * 4.0,
        alat_bohr=4.0,
        kpoints_cart_alat=np.array([[0.5, 0.0, 0.0]]),
        **kwargs,
    )
    return save_dir


def test_weights_infer_the_band_count_from_the_xml(tmp_path: Path):
    kmap = _kmap(tmp_path)
    save_dir = _save(tmp_path)

    diagnostics = compute_backend_weights(
        "qe",
        kmap=kmap,
        transform=TRANSFORM,
        out=tmp_path / "weights.dat",
        qe_save_dir=save_dir,
    )

    assert diagnostics.n_kpoints == 1
    assert diagnostics.n_bands == 2


def test_unfold_takes_the_energies_from_the_xml(tmp_path: Path):
    kmap = _kmap(tmp_path)
    save_dir = _save(tmp_path)

    result = unfold_backend_bands(
        "qe",
        kmap=kmap,
        transform=TRANSFORM,
        qe_save_dir=save_dir,
        out=tmp_path / "unfolded.dat",
        plot=None,
    )

    assert np.allclose(result.ebs.energies, EIGENVALUES_HA * HARTREE_EV)
    assert "data-file-schema.xml" in result.ebs.metadata["bands"]
    assert np.allclose(result.ebs.weights, [[5.0 / 6.0, 0.0]])


def test_unfold_without_bands_or_a_save_directory_is_rejected(tmp_path: Path):
    kmap = _kmap(tmp_path)

    with pytest.raises(ValueError, match="requires bands"):
        unfold_backend_bands(
            "qe",
            kmap=kmap,
            transform=TRANSFORM,
            out=tmp_path / "unfolded.dat",
            plot=None,
        )


def test_a_spin_channel_must_be_chosen_for_lsda(tmp_path: Path):
    save_dir = _save(
        tmp_path,
        lsda=True,
        eigenvalues_hartree=np.array([[-0.2, 0.1, -0.15, 0.12]]),
    )
    metadata = read_qe_xml(save_dir)

    assert metadata.lsda is True
    with pytest.raises(ValueError, match="spin-polarized"):
        eigenvalues_for_spin(metadata, None)
    up = eigenvalues_for_spin(metadata, 1)
    down = eigenvalues_for_spin(metadata, 2)
    assert np.allclose(up, np.array([-0.2, 0.1]) * HARTREE_EV)
    assert np.allclose(down, np.array([-0.15, 0.12]) * HARTREE_EV)


def test_unfold_cli_needs_neither_bands_nor_lattice_alat(tmp_path: Path):
    from typer.testing import CliRunner

    from unfoldlab.cli.main import app

    kmap = _kmap(tmp_path)
    save_dir = _save(tmp_path)
    out = tmp_path / "unfolded.dat"

    result = CliRunner().invoke(
        app,
        [
            "unfold",
            "--code",
            "qe",
            "--kmap",
            str(kmap),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--qe-save-dir",
            str(save_dir),
            "--out",
            str(out),
            "--plot",
            str(tmp_path / "unfolded.png"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
