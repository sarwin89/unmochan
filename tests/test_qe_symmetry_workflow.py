"""Symmetry-reduced QE unfolding and the QE k-point cross-check from the workflows.

The QE reader has supported both for a while, but neither was reachable from the
backend dispatcher or the command line: the dispatcher rejected ``operations``
for QE outright and had no way to pass QE's ``at`` matrix, without which the
stored Cartesian k-points cannot be converted to fractional coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_qe_wfc_kpoints import write_synthetic_wfc_dat
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.io.qe import write_qe_path_files
from unmochan.workflows.backend import compute_backend_weights

runner = CliRunner()

MILLER = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
COEFFS = np.array([[1.0, 1.0, 2.0], [0.0, 1.0, 0.0]], dtype=complex)
IDENTITY = np.eye(3)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A k-map asking for two k-points and a save directory holding only one."""

    path_json = tmp_path / "path.json"
    path_json.write_text(
        '{"transformation_matrix": [[2,0,0],[0,1,0],[0,0,1]], '
        '"path": [{"label":"G","k":[0,0,0],"n":2},{"label":"G","k":[0,0,0]}]}'
    )
    kmap = tmp_path / "kmap.tsv"
    write_qe_path_files(
        path_json,
        qe_kpoints=tmp_path / "qe_kpoints.in",
        kmap=kmap,
        ticks=tmp_path / "ticks.tsv",
    )
    save_dir = tmp_path / "pwscf.save"
    save_dir.mkdir()
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.0, 0.0, 0.0), MILLER, COEFFS)
    operations = tmp_path / "operations.txt"
    operations.write_text("# identity is implicit\n")
    return kmap, save_dir, operations


def test_qe_symmetry_mode_serves_a_longer_path_from_one_stored_kpoint(tmp_path: Path):
    kmap, save_dir, _operations = _fixture(tmp_path)
    out = tmp_path / "weights.dat"

    # Without symmetry mode the file count must match the k-map exactly.
    with pytest.raises(ValueError, match="wavefunction files"):
        compute_backend_weights(
            "qe",
            kmap=kmap,
            transform=np.diag([2.0, 1.0, 1.0]),
            out=out,
            nbnd=2,
            qe_save_dir=save_dir,
        )

    diagnostics = compute_backend_weights(
        "qe",
        kmap=kmap,
        transform=np.diag([2.0, 1.0, 1.0]),
        out=out,
        nbnd=2,
        qe_save_dir=save_dir,
        qe_lattice_alat=IDENTITY,
        operations=np.zeros((0, 3, 3), dtype=int),
    )

    assert diagnostics.n_kpoints == 2
    assert diagnostics.n_bands == 2
    assert 0.0 <= diagnostics.min_weight <= diagnostics.max_weight <= 1.0
    assert out.exists()


def test_qe_symmetry_mode_requires_the_alat_lattice(tmp_path: Path):
    kmap, save_dir, _operations = _fixture(tmp_path)

    with pytest.raises(ValueError, match="lattice_alat"):
        compute_backend_weights(
            "qe",
            kmap=kmap,
            transform=np.diag([2.0, 1.0, 1.0]),
            out=tmp_path / "weights.dat",
            nbnd=2,
            qe_save_dir=save_dir,
            operations=np.zeros((0, 3, 3), dtype=int),
        )


def test_qe_lattice_alat_enables_the_kpoint_cross_check(tmp_path: Path):
    # A single-point k-map, and a wavefunction stored at a different k-point.
    kmap = tmp_path / "kmap.tsv"
    header = (
        "ik\ts_pc\tkpc_1\tkpc_2\tkpc_3\t"
        "Ksc_unfold_1\tKsc_unfold_2\tKsc_unfold_3\t"
        "Ksc_fold_1\tKsc_fold_2\tKsc_fold_3\tlabel"
    )
    row = "\t".join(["1", "0.0", "0.0", "0.0", "0.0"] + ["0.0"] * 6 + ["G"])
    kmap.write_text(f"{header}\n{row}\n")
    save_dir = tmp_path / "pwscf.save"
    save_dir.mkdir()
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.25, 0.0, 0.0), MILLER, COEFFS)

    with pytest.raises(ValueError, match="not the same calculation"):
        compute_backend_weights(
            "qe",
            kmap=kmap,
            transform=np.diag([2.0, 1.0, 1.0]),
            out=tmp_path / "weights.dat",
            nbnd=2,
            qe_save_dir=save_dir,
            qe_lattice_alat=IDENTITY,
        )


def test_weights_cli_accepts_symmetry_and_lattice_alat_for_qe(tmp_path: Path):
    kmap, save_dir, operations = _fixture(tmp_path)
    out = tmp_path / "weights.dat"

    result = runner.invoke(
        app,
        [
            "weights",
            "--code",
            "qe",
            "--kmap",
            str(kmap),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--nbnd",
            "2",
            "--qe-save-dir",
            str(save_dir),
            "--symmetry",
            str(operations),
            "--lattice-alat",
            "1 0 0 0 1 0 0 0 1",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["n_kpoints"] == 2
    assert payload["max_weight"] <= 1.0
