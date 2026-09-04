"""Projected unfolding through the backend workflow and the command line."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic_procar import write_synthetic_procar
from synthetic_wavecar import write_synthetic_wavecar
from typer.testing import CliRunner

from unmochan.cli.main import app, main
from unmochan.io.vasp import write_vasp_path_files
from unmochan.io.vasp_wfc import generate_vasp_g_vectors
from unmochan.workflows.backend import unfold_backend_bands
from unmochan.workflows.vasp import build_vasp_effective_band_structure

runner = CliRunner()

ENCUT = 200.0
LATTICE = np.eye(3)


def test_backend_projection_splits_the_weight_between_the_two_sites(tmp_path: Path):
    kmap, wavecar, procar, poscar = _fixture(tmp_path)

    plain, plain_mode = build_vasp_effective_band_structure(
        kmap=kmap, wavecar=wavecar, transform=np.diag([2.0, 1.0, 1.0])
    )
    first, mode = build_vasp_effective_band_structure(
        kmap=kmap,
        wavecar=wavecar,
        transform=np.diag([2.0, 1.0, 1.0]),
        projections=["atom:0"],
        procar=procar,
        supercell=poscar,
    )
    second, _ = build_vasp_effective_band_structure(
        kmap=kmap,
        wavecar=wavecar,
        transform=np.diag([2.0, 1.0, 1.0]),
        projections=["atom:1"],
        procar=procar,
        supercell=poscar,
    )

    assert np.allclose(first.weights, 0.25 * plain.weights)
    assert np.allclose(first.weights + second.weights, plain.weights)
    assert "projected onto atom:0" in mode
    assert "projected" not in plain_mode
    assert first.metadata["projection"] == "atom:0"


def test_backend_projection_requires_procar_and_supercell(tmp_path: Path):
    kmap, wavecar, _procar, _poscar = _fixture(tmp_path)

    with pytest.raises(ValueError, match="requires both procar and supercell"):
        build_vasp_effective_band_structure(
            kmap=kmap,
            wavecar=wavecar,
            transform=np.diag([2.0, 1.0, 1.0]),
            projections=["atom:0"],
        )


def test_backend_projection_is_rejected_for_the_qe_backend(tmp_path: Path):
    kmap, _wavecar, procar, poscar = _fixture(tmp_path)

    with pytest.raises(ValueError, match="not available"):
        unfold_backend_bands(
            "qe",
            kmap=kmap,
            projections=["atom:0"],
            procar=procar,
            supercell=poscar,
        )


def test_unfold_cli_accepts_projection_selectors(tmp_path: Path):
    kmap, wavecar, procar, poscar = _fixture(tmp_path)
    unfolded = tmp_path / "unfolded.dat"
    plot = tmp_path / "plot.svg"

    result = runner.invoke(
        app,
        [
            "unfold",
            "--code",
            "vasp",
            "--kmap",
            str(kmap),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--wavecar",
            str(wavecar),
            "--procar",
            str(procar),
            "--supercell",
            str(poscar),
            "--projection",
            "species:A",
            "--out",
            str(unfolded),
            "--plot",
            str(plot),
        ],
    )

    assert result.exit_code == 0, result.output
    # The console wraps long lines, so normalize whitespace before matching.
    assert "projected onto species:A" in " ".join(result.output.split())
    assert unfolded.exists()


def test_unfold_cli_reports_an_unknown_selector_without_traceback(tmp_path: Path, capsys):
    kmap, wavecar, procar, poscar = _fixture(tmp_path)

    exit_code = main(
        [
            "unfold",
            "--code",
            "vasp",
            "--kmap",
            str(kmap),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--wavecar",
            str(wavecar),
            "--procar",
            str(procar),
            "--supercell",
            str(poscar),
            "--projection",
            "nonsense:1",
            "--out",
            str(tmp_path / "unfolded.dat"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "unsupported projection namespace" in captured.out
    assert "Traceback" not in captured.out + captured.err


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    path_json = tmp_path / "path.json"
    kpoints = tmp_path / "KPOINTS"
    kmap = tmp_path / "kmap.tsv"
    ticks = tmp_path / "ticks.tsv"
    path_json.write_text(
        '{"transformation_matrix": [[2,0,0],[0,1,0],[0,0,1]], '
        '"path": [{"label":"G","k":[0,0,0],"n":2},{"label":"G","k":[0,0,0]}]}'
    )
    write_vasp_path_files(path_json, kpoints=kpoints, kmap=kmap, ticks=ticks)

    wavecar = tmp_path / "WAVECAR"
    g_vectors = generate_vasp_g_vectors(LATTICE, np.zeros(3), ENCUT)
    write_synthetic_wavecar(
        wavecar,
        lattice=LATTICE,
        encut=ENCUT,
        rtag=45200,
        kpoints=np.zeros((2, 3)),
        energies=np.array([[-1.0], [-0.5]]),
        occupations=np.ones((2, 1)),
        coefficients=np.ones((2, 1, 1, len(g_vectors)), dtype=np.complex64),
    )

    # A quarter of the character on the first site, three quarters on the second.
    projections = np.array([[[[[0.1, 0.1], [0.3, 0.3]]], [[[0.1, 0.1], [0.3, 0.3]]]]])
    procar = write_synthetic_procar(
        tmp_path / "PROCAR",
        kpoints=np.zeros((2, 3)),
        kpoint_weights=np.ones(2),
        energies=np.array([[[-1.0], [-0.5]]]),
        occupations=np.ones((1, 2, 1)),
        projections=projections,
        orbital_labels=("s", "px"),
    )

    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "\n".join(
            ["supercell", "1.0", "2 0 0", "0 1 0", "0 0 1", "A", "2", "Direct", "0 0 0", "0.5 0 0"]
        )
    )
    return kmap, wavecar, procar, poscar
