from pathlib import Path

from typer.testing import CliRunner

from unfoldlab.cli.main import app, main

runner = CliRunner()


def test_vasp_map_cli(tmp_path: Path):
    primitive = tmp_path / "POSCAR.prim"
    supercell = tmp_path / "POSCAR.sc"
    primitive.write_text(_poscar("primitive", ["1 0 0", "0 1 0", "0 0 1"]))
    supercell.write_text(_poscar("supercell", ["2 0 0", "0 2 0", "0 0 1"]))

    result = runner.invoke(
        app,
        ["vasp", "map", "--primitive", str(primitive), "--supercell", str(supercell)],
    )

    assert result.exit_code == 0, result.output
    assert '"multiplicity": 4' in result.output


def test_no_args_launches_guided_menu_and_exits_cleanly():
    result = runner.invoke(app, [], input="vasp\n0\n")

    assert result.exit_code == 0, result.output
    assert "Unmochan Guided Mode" in result.output
    assert "Leaving guided mode." in result.output


def test_guide_command_has_workflow_options_only():
    result = runner.invoke(app, ["guide"], input="vasp\n0\n")

    assert result.exit_code == 0, result.output
    assert "make-kpoints" in result.output
    assert "weights" in result.output
    assert "unfold" in result.output
    assert "Theory/unification" not in result.output


def test_main_reports_bad_command_without_traceback(capsys):
    exit_code = main(["unfoldlab"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "No such command 'unfoldlab'" in captured.err
    assert "Traceback" not in captured.err


def test_qe_make_kpoints_cli(tmp_path: Path):
    path_json = tmp_path / "path.json"
    qek = tmp_path / "qe_kpoints.in"
    kmap = tmp_path / "kmap.tsv"
    ticks = tmp_path / "ticks.tsv"
    path_json.write_text(
        '{"transformation_matrix": [[2,0,0],[0,1,0],[0,0,1]], '
        '"path": [{"label":"G","k":[0,0,0],"n":2},{"label":"X","k":[0.5,0,0]}]}'
    )

    result = runner.invoke(
        app,
        [
            "qe",
            "make-kpoints",
            str(path_json),
            "--qe-kpoints",
            str(qek),
            "--kmap",
            str(kmap),
            "--ticks",
            str(ticks),
        ],
    )

    assert result.exit_code == 0, result.output
    assert qek.exists()
    assert kmap.exists()
    assert ticks.exists()


def test_unified_make_kpoints_cli_for_vasp_and_qe(tmp_path: Path):
    path_json = tmp_path / "path.json"
    path_json.write_text(
        '{"transformation_matrix": [[2,0,0],[0,1,0],[0,0,1]], '
        '"path": [{"label":"G","k":[0,0,0],"n":2},{"label":"X","k":[0.5,0,0]}]}'
    )
    for code in ("vasp", "qe"):
        kpoints = tmp_path / f"{code}.kpoints"
        result = runner.invoke(
            app,
            [
                "make-kpoints",
                "--code",
                code,
                str(path_json),
                "--kpoints",
                str(kpoints),
                "--kmap",
                str(tmp_path / f"{code}.kmap.tsv"),
                "--ticks",
                str(tmp_path / f"{code}.ticks.tsv"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert kpoints.exists()


def test_qe_weights_cli_from_coefficients(tmp_path: Path):
    kmap = _write_kmap(tmp_path)
    coeffs = tmp_path / "coefficients.dat"
    out = tmp_path / "weights.dat"
    coeffs.write_text(
        "# ik ib G1 G2 G3 Re Im\n1 1 0 0 0 1 0\n1 1 1 0 0 1 0\n2 1 0 0 0 1 0\n2 1 1 0 0 1 0\n"
    )

    result = runner.invoke(
        app,
        [
            "qe",
            "weights",
            "--kmap",
            str(kmap),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--coefficients",
            str(coeffs),
            "--nbnd",
            "1",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "0.500000000000" in out.read_text()


def test_qe_unfold_cli_with_existing_weights(tmp_path: Path):
    kmap = _write_kmap(tmp_path)
    bands = tmp_path / "bands.dat.gnu"
    weights = tmp_path / "weights.dat"
    out = tmp_path / "unfolded.dat"
    plot = tmp_path / "plot.svg"
    bands.write_text("0 -1\n1 -0.5\n\n0 1\n1 1.5\n")
    weights.write_text("1 1 1.0\n1 2 0.5\n2 1 0.25\n2 2 0.75\n")

    result = runner.invoke(
        app,
        [
            "qe",
            "unfold",
            "--bands",
            str(bands),
            "--kmap",
            str(kmap),
            "--weights",
            str(weights),
            "--out",
            str(out),
            "--plot",
            str(plot),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert plot.exists()
    assert "Weight mode: read spectral weights" in result.output


def _poscar(name: str, lattice: list[str]) -> str:
    return "\n".join(
        [
            name,
            "1.0",
            *lattice,
            "X",
            "1",
            "Direct",
            "0 0 0",
        ]
    )


def _write_kmap(tmp_path: Path) -> Path:
    path_json = tmp_path / "path.json"
    qek = tmp_path / "qe_kpoints.in"
    kmap = tmp_path / "kmap.tsv"
    ticks = tmp_path / "ticks.tsv"
    path_json.write_text(
        '{"transformation_matrix": [[2,0,0],[0,1,0],[0,0,1]], '
        '"path": [{"label":"G","k":[0,0,0],"n":2},{"label":"X","k":[0.5,0,0]}]}'
    )
    result = runner.invoke(
        app,
        [
            "qe",
            "make-kpoints",
            str(path_json),
            "--qe-kpoints",
            str(qek),
            "--kmap",
            str(kmap),
            "--ticks",
            str(ticks),
        ],
    )
    assert result.exit_code == 0, result.output
    return kmap


def test_qe_info_reports_the_save_metadata(tmp_path: Path):
    import numpy as np

    from tests.synthetic_qe_xml import write_data_file_schema

    save = tmp_path / "pwscf.save"
    write_data_file_schema(
        save / "data-file-schema.xml",
        lattice_bohr=np.diag([4.0, 4.0, 8.0]),
        alat_bohr=4.0,
        kpoints_cart_alat=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        eigenvalues_hartree=np.zeros((2, 3)),
        paw=True,
    )

    result = runner.invoke(app, ["qe-info", "--save", str(tmp_path), "--kpoints"])

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "alat: 4.00000000 bohr" in output
    assert "k-points: 2" in output
    assert "Bands: 3" in output
    assert "PAW" in output
    assert "--lattice-alat" in output


def test_qe_info_rejects_a_directory_without_a_save(tmp_path: Path, capsys):
    exit_code = main(["qe-info", "--save", str(tmp_path)])

    assert exit_code == 2
    assert "no QE XML" in capsys.readouterr().out
