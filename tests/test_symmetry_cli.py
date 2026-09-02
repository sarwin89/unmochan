"""CLI surface of the symmetry check: `check-symmetry`, and `--magmom`.

The physics is in ``tests/test_symmetry_validation.py`` and in
``RequestProject/Unfolding/SymmetryHypothesis.lean``; here only the plumbing is
checked -- that a bad operation file makes the command fail, that a good one
does not, and that `weights` and `unfold` refuse a symmetry file that the
structure they are given does not support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from unfoldlab.cli.common import parse_magmoms
from unfoldlab.cli.main import app

runner = CliRunner()

pytestmark = [pytest.mark.cli, pytest.mark.unit]

TETRAGONAL_POSCAR = "\n".join(
    [
        "two sites, unequal in-plane axes",
        "1.0",
        "3.0 0.0 0.0",
        "0.0 4.0 0.0",
        "0.0 0.0 10.0",
        "Fe",
        "2",
        "Direct",
        "0.0 0.0 0.0",
        "0.5 0.5 0.0",
        "",
    ]
)

CHAIN_POSCAR = "\n".join(
    [
        "three sites along z",
        "1.0",
        "3.0 0.0 0.0",
        "0.0 3.0 0.0",
        "0.0 0.0 9.0",
        "Fe",
        "3",
        "Direct",
        "0.0 0.0 0.0",
        "0.0 0.0 0.3333333333333333",
        "0.0 0.0 0.6666666666666666",
        "",
    ]
)


def _flat(text: str) -> str:
    """Rich wraps error text into a box; recover the flowing sentence."""

    return " ".join(text.replace("\u2502", " ").split())


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_parse_magmoms_collinear_and_noncollinear() -> None:
    assert np.allclose(parse_magmoms("2*1.0 2*-1.0", 4), [1.0, 1.0, -1.0, -1.0])
    assert parse_magmoms("0 0 1 0 0 -1", 2).shape == (2, 3)


def test_parse_magmoms_rejects_a_wrong_count() -> None:
    with pytest.raises(Exception, match="one value per site"):
        parse_magmoms("1.0 1.0", 3)


def test_check_symmetry_accepts_the_detected_file(tmp_path: Path) -> None:
    poscar = _write(tmp_path / "POSCAR", TETRAGONAL_POSCAR)
    ops = tmp_path / "ops.txt"
    detect = runner.invoke(
        app,
        [
            "detect-symmetry",
            "--structure",
            str(poscar),
            "--matrix",
            "1 0 0 0 1 0 0 0 1",
            "--out",
            str(ops),
        ],
    )
    assert detect.exit_code == 0, detect.output

    result = runner.invoke(
        app,
        [
            "check-symmetry",
            "--structure",
            str(poscar),
            "--matrix",
            "1 0 0 0 1 0 0 0 1",
            "--symmetry",
            str(ops),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "All supplied operations are symmetries" in result.output


def test_check_symmetry_rejects_an_axis_exchange(tmp_path: Path) -> None:
    poscar = _write(tmp_path / "POSCAR", TETRAGONAL_POSCAR)
    ops = _write(tmp_path / "ops.txt", "0 1 0 1 0 0 0 0 1\n")

    result = runner.invoke(
        app,
        [
            "check-symmetry",
            "--structure",
            str(poscar),
            "--matrix",
            "1 0 0 0 1 0 0 0 1",
            "--symmetry",
            str(ops),
        ],
    )
    assert result.exit_code == 1
    assert "REJECTED" in result.output


def test_detect_symmetry_with_magmom_returns_fewer_operations(tmp_path: Path) -> None:
    poscar = _write(tmp_path / "POSCAR", CHAIN_POSCAR)
    base = ["detect-symmetry", "--structure", str(poscar), "--matrix", "1 0 0 0 1 0 0 0 3"]

    plain = tmp_path / "plain.txt"
    magnetic = tmp_path / "magnetic.txt"
    assert runner.invoke(app, [*base, "--out", str(plain)]).exit_code == 0
    result = runner.invoke(app, [*base, "--magmom", "1.0 2.0 3.0", "--out", str(magnetic)])
    assert result.exit_code == 0, result.output

    assert len(magnetic.read_text().splitlines()) < len(plain.read_text().splitlines())


def test_detect_symmetry_refuses_time_reversal_for_a_magnet(tmp_path: Path) -> None:
    poscar = _write(tmp_path / "POSCAR", CHAIN_POSCAR)
    result = runner.invoke(
        app,
        [
            "detect-symmetry",
            "--structure",
            str(poscar),
            "--matrix",
            "1 0 0 0 1 0 0 0 3",
            "--magmom",
            "1.0 2.0 3.0",
            "--time-reversal",
        ],
    )
    assert result.exit_code != 0
    assert "magnetic" in result.output


def test_weights_refuses_a_symmetry_file_the_structure_does_not_support(
    tmp_path: Path,
) -> None:
    poscar = _write(tmp_path / "POSCAR", TETRAGONAL_POSCAR)
    ops = _write(tmp_path / "ops.txt", "0 1 0 1 0 0 0 0 1\n")
    kmap = _write(tmp_path / "kmap.txt", "0.5 0.0 0.0 0.5 0.0 0.0 1\n")

    result = runner.invoke(
        app,
        [
            "weights",
            "--code",
            "vasp",
            "--kmap",
            str(kmap),
            "--matrix",
            "1 0 0 0 1 0 0 0 1",
            "--symmetry",
            str(ops),
            "--structure",
            str(poscar),
            "--out",
            str(tmp_path / "w.dat"),
        ],
    )
    assert result.exit_code != 0
    assert "not symmetries of the structure" in _flat(result.output)


def test_unfold_refuses_a_symmetry_file_the_supercell_does_not_support(
    tmp_path: Path,
) -> None:
    poscar = _write(tmp_path / "POSCAR", TETRAGONAL_POSCAR)
    ops = _write(tmp_path / "ops.txt", "0 1 0 1 0 0 0 0 1\n")
    kmap = _write(tmp_path / "kmap.txt", "0.5 0.0 0.0 0.5 0.0 0.0 1\n")

    result = runner.invoke(
        app,
        [
            "unfold",
            "--code",
            "vasp",
            "--kmap",
            str(kmap),
            "--matrix",
            "1 0 0 0 1 0 0 0 1",
            "--symmetry",
            str(ops),
            "--supercell",
            str(poscar),
            "--out",
            str(tmp_path / "bands.dat"),
        ],
    )
    assert result.exit_code != 0
    assert "not symmetries of the structure" in _flat(result.output)
