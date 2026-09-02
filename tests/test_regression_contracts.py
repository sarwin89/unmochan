from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.io.qe import BOHR_TO_ANGSTROM, read_pw_input_structure, read_weight_table

pytestmark = pytest.mark.unit


def test_one_band_weight_table_keeps_two_dimensional_shape(tmp_path: Path):
    weights_file = tmp_path / "weights.dat"
    weights_file.write_text("# ik band spectral_weight\n1 1 0.25\n2 1 0.75\n")

    weights = read_weight_table(weights_file, n_kpoints=2, n_bands=1)

    assert weights.shape == (2, 1)
    assert np.allclose(weights, [[0.25], [0.75]])


def test_weight_table_requires_all_explicit_rows(tmp_path: Path):
    weights_file = tmp_path / "weights.dat"
    weights_file.write_text("1 1 1.0\n")

    with pytest.raises(ValueError, match="missing"):
        read_weight_table(weights_file, n_kpoints=2, n_bands=1)


@pytest.mark.parametrize(
    ("cell_unit", "position_unit", "alat_line", "expected_lattice"),
    [
        ("angstrom", "angstrom", "", 2.0),
        ("bohr", "bohr", "", 2.0 * BOHR_TO_ANGSTROM),
        ("alat", "alat", "  A = 3.0,\n", 6.0),
    ],
)
def test_qe_pw_input_angstrom_bohr_and_alat_units(
    tmp_path: Path,
    cell_unit: str,
    position_unit: str,
    alat_line: str,
    expected_lattice: float,
):
    pw_input = tmp_path / "scf.in"
    pw_input.write_text(
        f"""
&system
  ibrav = 0,
{alat_line}/
CELL_PARAMETERS {cell_unit}
  2.0 0.0 0.0
  0.0 2.0 0.0
  0.0 0.0 2.0
ATOMIC_POSITIONS {position_unit}
  X 0.5 0.0 0.0
""".strip()
    )

    structure = read_pw_input_structure(pw_input)

    assert np.allclose(np.diag(structure.lattice), [expected_lattice] * 3)
    assert np.allclose(structure.frac_coords, [[0.25, 0.0, 0.0]])


@pytest.mark.cli
def test_unified_cli_bad_backend_reports_clean_error():
    result = CliRunner().invoke(app, ["make-kpoints", "--code", "elk", "path.json"])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "Invalid value" in result.output or "Choose from" in result.output
