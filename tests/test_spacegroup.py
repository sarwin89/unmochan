from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.core.spacegroup import (
    detect_primitive_operations,
    lattice_point_group,
    metric_tensor,
    reciprocal_operation,
    space_group_operations,
)
from unmochan.core.structures import Structure
from unmochan.core.symmetry import map_kpoints_to_stored, supercell_operation

runner = CliRunner()

CUBIC = np.eye(3)
TETRAGONAL = np.diag([1.0, 1.0, 2.3])
HEXAGONAL = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 2.0]])
TRICLINIC = np.array([[1.0, 0.0, 0.0], [0.3, 1.7, 0.0], [0.1, 0.2, 2.3]])


@pytest.mark.parametrize(
    ("lattice", "order"),
    [(CUBIC, 48), (TETRAGONAL, 16), (HEXAGONAL, 24), (TRICLINIC, 2)],
)
def test_lattice_point_group_has_the_expected_order(lattice, order):
    assert len(lattice_point_group(lattice)) == order


def test_point_group_operations_preserve_the_metric_exactly():
    metric = metric_tensor(HEXAGONAL)

    for m in lattice_point_group(HEXAGONAL):
        assert m.dtype == np.int64
        np.testing.assert_allclose(m.T @ metric @ m, metric, atol=1e-9)
        assert round(abs(float(np.linalg.det(m)))) == 1


def test_point_group_is_closed_under_multiplication_and_inverse():
    group = lattice_point_group(TETRAGONAL)
    known = {m.tobytes() for m in group}

    for a in group:
        assert (a @ a).astype(np.int64).tobytes() in known
        inverse = np.round(np.linalg.inv(a)).astype(np.int64)
        assert inverse.tobytes() in known

    assert any(np.array_equal(m, np.eye(3, dtype=np.int64)) for m in group)
    assert any(np.array_equal(m, -np.eye(3, dtype=np.int64)) for m in group)


def test_the_search_box_is_wide_enough_for_a_skewed_cell():
    """A strongly sheared description of the cubic lattice still finds 48 operations."""

    shear = np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    assert len(lattice_point_group(shear @ CUBIC)) == 48


def test_reciprocal_operation_is_the_inverse_transpose():
    for m in lattice_point_group(HEXAGONAL):
        s = reciprocal_operation(m)
        np.testing.assert_allclose(s.T @ m, np.eye(3), atol=1e-9)


def test_reciprocal_operation_rejects_a_non_unimodular_matrix():
    with pytest.raises(ValueError, match="unimodular"):
        reciprocal_operation(np.diag([2, 1, 1]))


def test_space_group_of_a_one_atom_cubic_cell_is_the_full_holohedry():
    cell = Structure(lattice=CUBIC, species=("A",), frac_coords=np.zeros((1, 3)))

    operations = space_group_operations(cell)

    assert len(operations) == 48
    assert all(op.is_symmorphic for op in operations)


def test_a_body_centred_cell_gains_non_symmorphic_operations():
    cell = Structure(
        lattice=CUBIC,
        species=("A", "A"),
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    )

    operations = space_group_operations(cell)

    assert len(operations) == 96
    assert sum(0 if op.is_symmorphic else 1 for op in operations) == 48


def test_decorating_the_body_centre_with_another_species_removes_the_translation():
    cell = Structure(
        lattice=CUBIC,
        species=("A", "B"),
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    )

    operations = space_group_operations(cell)

    assert len(operations) == 48
    assert all(op.is_symmorphic for op in operations)


def test_a_displaced_site_lowers_the_symmetry():
    cell = Structure(
        lattice=CUBIC,
        species=("A", "B"),
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]]),
    )

    operations = space_group_operations(cell)

    assert 0 < len(operations) < 48
    # Every kept rotation must fix the stacking axis up to a sign.
    for op in operations:
        np.testing.assert_array_equal(np.abs(op.rotation[:, 2]), np.array([0, 0, 1]))


def test_detected_operations_are_symmetries_of_the_supercell_matrix():
    transform = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    supercell = Structure(
        lattice=np.diag([2.0, 1.0, 1.0]),
        species=("A", "A"),
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )

    operations = detect_primitive_operations(supercell, transform)

    assert operations.shape == (16, 3, 3)
    assert any(np.array_equal(op, np.eye(3, dtype=np.int64)) for op in operations)
    for op in operations:
        # Integral by construction: this is what --symmetry requires.
        supercell_operation(op, transform)


def test_detected_operations_can_serve_a_symmetry_reduced_kpoint_set():
    """A k-point outside the stored wedge is reached by a detected operation."""

    transform = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    supercell = Structure(
        lattice=np.diag([2.0, 2.0, 1.0]),
        species=("A",) * 4,
        frac_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]),
    )
    operations = detect_primitive_operations(supercell, transform)

    stored = np.array([[0.25, 0.0, 0.0]])
    requested = np.array([[0.0, 0.125, 0.0]])  # folds to (0, 0.25, 0), a star partner

    with pytest.raises(ValueError):
        map_kpoints_to_stored(requested, stored, transform)

    matches = map_kpoints_to_stored(requested, stored, transform, operations)

    assert len(matches) == 1
    assert matches[0].index == 0


def test_detect_symmetry_cli_reports_and_writes_operations(tmp_path: Path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "\n".join(
            [
                "supercell",
                "1.0",
                "2.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "A",
                "2",
                "Direct",
                "0.0 0.0 0.0",
                "0.5 0.0 0.0",
                "",
            ]
        )
    )
    out = tmp_path / "ops.txt"

    result = runner.invoke(
        app,
        [
            "detect-symmetry",
            "--structure",
            str(poscar),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "16" in result.output
    rows = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 16
    assert all(len(line.split()) == 9 for line in rows)


def test_detect_symmetry_cli_reports_the_space_group_without_a_matrix(tmp_path: Path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "\n".join(
            [
                "bcc in a cubic cell",
                "1.0",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "A",
                "2",
                "Direct",
                "0.0 0.0 0.0",
                "0.5 0.5 0.5",
                "",
            ]
        )
    )

    result = runner.invoke(app, ["detect-symmetry", "--structure", str(poscar)])

    assert result.exit_code == 0, result.output
    assert "Space-group operations: 96" in result.output
    assert "Non-symmorphic among them: 48" in result.output
