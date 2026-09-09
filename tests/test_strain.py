"""Residual strain and approximate commensurability.

The statements checked here are the numerical face of
``RequestProject/Unfolding/Strain.lean``: commensurability is exactly
integrality of ``A_sc A_pc^-1``, and the k-space error left by residual strain
is linear in k and cannot be removed by any choice of integer transform.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.core.strain import (
    deformation_gradient,
    diagnose_commensurability,
    kpoint_shift,
    nearest_integer_transformation,
    rotation_angle,
    strain_tensor,
)
from unmochan.core.transformations import detect_transformation

runner = CliRunner()

PRIMITIVE = np.diag([3.0, 3.0, 5.0])


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _poscar(name: str, lattice: np.ndarray) -> str:
    rows = [" ".join(f"{value:.12f}" for value in row) for row in lattice]
    return "\n".join([name, "1.0", *rows, "X", "1", "Direct", "0.0 0.0 0.0", ""])


def test_exact_supercell_is_commensurate() -> None:
    supercell = np.diag([2, 2, 2]) @ PRIMITIVE
    report = diagnose_commensurability(PRIMITIVE, supercell)

    assert report.commensurate
    assert report.residual == pytest.approx(0.0)
    assert np.array_equal(report.transform, np.diag([2, 2, 2]))
    assert report.deformation_gradient == pytest.approx(np.eye(3))
    assert report.max_abs_strain == pytest.approx(0.0)
    assert report.max_kpoint_shift == pytest.approx(0.0)
    assert "commensurate" in report.summary()


def test_a_non_integer_transform_is_rejected_and_quantified() -> None:
    supercell = np.diag([6.06, 6.0, 10.0])
    with pytest.raises(ValueError, match="not related by an integer"):
        detect_transformation(PRIMITIVE, supercell)

    report = diagnose_commensurability(PRIMITIVE, supercell)
    assert not report.commensurate
    # 6.06 / 3 = 2.02, so the residual is 0.02 and the strain is 1%.
    assert report.residual == pytest.approx(0.02)
    assert report.strain[0, 0] == pytest.approx(0.01)
    assert report.max_abs_strain == pytest.approx(0.01)
    assert report.rotation_deg == pytest.approx(0.0, abs=1e-9)
    assert not report.ambiguous


def test_the_kpoint_error_is_linear_in_k_and_vanishes_at_gamma() -> None:
    gradient = np.diag([1.01, 1.0, 1.0])
    assert kpoint_shift(gradient, np.zeros(3)) == pytest.approx(np.zeros((1, 3)))

    q = np.array([0.3, -0.7, 1.1])
    single = kpoint_shift(gradient, q)
    doubled = kpoint_shift(gradient, 2.0 * q)
    assert doubled == pytest.approx(2.0 * single)

    # The error is exactly q - q @ inv(D).
    assert single[0] == pytest.approx(q - q @ np.linalg.inv(gradient))


def test_the_deformation_gradient_reproduces_the_supercell() -> None:
    transform = np.diag([2, 2, 2])
    supercell = np.diag([1.01, 0.99, 1.0]) @ (transform @ PRIMITIVE)
    gradient = deformation_gradient(PRIMITIVE, supercell, transform)
    assert (transform @ PRIMITIVE) @ gradient.T == pytest.approx(supercell)


def test_a_rigid_rotation_is_reported_as_rotation_not_strain() -> None:
    rotation = _rotation_z(2.0)
    supercell = (np.diag([2, 2, 2]) @ PRIMITIVE) @ rotation.T
    report = diagnose_commensurability(PRIMITIVE, supercell)

    assert not report.commensurate
    assert report.rotation_deg == pytest.approx(2.0)
    # The Green-Lagrange strain is exactly zero for a rigid rotation, unlike the
    # infinitesimal one, which picks up a second-order artefact.
    green = strain_tensor(report.deformation_gradient, kind="green-lagrange")
    assert np.max(np.abs(green)) == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(report.strain)) > 0.0
    # A rotation still misplaces k: the reported directions are rotated.
    assert report.max_kpoint_shift > 0.0


def test_rotation_angle_of_the_identity_is_zero() -> None:
    assert rotation_angle(np.eye(3)) == pytest.approx(0.0)


def test_rounding_is_the_best_integer_approximation() -> None:
    supercell = np.array([[7.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 10.0]])
    transform, residual = nearest_integer_transformation(PRIMITIVE, supercell)
    raw = supercell @ np.linalg.inv(PRIMITIVE)
    assert residual <= 0.5
    for candidate in (transform - 1, transform + 1, np.diag([1, 3, 2])):
        assert residual <= np.max(np.abs(raw - candidate)) + 1e-12
    # 7 / 3 = 2.333..., a residual large enough to call the transform a guess.
    assert diagnose_commensurability(PRIMITIVE, supercell).ambiguous


def test_strain_tensor_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown strain kind"):
        strain_tensor(np.eye(3), kind="cauchy")


def test_a_singular_primitive_lattice_is_rejected() -> None:
    singular = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="singular"):
        nearest_integer_transformation(singular, np.eye(3))


def test_strain_cli_reports_the_mismatch(tmp_path) -> None:
    primitive = tmp_path / "POSCAR_pc"
    supercell = tmp_path / "POSCAR_sc"
    primitive.write_text(_poscar("primitive", PRIMITIVE))
    supercell.write_text(_poscar("supercell", np.diag([6.06, 6.0, 10.0])))
    out = tmp_path / "strain.json"

    result = runner.invoke(
        app,
        [
            "strain",
            "--primitive",
            str(primitive),
            "--supercell",
            str(supercell),
            "--json",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "NOT commensurate" in result.output
    payload = json.loads(out.read_text())
    assert payload["commensurate"] is False
    assert payload["transform"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert payload["max_abs_strain"] == pytest.approx(0.01)
    assert payload["relative_kpoint_shift"] > 0.0
