"""Brillouin-zone meshes: which primitive k-points a supercell run resolves.

Each test names the Lean statement it exercises; the theory is in
``RequestProject/Unfolding/Mesh.lean``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.core.kpoints import fiber_kpoints
from unfoldlab.core.mesh import (
    MeshSpec,
    diagnose_mesh,
    mesh_distance,
    mesh_kpoint_mappings,
    nearest_mesh_point,
    unfolded_mesh,
    unfolded_mesh_points,
)
from unfoldlab.core.numerics import wrap_fractional
from unfoldlab.core.transformations import TransformationMatrix

runner = CliRunner()

TRANSFORM = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 1]], dtype=np.int64)


def test_gamma_centred_mesh_points() -> None:
    mesh = MeshSpec.from_divisions([2, 2, 2])
    points = mesh.points()

    assert mesh.count == 8
    assert points.shape == (8, 3)
    assert set(np.unique(points)) == {0.0, 0.5}
    assert mesh.contains(points).all()
    assert mesh.is_gamma_centred
    assert mesh.is_product_mesh


def test_shifted_mesh_misses_gamma_before_and_after_unfolding() -> None:
    """``UnfoldLab.zero_mem_Mesh_iff``: unfolding does not change the shift."""

    mesh = MeshSpec.from_divisions([2, 2, 2], shift=[0.5, 0.5, 0.5])

    assert not mesh.is_gamma_centred
    assert not mesh.contains(np.zeros(3))[0]
    assert not unfolded_mesh(mesh, TRANSFORM).contains(np.zeros(3))[0]
    assert np.allclose(np.unique(mesh.points()), [0.25, 0.75])


def test_unfolded_mesh_is_the_mesh_of_the_product() -> None:
    """``UnfoldLab.unfoldedMesh_eq`` and ``card_unfoldedMesh_repr``."""

    mesh = MeshSpec.from_divisions([2, 2, 2])
    unfolded = unfolded_mesh(mesh, TRANSFORM)

    assert np.array_equal(unfolded.generator, mesh.generator @ TRANSFORM)
    assert unfolded.count == mesh.count * abs(int(round(np.linalg.det(TRANSFORM))))
    assert np.allclose(unfolded.shift, mesh.shift)


def test_unfolded_points_are_the_union_of_the_fibers() -> None:
    """``UnfoldLab.IsFiberRepr.mesh_fiber_complete``, against `fiber_kpoints`."""

    mesh = MeshSpec.from_divisions([2, 2, 2])
    transform = TransformationMatrix.from_values(TRANSFORM)
    expected = np.concatenate([fiber_kpoints(point, transform) for point in mesh.points()], axis=0)
    got = unfolded_mesh_points(mesh, transform)

    assert got.shape == expected.shape
    order = np.lexsort((expected[:, 2], expected[:, 1], expected[:, 0]))
    assert np.allclose(got, expected[order], atol=1e-12)


def test_every_unfolded_point_folds_into_the_supercell_mesh() -> None:
    mesh = MeshSpec.from_divisions([3, 2, 1], shift=[0.5, 0.0, 0.0])
    points = unfolded_mesh_points(mesh, TRANSFORM)
    folded = wrap_fractional(points @ TRANSFORM.T)

    assert mesh.contains(folded).all()
    assert mesh.contains(points @ TRANSFORM.T).all()  # wrapping changes nothing

    keys = {tuple(np.round(row, 9)) for row in folded}
    assert len(keys) == mesh.count
    for key in keys:
        matches = np.all(np.abs(folded - np.array(key)) < 1e-9, axis=1).sum()
        assert matches == 2  # |det T|


def test_zone_boundary_needs_an_even_division() -> None:
    """``UnfoldLab.zoneBoundary_mem_iff``."""

    even = MeshSpec.from_divisions([4, 4, 4])
    odd = MeshSpec.from_divisions([3, 3, 3])

    assert even.contains([0.5, 0.0, 0.0])[0]
    assert not odd.contains([0.5, 0.0, 0.0])[0]

    mixed = MeshSpec.from_divisions([2, 3, 4])
    report = diagnose_mesh(mixed, np.eye(3, dtype=np.int64))
    assert report.zone_boundary_resolved == (True, False, True)


def test_resolution_is_half_a_mesh_spacing() -> None:
    """``UnfoldLab.mesh_resolution_bound``."""

    mesh = MeshSpec.from_divisions([4, 4, 4])
    lattice = 2.0 * np.pi * np.eye(3)

    assert mesh_distance(mesh, [1 / 8, 0.0, 0.0]) == pytest.approx(1 / 8)
    assert mesh_distance(mesh, [1 / 8, 0.0, 0.0], lattice) == pytest.approx(2 * np.pi / 8)

    rng = np.random.default_rng(0)
    for point in rng.random((20, 3)):
        assert mesh_distance(mesh, point) <= np.sqrt(3) / 8 + 1e-12


def test_nearest_mesh_point_is_a_mesh_point_of_a_hexagonal_cell() -> None:
    mesh = MeshSpec.from_divisions([3, 3, 2])
    lattice = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 0.4]])
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    kpoint = np.array([0.31, 0.47, 0.13])

    nearest = nearest_mesh_point(mesh, kpoint, reciprocal)

    assert mesh.contains(nearest)[0]
    delta = (kpoint - nearest) @ reciprocal
    assert float(np.linalg.norm(delta)) == pytest.approx(
        mesh_distance(mesh, kpoint, reciprocal), abs=1e-9
    )


def test_unfolded_mesh_is_not_a_monkhorst_pack_mesh() -> None:
    """``UnfoldLab.exists_unfoldedMesh_not_product``."""

    mesh = MeshSpec.from_divisions([2, 2, 2])
    unfolded = unfolded_mesh(mesh, TRANSFORM)

    assert unfolded.axis_divisions() == (4, 4, 2)
    assert not unfolded.is_product_mesh
    # the product mesh with those divisions has more points than the fibers hold
    assert 4 * 4 * 2 > unfolded.count

    points = unfolded.points()
    assert unfolded.contains([0.25, 0.25, 0.0])[0]
    assert unfolded.contains([0.0, 0.0, 0.0])[0]
    assert not unfolded.contains([0.25, 0.0, 0.0])[0]
    assert points.shape[0] == unfolded.count


def test_mesh_kpoint_mappings_cover_every_fiber_with_uniform_weights() -> None:
    mesh = MeshSpec.from_divisions([2, 2, 1])
    mappings = mesh_kpoint_mappings(mesh, TRANSFORM)

    assert len(mappings) == 8
    assert sum(mapping.primitive.weight for mapping in mappings) == pytest.approx(1.0)
    for mapping in mappings:
        assert mesh.contains(mapping.supercell.fractional)[0]
        folded = wrap_fractional(mapping.primitive.fractional @ TRANSFORM.T)
        assert np.allclose(folded, mapping.supercell.fractional, atol=1e-12)


def test_report_round_trips_through_json() -> None:
    mesh = MeshSpec.from_divisions([2, 2, 2], shift=[0.5, 0.0, 0.0])
    report = diagnose_mesh(mesh, TRANSFORM, requested=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["n_supercell_kpoints"] == 8
    assert payload["n_primitive_kpoints"] == 16
    assert payload["multiplicity"] == 2
    assert payload["gamma_resolved"] is False
    assert payload["cartesian_distances"] is False
    assert len(payload["distances"]) == 2


def test_invalid_meshes_are_rejected() -> None:
    with pytest.raises(ValueError):
        MeshSpec(generator=np.zeros((3, 3), dtype=np.int64), shift=np.zeros(3))
    with pytest.raises(ValueError):
        MeshSpec.from_divisions([2, 0, 2])
    with pytest.raises(ValueError):
        MeshSpec(generator=np.eye(3) * 1.5, shift=np.zeros(3))


def test_mesh_cli_reports_and_writes_kpoints(tmp_path: Path) -> None:
    kpoints = tmp_path / "primitive_kpoints.tsv"
    report = tmp_path / "mesh.json"

    result = runner.invoke(
        app,
        [
            "mesh",
            "--matrix",
            "1,1,0,0,2,0,0,0,1",
            "--divisions",
            "2,2,2",
            "--point",
            "0.25,0.25,0",
            "--point",
            "0.25,0,0",
            "--kpoints",
            str(kpoints),
            "--json",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "primitive k-points: 16" in result.output.replace("\n", "")
    lines = kpoints.read_text().strip().splitlines()
    assert len(lines) == 17  # header plus |det M| |det T| k-points
    payload = json.loads(report.read_text())
    assert payload["resolved"] == [True, False]
    assert payload["n_primitive_kpoints"] == 16
    assert payload["product_mesh"] is False
