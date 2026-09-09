"""First-Brillouin-zone representatives.

The mathematics is in ``RequestProject/Unfolding/BrillouinZone.lean``; this file
checks that :mod:`unmochan.core.brillouin` implements it: that the reduction
really is the minimum over the lattice (checked against brute force), that the
Bragg-plane test agrees with it, that rounding is exact for an orthogonal cell
and wrong for a hexagonal one, and that the CLI reports all of that.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.core.brillouin import (
    diagnose_bz_reduction,
    has_orthogonal_reciprocal_basis,
    inscribed_radius,
    is_in_first_bz,
    reduce_kpoints_to_first_bz,
    reduce_to_first_bz,
    shortest_lattice_vector_length,
    zone_boundary_distance,
)
from unmochan.core.valleys import certified_minimum_image_shift, minimum_image_distance

pytestmark = [pytest.mark.unit]

runner = CliRunner()


def hexagonal_reciprocal(a: float = 2.5, c: float = 10.0) -> np.ndarray:
    """Reciprocal lattice (rows) of a hexagonal cell."""

    direct = np.array([[a, 0.0, 0.0], [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, c]])
    return 2.0 * np.pi * np.linalg.inv(direct).T


HEXAGONAL_POSCAR = "\n".join(
    [
        "hexagonal primitive cell",
        "1.0",
        "2.5 0.0 0.0",
        "-1.25 2.165063509461097 0.0",
        "0.0 0.0 10.0",
        "C",
        "1",
        "Direct",
        "0.0 0.0 0.0",
        "",
    ]
)


def brute_force_shortest(kpoint: np.ndarray, matrix: np.ndarray, radius: int = 3) -> float:
    """Shortest Cartesian length of a lattice translate, by exhaustive search."""

    span = range(-radius, radius + 1)
    return min(
        float(np.linalg.norm((kpoint + np.array(shift, dtype=float)) @ matrix))
        for shift in itertools.product(span, span, span)
    )


def test_rounding_is_wrong_for_a_hexagonal_cell() -> None:
    matrix = hexagonal_reciprocal()
    result = reduce_to_first_bz([0.5, 0.5, 0.0], matrix)

    assert not result.rounding_suffices
    assert result.shift.tolist() in ([-1, 0, 0], [0, -1, 0])
    # The rounded representative is sqrt(3) times too long.
    assert result.rounded_length / result.length == pytest.approx(np.sqrt(3.0))
    assert result.length == pytest.approx(brute_force_shortest(np.array([0.5, 0.5, 0.0]), matrix))


def test_reduction_matches_brute_force_on_random_skewed_cells() -> None:
    rng = np.random.default_rng(20240918)
    for _ in range(20):
        matrix = np.eye(3) + 0.4 * rng.standard_normal((3, 3))
        if abs(np.linalg.det(matrix)) < 0.2:
            continue
        kpoint = rng.uniform(-1.5, 1.5, size=3)
        result = reduce_to_first_bz(kpoint, matrix)
        assert result.length == pytest.approx(brute_force_shortest(kpoint, matrix), abs=1e-10)
        # The reduction is a lattice translation of the input.
        assert np.allclose(result.fractional, kpoint + result.shift)
        assert np.allclose(result.cartesian, result.fractional @ matrix)


def test_reduction_is_idempotent_and_agrees_with_the_bragg_test() -> None:
    matrix = hexagonal_reciprocal()
    rng = np.random.default_rng(7)
    for _ in range(20):
        kpoint = rng.uniform(-2.0, 2.0, size=3)
        reduced = reduce_to_first_bz(kpoint, matrix)
        again = reduce_to_first_bz(reduced.fractional, matrix)
        assert np.allclose(again.shift, 0)
        assert is_in_first_bz(reduced.fractional, matrix)
        assert zone_boundary_distance(reduced.fractional, matrix) >= -1e-9


def test_zone_boundary_distance_is_zero_on_the_boundary() -> None:
    matrix = np.diag([1.0, 2.0, 3.0])
    assert zone_boundary_distance([0.5, 0.0, 0.0], matrix) == pytest.approx(0.0, abs=1e-12)
    assert zone_boundary_distance([0.0, 0.0, 0.0], matrix) == pytest.approx(0.5)
    assert not is_in_first_bz([0.6, 0.0, 0.0], matrix)
    assert is_in_first_bz([0.5, 0.0, 0.0], matrix)


def test_inscribed_radius_is_half_the_shortest_vector() -> None:
    matrix = np.diag([1.0, 2.0, 3.0])
    assert shortest_lattice_vector_length(matrix) == pytest.approx(1.0)
    assert inscribed_radius(matrix) == pytest.approx(0.5)

    hexagonal = hexagonal_reciprocal()
    radius = inscribed_radius(hexagonal)
    rng = np.random.default_rng(3)
    for _ in range(20):
        direction = rng.standard_normal(3)
        cartesian = 0.99 * radius * direction / np.linalg.norm(direction)
        fractional = cartesian @ np.linalg.inv(hexagonal)
        assert is_in_first_bz(fractional, hexagonal)


def test_rounding_is_exact_for_an_orthogonal_basis() -> None:
    matrix = np.diag([1.0, 2.0, 3.0])
    assert has_orthogonal_reciprocal_basis(matrix)
    assert not has_orthogonal_reciprocal_basis(hexagonal_reciprocal())

    rng = np.random.default_rng(11)
    points = rng.uniform(-2.0, 2.0, size=(25, 3))
    reduced = reduce_kpoints_to_first_bz(points, matrix)
    assert np.allclose(reduced, points - np.rint(points))

    report = diagnose_bz_reduction(points, matrix)
    assert report.rounding_is_exact
    assert report.n_moved == 0
    assert report.max_length_ratio == pytest.approx(1.0)
    assert "orthogonal" in report.summary()


def test_diagnose_reports_the_hexagonal_discrepancy() -> None:
    matrix = hexagonal_reciprocal()
    report = diagnose_bz_reduction([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], matrix)

    assert report.n_points == 3
    assert report.n_moved == 1
    assert report.max_length_ratio == pytest.approx(np.sqrt(3.0))
    assert not report.rounding_is_exact
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["orthogonal_basis"] is False
    assert len(payload["reduced"]) == 3


def test_default_lattice_is_the_fractional_norm() -> None:
    result = reduce_to_first_bz([1.4, -0.6, 0.2])
    assert np.allclose(result.fractional, [0.4, 0.4, 0.2])
    assert result.length == pytest.approx(np.linalg.norm([0.4, 0.4, 0.2]))


def test_shift_and_distance_are_consistent_with_the_valley_search() -> None:
    matrix = hexagonal_reciprocal()
    delta = np.array([0.5, 0.5, 0.1])
    shift, distance = certified_minimum_image_shift(delta, matrix)
    assert distance == pytest.approx(minimum_image_distance(delta, matrix))
    assert np.linalg.norm((delta + shift) @ matrix) == pytest.approx(distance)


def test_bad_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        reduce_kpoints_to_first_bz(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        diagnose_bz_reduction(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        certified_minimum_image_shift([0.0, 0.0, 0.0], np.zeros((3, 3)))


def test_cli_reports_the_reduction(tmp_path: Path) -> None:
    structure = tmp_path / "POSCAR"
    structure.write_text(HEXAGONAL_POSCAR)
    out = tmp_path / "reduced.tsv"
    report = tmp_path / "bz.json"

    result = runner.invoke(
        app,
        [
            "bz",
            "--structure",
            str(structure),
            "--point",
            "0.5,0.5,0.0",
            "--point",
            "0.0,0.0,0.0",
            "--out",
            str(out),
            "--json",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report.read_text())
    assert payload["n_moved"] == 1
    assert payload["rounding_is_exact"] is False
    lines = out.read_text().strip().splitlines()
    assert lines[0].split() == ["k1", "k2", "k3"]
    assert len(lines) == 3


def test_cli_reads_a_kpoint_file_and_needs_input(tmp_path: Path) -> None:
    listing = tmp_path / "kpoints.tsv"
    listing.write_text("k1\tk2\tk3\tweight\n0.5\t0.5\t0.0\t1.0\n0.25\t0.0\t0.0\t1.0\n")

    result = runner.invoke(app, ["bz", "--kpoints", str(listing)])
    assert result.exit_code == 0, result.output

    empty = runner.invoke(app, ["bz"])
    assert empty.exit_code != 0


def test_mesh_command_can_report_zone_representatives(tmp_path: Path) -> None:
    structure = tmp_path / "POSCAR"
    structure.write_text(HEXAGONAL_POSCAR)
    listing = tmp_path / "kpoints.tsv"

    result = runner.invoke(
        app,
        [
            "mesh",
            "--matrix",
            "2,0,0,0,2,0,0,0,1",
            "--divisions",
            "2,2,1",
            "--structure",
            str(structure),
            "--kpoints",
            str(listing),
            "--first-bz",
        ],
    )

    assert result.exit_code == 0, result.output
    lines = listing.read_text().strip().splitlines()
    assert lines[0].split() == ["k1", "k2", "k3", "weight", "K1", "K2", "K3", "b1", "b2", "b3"]

    reciprocal = hexagonal_reciprocal()
    for line in lines[1:]:
        values = [float(token) for token in line.split()]
        primitive, reduced = np.array(values[:3]), np.array(values[7:10])
        assert is_in_first_bz(reduced, reciprocal)
        # A zone representative differs from the listed k-point by a lattice vector.
        assert np.allclose(reduced - primitive, np.rint(reduced - primitive))
