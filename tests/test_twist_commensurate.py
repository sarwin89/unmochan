"""Commensurate twisted stacks: angles, moire cells, transfer and layer weights.

The reference numbers are the classic twisted-bilayer-graphene table (7 cells at
21.79 degrees, 19 at 13.17, 37 at 9.43, 61 at 7.34), which is what the closed
form in ``RequestProject/Unfolding/Twist.lean`` has to reproduce if the Lean
statement and the Python implementation agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.core.numerics import integer_det3
from unfoldlab.twist.commensurate import (
    commensurate_stack_transforms,
    diagnose_stack,
    hexagonal_commensurate_twists,
    layer_projected_weights,
    moire_cell_size,
    transfer_kpoint,
)

runner = CliRunner()

LATTICE_CONSTANT = 2.46


def hexagonal_lattice(angle_deg: float = 0.0, *, height: float = 20.0) -> np.ndarray:
    """A 120-degree hexagonal cell, rotated in plane by ``angle_deg``."""

    a = LATTICE_CONSTANT
    lattice = np.array([[a, 0.0, 0.0], [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, height]])
    theta = np.radians(angle_deg)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return lattice @ rotation.T


def test_moire_cell_size_matches_the_norm_form() -> None:
    assert moire_cell_size(2, 3) == 7
    assert moire_cell_size(2, 5) == 19
    assert moire_cell_size(3, 7) == 37
    # symmetric in its arguments, as the norm form of the Eisenstein integers is
    assert moire_cell_size(4, 9) == moire_cell_size(9, 4) == 61


def test_classic_twisted_bilayer_table_is_reproduced() -> None:
    twists = hexagonal_commensurate_twists(12)
    table = {twist.cells: twist.equivalent_angle_deg for twist in twists}
    for cells, angle in ((7, 21.7868), (13, 27.7958), (19, 13.1736), (37, 9.4300), (61, 7.3411)):
        assert table[cells] == pytest.approx(angle, abs=1e-3)


def test_enumeration_drops_aligned_and_oversized_duplicates() -> None:
    twists = hexagonal_commensurate_twists(6)
    # (1, 2) is a 60-degree rotation, i.e. no twist at all
    assert all(not twist.is_aligned for twist in twists)
    assert (1, 2) not in {(twist.m, twist.n) for twist in twists}
    # (1, 5) is the 21.79-degree stack in 21 cells instead of 7
    angles = [twist.equivalent_angle_deg for twist in twists]
    assert len(angles) == len(set(round(value, 9) for value in angles))
    smallest = min(twists, key=lambda twist: twist.cells)
    assert smallest.cells == 7
    assert hexagonal_commensurate_twists(6, include_aligned=True) != twists


def test_enumeration_rejects_a_nonsensical_range() -> None:
    with pytest.raises(ValueError):
        hexagonal_commensurate_twists(0)


def test_cos_angle_is_the_closed_form() -> None:
    (twist,) = [t for t in hexagonal_commensurate_twists(3) if t.cells == 7]
    m, n = twist.m, twist.n
    expected = (4 * m * n - m * m - n * n) / (2 * moire_cell_size(m, n))
    assert twist.cos_angle == pytest.approx(expected)
    assert twist.to_dict()["cells"] == 7


def test_stack_transforms_have_equal_determinants() -> None:
    for m, n in ((2, 3), (2, 5), (3, 7), (4, 9)):
        transform_a, transform_b = commensurate_stack_transforms(m, n)
        assert integer_det3(transform_a) == moire_cell_size(m, n)
        assert integer_det3(transform_b) == moire_cell_size(m, n)


def test_the_two_transforms_describe_the_same_moire_cell() -> None:
    """``T_a A_a = T_b A_b`` for the rotated layer, with the predicted angle."""

    m, n = 2, 3
    transform_a, transform_b = commensurate_stack_transforms(m, n)
    layer_a = hexagonal_lattice()
    moire = transform_a.astype(float) @ layer_a
    layer_b = np.linalg.solve(transform_b.astype(float), moire)

    lengths_a = np.linalg.norm(layer_a[:2], axis=1)
    lengths_b = np.linalg.norm(layer_b[:2], axis=1)
    assert lengths_b == pytest.approx(lengths_a)

    cosine = float(np.dot(layer_a[0, :2], layer_b[0, :2]) / (lengths_a[0] * lengths_b[0]))
    assert cosine == pytest.approx(11.0 / 14.0)


def test_moire_cell_size_zero_is_rejected() -> None:
    with pytest.raises(ValueError):
        commensurate_stack_transforms(0, 0)


def test_transfer_kpoint_round_trips_and_preserves_cartesian_k() -> None:
    m, n = 2, 3
    transform_a, transform_b = commensurate_stack_transforms(m, n)
    layer_a = hexagonal_lattice()
    moire = transform_a.astype(float) @ layer_a
    layer_b = np.linalg.solve(transform_b.astype(float), moire)

    kpoints = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.0], [0.3, -0.1, 0.25]])
    transferred = transfer_kpoint(kpoints, transform_a, transform_b)
    assert transfer_kpoint(transferred, transform_b, transform_a) == pytest.approx(kpoints)

    reciprocal_a = 2.0 * np.pi * np.linalg.inv(layer_a).T
    reciprocal_b = 2.0 * np.pi * np.linalg.inv(layer_b).T
    assert kpoints @ reciprocal_a == pytest.approx(transferred @ reciprocal_b)


def test_transfer_kpoint_validates_its_arguments() -> None:
    transform_a, transform_b = commensurate_stack_transforms(2, 3)
    with pytest.raises(ValueError):
        transfer_kpoint(np.zeros((2, 2)), transform_a, transform_b)
    with pytest.raises(ValueError):
        transfer_kpoint([0.0, 0.0, 0.0], transform_a, np.zeros((3, 3), dtype=int))


def test_diagnose_stack_reports_both_layers() -> None:
    transform_a, transform_b = commensurate_stack_transforms(2, 3)
    layer_a = hexagonal_lattice()
    moire = transform_a.astype(float) @ layer_a
    layer_b = np.linalg.solve(transform_b.astype(float), moire)

    report = diagnose_stack(moire, {"top": layer_a, "bottom": layer_b})
    assert report.commensurate
    assert report.multiplicities == {"top": 7, "bottom": 7}
    assert report.twist_angles["top"] == pytest.approx(0.0)
    assert abs(report.twist_angles["bottom"]) == pytest.approx(
        np.degrees(np.arccos(11.0 / 14.0)), abs=1e-6
    )
    assert "EVERY layer" in report.summary()
    assert report.to_dict()["reference"] == "top"
    assert report.layer("top").transform.shape == (3, 3)

    kpoint = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]])
    assert report.transfer(kpoint, "top", "bottom") == pytest.approx(
        transfer_kpoint(kpoint, transform_a, transform_b)
    )


def test_diagnose_stack_flags_an_incommensurate_layer() -> None:
    transform_a, _ = commensurate_stack_transforms(2, 3)
    layer_a = hexagonal_lattice()
    moire = transform_a.astype(float) @ layer_a
    strained = hexagonal_lattice() * 1.037
    report = diagnose_stack(moire, {"top": layer_a, "bad": strained})
    assert not report.commensurate
    assert report.layer("top").commensurate
    assert not report.layer("bad").commensurate
    assert "NOT commensurate" in report.summary()


def test_diagnose_stack_validates_its_arguments() -> None:
    layer = hexagonal_lattice()
    with pytest.raises(ValueError):
        diagnose_stack(layer, {})
    with pytest.raises(KeyError):
        diagnose_stack(layer, {"a": layer}, reference="b")


def test_layer_projected_weights_partition_the_total() -> None:
    """A partition into layers adds up to the total tight-binding weight."""

    rng = np.random.default_rng(7)
    cells = np.array([[index, 0, 0] for index in range(4)], dtype=np.int64)
    n_states, n_orbitals = 3, 6
    coefficients = rng.normal(size=(n_states, n_orbitals, 4)) + 1j * rng.normal(
        size=(n_states, n_orbitals, 4)
    )
    kpoints = np.array([[index / 4.0, 0.0, 0.0] for index in range(4)])
    assignment = [0, 0, 0, 1, 1, 1]

    weights, labels = layer_projected_weights(cells, coefficients, kpoints, assignment)
    assert labels == [0, 1]
    assert weights.shape == (2, 4, n_states)
    assert np.all(weights >= -1e-12)

    everything, single = layer_projected_weights(cells, coefficients, kpoints, [0] * n_orbitals)
    assert single == [0]
    assert weights.sum(axis=0) == pytest.approx(everything[0])

    # over the complete fiber the total weight of every state is one
    assert everything[0].sum(axis=0) == pytest.approx(np.ones(n_states))


def test_layer_projected_weights_reject_a_bad_assignment() -> None:
    cells = np.array([[0, 0, 0]], dtype=np.int64)
    coefficients = np.zeros((1, 2, 1), dtype=complex)
    kpoints = np.zeros((1, 3))
    with pytest.raises(ValueError):
        layer_projected_weights(cells, coefficients, kpoints, [])
    with pytest.raises(ValueError):
        layer_projected_weights(cells, coefficients, kpoints, [0, 1, 1])
    with pytest.raises(ValueError):
        layer_projected_weights(cells, np.zeros((1, 2)), kpoints, [0, 1])


def test_transferred_kpoint_folds_onto_the_same_supercell_point() -> None:
    """``UnfoldLab.foldsTo_transferK``: the transfer stays inside one fiber."""

    transform_a, transform_b = commensurate_stack_transforms(2, 3)
    kpoints = np.array([[0.2, -0.35, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.5]])
    transferred = transfer_kpoint(kpoints, transform_a, transform_b)
    folded_a = kpoints @ transform_a.astype(float).T
    folded_b = transferred @ transform_b.astype(float).T
    assert folded_a == pytest.approx(folded_b)


def _poscar(name: str, lattice: np.ndarray) -> str:
    rows = [" ".join(f"{value:.12f}" for value in row) for row in lattice]
    return "\n".join([name, "1.0", *rows, "X", "1", "Direct", "0.0 0.0 0.0", ""])


def test_cli_lists_commensurate_angles(tmp_path: Path) -> None:
    json_out = tmp_path / "twists.json"
    result = runner.invoke(app, ["twist", "--max-index", "6", "--json", str(json_out)])
    assert result.exit_code == 0, result.output
    entries = json.loads(json_out.read_text())
    assert {entry["cells"] for entry in entries} >= {7, 13, 19}


def test_cli_diagnoses_a_stack(tmp_path: Path) -> None:
    transform_a, transform_b = commensurate_stack_transforms(2, 3)
    layer_a = hexagonal_lattice()
    moire = transform_a.astype(float) @ layer_a
    layer_b = np.linalg.solve(transform_b.astype(float), moire)

    moire_file = tmp_path / "POSCAR_moire"
    top_file = tmp_path / "POSCAR_top"
    bottom_file = tmp_path / "POSCAR_bottom"
    moire_file.write_text(_poscar("moire", moire))
    top_file.write_text(_poscar("top", layer_a))
    bottom_file.write_text(_poscar("bottom", layer_b))

    json_out = tmp_path / "stack.json"
    result = runner.invoke(
        app,
        [
            "twist",
            "--moire",
            str(moire_file),
            "--layer",
            f"top={top_file}",
            "--layer",
            f"bottom={bottom_file}",
            "--json",
            str(json_out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(json_out.read_text())
    assert payload["commensurate"]
    assert [layer["multiplicity"] for layer in payload["layers"]] == [7, 7]


def test_cli_rejects_layers_without_a_moire_cell() -> None:
    result = runner.invoke(app, ["twist", "--layer", "top=POSCAR"])
    assert result.exit_code != 0
