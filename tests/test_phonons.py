"""Phonon unfolding from primitive force constants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.core.phonons import (
    ForceConstantModel,
    eigenvalues_to_frequencies,
    unfold_phonon_fiber,
    unfold_phonon_path,
)
from unfoldlab.core.transformations import TransformationMatrix
from unfoldlab.io.force_constants import read_force_constant_model

runner = CliRunner()

MATRICES = [
    np.diag([3.0, 1.0, 1.0]),
    np.diag([2.0, 2.0, 1.0]),
    np.array([[2.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
]


def _spring_chain(spring: float = 1.0, mass: float = 2.0) -> ForceConstantModel:
    """A monatomic chain of springs along x, coupled in that direction only."""

    onsite = np.zeros((3, 3))
    onsite[0, 0] = 2.0 * spring
    hop = np.zeros((3, 3))
    hop[0, 0] = -spring
    return ForceConstantModel(
        np.array([mass]),
        {(0, 0, 0): onsite, (1, 0, 0): hop, (-1, 0, 0): hop},
    )


def _chain_frequency(kx: float, spring: float = 1.0, mass: float = 2.0) -> float:
    return 2.0 * np.sqrt(spring / mass) * abs(np.sin(np.pi * kx))


def test_signed_square_root_keeps_unstable_modes_visible() -> None:
    assert eigenvalues_to_frequencies([4.0, 0.0, -9.0]) == pytest.approx([2.0, 0.0, -3.0])


def test_chain_reproduces_the_analytic_dispersion() -> None:
    model = _spring_chain()
    points = np.array([[t, 0.0, 0.0] for t in [0.0, 0.1, 0.25, 0.5]])

    frequencies = model.frequencies(points)
    assert frequencies.shape == (4, 3)
    for index, point in enumerate(points):
        # Two transverse branches are flat at zero, the longitudinal one disperses.
        assert frequencies[index, 0] == pytest.approx(0.0, abs=1e-8)
        assert frequencies[index, 1] == pytest.approx(0.0, abs=1e-8)
        assert frequencies[index, 2] == pytest.approx(_chain_frequency(point[0]))


def test_heavier_atoms_soften_the_branch() -> None:
    light = _spring_chain(mass=1.0).frequencies([[0.5, 0.0, 0.0]])[0, 2]
    heavy = _spring_chain(mass=4.0).frequencies([[0.5, 0.0, 0.0]])[0, 2]
    assert heavy == pytest.approx(light / 2.0)


def test_acoustic_sum_rule_is_measured_and_repairable() -> None:
    model = _spring_chain()
    assert model.acoustic_sum_rule_deviation() == pytest.approx(0.0)

    broken_onsite = np.zeros((3, 3))
    broken_onsite[0, 0] = 2.3  # should be 2.0
    hop = np.zeros((3, 3))
    hop[0, 0] = -1.0
    broken = ForceConstantModel(
        np.array([2.0]), {(0, 0, 0): broken_onsite, (1, 0, 0): hop, (-1, 0, 0): hop}
    )
    assert broken.acoustic_sum_rule_deviation() == pytest.approx(0.3)
    # The acoustic branch misses zero at Gamma exactly because of that.
    assert broken.frequencies([[0.0, 0.0, 0.0]])[0, 2] > 1e-3

    repaired = broken.enforce_acoustic_sum_rule()
    assert repaired.acoustic_sum_rule_deviation() == pytest.approx(0.0, abs=1e-12)
    assert repaired.frequencies([[0.0, 0.0, 0.0]])[0, 2] == pytest.approx(0.0, abs=1e-8)
    # Only the on-site block moved; the inter-cell force constants are untouched.
    assert repaired.force_constants[(1, 0, 0)] == pytest.approx(hop)


def test_symmetrized_fills_in_partners_at_full_strength_and_is_idempotent() -> None:
    onsite = np.zeros((3, 3))
    onsite[0, 0] = 2.0
    hop = np.zeros((3, 3))
    hop[0, 0] = -1.0
    one_sided = ForceConstantModel(np.array([2.0]), {(0, 0, 0): onsite, (1, 0, 0): hop})

    fixed = one_sided.symmetrized()
    assert fixed.force_constants[(-1, 0, 0)] == pytest.approx(hop)
    assert fixed.force_constants[(1, 0, 0)] == pytest.approx(hop)
    assert fixed.frequencies([[0.5, 0.0, 0.0]])[0, 2] == pytest.approx(_chain_frequency(0.5))
    again = fixed.symmetrized()
    for cell, block in fixed.force_constants.items():
        assert again.force_constants[cell] == pytest.approx(block)


def test_dynamical_matrix_is_hermitian() -> None:
    model = _spring_chain()
    matrix = model.dynamical_matrix([0.23, 0.0, 0.0])
    assert np.allclose(matrix, matrix.conj().T)


@pytest.mark.parametrize("matrix", MATRICES)
def test_perfect_supercell_unfolds_to_the_primitive_branches(matrix: np.ndarray) -> None:
    transform = TransformationMatrix(matrix)
    model = _spring_chain()
    path = np.array([[t, 0.0, 0.0] for t in np.linspace(0.05, 0.45, 5)])

    frequencies, weights = unfold_phonon_path(model, transform, path)
    assert frequencies.shape == (5, transform.multiplicity * model.n_modes)
    # The sum rule across branches survives the supercell.
    assert weights.sum(axis=1) == pytest.approx(np.full(5, float(model.n_modes)))
    for index, point in enumerate(path):
        expected = _chain_frequency(point[0])
        heavy = np.flatnonzero(weights[index] > 1e-8)
        # The dispersing branch is the largest frequency carrying weight.
        assert np.max(frequencies[index][heavy]) == pytest.approx(expected, abs=1e-6)


def test_fiber_unfolding_matches_the_path_unfolding() -> None:
    transform = TransformationMatrix(np.diag([4.0, 1.0, 1.0]))
    model = _spring_chain()
    kpoint = np.array([0.07, 0.0, 0.0])

    frequencies, weights = unfold_phonon_path(model, transform, kpoint[np.newaxis, :])
    fiber, fiber_frequencies, fiber_weights = unfold_phonon_fiber(model, transform, [kpoint])
    offsets = fiber - kpoint
    row = int(np.argmin(np.linalg.norm(offsets - np.round(offsets), axis=1)))
    assert frequencies[0] == pytest.approx(fiber_frequencies[row])
    assert weights[0] == pytest.approx(fiber_weights[row])


def test_a_stiffened_cell_spreads_the_weight() -> None:
    transform = TransformationMatrix(np.diag([4.0, 1.0, 1.0]))
    model = _spring_chain()
    path = np.array([[t, 0.0, 0.0] for t in np.linspace(0.05, 0.45, 5)])

    _, clean = unfold_phonon_path(model, transform, path)
    _, defect = unfold_phonon_path(model, transform, path, perturbations=[(1, 0, 0.75)])

    assert np.max(clean) == pytest.approx(1.0)
    assert np.max(defect) < 1.0
    assert defect.sum(axis=1) == pytest.approx(np.full(5, float(model.n_modes)))


def test_an_unstable_supercell_shows_a_negative_frequency() -> None:
    """Softening one cell below stability must be visible, not silently dropped."""

    transform = TransformationMatrix(np.diag([3.0, 1.0, 1.0]))
    model = _spring_chain()
    frequencies, _ = unfold_phonon_path(
        model, transform, [[0.0, 0.0, 0.0]], perturbations=[(0, 0, -5.0)]
    )
    assert np.min(frequencies) < -0.1


def test_invalid_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        ForceConstantModel(np.array([0.0]), {})
    with pytest.raises(ValueError, match="at least one atom"):
        ForceConstantModel(np.array([]), {})
    with pytest.raises(ValueError, match="shape"):
        ForceConstantModel(np.array([1.0]), {(0, 0, 0): np.zeros((2, 2))})
    with pytest.raises(ValueError, match="integer 3-vectors"):
        ForceConstantModel(np.array([1.0]), {(0, 0): np.zeros((3, 3))})  # type: ignore[dict-item]


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "phonons.json"
    path.write_text(json.dumps(payload))
    return path


CHAIN_JSON = {
    "masses": [2.0],
    "force_constants": [
        {"cell": [0, 0, 0], "matrix": (np.diag([2.0, 0.0, 0.0])).tolist()},
        {"cell": [1, 0, 0], "matrix": (np.diag([-1.0, 0.0, 0.0])).tolist()},
    ],
}


def test_reader_symmetrizes_and_reads_perturbations(tmp_path: Path) -> None:
    payload = {**CHAIN_JSON, "perturbations": [{"cell": 1, "mode": 0, "value": 0.5}]}
    model, perturbations = read_force_constant_model(_write(tmp_path, payload))

    assert perturbations == [(1, 0, 0.5)]
    assert set(model.force_constants) == {(0, 0, 0), (1, 0, 0), (-1, 0, 0)}
    assert model.frequencies([[0.5, 0.0, 0.0]])[0, 2] == pytest.approx(_chain_frequency(0.5))


def test_reader_can_enforce_the_acoustic_sum_rule(tmp_path: Path) -> None:
    broken = {
        "masses": [2.0],
        "force_constants": [
            {"cell": [0, 0, 0], "matrix": (np.diag([2.4, 0.0, 0.0])).tolist()},
            {"cell": [1, 0, 0], "matrix": (np.diag([-1.0, 0.0, 0.0])).tolist()},
        ],
    }
    raw, _ = read_force_constant_model(_write(tmp_path, broken))
    assert raw.acoustic_sum_rule_deviation() == pytest.approx(0.4)

    fixed, _ = read_force_constant_model(
        _write(tmp_path, {**broken, "enforce_acoustic_sum_rule": True})
    )
    assert fixed.acoustic_sum_rule_deviation() == pytest.approx(0.0, abs=1e-12)


def test_reader_rejects_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="masses"):
        read_force_constant_model(_write(tmp_path, {"force_constants": []}))
    with pytest.raises(ValueError, match="JSON object"):
        read_force_constant_model(_write(tmp_path, [1, 2, 3]))  # type: ignore[arg-type]
    too_small = {
        "masses": [1.0],
        "force_constants": [{"cell": [0, 0, 0], "matrix": [[1.0]]}],
    }
    with pytest.raises(ValueError, match="3x3"):
        read_force_constant_model(_write(tmp_path, too_small))


def test_phonon_cli_unfolds_the_shipped_examples(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "examples"
    output = tmp_path / "phonons_ebs.json"
    result = runner.invoke(
        app,
        [
            "phonon",
            "bands",
            "--model",
            str(root / "toy_1d_phonon_chain.json"),
            "--matrix",
            "3 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0",
            "--points",
            "6",
            "--json",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.split("Wrote")[0])
    assert payload["n_modes"] == 3
    assert payload["acoustic_sum_rule_deviation"] < 1e-12
    assert payload["max_band_deviation"] < 1e-10
    stored = json.loads(output.read_text())
    frequencies = np.array(stored["energies"])
    weights = np.array(stored["weights"])
    for index, point in enumerate(stored["kpoints"]):
        heavy = np.flatnonzero(weights[index] > 1e-8)
        assert np.max(frequencies[index][heavy]) == pytest.approx(
            _chain_frequency(point[0]), abs=1e-6
        )

    defect = runner.invoke(
        app,
        [
            "phonon",
            "bands",
            "--model",
            str(root / "toy_1d_phonon_defect.json"),
            "--matrix",
            "3 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0",
            "--points",
            "6",
        ],
    )
    assert defect.exit_code == 0, defect.output
    assert json.loads(defect.output)["max_band_deviation"] < 1e-10


def test_phonon_cli_warns_about_a_broken_acoustic_sum_rule(tmp_path: Path) -> None:
    model = _write(
        tmp_path,
        {
            "masses": [2.0],
            "force_constants": [
                {"cell": [0, 0, 0], "matrix": (np.diag([2.4, 0.0, 0.0])).tolist()},
                {"cell": [1, 0, 0], "matrix": (np.diag([-1.0, 0.0, 0.0])).tolist()},
            ],
        },
    )
    result = runner.invoke(
        app,
        [
            "phonon",
            "bands",
            "--model",
            str(model),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0",
            "--points",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "acoustic" in result.output
    assert "--mode-term" not in result.output


def test_phonon_cli_accepts_repeatable_mode_terms(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "examples"
    common = [
        "phonon",
        "bands",
        "--model",
        str(root / "toy_1d_phonon_chain.json"),
        "--matrix",
        "3 0 0 0 1 0 0 0 1",
        "--path",
        "0,0,0:0.5,0,0",
        "--points",
        "3",
    ]
    result = runner.invoke(app, [*common, "--mode-term", "1,0,0.5", "--mode-term", "2,0,0.25"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["max_band_deviation"] < 1e-10

    bad = runner.invoke(app, [*common, "--mode-term", "1,0"])
    assert bad.exit_code != 0
    assert "cell,mode,value" in bad.output
