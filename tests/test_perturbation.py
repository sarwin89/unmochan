"""Shadow bands from a weak superlattice perturbation.

The closed forms in ``unfoldlab.core.perturbation`` are checked against a dense
diagonalization of the same 2x2 Hamiltonian, against the bounds proved in
``RequestProject/Unfolding/Perturbation.lean``, and end to end: a model
structure built by :func:`shadow_band_structure` is handed back to
:func:`diagnose_shadow_bands`, which must recover the coupling it was built
with.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from unfoldlab.core.perturbation import (
    coupling_from_peaks,
    diagnose_shadow_bands,
    shadow_band_structure,
    shadow_weight_bounds,
    two_level_hamiltonian,
    two_level_solution,
)
from unfoldlab.core.spectral import EffectiveBandStructure


def _reference_solution(mean: float, detuning: float, coupling: complex):
    """Energies and ``|k>``-weights from a dense Hermitian diagonalization."""

    values, vectors = np.linalg.eigh(two_level_hamiltonian(mean, detuning, coupling))
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    weights = np.abs(vectors[0, :]) ** 2
    return values, weights


@pytest.mark.parametrize(
    ("mean", "detuning", "coupling"),
    [
        (0.0, 1.0, 0.1),
        (-2.5, 0.25, 0.4),
        (1.0, 0.0, 0.3),
        (0.0, 3.0, 0.0),
        (0.7, -1.5, 0.2 + 0.35j),
    ],
)
def test_closed_form_matches_dense_diagonalization(
    mean: float, detuning: float, coupling: complex
) -> None:
    solution = two_level_solution(mean, detuning, coupling)
    values, weights = _reference_solution(mean, detuning, coupling)
    assert solution.upper_energy == pytest.approx(values[0])
    assert solution.lower_energy == pytest.approx(values[1])
    assert float(solution.upper_weight) == pytest.approx(weights[0], abs=1e-12)
    assert float(solution.lower_weight) == pytest.approx(weights[1], abs=1e-12)


def test_weights_obey_the_fiber_sum_rule() -> None:
    rng = np.random.default_rng(20240901)
    mean = rng.normal(size=64)
    detuning = rng.normal(size=64)
    coupling = rng.normal(size=64)
    solution = two_level_solution(mean, detuning, coupling)
    total = solution.upper_weight + solution.lower_weight
    assert np.allclose(total, 1.0)
    assert np.all(solution.upper_weight >= 0.0)
    assert np.all(solution.lower_weight >= 0.0)
    assert np.all(solution.splitting >= 2.0 * np.abs(coupling) - 1e-12)


def test_degenerate_pair_splits_evenly_and_repels_by_twice_the_coupling() -> None:
    solution = two_level_solution(0.0, 0.0, 0.35)
    assert float(solution.upper_weight) == pytest.approx(0.5)
    assert float(solution.lower_weight) == pytest.approx(0.5)
    assert float(solution.splitting) == pytest.approx(0.7)


def test_uncoupled_degenerate_point_is_reported_as_an_even_split() -> None:
    solution = two_level_solution(1.0, 0.0, 0.0)
    assert float(solution.upper_weight) == pytest.approx(0.5)
    assert float(solution.lower_weight) == pytest.approx(0.5)


def test_only_the_modulus_of_a_complex_coupling_matters() -> None:
    plain = two_level_solution(0.0, 0.8, 0.2)
    phased = two_level_solution(0.0, 0.8, 0.2 * np.exp(1.234j))
    assert float(phased.lower_weight) == pytest.approx(float(plain.lower_weight))
    assert float(phased.upper_energy) == pytest.approx(float(plain.upper_energy))


def test_shadow_weight_is_second_order_in_the_coupling() -> None:
    detuning = 1.0
    weights = [float(two_level_solution(0.0, detuning, v).lower_weight) for v in (0.1, 0.05, 0.025)]
    ratios = [weights[0] / weights[1], weights[1] / weights[2]]
    assert ratios[0] == pytest.approx(4.0, rel=2e-2)
    assert ratios[1] == pytest.approx(4.0, rel=2e-2)


def test_shadow_weight_lies_between_the_proved_bounds() -> None:
    detuning = np.array([0.0, 0.1, 0.5, 1.0, 4.0])
    coupling = np.array([0.3, 0.3, 0.05, 0.2, 0.7])
    lower, upper = shadow_weight_bounds(detuning, coupling)
    actual = two_level_solution(np.zeros_like(detuning), detuning, coupling).lower_weight
    assert np.all(actual >= lower - 1e-12)
    assert np.all(actual <= upper + 1e-12)
    assert np.isinf(upper[0])


def test_bounds_reject_a_negative_detuning() -> None:
    with pytest.raises(ValueError, match="non-negative detuning"):
        shadow_weight_bounds(-1.0, 0.2)


def test_inversion_recovers_the_coupling_and_the_unperturbed_levels() -> None:
    mean, detuning, coupling = 0.4, 1.3, 0.27
    solution = two_level_solution(mean, detuning, coupling)
    recovered, main, partner = coupling_from_peaks(
        solution.upper_energy,
        solution.lower_energy,
        solution.upper_weight,
        solution.lower_weight,
    )
    assert float(recovered) == pytest.approx(coupling)
    assert float(main) == pytest.approx(mean + detuning)
    assert float(partner) == pytest.approx(mean - detuning)


def test_inversion_rejects_inverted_peaks_and_negative_weights() -> None:
    with pytest.raises(ValueError, match="must not lie below"):
        coupling_from_peaks(0.0, 1.0, 0.5, 0.5)
    with pytest.raises(ValueError, match="non-negative"):
        coupling_from_peaks(1.0, 0.0, -0.1, 0.5)
    with pytest.raises(ValueError, match="some weight"):
        coupling_from_peaks(1.0, 0.0, 0.0, 0.0)


def _model_structure(coupling: float) -> EffectiveBandStructure:
    """A cosine band and its ``Q = pi/a`` partner, coupled by ``coupling``."""

    n = 21
    kx = np.linspace(0.0, 0.5, n)
    kpoints = np.stack([kx, np.zeros(n), np.zeros(n)], axis=1)
    main = -2.0 * np.cos(2.0 * np.pi * kx)
    partner = -2.0 * np.cos(2.0 * np.pi * (kx - 0.5))
    return shadow_band_structure(
        kpoints, main, partner, coupling, distances=kx, metadata={"coupling": coupling}
    )


def test_model_structure_carries_unit_weight_at_every_kpoint() -> None:
    structure = _model_structure(0.15)
    assert structure.n_bands == 2
    assert np.allclose(structure.weights.sum(axis=1), 1.0)
    assert np.all(np.diff(structure.energies, axis=1) >= -1e-12)


def test_diagnosis_recovers_the_coupling_of_a_model_structure() -> None:
    coupling = 0.18
    structure = _model_structure(coupling)
    diagnosis = diagnose_shadow_bands(structure)
    assert np.allclose(diagnosis.couplings, coupling)
    assert np.allclose(diagnosis.residual_weights, 0.0, atol=1e-12)
    assert diagnosis.coupling_estimate() == pytest.approx(coupling)
    assert np.all(diagnosis.main_weights >= diagnosis.shadow_weights)


def test_diagnosis_reports_the_weight_left_outside_the_pair() -> None:
    """An unrelated state outside the window is residual weight, not a partner."""

    structure = _model_structure(0.2)
    energies = np.concatenate([structure.energies, np.full((structure.n_kpoints, 1), 6.0)], axis=1)
    weights = np.concatenate(
        [0.7 * structure.weights, np.full((structure.n_kpoints, 1), 0.3)], axis=1
    )
    contaminated = EffectiveBandStructure(
        kpoints=structure.kpoints,
        energies=energies,
        weights=weights,
        distances=structure.distances,
    )
    diagnosis = diagnose_shadow_bands(contaminated, energy_window=5.0)
    assert np.allclose(diagnosis.pair_fraction, 0.7)
    assert np.allclose(diagnosis.residual_weights, 0.3)
    assert np.allclose(diagnosis.couplings, 0.2)
    assert np.isnan(diagnosis.coupling_estimate(minimum_pair_fraction=0.9))
    assert diagnosis.coupling_estimate(minimum_pair_fraction=0.5) == pytest.approx(0.2)


def test_a_bright_unrelated_state_is_picked_up_without_a_window() -> None:
    """The documented limitation: the pair is chosen by weight, not by physics.

    A bright state that is not the shadow partner displaces the real one from
    the pair, and nothing in the weights says so -- ``pair_fraction`` is then
    close to one and reassuring.  An energy window is the only defence, which is
    why :func:`diagnose_shadow_bands` offers one.
    """

    structure = _model_structure(0.05)
    energies = np.concatenate([structure.energies, np.full((structure.n_kpoints, 1), 6.0)], axis=1)
    weights = np.concatenate([structure.weights, np.full((structure.n_kpoints, 1), 0.3)], axis=1)
    contaminated = EffectiveBandStructure(
        kpoints=structure.kpoints, energies=energies, weights=weights
    )
    unwindowed = diagnose_shadow_bands(contaminated)
    windowed = diagnose_shadow_bands(contaminated, energy_window=5.0)
    assert unwindowed.pair_fraction[0] > 0.95
    assert unwindowed.couplings[0] != pytest.approx(0.05, rel=0.1)
    assert np.allclose(windowed.couplings, 0.05)


def test_diagnosis_needs_two_states() -> None:
    single = EffectiveBandStructure(
        kpoints=np.zeros((2, 3)), energies=np.zeros((2, 1)), weights=np.ones((2, 1))
    )
    with pytest.raises(ValueError, match="at least two states"):
        diagnose_shadow_bands(single)
    with pytest.raises(ValueError, match="fewer than two states"):
        diagnose_shadow_bands(_model_structure(0.2), energy_window=1e-6)


def test_shadow_structure_validates_its_shapes() -> None:
    with pytest.raises(ValueError, match=r"\(n_kpoints, 3\)"):
        shadow_band_structure(np.zeros((3, 2)), np.zeros(3), np.zeros(3), 0.1)
    with pytest.raises(ValueError, match=r"\(n_kpoints,\)"):
        shadow_band_structure(np.zeros((3, 3)), np.zeros(2), np.zeros(3), 0.1)


def test_diagnosis_serializes() -> None:
    diagnosis = diagnose_shadow_bands(_model_structure(0.1))
    payload = diagnosis.to_dict()
    assert set(payload) == {
        "distances",
        "main_energies",
        "shadow_energies",
        "main_weights",
        "shadow_weights",
        "couplings",
        "residual_weights",
        "pair_fraction",
    }
    assert len(payload["couplings"]) == diagnosis.distances.size


def test_cli_shadow_reports_and_writes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from typer.testing import CliRunner

    from unfoldlab.cli.main import app
    from unfoldlab.io.serialization import write_ebs

    run_path = tmp_path / "run.json"
    json_path = tmp_path / "shadow.json"
    write_ebs(run_path, _model_structure(0.22))
    result = CliRunner().invoke(
        app,
        ["shadow", "--input", str(run_path), "--json", str(json_path)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(json_path.read_text())
    assert payload["coupling_estimate"] == pytest.approx(0.22)
    assert len(payload["couplings"]) == 21


def test_cli_shadow_says_so_when_the_two_level_picture_fails(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from typer.testing import CliRunner

    from unfoldlab.cli.main import app
    from unfoldlab.io.serialization import write_ebs

    structure = _model_structure(0.2)
    energies = np.concatenate([structure.energies, np.full((structure.n_kpoints, 1), 6.0)], axis=1)
    weights = np.concatenate(
        [0.4 * structure.weights, np.full((structure.n_kpoints, 1), 0.6)], axis=1
    )
    run_path = tmp_path / "run.json"
    write_ebs(
        run_path,
        EffectiveBandStructure(kpoints=structure.kpoints, energies=energies, weights=weights),
    )
    result = CliRunner().invoke(app, ["shadow", "--input", str(run_path), "--window", "5.0"])
    assert result.exit_code == 0, result.output
    assert "does not apply here" in result.output
