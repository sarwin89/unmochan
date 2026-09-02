"""Configurational averaging: alloy, defect-ensemble and SQS disorder."""

from __future__ import annotations

import json

import numpy as np
import pytest

from unfoldlab.cli.main import main
from unfoldlab.core.ensemble import (
    band_moments,
    configuration_weights,
    disorder_broadening,
    ensemble_spectral_function,
    stack_configurations,
)
from unfoldlab.core.spectral import EffectiveBandStructure
from unfoldlab.io.serialization import write_ebs


def _ebs(energies, weights=None, *, reference: float = 0.0) -> EffectiveBandStructure:
    energies = np.asarray(energies, dtype=float)
    n_kpoints = energies.shape[0]
    kpoints = np.zeros((n_kpoints, 3))
    kpoints[:, 0] = np.linspace(0.0, 0.5, n_kpoints)
    return EffectiveBandStructure(
        kpoints=kpoints,
        energies=energies,
        weights=weights,
        reference_energy=reference,
    )


def _shifted_chain(shift: float, n_kpoints: int = 5) -> EffectiveBandStructure:
    """A one-band cosine chain rigidly shifted in energy."""

    k = np.linspace(0.0, 0.5, n_kpoints)
    energies = (-2.0 * np.cos(2.0 * np.pi * k) + shift)[:, None]
    return _ebs(energies, np.ones((n_kpoints, 1)))


class TestConfigurationWeights:
    def test_default_is_uniform(self) -> None:
        assert configuration_weights(None, 4) == pytest.approx([0.25] * 4)

    def test_multiplicities_are_normalized(self) -> None:
        assert configuration_weights([2, 1, 1], 3) == pytest.approx([0.5, 0.25, 0.25])

    @pytest.mark.parametrize(
        "weights, message",
        [
            ([1.0, -1.0], "non-negative"),
            ([0.0, 0.0], "not all be zero"),
            ([np.nan, 1.0], "finite"),
            ([1.0], "expected 2"),
        ],
    )
    def test_invalid_weights_are_rejected(self, weights, message) -> None:
        with pytest.raises(ValueError, match=message):
            configuration_weights(weights, 2)


class TestStacking:
    def test_stacked_weights_are_the_mixture(self) -> None:
        a = _shifted_chain(0.0)
        b = _shifted_chain(0.4)
        merged = stack_configurations([a, b], [3.0, 1.0])
        assert merged.n_bands == a.n_bands + b.n_bands
        assert merged.weights[:, 0] == pytest.approx(0.75 * a.weights[:, 0])
        assert merged.weights[:, 1] == pytest.approx(0.25 * b.weights[:, 0])
        assert merged.metadata["ensemble"]["configuration_weights"] == pytest.approx([0.75, 0.25])

    def test_sum_rule_survives_averaging(self) -> None:
        # Each configuration carries unit weight per k-point, so the average
        # does too: UnfoldLab.integral_mixture_spectralFunction_one.
        members = [_shifted_chain(shift) for shift in (-0.3, 0.0, 0.25)]
        merged = stack_configurations(members)
        assert merged.weights.sum(axis=1) == pytest.approx(np.ones(merged.n_kpoints))

    def test_spectral_function_of_the_stack_is_the_average(self) -> None:
        members = [_shifted_chain(shift) for shift in (-0.3, 0.4)]
        grid = np.linspace(-4.0, 4.0, 401)
        merged = stack_configurations(members, [0.7, 0.3])
        direct = 0.7 * members[0].spectral_function(grid, broadening=0.1) + (
            0.3 * members[1].spectral_function(grid, broadening=0.1)
        )
        assert merged.spectral_function(grid, broadening=0.1) == pytest.approx(direct)

    def test_each_configuration_keeps_its_own_reference_energy(self) -> None:
        a = _ebs([[1.0]], reference=1.0)
        b = _ebs([[2.0]], reference=2.0)
        merged = stack_configurations([a, b])
        assert merged.reference_energy == 0.0
        assert merged.energies == pytest.approx(np.zeros((1, 2)))

    def test_configurations_must_share_the_kpoints(self) -> None:
        a = _shifted_chain(0.0, n_kpoints=4)
        b = _shifted_chain(0.0, n_kpoints=5)
        with pytest.raises(ValueError, match="share the k-point list"):
            stack_configurations([a, b])

    def test_empty_ensemble_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one configuration"):
            stack_configurations([])

    def test_configurations_may_have_different_band_counts(self) -> None:
        a = _ebs([[0.0, 1.0]], [[0.5, 0.5]])
        b = _ebs([[0.5]], [[1.0]])
        merged = stack_configurations([a, b])
        assert merged.n_bands == 3
        assert merged.weights.sum() == pytest.approx(1.0)


class TestBandMoments:
    def test_moments_of_a_single_line(self) -> None:
        moments = band_moments(_ebs([[1.5]], [[1.0]]))
        assert moments.total == pytest.approx([1.0])
        assert moments.centre == pytest.approx([1.5])
        assert moments.spread == pytest.approx([0.0])

    def test_two_equal_lines(self) -> None:
        moments = band_moments(_ebs([[-0.5, 0.5]], [[0.5, 0.5]]))
        assert moments.centre == pytest.approx([0.0])
        assert moments.spread == pytest.approx([0.25])
        assert moments.width == pytest.approx([0.5])

    def test_moments_are_measured_from_the_reference(self) -> None:
        moments = band_moments(_ebs([[3.0]], [[1.0]], reference=1.0))
        assert moments.centre == pytest.approx([2.0])

    def test_a_kpoint_with_no_weight_gives_zero(self) -> None:
        moments = band_moments(_ebs([[0.0, 1.0]], [[0.0, 0.0]]))
        assert moments.total == pytest.approx([0.0])
        assert moments.centre == pytest.approx([0.0])
        assert moments.spread == pytest.approx([0.0])


class TestDisorderBroadening:
    def test_rigid_shifts_give_pure_disorder_broadening(self) -> None:
        # Every configuration is a single sharp line, so the intrinsic width is
        # zero and all of the averaged width is disorder.
        shifts = np.array([-0.2, 0.0, 0.2])
        members = [_shifted_chain(shift) for shift in shifts]
        split = disorder_broadening(members)
        assert split.intrinsic == pytest.approx(np.zeros(5), abs=1e-12)
        assert split.disorder == pytest.approx(np.full(5, shifts.var()))
        assert split.disorder_width == pytest.approx(np.full(5, shifts.std()))
        assert split.residual < 1e-12

    def test_law_of_total_variance_holds(self) -> None:
        rng = np.random.default_rng(3)
        members = [_ebs(rng.normal(size=(4, 3)), rng.random((4, 3)) + 0.1) for _ in range(5)]
        normalized = [
            EffectiveBandStructure(
                kpoints=ebs.kpoints,
                energies=ebs.energies,
                weights=ebs.weights / ebs.weights.sum(axis=1, keepdims=True),
            )
            for ebs in members
        ]
        split = disorder_broadening(normalized, [3.0, 1.0, 1.0, 2.0, 1.0])
        assert split.total == pytest.approx(split.intrinsic + split.disorder)
        assert split.residual < 1e-12

    def test_disorder_never_sharpens_the_average(self) -> None:
        members = [_ebs([[-0.5, 0.5]], [[0.5, 0.5]]), _ebs([[0.2, 1.2]], [[0.5, 0.5]])]
        split = disorder_broadening(members)
        assert np.all(split.total >= split.intrinsic - 1e-12)

    def test_identical_configurations_have_no_disorder_broadening(self) -> None:
        member = _ebs([[-0.5, 0.5]], [[0.5, 0.5]])
        split = disorder_broadening([member, member, member])
        assert split.disorder == pytest.approx([0.0], abs=1e-12)
        assert split.total == pytest.approx(band_moments(member).spread)

    def test_weights_pick_out_a_single_configuration(self) -> None:
        a = _shifted_chain(0.0)
        b = _shifted_chain(1.0)
        split = disorder_broadening([a, b], [1.0, 0.0])
        assert split.centre == pytest.approx(band_moments(a).centre)
        assert split.disorder == pytest.approx(np.zeros(5), abs=1e-12)


class TestEnsembleSpectralFunction:
    def test_mean_conserves_spectral_weight(self) -> None:
        members = [_shifted_chain(shift) for shift in (-0.3, 0.0, 0.4)]
        grid = np.linspace(-6.0, 6.0, 2401)
        result = ensemble_spectral_function(members, grid, broadening=0.08)
        integral = np.trapezoid(result.mean, grid, axis=1)
        assert integral == pytest.approx(np.ones(5), abs=1e-6)
        assert np.all(result.mean >= 0.0)

    def test_identical_configurations_have_zero_scatter(self) -> None:
        member = _shifted_chain(0.0)
        grid = np.linspace(-4.0, 4.0, 201)
        result = ensemble_spectral_function([member] * 3, grid, broadening=0.1)
        assert result.variance == pytest.approx(np.zeros_like(result.variance), abs=1e-18)
        assert result.standard_deviation.max() == pytest.approx(0.0)
        assert result.mean == pytest.approx(member.spectral_function(grid, broadening=0.1))

    def test_per_configuration_broadening(self) -> None:
        members = [_shifted_chain(0.0), _shifted_chain(0.5)]
        grid = np.linspace(-4.0, 4.0, 801)
        result = ensemble_spectral_function(members, grid, broadening=[0.05, 0.2])
        direct = 0.5 * members[0].spectral_function(grid, broadening=0.05) + (
            0.5 * members[1].spectral_function(grid, broadening=0.2)
        )
        assert result.mean == pytest.approx(direct)

    def test_wrong_number_of_broadenings_is_rejected(self) -> None:
        members = [_shifted_chain(0.0), _shifted_chain(0.5)]
        with pytest.raises(ValueError, match="one entry per configuration"):
            ensemble_spectral_function(
                members, np.linspace(-1.0, 1.0, 5), broadening=[0.1, 0.1, 0.1]
            )

    def test_lorentzian_is_available(self) -> None:
        members = [_shifted_chain(0.0), _shifted_chain(0.3)]
        grid = np.linspace(-4.0, 4.0, 401)
        result = ensemble_spectral_function(members, grid, broadening=0.1, kind="lorentzian")
        assert np.all(result.mean >= 0.0)


class TestEnsembleCLI:
    def test_cli_averages_and_reports(self, tmp_path) -> None:
        paths = []
        for index, shift in enumerate((-0.2, 0.2)):
            path = tmp_path / f"config{index}.json"
            write_ebs(path, _shifted_chain(shift))
            paths.append(path)
        out = tmp_path / "average.json"
        report = tmp_path / "report.json"
        code = main(
            [
                "ensemble",
                "--input",
                str(paths[0]),
                "--input",
                str(paths[1]),
                "--out",
                str(out),
                "--report",
                str(report),
            ]
        )
        assert code == 0
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["configuration_weights"] == pytest.approx([0.5, 0.5])
        assert payload["disorder_variance"] == pytest.approx([0.04] * 5)
        assert payload["intrinsic_variance"] == pytest.approx([0.0] * 5, abs=1e-12)
        assert payload["residual"] < 1e-12
        assert out.exists()

    def test_cli_rejects_mismatched_weight_count(self, tmp_path) -> None:
        path = tmp_path / "config.json"
        write_ebs(path, _shifted_chain(0.0))
        code = main(["ensemble", "--input", str(path), "--weight", "1", "--weight", "2"])
        assert code != 0
