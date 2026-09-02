"""Constant-energy cuts and closed-form energy windows."""

from __future__ import annotations

import math

import numpy as np
import pytest

from unfoldlab.cli.main import main
from unfoldlab.core.spectral import EffectiveBandStructure
from unfoldlab.core.windows import (
    band_filling,
    constant_energy_cut,
    energy_window_weight,
)
from unfoldlab.io.serialization import write_ebs


def _ebs(energies, weights=None, *, reference: float = 0.0) -> EffectiveBandStructure:
    energies = np.asarray(energies, dtype=float)
    kpoints = np.zeros((energies.shape[0], 3))
    kpoints[:, 0] = np.linspace(0.0, 0.5, energies.shape[0])
    return EffectiveBandStructure(
        kpoints=kpoints, energies=energies, weights=weights, reference_energy=reference
    )


class TestConstantEnergyCut:
    @pytest.mark.parametrize("kind", ["gaussian", "lorentzian"])
    def test_cut_matches_the_spectral_function(self, kind) -> None:
        structure = _ebs([[-0.4, 0.6], [0.1, 1.0]], [[0.3, 0.7], [0.5, 0.5]])
        grid = np.array([-1.0, 0.0, 0.25, 2.0])
        reference = structure.spectral_function(grid, broadening=0.12, kind=kind)
        for index, energy in enumerate(grid):
            cut = constant_energy_cut(structure, energy, broadening=0.12, kind=kind)
            assert cut == pytest.approx(reference[:, index])

    def test_peak_of_a_single_line_is_the_kernel_peak(self) -> None:
        structure = _ebs([[0.5]], [[1.0]])
        peak = constant_energy_cut(structure, 0.5, broadening=0.2, kind="gaussian")
        assert peak == pytest.approx([1.0 / (0.2 * math.sqrt(2.0 * math.pi))])

    def test_cut_is_measured_from_the_reference(self) -> None:
        structure = _ebs([[3.0]], [[1.0]], reference=1.0)
        assert constant_energy_cut(structure, 2.0, broadening=0.1) > constant_energy_cut(
            structure, 3.0, broadening=0.1
        )

    def test_per_state_widths_are_accepted(self) -> None:
        structure = _ebs([[0.0, 1.0]], [[0.5, 0.5]])
        widths = np.array([[0.05, 0.5]])
        cut = constant_energy_cut(structure, 0.0, broadening=widths)
        assert cut == pytest.approx(
            structure.spectral_function(np.array([0.0]), broadening=widths)[:, 0]
        )

    def test_invalid_broadening_is_rejected(self) -> None:
        structure = _ebs([[0.0]], [[1.0]])
        with pytest.raises(ValueError, match="positive"):
            constant_energy_cut(structure, 0.0, broadening=0.0)

    def test_unknown_kernel_is_rejected(self) -> None:
        structure = _ebs([[0.0]], [[1.0]])
        with pytest.raises(ValueError, match="unsupported broadening kind"):
            constant_energy_cut(structure, 0.0, broadening=0.1, kind="voigt")


class TestEnergyWindow:
    @pytest.mark.parametrize("kind", ["gaussian", "lorentzian"])
    def test_infinite_window_gives_the_sum_rule(self, kind) -> None:
        structure = _ebs([[-1.0, 0.0, 1.0]], [[0.2, 0.3, 0.5]])
        total = energy_window_weight(structure, -np.inf, np.inf, broadening=0.3, kind=kind)
        assert total == pytest.approx([1.0])

    @pytest.mark.parametrize("kind", ["gaussian", "lorentzian"])
    def test_window_matches_numerical_integration(self, kind) -> None:
        structure = _ebs([[-0.4, 0.6], [0.1, 1.0]], [[0.3, 0.7], [0.5, 0.5]])
        grid = np.linspace(-0.5, 0.8, 40001)
        numeric = np.trapezoid(
            structure.spectral_function(grid, broadening=0.15, kind=kind), grid, axis=1
        )
        exact = energy_window_weight(structure, -0.5, 0.8, broadening=0.15, kind=kind)
        assert exact == pytest.approx(numeric, abs=1e-6)

    @pytest.mark.parametrize("kind", ["gaussian", "lorentzian"])
    def test_adjacent_windows_add(self, kind) -> None:
        # UnfoldLab.windowWeight_add_adjacent
        structure = _ebs([[-0.4, 0.6], [0.1, 1.0]], [[0.3, 0.7], [0.5, 0.5]])
        left = energy_window_weight(structure, -2.0, 0.3, broadening=0.2, kind=kind)
        right = energy_window_weight(structure, 0.3, 2.0, broadening=0.2, kind=kind)
        whole = energy_window_weight(structure, -2.0, 2.0, broadening=0.2, kind=kind)
        assert left + right == pytest.approx(whole)

    @pytest.mark.parametrize("kind", ["gaussian", "lorentzian"])
    def test_window_is_between_zero_and_the_total(self, kind) -> None:
        # UnfoldLab.windowWeight_nonneg and windowWeight_le_total
        structure = _ebs([[-0.4, 0.6], [0.1, 1.0]], [[0.3, 0.7], [0.5, 0.5]])
        value = energy_window_weight(structure, -0.2, 0.2, broadening=0.25, kind=kind)
        assert np.all(value >= 0.0)
        assert np.all(value <= structure.weights.sum(axis=1) + 1e-12)

    @pytest.mark.parametrize("kind", ["gaussian", "lorentzian"])
    def test_widening_a_window_cannot_lose_weight(self, kind) -> None:
        # UnfoldLab.windowWeight_mono
        structure = _ebs([[-0.4, 0.6], [0.1, 1.0]], [[0.3, 0.7], [0.5, 0.5]])
        narrow = energy_window_weight(structure, -0.1, 0.1, broadening=0.2, kind=kind)
        wide = energy_window_weight(structure, -0.5, 0.7, broadening=0.2, kind=kind)
        assert np.all(wide >= narrow - 1e-12)

    def test_empty_window_carries_no_weight(self) -> None:
        structure = _ebs([[0.0]], [[1.0]])
        assert energy_window_weight(structure, 0.0, 0.0, broadening=0.1) == pytest.approx([0.0])

    def test_a_line_on_the_edge_contributes_half(self) -> None:
        structure = _ebs([[0.0]], [[1.0]])
        below = energy_window_weight(structure, -np.inf, 0.0, broadening=0.3)
        assert below == pytest.approx([0.5])

    def test_reversed_window_is_rejected(self) -> None:
        structure = _ebs([[0.0]], [[1.0]])
        with pytest.raises(ValueError, match="must not be below"):
            energy_window_weight(structure, 1.0, 0.0, broadening=0.1)

    def test_lorentzian_leaks_more_into_a_gap_than_a_gaussian(self) -> None:
        # Both kernels have the same width; the Lorentzian tail is heavier, so a
        # nominally empty window picks up much more weight from it.
        structure = _ebs([[-1.0, 1.0]], [[0.5, 0.5]])
        gaussian = energy_window_weight(structure, -0.2, 0.2, broadening=0.2, kind="gaussian")
        lorentzian = energy_window_weight(structure, -0.2, 0.2, broadening=0.2, kind="lorentzian")
        assert lorentzian[0] > 10.0 * gaussian[0]


class TestBandFilling:
    def test_occupied_and_empty_add_up_to_the_sum_rule(self) -> None:
        structure = _ebs([[-1.0, -0.2, 0.7]], [[1.0, 1.0, 1.0]])
        occupied = band_filling(structure, 0.0, broadening=0.1)
        empty = energy_window_weight(structure, 0.0, np.inf, broadening=0.1)
        assert occupied + empty == pytest.approx(structure.weights.sum(axis=1))

    def test_sharp_filling_counts_the_states_below(self) -> None:
        structure = _ebs([[-1.0, -0.2, 0.7]], [[1.0, 1.0, 1.0]])
        assert band_filling(structure, 0.0, broadening=1e-3) == pytest.approx([2.0])


class TestCutCLI:
    def test_cli_writes_a_cut(self, tmp_path) -> None:
        path = tmp_path / "ebs.json"
        write_ebs(path, _ebs([[-0.4, 0.6], [0.1, 1.0]], [[0.3, 0.7], [0.5, 0.5]]))
        out = tmp_path / "cut.dat"
        code = main(
            [
                "cut",
                "--input",
                str(path),
                "--energy",
                "0.1",
                "--broadening",
                "0.2",
                "--out",
                str(out),
            ]
        )
        assert code == 0
        rows = [line for line in out.read_text().splitlines() if not line.startswith("#")]
        assert len(rows) == 2

    def test_cli_writes_a_window(self, tmp_path) -> None:
        path = tmp_path / "ebs.json"
        write_ebs(path, _ebs([[-0.4, 0.6]], [[0.3, 0.7]]))
        out = tmp_path / "window.dat"
        code = main(
            [
                "cut",
                "--input",
                str(path),
                "--window",
                "-inf,inf",
                "--broadening",
                "0.2",
                "--out",
                str(out),
            ]
        )
        assert code == 0
        row = [line for line in out.read_text().splitlines() if not line.startswith("#")][0]
        assert float(row.split()[-1]) == pytest.approx(1.0)

    def test_cli_needs_exactly_one_of_energy_and_window(self, tmp_path) -> None:
        path = tmp_path / "ebs.json"
        write_ebs(path, _ebs([[0.0]], [[1.0]]))
        assert main(["cut", "--input", str(path), "--broadening", "0.1"]) != 0
        assert (
            main(
                [
                    "cut",
                    "--input",
                    str(path),
                    "--broadening",
                    "0.1",
                    "--energy",
                    "0",
                    "--window",
                    "-1,1",
                ]
            )
            != 0
        )
