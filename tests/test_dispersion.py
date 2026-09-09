"""Extracting a dispersion, and its effective mass, from a spectral function.

The claims exercised here are the numerical face of
``RequestProject/Unfolding/Dispersion.lean``: parabolic refinement is exact for
a quadratic, the symmetric second difference is exact for a cubic, and the
broadening width is a hard resolution limit.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.core.dispersion import (
    HBAR_SQ_OVER_ME,
    band_extremum,
    effective_mass,
    parabolic_vertex,
    peaks_to_arrays,
    second_derivative,
    spectral_peaks,
    unresolvable_pairs,
)
from unmochan.core.spectral import EffectiveBandStructure
from unmochan.io.serialization import write_ebs

runner = CliRunner()


def _structure(energies: np.ndarray, weights: np.ndarray | None = None) -> EffectiveBandStructure:
    n_kpoints = energies.shape[0]
    kpoints = np.zeros((n_kpoints, 3))
    kpoints[:, 0] = np.linspace(0.0, 0.5, n_kpoints)
    return EffectiveBandStructure(
        kpoints=kpoints,
        energies=energies,
        weights=np.ones_like(energies) if weights is None else weights,
        distances=np.linspace(0.0, 1.0, n_kpoints),
    )


def test_parabolic_refinement_is_exact_for_a_quadratic() -> None:
    a, vertex, offset = -2.5, 0.37, 0.11
    x0, h = vertex + offset, 0.05
    samples = [a * (x0 + delta - vertex) ** 2 + 1.25 for delta in (-h, 0.0, h)]
    position, value = parabolic_vertex(x0, h, *samples)
    assert position == pytest.approx(vertex)
    assert value == pytest.approx(1.25)


def test_parabolic_refinement_returns_the_centre_for_a_flat_triple() -> None:
    assert parabolic_vertex(0.5, 0.1, 1.0, 1.0, 1.0) == (0.5, 1.0)


def test_parabolic_refinement_rejects_a_zero_step() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        parabolic_vertex(0.0, 0.0, 1.0, 2.0, 1.0)


def test_the_second_difference_is_exact_for_a_cubic() -> None:
    x = np.linspace(-1.0, 1.0, 5)
    a, b, c, d = 0.7, -1.3, 2.0, 0.5
    y = a * x**3 + b * x**2 + c * x + d
    for index in (1, 2, 3):
        assert second_derivative(x, y, index) == pytest.approx(6.0 * a * x[index] + 2.0 * b)


def test_the_second_difference_handles_unequal_spacing() -> None:
    x = np.array([0.0, 0.1, 0.4])
    y = 3.0 * x**2 - x + 2.0
    assert second_derivative(x, y, 1) == pytest.approx(6.0)


def test_second_derivative_rejects_an_edge_index() -> None:
    x = np.linspace(0.0, 1.0, 4)
    with pytest.raises(ValueError, match="interior"):
        second_derivative(x, x**2, 0)


def test_a_free_electron_band_has_unit_mass() -> None:
    k = np.linspace(-0.3, 0.3, 7)
    energies = 0.5 * HBAR_SQ_OVER_ME * k**2
    assert effective_mass(k, energies, 3) == pytest.approx(1.0)

    extremum = band_extremum(k, energies)
    assert extremum.energy == pytest.approx(0.0, abs=1e-12)
    assert extremum.distance == pytest.approx(0.0, abs=1e-12)
    assert extremum.effective_mass == pytest.approx(1.0)
    assert not extremum.is_maximum


def test_a_band_maximum_gives_a_negative_mass() -> None:
    k = np.linspace(-0.3, 0.3, 7)
    energies = 1.0 - 0.5 * HBAR_SQ_OVER_ME * k**2 / 0.5
    extremum = band_extremum(k, energies, maximum=True)
    assert extremum.is_maximum
    assert extremum.energy == pytest.approx(1.0)
    assert extremum.effective_mass == pytest.approx(-0.5)


def test_a_flat_band_has_infinite_mass() -> None:
    k = np.linspace(0.0, 1.0, 5)
    assert effective_mass(k, np.ones_like(k), 2) == float("inf")


def test_an_extremum_at_the_edge_has_no_mass() -> None:
    k = np.linspace(0.0, 1.0, 5)
    extremum = band_extremum(k, k)
    assert extremum.index == 0
    assert np.isnan(extremum.effective_mass)


def test_peaks_sit_on_the_band_energies_when_they_are_far_apart() -> None:
    energies = np.array([[-1.0, 1.0], [-0.8, 1.2]])
    structure = _structure(energies)
    grid = np.linspace(-2.0, 2.0, 2001)
    peaks = spectral_peaks(structure, grid, broadening=0.05)

    assert len(peaks) == 4
    found = sorted((peak.kpoint_index, peak.energy) for peak in peaks)
    expected = [(0, -1.0), (0, 1.0), (1, -0.8), (1, 1.2)]
    for (index, energy), (index_ref, energy_ref) in zip(found, expected, strict=True):
        assert index == index_ref
        assert energy == pytest.approx(energy_ref, abs=2e-3)

    columns = peaks_to_arrays(peaks)
    assert columns["energy"].shape == (4,)


def test_two_close_bands_report_one_peak_between_them() -> None:
    # Separation 0.04 with a width of 0.05: below the resolution limit.
    energies = np.array([[-0.02, 0.02]])
    structure = _structure(energies)
    grid = np.linspace(-1.0, 1.0, 4001)
    peaks = spectral_peaks(structure, grid, broadening=0.05)

    assert len(peaks) == 1
    assert peaks[0].energy == pytest.approx(0.0, abs=1e-3)
    # And no state lives there.
    assert not np.any(np.isclose(energies[0], peaks[0].energy, atol=1e-3))

    merged = unresolvable_pairs(structure, broadening=0.05)
    assert len(merged) == 1
    assert merged[0].separation == pytest.approx(0.04)


def test_two_bands_two_widths_apart_are_resolved() -> None:
    energies = np.array([[-0.06, 0.06]])
    structure = _structure(energies)
    grid = np.linspace(-1.0, 1.0, 4001)
    peaks = spectral_peaks(structure, grid, broadening=0.05)

    assert len(peaks) == 2
    assert not unresolvable_pairs(structure, broadening=0.05)


def test_states_without_weight_are_not_reported_as_unresolvable() -> None:
    energies = np.array([[-0.02, 0.0, 0.02]])
    weights = np.array([[1.0, 0.0, 1.0]])
    structure = _structure(energies, weights)
    merged = unresolvable_pairs(structure, broadening=0.05)
    assert len(merged) == 1
    assert (merged[0].lower_band, merged[0].upper_band) == (0, 2)


def test_spectral_peaks_validates_its_grid() -> None:
    structure = _structure(np.array([[0.0]]))
    with pytest.raises(ValueError, match="at least three"):
        spectral_peaks(structure, [0.0, 1.0], broadening=0.1)
    with pytest.raises(ValueError, match="increasing"):
        spectral_peaks(structure, [0.0, 1.0, 0.5], broadening=0.1)


def test_dispersion_cli_writes_a_peak_table(tmp_path) -> None:
    structure = _structure(np.array([[-1.0, 1.0], [-0.8, 1.2]]))
    run = tmp_path / "run.json"
    write_ebs(run, structure)
    out = tmp_path / "peaks.dat"

    result = runner.invoke(
        app,
        [
            "dispersion",
            "--input",
            str(run),
            "--broadening",
            "0.05",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Peaks found: 4" in result.output
    rows = [line for line in out.read_text().splitlines() if not line.startswith("#")]
    assert len(rows) == 4


def test_dispersion_cli_warns_about_unresolvable_states(tmp_path) -> None:
    structure = _structure(np.array([[-0.02, 0.02]]))
    run = tmp_path / "run.json"
    write_ebs(run, structure)

    result = runner.invoke(app, ["dispersion", "--input", str(run), "--broadening", "0.05"])

    assert result.exit_code == 0, result.output
    assert "closer than the broadening width" in result.output
