"""The Fermi level of an unfolded run, and the energy reference.

The theorems behind these tests are in `RequestProject/Unfolding/Fermi.lean`:
`existsUnique_fermiLevel` (a unique root between zero and the total weight),
`filling_strictMono` and the two limits (why bracketing works),
`filling_add_empty` (occupied plus empty is the total), `filling_shift` /
`fermiLevel_shift` (equivariance under a change of reference) and
`abs_filling_sub_le` (the Lipschitz constant reported as `sensitivity`).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.core.fermi import (
    align_reference,
    electron_count,
    electrons_per_primitive_cell,
    fermi_dirac_occupation,
    find_fermi_level,
    state_occupations,
    total_weight,
)
from unmochan.core.spectral import EffectiveBandStructure
from unmochan.io.serialization import write_ebs


def two_band_structure(gap: float = 4.0, n_kpoints: int = 8) -> EffectiveBandStructure:
    """A filled band at `-gap/2` and an empty one at `+gap/2`, both dispersing."""

    kpoints = np.zeros((n_kpoints, 3))
    kpoints[:, 0] = np.linspace(0.0, 0.5, n_kpoints)
    dispersion = 0.5 * np.cos(2.0 * np.pi * kpoints[:, 0])
    energies = np.column_stack([-0.5 * gap + dispersion, 0.5 * gap + dispersion])
    return EffectiveBandStructure(
        kpoints=kpoints, energies=energies, weights=np.ones_like(energies)
    )


def test_occupation_is_stable_at_large_argument() -> None:
    """The naive `1 / (1 + exp(x))` overflows and warns above about x = 709."""

    with np.errstate(over="raise"):
        values = fermi_dirac_occupation([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    assert values[0] == pytest.approx(1.0)
    assert values[2] == pytest.approx(0.5)
    assert values[-1] == pytest.approx(0.0)
    assert np.all(np.diff(values) < 0.0)


def test_occupation_is_particle_hole_symmetric() -> None:
    """`fermiDirac_add_neg`: f(x) + f(-x) = 1."""

    x = np.array([-3.0, -0.7, 0.0, 0.7, 3.0])
    assert np.allclose(fermi_dirac_occupation(x) + fermi_dirac_occupation(-x), 1.0)


def test_state_occupations_rejects_a_zero_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        state_occupations(two_band_structure(), 0.0, temperature=0.0)


def test_occupied_plus_empty_is_the_total_weight() -> None:
    """`filling_add_empty`, evaluated on the whole structure."""

    structure = two_band_structure()
    occupied = electron_count(structure, 0.3, temperature=0.1, spin_degeneracy=2.0)
    empty_occupations = 1.0 - state_occupations(structure, 0.3, temperature=0.1)
    empty = 2.0 * float((structure.weights * empty_occupations).sum(axis=1).mean())
    assert occupied + empty == pytest.approx(total_weight(structure, spin_degeneracy=2.0))


def test_electron_count_is_strictly_increasing() -> None:
    """`filling_strictMono`: the count rises with the chemical potential."""

    structure = two_band_structure()
    counts = [
        electron_count(structure, energy, temperature=0.1, spin_degeneracy=2.0)
        for energy in np.linspace(-6.0, 6.0, 25)
    ]
    assert np.all(np.diff(counts) > 0.0)
    assert counts[0] == pytest.approx(0.0, abs=1e-9)
    assert counts[-1] == pytest.approx(4.0, abs=1e-9)


def test_fermi_level_of_a_gapped_structure_sits_in_the_gap() -> None:
    structure = two_band_structure(gap=4.0)
    report = find_fermi_level(structure, 2.0, temperature=0.05, spin_degeneracy=2.0)
    assert report.converged
    assert -2.0 < report.fermi < 2.0
    assert electron_count(
        structure, report.fermi, temperature=0.05, spin_degeneracy=2.0
    ) == pytest.approx(2.0, abs=1e-9)


def test_fermi_level_of_a_metal_lands_on_the_band() -> None:
    """One electron in a two-band structure puts the level inside the lower band."""

    structure = two_band_structure(gap=4.0)
    report = find_fermi_level(structure, 1.0, temperature=0.05, spin_degeneracy=2.0)
    assert report.converged
    assert -2.6 < report.fermi < -1.4


def test_the_root_is_unique() -> None:
    """`existsUnique_fermiLevel`: no other energy reproduces the count."""

    structure = two_band_structure()
    report = find_fermi_level(structure, 1.5, temperature=0.2, spin_degeneracy=2.0)
    for offset in (-1.0, -0.1, 0.1, 1.0):
        other = electron_count(
            structure, report.fermi + offset, temperature=0.2, spin_degeneracy=2.0
        )
        assert abs(other - 1.5) > 1e-6


def test_fermi_level_is_equivariant_under_a_change_of_reference() -> None:
    """`fermiLevel_shift`: shifting the reference shifts the level by the same amount."""

    structure = two_band_structure()
    shifted = EffectiveBandStructure(
        kpoints=structure.kpoints,
        energies=structure.energies + 3.0,
        weights=structure.weights,
    )
    base = find_fermi_level(structure, 2.0, temperature=0.05, spin_degeneracy=2.0)
    moved = find_fermi_level(shifted, 2.0, temperature=0.05, spin_degeneracy=2.0)
    assert moved.fermi == pytest.approx(base.fermi + 3.0, abs=1e-9)


def test_align_reference_puts_the_level_at_zero() -> None:
    structure = two_band_structure()
    report = find_fermi_level(structure, 2.0, temperature=0.05, spin_degeneracy=2.0)
    aligned = align_reference(structure, report.fermi)
    assert aligned.reference_energy == pytest.approx(report.fermi)
    again = find_fermi_level(aligned, 2.0, temperature=0.05, spin_degeneracy=2.0)
    assert again.fermi == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(aligned.energies, structure.energies)


def test_the_reported_sensitivity_bounds_the_actual_slope() -> None:
    """`abs_filling_sub_le`: the count is Lipschitz with constant (sum w) / (4 T)."""

    structure = two_band_structure()
    temperature = 0.1
    report = find_fermi_level(structure, 2.0, temperature=temperature, spin_degeneracy=2.0)
    step = 1e-3
    for centre in (-2.0, 0.0, 2.0):
        left = electron_count(
            structure, centre - step, temperature=temperature, spin_degeneracy=2.0
        )
        right = electron_count(
            structure, centre + step, temperature=temperature, spin_degeneracy=2.0
        )
        assert abs(right - left) <= report.sensitivity * 2.0 * step + 1e-12


def test_colder_smearing_makes_a_sharper_root() -> None:
    warm = find_fermi_level(two_band_structure(), 2.0, temperature=0.5, spin_degeneracy=2.0)
    cold = find_fermi_level(two_band_structure(), 2.0, temperature=0.05, spin_degeneracy=2.0)
    assert cold.sensitivity > warm.sensitivity


def test_unattainable_electron_counts_are_rejected() -> None:
    structure = two_band_structure()
    with pytest.raises(ValueError, match="not attainable"):
        find_fermi_level(structure, 0.0, temperature=0.1, spin_degeneracy=2.0)
    with pytest.raises(ValueError, match="not attainable"):
        find_fermi_level(structure, 4.0, temperature=0.1, spin_degeneracy=2.0)


def test_find_fermi_level_validates_its_arguments() -> None:
    structure = two_band_structure()
    with pytest.raises(ValueError, match="temperature must be positive"):
        find_fermi_level(structure, 2.0, temperature=0.0)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        find_fermi_level(structure, 2.0, temperature=0.1, tolerance=0.0)


def test_kpoint_weights_are_normalized_and_validated() -> None:
    structure = two_band_structure(n_kpoints=4)
    uniform = electron_count(structure, 0.0, temperature=0.1)
    scaled = electron_count(structure, 0.0, temperature=0.1, kpoint_weights=[2.0] * 4)
    assert scaled == pytest.approx(uniform)
    with pytest.raises(ValueError, match="one entry per k-point"):
        electron_count(structure, 0.0, temperature=0.1, kpoint_weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        electron_count(structure, 0.0, temperature=0.1, kpoint_weights=[1.0, -1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="must not sum to zero"):
        electron_count(structure, 0.0, temperature=0.1, kpoint_weights=[0.0] * 4)


def test_weighting_one_kpoint_reproduces_its_own_count() -> None:
    structure = two_band_structure(n_kpoints=4)
    weights = [1.0, 0.0, 0.0, 0.0]
    selected = electron_count(structure, -2.0, temperature=0.1, kpoint_weights=weights)
    occupations = state_occupations(structure, -2.0, temperature=0.1)
    assert selected == pytest.approx(float((structure.weights * occupations)[0].sum()))


def test_electrons_per_primitive_cell_divides_and_checks() -> None:
    assert electrons_per_primitive_cell(16.0, 4) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="not an integer"):
        electrons_per_primitive_cell(15.0, 4)
    with pytest.raises(ValueError, match="positive integer"):
        electrons_per_primitive_cell(16.0, 0)


def test_total_weight_and_spin_degeneracy_validation() -> None:
    structure = two_band_structure()
    assert total_weight(structure) == pytest.approx(2.0)
    assert total_weight(structure, spin_degeneracy=2.0) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="spin_degeneracy must be positive"):
        total_weight(structure, spin_degeneracy=0.0)
    with pytest.raises(ValueError, match="spin_degeneracy must be positive"):
        electron_count(structure, 0.0, temperature=0.1, spin_degeneracy=-1.0)


def test_cli_fermi_reports_and_aligns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_path = tmp_path / "run.json"
    aligned_path = tmp_path / "aligned.json"
    report_path = tmp_path / "report.json"
    write_ebs(run_path, two_band_structure())
    result = CliRunner().invoke(
        app,
        [
            "fermi",
            "--input",
            str(run_path),
            "--supercell-electrons",
            "8",
            "--multiplicity",
            "4",
            "--temperature",
            "0.05",
            "--align",
            str(aligned_path),
            "--json",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "per primitive cell" in result.output
    payload = json.loads(report_path.read_text())
    assert payload["converged"] is True
    assert payload["electrons"] == pytest.approx(2.0)
    assert aligned_path.exists()


def test_cli_fermi_requires_exactly_one_electron_count(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_path = tmp_path / "run.json"
    write_ebs(run_path, two_band_structure())
    result = CliRunner().invoke(app, ["fermi", "--input", str(run_path)])
    assert result.exit_code != 0
