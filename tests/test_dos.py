"""The unfolded density of states and its conservation.

The theorems behind these tests are in `RequestProject/Unfolding/Dos.lean`:
`IsFiberRepr.fiberDos_eq_bandDos` (summing over a complete fiber returns the
supercell density of states), `IsFiberRepr.fiberDos_div_card` (per primitive
cell it is that curve divided by `|det T|`), `integral_bandDos` and
`IsFiberRepr.integral_fiberDos` (both integrate to the band count) and
`IsFiberRepr.fiberDos_subset_le` / `meshDos_le` (an incomplete fiber can only
undercount).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.core.dos import (
    default_energy_grid,
    diagnose_dos_conservation,
    supercell_dos,
    unfolded_dos,
)
from unmochan.core.spectral import EffectiveBandStructure
from unmochan.core.tight_binding import TightBindingModel, unfold_tight_binding_model
from unmochan.core.transformations import TransformationMatrix
from unmochan.io.serialization import write_ebs

RNG = np.random.default_rng(20240901)


def chain_model(hopping: float = -1.0) -> TightBindingModel:
    """Single-orbital chain, ``E(k) = 2 * hopping * cos(2 pi k)``."""

    return TightBindingModel(
        1,
        {
            (0, 0, 0): [[0.0]],
            (1, 0, 0): [[hopping]],
            (-1, 0, 0): [[hopping]],
        },
    )


def complete_fiber_structure(
    n_supercell_kpoints: int = 5, multiplicity: int = 3, n_bands: int = 4
) -> EffectiveBandStructure:
    """An unfolded run whose fibers are complete: weights sum to one per band.

    Every supercell k-point contributes ``multiplicity`` primitive k-points, all
    carrying the same supercell eigenvalues and a random partition of unity as
    their weights -- exactly the structure `IsFiberRepr.sum_weight_eq_one`
    describes.
    """

    kpoints = []
    energies = []
    weights = []
    for index in range(n_supercell_kpoints):
        levels = np.sort(RNG.uniform(-3.0, 3.0, n_bands))
        share = RNG.random((multiplicity, n_bands))
        share /= share.sum(axis=0, keepdims=True)
        for member in range(multiplicity):
            kpoints.append([(index * multiplicity + member) / 16.0, 0.0, 0.0])
            energies.append(levels)
            weights.append(share[member])
    return EffectiveBandStructure(
        kpoints=np.asarray(kpoints, dtype=float),
        energies=np.asarray(energies, dtype=float),
        weights=np.asarray(weights, dtype=float),
    )


def single_state_structure(energy: float = 1.25) -> EffectiveBandStructure:
    return EffectiveBandStructure(
        kpoints=np.zeros((1, 3)),
        energies=np.array([[energy]]),
        weights=np.array([[1.0]]),
    )


@pytest.mark.parametrize("kind", ["gaussian", "lorentzian"])
def test_one_state_carries_one_state(kind: str) -> None:
    """`integral_bandDos`: a normalized kernel integrates to one state."""

    curve = unfolded_dos(single_state_structure(), broadening=0.1, kind=kind)
    assert curve.states == pytest.approx(1.0)
    assert curve.integrated[0] < 0.1
    assert np.all(np.diff(curve.integrated) >= -1e-12)


def test_gaussian_dos_matches_its_quadrature() -> None:
    """The closed-form integrated curve agrees with integrating the plotted one."""

    curve = unfolded_dos(single_state_structure(), broadening=0.2, n_points=4001)
    quadrature = np.trapezoid(curve.dos, curve.energies)
    assert quadrature == pytest.approx(curve.integrated[-1], rel=1e-5)
    assert curve.tail_loss == pytest.approx(0.0, abs=1e-6)


def test_a_lorentzian_loses_weight_off_the_grid() -> None:
    """The algebraic tail is why the state count is not read off the grid."""

    gaussian = unfolded_dos(single_state_structure(), broadening=0.2, kind="gaussian")
    lorentzian = unfolded_dos(single_state_structure(), broadening=0.2, kind="lorentzian")
    assert gaussian.tail_loss < 1e-6
    assert lorentzian.tail_loss > 1e-2
    assert lorentzian.states == pytest.approx(gaussian.states)


def test_unfolding_conserves_the_density_of_states() -> None:
    """`IsFiberRepr.fiberDos_eq_bandDos`: |det T| times the unfolded curve is
    the supercell curve, at every energy and for any broadening."""

    multiplicity = 3
    structure = complete_fiber_structure(multiplicity=multiplicity)
    unfolded = unfolded_dos(structure, broadening=0.15, multiplicity=multiplicity)
    reference = supercell_dos(structure, unfolded.energies, broadening=0.15)
    assert np.allclose(multiplicity * unfolded.dos, reference.dos, atol=1e-12)
    assert unfolded.states == pytest.approx(structure.n_bands / multiplicity)


def test_conservation_report_is_clean_for_a_complete_fiber() -> None:
    multiplicity = 3
    structure = complete_fiber_structure(multiplicity=multiplicity)
    report = diagnose_dos_conservation(structure, broadening=0.15, multiplicity=multiplicity)
    assert report.is_conserved()
    assert report.states_error == pytest.approx(0.0, abs=1e-12)
    assert report.max_excess < 1e-12
    assert report.max_fiber_excess < 1e-12
    assert report.max_fiber_deficit < 1e-12
    assert report.supercell_states == pytest.approx(structure.n_bands)
    assert report.expected_states == pytest.approx(structure.n_bands / multiplicity)


def test_an_incomplete_fiber_undercounts_but_never_exceeds() -> None:
    """`IsFiberRepr.fiberDos_subset_le`: dropping fiber members loses weight."""

    multiplicity = 3
    structure = complete_fiber_structure(multiplicity=multiplicity)
    keep = np.arange(structure.n_kpoints) % multiplicity != 0
    partial = EffectiveBandStructure(
        kpoints=structure.kpoints[keep],
        energies=structure.energies[keep],
        weights=structure.weights[keep],
    )
    report = diagnose_dos_conservation(partial, broadening=0.15, multiplicity=multiplicity)
    assert not report.is_conserved()
    assert report.states_error < 0.0
    # The bound on the weights survives an incomplete sample; the equality with
    # the supercell curve does not, so `max_fiber_excess` is positive and is not
    # a fault.
    assert report.is_bounded()
    assert report.max_fiber_excess > 0.0
    assert report.max_fiber_deficit > 0.0


def test_weights_above_one_are_caught_as_an_excess() -> None:
    """`meshDos_le` fails only if a weight leaves [0, 1] -- a bug, not a sample."""

    structure = complete_fiber_structure(multiplicity=2)
    inflated = EffectiveBandStructure(
        kpoints=structure.kpoints,
        energies=structure.energies,
        weights=structure.weights * 4.0,
    )
    report = diagnose_dos_conservation(inflated, broadening=0.15, multiplicity=2)
    assert not report.is_bounded()
    assert report.max_excess > 1e-6


def test_kpoint_weights_are_honoured_and_validated() -> None:
    structure = complete_fiber_structure(multiplicity=2, n_supercell_kpoints=2)
    single = np.zeros(structure.n_kpoints)
    single[0] = 1.0
    curve = unfolded_dos(structure, broadening=0.1, kpoint_weights=single)
    assert curve.states == pytest.approx(structure.weights[0].sum())

    # Unnormalized weights are normalized, not rejected.
    doubled = unfolded_dos(structure, broadening=0.1, kpoint_weights=2.0 * single)
    assert np.allclose(doubled.dos, curve.dos)

    with pytest.raises(ValueError, match="one entry per k-point"):
        unfolded_dos(structure, broadening=0.1, kpoint_weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        unfolded_dos(structure, broadening=0.1, kpoint_weights=-np.ones(structure.n_kpoints))
    with pytest.raises(ValueError, match="sum to zero"):
        unfolded_dos(structure, broadening=0.1, kpoint_weights=np.zeros(structure.n_kpoints))


def test_arguments_are_validated() -> None:
    structure = single_state_structure()
    with pytest.raises(ValueError, match="unsupported broadening kind"):
        unfolded_dos(structure, broadening=0.1, kind="voigt")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="multiplicity"):
        unfolded_dos(structure, broadening=0.1, multiplicity=0)
    with pytest.raises(ValueError, match="broadening must be positive"):
        unfolded_dos(structure, broadening=0.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        unfolded_dos(structure, np.zeros((2, 2)), broadening=0.1)
    with pytest.raises(ValueError, match="must not be empty"):
        unfolded_dos(structure, np.zeros(0), broadening=0.1)
    with pytest.raises(ValueError, match="n_points"):
        default_energy_grid(structure, broadening=0.1, n_points=1)
    with pytest.raises(ValueError, match="padding"):
        default_energy_grid(structure, broadening=0.1, padding=-1.0)


def test_a_supplied_grid_is_used_verbatim() -> None:
    grid = np.linspace(-1.0, 2.0, 17)
    curve = unfolded_dos(single_state_structure(), grid, broadening=0.1)
    assert np.array_equal(curve.energies, grid)
    assert curve.dos.shape == grid.shape


def test_serialization_round_trips_the_curve() -> None:
    curve = unfolded_dos(single_state_structure(), broadening=0.1, multiplicity=2)
    payload = curve.to_dict()
    assert payload["multiplicity"] == 2
    assert payload["states"] == pytest.approx(1.0)
    assert len(payload["dos"]) == len(payload["energies"])


def test_cli_dos_reports_and_writes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_path = tmp_path / "run.json"
    out_path = tmp_path / "dos.dat"
    report_path = tmp_path / "dos.json"
    write_ebs(run_path, complete_fiber_structure(multiplicity=3))
    result = CliRunner().invoke(
        app,
        [
            "dos",
            "--input",
            str(run_path),
            "--broadening",
            "0.15",
            "--multiplicity",
            "3",
            "--points",
            "201",
            "--out",
            str(out_path),
            "--json",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(report_path.read_text())
    assert payload["states_error"] == pytest.approx(0.0, abs=1e-9)
    assert payload["max_excess"] < 1e-9
    assert payload["max_fiber_excess"] < 1e-9
    table = np.loadtxt(out_path)
    assert table.shape == (201, 4)
    assert np.allclose(3.0 * table[:, 1], table[:, 3], atol=1e-12)


def test_cli_dos_rejects_a_half_given_energy_range(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_path = tmp_path / "run.json"
    write_ebs(run_path, single_state_structure())
    result = CliRunner().invoke(
        app, ["dos", "--input", str(run_path), "--broadening", "0.1", "--emin", "-1"]
    )
    assert result.exit_code != 0


def test_end_to_end_defect_chain_conserves_the_density_of_states() -> None:
    """The whole pipeline, not a synthetic table: a defect chain unfolded.

    A four-site supercell of a one-orbital chain with an on-site defect is
    diagonalized and unfolded onto the full fiber of every supercell k-point of
    a uniform mesh.  The defect scatters weight across the fiber -- the ideal
    one-fiber-member answer is gone -- but the density of states is untouched:
    four times the unfolded curve is the supercell curve, and the unfolded one
    carries exactly one state per primitive cell.
    """

    multiplicity = 4
    transform = TransformationMatrix(np.diag([multiplicity, 1, 1]))
    mesh = np.zeros((6, 3))
    mesh[:, 0] = np.arange(6) / 6.0
    kpoints, energies, weights = unfold_tight_binding_model(
        chain_model(), transform, mesh, perturbations=[(0, 0, 1.5)]
    )
    structure = EffectiveBandStructure(kpoints=kpoints, energies=energies, weights=weights)

    unfolded = unfolded_dos(structure, broadening=0.1, multiplicity=multiplicity)
    reference = supercell_dos(structure, unfolded.energies, broadening=0.1)
    assert np.allclose(multiplicity * unfolded.dos, reference.dos, atol=1e-12)

    report = diagnose_dos_conservation(structure, broadening=0.1, multiplicity=multiplicity)
    assert report.is_conserved()
    assert report.unfolded_states == pytest.approx(1.0)
    assert np.min(weights) < 0.99
