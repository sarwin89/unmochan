"""Tracking spectral peaks into branches.

The statements behind these tests are in
`RequestProject/Unfolding/Tracking.lean`: `matchCost_id_le` (for ordered peak
lists the order-preserving matching is optimal among all permutations),
`matchCost_greedy_gt` (nearest-free-partner tracking is strictly worse on an
explicit example), `exists_matchCost_tie` (a degeneracy leaves the branch labels
undetermined) and `abs_pair_exchange` (the same exchange inequality for the
absolute cost).
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.core.dispersion import SpectralPeak
from unfoldlab.core.spectral import EffectiveBandStructure
from unfoldlab.core.tracking import track_branches, track_spectral_peaks
from unfoldlab.io.serialization import write_ebs


def two_band_structure(n_kpoints: int = 21) -> EffectiveBandStructure:
    """Two well-separated cosine bands, no crossing."""

    kpoints = np.zeros((n_kpoints, 3))
    kpoints[:, 0] = np.linspace(0.0, 0.5, n_kpoints)
    lower = -np.cos(2.0 * np.pi * kpoints[:, 0])
    upper = 2.0 + 0.5 * np.cos(2.0 * np.pi * kpoints[:, 0])
    energies = np.column_stack([lower, upper])
    return EffectiveBandStructure(
        kpoints=kpoints, energies=energies, weights=np.ones_like(energies)
    )


def crossing_structure(n_kpoints: int = 11) -> EffectiveBandStructure:
    """Two straight bands crossing exactly in the middle of the path."""

    kpoints = np.zeros((n_kpoints, 3))
    x = np.linspace(0.0, 1.0, n_kpoints)
    kpoints[:, 0] = 0.5 * x
    energies = np.column_stack([-1.0 + 2.0 * x, 1.0 - 2.0 * x])
    return EffectiveBandStructure(
        kpoints=kpoints, energies=energies, weights=np.ones_like(energies)
    )


def peaks_from_energies(energies: list[list[float]]) -> list[SpectralPeak]:
    return [
        SpectralPeak(kpoint_index=index, distance=float(index), energy=value, intensity=1.0)
        for index, row in enumerate(energies)
        for value in row
    ]


def match_cost(a: np.ndarray, b: np.ndarray, permutation: tuple[int, ...]) -> float:
    return float(sum((a[i] - b[permutation[i]]) ** 2 for i in range(a.size)))


def test_sorted_matching_beats_every_permutation() -> None:
    """`matchCost_id_le`, checked exhaustively on random ordered lists."""

    rng = np.random.default_rng(7)
    for _ in range(20):
        a = np.sort(rng.normal(size=5))
        b = np.sort(rng.normal(size=5))
        identity = match_cost(a, b, tuple(range(5)))
        for permutation in itertools.permutations(range(5)):
            assert identity <= match_cost(a, b, permutation) + 1e-12


def test_greedy_nearest_partner_is_worse() -> None:
    """`matchCost_greedy_gt`: the example the tracker refuses to use."""

    a = np.array([0.0, 1.0])
    b = np.array([-5.0, 0.1])
    # Walking `a` in order and taking the nearest free partner pairs 0 with 0.1.
    greedy = match_cost(a, b, (1, 0))
    ordered = match_cost(a, b, (0, 1))
    assert ordered < greedy
    assert ordered == pytest.approx(25.81)
    assert greedy == pytest.approx(36.01)


def test_two_separated_bands_give_two_full_branches() -> None:
    structure = two_band_structure()
    tracking = track_spectral_peaks(structure, np.linspace(-2.0, 3.5, 801), broadening=0.05)
    assert tracking.n_branches == 2
    assert [branch.length for branch in tracking.branches] == [21, 21]
    assert tracking.crossings == []
    lower = tracking.branches[0].energies
    assert lower[0] == pytest.approx(-1.0, abs=2e-3)
    assert lower[-1] == pytest.approx(1.0, abs=2e-3)
    assert np.all(np.diff(lower) > 0.0)


def test_a_crossing_is_reported_and_not_resolved() -> None:
    """At the crossing the two peaks merge, so the continuation is a choice."""

    structure = crossing_structure()
    tracking = track_spectral_peaks(structure, np.linspace(-2.0, 2.0, 1601), broadening=0.05)
    # The sorted matching follows the lower and upper envelopes; the middle
    # k-point has a single peak, so one branch ends there and another begins.
    termini = [c for c in tracking.crossings if c.kind == "terminus"]
    assert termini
    assert {c.kpoint_index for c in termini} <= {4, 5, 6}
    envelope = tracking.branches[0].energies
    assert envelope[0] == pytest.approx(-1.0, abs=1e-2)
    assert envelope[-1] == pytest.approx(-1.0, abs=1e-2)
    assert envelope[5] == pytest.approx(0.0, abs=1e-2)


def test_close_peaks_are_flagged_as_degenerate() -> None:
    peaks = peaks_from_energies([[0.0, 0.02], [0.0, 1.0]])
    tracking = track_branches(peaks, 2, degeneracy_tolerance=0.1)
    degenerate = [c for c in tracking.crossings if c.kind == "degenerate"]
    assert len(degenerate) == 1
    assert degenerate[0].kpoint_index == 0
    assert degenerate[0].separation == pytest.approx(0.02)


def test_max_jump_ends_a_branch() -> None:
    peaks = peaks_from_energies([[0.0], [0.1], [5.0], [5.1]])
    together = track_branches(peaks, 4)
    assert together.n_branches == 1
    apart = track_branches(peaks, 4, max_jump=1.0)
    assert apart.n_branches == 2
    assert [branch.length for branch in apart.branches] == [2, 2]
    assert apart.max_jump == pytest.approx(0.1)


def test_a_missing_peak_splits_the_branch() -> None:
    """A k-point with no peak breaks the chain, and the break is reported."""

    peaks = peaks_from_energies([[0.0], [], [0.05]])
    tracking = track_branches(peaks, 3)
    # With no peak in the middle the two ends cannot be linked by a matching --
    # they are separate branches, and the terminus report says where.
    assert tracking.n_branches == 2
    assert [c.kind for c in tracking.crossings] == ["terminus", "terminus"]


def test_min_length_drops_stray_peaks() -> None:
    peaks = peaks_from_energies([[0.0], [0.1, 9.0], [0.2]])
    assert track_branches(peaks, 3).n_branches == 2
    assert track_branches(peaks, 3, min_length=2).n_branches == 1


def test_branch_geometry_and_serialization() -> None:
    peaks = peaks_from_energies([[0.0], [0.4], [1.0]])
    tracking = track_branches(peaks, 3, distances=[0.0, 0.5, 1.0])
    branch = tracking.branches[0]
    assert branch.span == (0, 2)
    assert branch.length == 3
    assert branch.max_jump == pytest.approx(0.6)
    payload = tracking.to_dict()
    assert payload["n_branches"] == 1
    assert payload["max_jump"] == pytest.approx(0.6)
    assert payload["branches"][0]["energies"] == [0.0, 0.4, 1.0]
    assert tracking.energy_table().shape == (1, 3)


def test_arguments_are_validated() -> None:
    peaks = peaks_from_energies([[0.0]])
    with pytest.raises(ValueError, match="n_kpoints"):
        track_branches(peaks, 0)
    with pytest.raises(ValueError, match="max_jump"):
        track_branches(peaks, 1, max_jump=0.0)
    with pytest.raises(ValueError, match="degeneracy_tolerance"):
        track_branches(peaks, 1, degeneracy_tolerance=-1.0)
    with pytest.raises(ValueError, match="min_length"):
        track_branches(peaks, 1, min_length=0)
    with pytest.raises(ValueError, match="one entry per k-point"):
        track_branches(peaks, 1, distances=[0.0, 1.0])
    with pytest.raises(ValueError, match="outside the run"):
        track_branches(peaks_from_energies([[0.0], [1.0]]), 1)


def test_empty_peak_table_gives_no_branches() -> None:
    tracking = track_branches([], 4)
    assert tracking.n_branches == 0
    assert tracking.max_jump == 0.0
    assert tracking.energy_table().shape == (0, 4)


def test_infinite_max_jump_never_forbids_a_pairing() -> None:
    peaks = peaks_from_energies([[0.0], [100.0]])
    tracking = track_branches(peaks, 2, max_jump=math.inf)
    assert tracking.n_branches == 1
    assert tracking.max_jump == pytest.approx(100.0)


def test_cli_branches_reports_and_writes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_path = tmp_path / "run.json"
    out_path = tmp_path / "branches.dat"
    json_path = tmp_path / "branches.json"
    write_ebs(run_path, two_band_structure())
    result = CliRunner().invoke(
        app,
        [
            "branches",
            "--input",
            str(run_path),
            "--broadening",
            "0.05",
            "--points",
            "401",
            "--out",
            str(out_path),
            "--json",
            str(json_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(json_path.read_text())
    assert payload["n_branches"] == 2
    table = np.loadtxt(out_path)
    assert table.shape == (21, 3)


def test_cli_branches_rejects_a_half_given_energy_range(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_path = tmp_path / "run.json"
    write_ebs(run_path, two_band_structure())
    result = CliRunner().invoke(
        app,
        ["branches", "--input", str(run_path), "--broadening", "0.05", "--emax", "1.0"],
    )
    assert result.exit_code != 0
