"""Degeneracy grouping and gauge invariance of unfolded weights."""

from __future__ import annotations

import json

import numpy as np
import pytest

from unmochan.cli.main import main
from unmochan.core.degeneracy import (
    DegeneracyReport,
    average_degenerate_weights,
    degeneracy_averaged_weights,
    degeneracy_report,
    group_degenerate_bands,
    subspace_weights,
)
from unmochan.core.plane_waves import weights_from_coefficients
from unmochan.core.spectral import EffectiveBandStructure
from unmochan.io.serialization import read_ebs, write_ebs


def test_group_degenerate_bands_partitions_indices() -> None:
    energies = [0.0, 1.0, 1.00001, 3.0, 3.00002, 3.00003]
    groups = group_degenerate_bands(energies, tol=1e-4)
    assert groups == [[0], [1, 2], [3, 4, 5]]
    flattened = sorted(index for group in groups for index in group)
    assert flattened == list(range(len(energies)))


def test_group_degenerate_bands_is_order_independent() -> None:
    energies = np.array([3.0, 0.0, 1.00001, 3.00002, 1.0, 3.00003])
    groups = group_degenerate_bands(energies, tol=1e-4)
    assert groups == [[1], [2, 4], [0, 3, 5]]


def test_zero_tolerance_groups_only_exact_duplicates() -> None:
    energies = [1.0, 1.0, 1.0 + 1e-12]
    assert group_degenerate_bands(energies, tol=0.0) == [[0, 1], [2]]


def test_group_degenerate_bands_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        group_degenerate_bands(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        group_degenerate_bands([0.0, np.nan])
    with pytest.raises(ValueError, match="non-negative"):
        group_degenerate_bands([0.0, 1.0], tol=-1.0)


def test_averaging_preserves_the_total_weight() -> None:
    energies = np.array([[0.0, 1.0, 1.0, 2.0]])
    weights = np.array([[0.2, 0.9, 0.1, 0.5]])
    averaged = degeneracy_averaged_weights(energies, weights, tol=1e-6)
    assert averaged == pytest.approx(np.array([[0.2, 0.5, 0.5, 0.5]]))
    assert averaged.sum() == pytest.approx(weights.sum())


def test_averaging_accepts_a_single_kpoint_row() -> None:
    averaged = degeneracy_averaged_weights([0.0, 0.0], [1.0, 0.0], tol=1e-6)
    assert averaged.shape == (2,)
    assert averaged == pytest.approx([0.5, 0.5])


def test_subspace_weights_collapse_the_band_axis() -> None:
    energies = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 5.0]])
    weights = np.array([[0.25, 0.5, 0.25], [0.1, 0.2, 0.7]])
    groups = subspace_weights(energies, weights, tol=1e-6)
    assert groups[0] == [(0.0, 1, pytest.approx(0.25)), (1.0, 2, pytest.approx(0.75))]
    assert groups[1] == [(0.0, 2, pytest.approx(0.30000000000000004)), (5.0, 1, pytest.approx(0.7))]


def test_report_measures_gauge_sensitivity() -> None:
    energies = np.array([[0.0, 1.0, 1.0, 4.0]])
    weights = np.array([[1.0, 0.8, 0.2, 0.3]])
    report = degeneracy_report(energies, weights, tol=1e-6)
    assert isinstance(report, DegeneracyReport)
    assert report.n_groups == 3
    assert report.n_degenerate_groups == 1
    assert report.max_multiplicity == 2
    assert report.gauge_sensitive
    # 0.8 and 0.2 average to 0.5, so either band could be off by 0.3.
    assert report.max_spread == pytest.approx(0.3)
    assert report.max_group_width == pytest.approx(0.0)


def test_report_on_a_nondegenerate_structure_is_clean() -> None:
    energies = np.array([[0.0, 1.0, 2.0]])
    weights = np.array([[0.5, 0.25, 0.25]])
    report = degeneracy_report(energies, weights, tol=1e-6)
    assert not report.gauge_sensitive
    assert report.max_spread == 0.0
    assert report.to_dict()["n_groups"] == 3


def _random_unitary(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    q, r = np.linalg.qr(matrix)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def test_individual_weights_are_gauge_dependent_but_the_sum_is_not() -> None:
    """The numerical counterpart of ``UnfoldLab.subspaceWeight_mix``."""

    transform = np.diag([2, 1, 1])
    g_vectors = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [-1, 0, 0], [-2, 0, 0]],
        dtype=int,
    )
    primitive = np.array([0.25, 0.0, 0.0])
    folded = primitive @ transform.T

    rng = np.random.default_rng(20240514)
    n_bands = 3
    coefficients = rng.normal(size=(n_bands, 1, len(g_vectors))) + 1j * rng.normal(
        size=(n_bands, 1, len(g_vectors))
    )
    # Orthonormalize the multiplet, as a diagonalizer would.
    flat = coefficients.reshape(n_bands, -1)
    q, _ = np.linalg.qr(flat.T)
    coefficients = q.T.reshape(n_bands, 1, len(g_vectors))

    unitary = _random_unitary(n_bands, seed=7)
    mixed = np.einsum("jl,lcg->jcg", unitary, coefficients)

    kwargs = dict(
        primitive_kpoint=primitive,
        folded_supercell_kpoint=folded,
        transform=transform.astype(float),
    )
    original = weights_from_coefficients(g_vectors, coefficients, **kwargs)
    rotated = weights_from_coefficients(g_vectors, mixed, **kwargs)

    # The mixing is a genuine gauge change: the individual weights move.
    assert not np.allclose(original, rotated, atol=1e-6)
    # But the subspace weight does not.
    assert original.sum() == pytest.approx(rotated.sum(), abs=1e-10)

    energies = np.zeros((1, n_bands))
    averaged_original = degeneracy_averaged_weights(energies, original[np.newaxis, :])
    averaged_rotated = degeneracy_averaged_weights(energies, rotated[np.newaxis, :])
    assert averaged_original == pytest.approx(averaged_rotated, abs=1e-10)


def test_gauge_change_preserves_the_fiber_sum_rule() -> None:
    """Averaging cannot break the sum rule, since it preserves group totals."""

    transform = np.diag([3, 1, 1])
    g_vectors = np.array([[m, 0, 0] for m in range(-4, 5)], dtype=int)
    rng = np.random.default_rng(99)
    n_bands = 2
    coefficients = rng.normal(size=(n_bands, 1, len(g_vectors))) + 1j * rng.normal(
        size=(n_bands, 1, len(g_vectors))
    )
    fiber = [np.array([x, 0.0, 0.0]) for x in (0.0, 1.0 / 3.0, 2.0 / 3.0)]
    folded = np.zeros(3)

    per_member = np.array(
        [
            weights_from_coefficients(
                g_vectors,
                coefficients,
                primitive_kpoint=k,
                folded_supercell_kpoint=folded,
                transform=transform.astype(float),
            )
            for k in fiber
        ]
    )
    assert per_member.sum(axis=0) == pytest.approx(np.ones(n_bands))

    energies = np.zeros((len(fiber), n_bands))
    averaged = degeneracy_averaged_weights(energies, per_member)
    assert averaged.sum(axis=0) == pytest.approx(np.ones(n_bands))


def test_average_degenerate_weights_on_an_effective_band_structure() -> None:
    structure = EffectiveBandStructure(
        kpoints=np.zeros((2, 3)),
        energies=np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 2.0]]),
        weights=np.array([[0.9, 0.1, 0.4], [0.2, 0.7, 0.3]]),
    )
    averaged = average_degenerate_weights(structure, tol=1e-6)
    assert averaged.weights == pytest.approx(np.array([[0.5, 0.5, 0.4], [0.2, 0.5, 0.5]]))
    assert averaged.energies == pytest.approx(structure.energies)
    assert averaged.metadata["degeneracy_tol"] == pytest.approx(1e-6)
    # The input is untouched.
    assert structure.weights == pytest.approx(np.array([[0.9, 0.1, 0.4], [0.2, 0.7, 0.3]]))


class TestDegeneracyCLI:
    """`unmochan degeneracy` on a serialized effective band structure."""

    @staticmethod
    def _write(path, energies, weights):
        energies = np.asarray(energies, dtype=float)
        kpoints = np.zeros((energies.shape[0], 3))
        kpoints[:, 0] = np.linspace(0.0, 0.5, energies.shape[0])
        write_ebs(
            path,
            EffectiveBandStructure(kpoints=kpoints, energies=energies, weights=weights),
        )

    def test_cli_averages_a_multiplet(self, tmp_path) -> None:
        path = tmp_path / "ebs.json"
        self._write(path, [[0.0, 0.0, 1.0]], [[0.9, 0.1, 0.5]])
        out = tmp_path / "averaged.json"
        report = tmp_path / "report.json"
        code = main(
            [
                "degeneracy",
                "--input",
                str(path),
                "--tol",
                "1e-6",
                "--out",
                str(out),
                "--report",
                str(report),
            ]
        )
        assert code == 0

        averaged, _ = read_ebs(out)
        assert averaged.weights == pytest.approx(np.array([[0.5, 0.5, 0.5]]))
        assert averaged.energies == pytest.approx(np.array([[0.0, 0.0, 1.0]]))

        payload = json.loads(report.read_text())
        assert payload["summary"]["max_multiplicity"] == 2
        assert payload["summary"]["max_spread"] == pytest.approx(0.4)
        assert payload["multiplets"][0][0]["multiplicity"] == 2
        assert payload["multiplets"][0][0]["weight"] == pytest.approx(1.0)

    def test_cli_reports_a_clean_structure(self, tmp_path) -> None:
        path = tmp_path / "ebs.json"
        self._write(path, [[0.0, 1.0]], [[0.4, 0.6]])
        assert main(["degeneracy", "--input", str(path), "--tol", "1e-6"]) == 0

    def test_cli_rejects_a_negative_tolerance(self, tmp_path) -> None:
        path = tmp_path / "ebs.json"
        self._write(path, [[0.0, 1.0]], [[0.4, 0.6]])
        assert main(["degeneracy", "--input", str(path), "--tol", "-1"]) != 0
