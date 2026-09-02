"""Layer-resolved unfolding of a plane-wave slab.

The statements checked here are the numerical face of
``RequestProject/Unfolding/Layers.lean``: the layer split of the weight is exact
at a fixed k-point, and summing it over the fiber does *not* return the layer's
charge, because a layer is not diagonal in the plane-wave basis.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from synthetic_wavecar import write_synthetic_wavecar
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.core.layers import (
    density_fourier_coefficients,
    diagnose_layer_conservation,
    layer_charges,
    layer_edges_from_structure,
    layer_resolved_weights,
    planar_average_density,
)
from unfoldlab.core.plane_waves import weights_from_coefficients
from unfoldlab.core.structures import Structure
from unfoldlab.io.vasp_wfc import generate_vasp_g_vectors

runner = CliRunner()

TRANSFORM = np.diag([1, 1, 2]).astype(np.int64)


def _g_vectors(limit: int = 3) -> np.ndarray:
    return np.array([[0, 0, m] for m in range(-limit, limit + 1)], dtype=np.int64)


def _state(g_vectors: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=(2, 1, len(g_vectors))) + 1j * rng.normal(size=(2, 1, len(g_vectors)))
    norms = np.sqrt((np.abs(coeffs) ** 2).sum(axis=(1, 2)))
    return (coeffs / norms[:, None, None]).astype(np.complex128)


def test_density_is_nonnegative_and_normalized() -> None:
    g_vectors = _g_vectors()
    coeffs = _state(g_vectors)
    orders, rho = density_fourier_coefficients(g_vectors, coeffs)

    assert orders[0] == -(2 * 3)
    assert np.allclose(rho[:, orders == 0].real.ravel(), 1.0)

    _, profile = planar_average_density(g_vectors, coeffs, n_points=64)
    assert np.all(profile > -1e-12)
    assert np.allclose(profile.mean(axis=1), 1.0)


def test_density_of_a_single_plane_wave_is_flat() -> None:
    g_vectors = _g_vectors(1)
    coeffs = np.zeros((1, 1, len(g_vectors)), dtype=np.complex128)
    coeffs[0, 0, 2] = 1.0  # G_z = 1
    _, profile = planar_average_density(g_vectors, coeffs, n_points=32)
    assert np.allclose(profile, 1.0)


def test_layer_charges_partition_the_norm() -> None:
    """``UnfoldLab.sum_regionCharge_layers`` with the whole state."""

    g_vectors = _g_vectors()
    coeffs = _state(g_vectors, seed=1)
    charges = layer_charges(g_vectors, coeffs, [0.0, 0.3, 0.65])
    assert charges.shape == (2, 3)
    assert np.all(charges > -1e-12)
    assert np.allclose(charges.sum(axis=1), 1.0)


def test_layer_charge_of_a_localized_state() -> None:
    """A state built to sit in the lower half of the cell is found there."""

    limit = 8
    g_vectors = np.array([[0, 0, m] for m in range(-limit, limit + 1)], dtype=np.int64)
    z = np.linspace(0.0, 1.0, 512, endpoint=False)
    envelope = np.exp(-(((z - 0.25) / 0.05) ** 2))
    coeffs = np.array(
        [[[np.mean(envelope * np.exp(-2j * np.pi * m * z)) for m in range(-limit, limit + 1)]]]
    )
    coeffs = coeffs / np.sqrt((np.abs(coeffs) ** 2).sum())
    charges = layer_charges(g_vectors, coeffs, [0.0, 0.5])
    assert charges[0, 0] > 0.99
    assert charges[0, 1] < 0.01


def test_layer_resolved_weights_split_the_weight() -> None:
    """``UnfoldLab.sum_regionCharge_layers`` for the matched component."""

    g_vectors = _g_vectors()
    coeffs = _state(g_vectors, seed=2)
    primitive = np.array([0.0, 0.0, 0.5])
    folded = np.zeros(3)
    edges = [0.0, 0.25, 0.5, 0.75]

    resolved = layer_resolved_weights(g_vectors, coeffs, primitive, folded, TRANSFORM, edges)
    weight = weights_from_coefficients(g_vectors, coeffs, primitive, folded, TRANSFORM)
    assert resolved.shape == (2, 4)
    assert np.allclose(resolved.sum(axis=1), weight)


def test_layer_weights_are_nonnegative() -> None:
    g_vectors = _g_vectors()
    coeffs = _state(g_vectors, seed=5)
    resolved = layer_resolved_weights(
        g_vectors, coeffs, np.zeros(3), np.zeros(3), TRANSFORM, [0.0, 0.5]
    )
    assert np.all(resolved > -1e-12)


def test_zero_state_gives_zero_weights() -> None:
    g_vectors = _g_vectors(1)
    coeffs = np.zeros((1, 1, len(g_vectors)), dtype=np.complex128)
    resolved = layer_resolved_weights(
        g_vectors, coeffs, np.zeros(3), np.zeros(3), TRANSFORM, [0.0, 0.5]
    )
    assert np.allclose(resolved, 0.0)


def test_k_summed_layer_weights_miss_the_cross_term() -> None:
    """``UnfoldLab.regionCharge_eq_blockSum_add_crossForm``: the gap is real."""

    g_vectors = _g_vectors()
    coeffs = _state(g_vectors, seed=4)
    fiber = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    report = diagnose_layer_conservation(
        g_vectors, coeffs, fiber, np.zeros(3), TRANSFORM, [0.0, 0.5]
    )

    assert report.fiber_complete
    # The total charge is conserved -- it is the *layer* resolution that is not.
    assert np.allclose(report.layer_charges.sum(axis=1), 1.0)
    assert np.allclose(report.summed_weights.sum(axis=1), 1.0)
    assert report.max_abs_cross_term > 1e-3
    assert "cross term" in report.summary()
    assert report.to_dict()["fiber_complete"] is True


def test_a_diagonal_layer_has_no_cross_term() -> None:
    """One layer is the whole cell, which *is* diagonal: then the split is exact.

    The Lean statement is ``sum_regionCharge_blocks_of_diagonal``; the whole cell is
    the simplest region whose projector is the identity, and the k-summed weight
    then returns the charge exactly.
    """

    g_vectors = _g_vectors()
    coeffs = _state(g_vectors, seed=6)
    fiber = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    report = diagnose_layer_conservation(g_vectors, coeffs, fiber, np.zeros(3), TRANSFORM, [0.0])
    assert report.max_abs_cross_term == pytest.approx(0.0, abs=1e-12)


def test_incomplete_fiber_is_flagged() -> None:
    g_vectors = _g_vectors()
    coeffs = _state(g_vectors, seed=8)
    report = diagnose_layer_conservation(
        g_vectors, coeffs, np.array([[0.0, 0.0, 0.0]]), np.zeros(3), TRANSFORM, [0.0, 0.5]
    )
    assert not report.fiber_complete
    assert "complete fiber" in report.summary()


def test_layer_edges_from_structure_separate_the_planes() -> None:
    lattice = np.diag([3.0, 3.0, 10.0])
    frac = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.5]])
    structure = Structure(lattice=lattice, species=("A", "A"), frac_coords=frac)
    edges = layer_edges_from_structure(structure)
    assert edges.shape == (2,)
    assert np.allclose(np.sort(edges), [0.3, 0.8])


def test_bad_inputs_are_rejected() -> None:
    g_vectors = _g_vectors(1)
    coeffs = _state(g_vectors)
    with pytest.raises(ValueError, match="axis must be"):
        layer_charges(g_vectors, coeffs, [0.0], axis=3)
    with pytest.raises(ValueError, match="at least one boundary"):
        layer_charges(g_vectors, coeffs, [])
    with pytest.raises(ValueError, match="distinct"):
        layer_charges(g_vectors, coeffs, [0.25, 1.25])
    with pytest.raises(ValueError, match="n_points must be positive"):
        planar_average_density(g_vectors, coeffs, n_points=0)


def test_cli_layers_reports_the_cross_term(tmp_path) -> None:
    lattice = np.diag([1.0, 1.0, 4.0])
    encut = 200.0
    g_vectors = generate_vasp_g_vectors(lattice, np.zeros(3), encut)
    rng = np.random.default_rng(11)
    coefficients = (
        rng.normal(size=(1, 2, 1, len(g_vectors))) + 1j * rng.normal(size=(1, 2, 1, len(g_vectors)))
    ).astype(np.complex64)
    wavecar = tmp_path / "WAVECAR"
    write_synthetic_wavecar(
        wavecar,
        lattice=lattice,
        encut=encut,
        rtag=45200,
        kpoints=np.zeros((1, 3)),
        energies=np.array([[-1.0, 0.5]]),
        occupations=np.ones((1, 2)),
        coefficients=coefficients,
    )
    out = tmp_path / "layers.json"

    result = runner.invoke(
        app,
        [
            "layers",
            "--wavecar",
            str(wavecar),
            "--matrix",
            "1 0 0 0 1 0 0 0 2",
            "--edges",
            "0.0,0.5",
            "--json",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["fiber_complete"] is True
    assert len(payload["fiber"]) == 2
    assert np.allclose(np.sum(payload["layer_charges"], axis=1), 1.0)
    assert payload["max_abs_cross_term"] > 0.0


def test_cli_layers_needs_edges_or_structure(tmp_path) -> None:
    lattice = np.diag([1.0, 1.0, 4.0])
    encut = 200.0
    g_vectors = generate_vasp_g_vectors(lattice, np.zeros(3), encut)
    coefficients = np.ones((1, 1, 1, len(g_vectors)), dtype=np.complex64)
    wavecar = tmp_path / "WAVECAR"
    write_synthetic_wavecar(
        wavecar,
        lattice=lattice,
        encut=encut,
        rtag=45200,
        kpoints=np.zeros((1, 3)),
        energies=np.array([[0.0]]),
        occupations=np.ones((1, 1)),
        coefficients=coefficients,
    )

    result = runner.invoke(
        app, ["layers", "--wavecar", str(wavecar), "--matrix", "1 0 0 0 1 0 0 0 2"]
    )
    assert result.exit_code != 0
