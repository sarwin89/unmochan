"""Slabs, wires and other partially periodic cells.

The statements checked here are the numerical face of
``RequestProject/Unfolding/Slab.lean``: a slab fiber is flat, the perpendicular
plane-wave index does not enter the matching test (so it may be summed over
first), and a transform that folds along the vacuum direction invents a
dispersion that is a function of the vacuum thickness alone.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.core.plane_waves import weights_from_coefficients
from unmochan.core.slab import (
    collapse_perpendicular,
    detect_vacuum_axes,
    diagnose_slab_transform,
    embed_parallel_transform,
    flat_kpoint_deviation,
    is_slab_transform,
    slab_weights_from_coefficients,
    vacuum_gap,
)
from unmochan.core.structures import Structure

runner = CliRunner()

SLAB_TRANSFORM = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.int64)


def _slab_structure(vacuum: float = 12.0, thickness: float = 4.0) -> Structure:
    """Two atomic planes separated by ``thickness``, with ``vacuum`` above."""

    height = thickness + vacuum
    lattice = np.diag([3.0, 3.0, height])
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, thickness / height]])
    return Structure(lattice=lattice, species=("A", "A"), frac_coords=frac)


def _plane_wave_state(g_vectors: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.normal(size=(3, 2, len(g_vectors)))
    imag = rng.normal(size=(3, 2, len(g_vectors)))
    return (real + 1j * imag).astype(np.complex128)


def _g_grid(limit: int = 2, perp_limit: int = 4) -> np.ndarray:
    values = range(-limit, limit + 1)
    perp = range(-perp_limit, perp_limit + 1)
    return np.array([[i, j, k] for i in values for j in values for k in perp], dtype=np.int64)


def test_vacuum_gap_measures_the_empty_stretch() -> None:
    structure = _slab_structure(vacuum=12.0, thickness=4.0)
    assert vacuum_gap(structure, 2) == pytest.approx(12.0)
    # In the periodic directions the two atoms sit half a cell apart.
    assert vacuum_gap(structure, 0) == pytest.approx(1.5)


def test_vacuum_gap_is_cyclic() -> None:
    """A slab straddling the cell boundary has the same gap."""

    lattice = np.diag([3.0, 3.0, 16.0])
    frac = np.array([[0.0, 0.0, 0.9], [0.0, 0.0, 0.15]])
    structure = Structure(lattice=lattice, species=("A", "A"), frac_coords=frac)
    assert vacuum_gap(structure, 2) == pytest.approx(12.0)


def test_detect_vacuum_axes_finds_the_stacking_direction() -> None:
    assert detect_vacuum_axes(_slab_structure()) == (2,)


def test_detect_vacuum_axes_is_empty_for_a_bulk_cell() -> None:
    bulk = Structure(
        lattice=np.diag([3.0, 3.0, 3.0]),
        species=("A",),
        frac_coords=np.zeros((1, 3)),
    )
    assert detect_vacuum_axes(bulk) == ()


def test_is_slab_transform_accepts_a_parallel_supercell() -> None:
    assert is_slab_transform(SLAB_TRANSFORM, 2)


def test_is_slab_transform_rejects_folding_along_the_vacuum() -> None:
    assert not is_slab_transform(np.diag([2, 2, 2]), 2)


def test_is_slab_transform_rejects_mixing_with_the_vacuum_axis() -> None:
    mixed = np.array([[2, 0, 0], [0, 2, 0], [1, 0, 1]], dtype=np.int64)
    assert not is_slab_transform(mixed, 2)


def test_embed_parallel_transform_is_a_slab_transform() -> None:
    matrix = embed_parallel_transform([[2, 1], [0, 3]], 1)
    assert is_slab_transform(matrix, 1)
    assert matrix.tolist() == [[2, 0, 1], [0, 1, 0], [0, 0, 3]]


def test_diagnose_slab_transform_is_valid_for_a_parallel_supercell() -> None:
    report = diagnose_slab_transform(SLAB_TRANSFORM, 2)
    assert report.valid
    assert report.spurious_kpoints == 1
    assert report.mixed_axes == ()
    assert "valid slab transform" in report.summary()
    assert report.to_dict()["valid"] is True


def test_diagnose_slab_transform_counts_spurious_kpoints() -> None:
    report = diagnose_slab_transform(np.diag([2, 2, 3]), 2)
    assert not report.valid
    assert report.folding_factors == (3,)
    assert report.spurious_kpoints == 3
    assert "3-fold" in report.summary()


def test_diagnose_slab_transform_reports_mixing() -> None:
    mixed = np.array([[2, 0, 0], [0, 2, 0], [1, 0, 1]], dtype=np.int64)
    report = diagnose_slab_transform(mixed, 2)
    assert report.mixed_axes == (2,)
    assert not report.valid
    assert "mixed with a periodic direction" in report.summary()


def test_diagnose_slab_transform_of_a_bulk_cell_is_vacuous() -> None:
    report = diagnose_slab_transform(np.diag([2, 2, 2]), [])
    assert report.valid
    assert "fully periodic" in report.summary()


def test_flat_kpoint_deviation_vanishes_on_a_flat_path() -> None:
    kpoints = np.array([[0.0, 0.0, 0.25], [0.5, 0.5, 1.25], [0.25, 0.0, -0.75]])
    assert flat_kpoint_deviation(kpoints, [0.0, 0.0, 0.25], 2) == pytest.approx(0.0)


def test_flat_kpoint_deviation_detects_vacuum_dispersion() -> None:
    kpoints = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    assert flat_kpoint_deviation(kpoints, [0.0, 0.0, 0.0], 2) == pytest.approx(0.5)


def test_collapse_perpendicular_conserves_the_norm() -> None:
    g_vectors = _g_grid()
    coeffs = _plane_wave_state(g_vectors)
    reduced_g, reduced = collapse_perpendicular(g_vectors, coeffs, 2)

    assert reduced_g.shape[0] == 25  # 5 x 5 parallel vectors, 9 shells collapsed
    assert np.all(reduced_g[:, 2] == 0)
    full = (np.abs(coeffs) ** 2).sum(axis=(1, 2))
    assert np.allclose((reduced**2).sum(axis=(1, 2)), full)


def test_collapse_perpendicular_matches_the_full_weights() -> None:
    """The optimization is exact, not an approximation."""

    g_vectors = _g_grid()
    coeffs = _plane_wave_state(g_vectors, seed=3)
    primitive = np.array([0.25, 0.0, 0.1])
    folded = primitive @ SLAB_TRANSFORM.T

    reference = weights_from_coefficients(g_vectors, coeffs, primitive, folded, SLAB_TRANSFORM)
    collapsed = slab_weights_from_coefficients(
        g_vectors, coeffs, primitive, folded, SLAB_TRANSFORM, 2
    )
    assert np.allclose(collapsed, reference)


def test_slab_weights_obey_the_fiber_sum_rule() -> None:
    g_vectors = _g_grid()
    coeffs = _plane_wave_state(g_vectors, seed=7)
    folded = np.array([0.0, 0.0, 0.2])
    fiber = [
        np.array([shift_x / 2.0, shift_y / 2.0, 0.2]) for shift_x in (0, 1) for shift_y in (0, 1)
    ]
    total = sum(
        slab_weights_from_coefficients(g_vectors, coeffs, k, folded, SLAB_TRANSFORM, 2)
        for k in fiber
    )
    assert np.allclose(total, 1.0)


def test_slab_weights_reject_a_vacuum_folding_transform() -> None:
    g_vectors = _g_grid(limit=1, perp_limit=1)
    coeffs = _plane_wave_state(g_vectors)
    with pytest.raises(ValueError, match="not trivial along the non-periodic axes"):
        slab_weights_from_coefficients(
            g_vectors, coeffs, np.zeros(3), np.zeros(3), np.diag([2, 2, 2]), 2
        )


def test_vacuum_folding_invents_a_dispersion() -> None:
    """The pitfall of ``UnfoldLab.weight_vacuum_fold_half``, in three dimensions.

    A single plane wave along the vacuum direction is handed all of its weight
    at ``k_perp = 1/2`` by a transform that doubles the cell there -- a "band"
    whose only parameter is the vacuum thickness.
    """

    g_vectors = np.array([[0, 0, 1]], dtype=np.int64)
    coeffs = np.ones((1, 1, 1), dtype=np.complex128)
    transform = np.diag([1, 1, 2])
    at_half = weights_from_coefficients(
        g_vectors, coeffs, np.array([0.0, 0.0, 0.5]), np.zeros(3), transform
    )
    at_zero = weights_from_coefficients(g_vectors, coeffs, np.zeros(3), np.zeros(3), transform)
    assert at_half == pytest.approx(1.0)
    assert at_zero == pytest.approx(0.0)


def test_axes_must_be_valid() -> None:
    with pytest.raises(ValueError, match="must be 0, 1 or 2"):
        is_slab_transform(SLAB_TRANSFORM, 3)
    with pytest.raises(ValueError, match="distinct"):
        is_slab_transform(SLAB_TRANSFORM, [2, 2])


def _poscar(lattice: np.ndarray, frac: np.ndarray) -> str:
    rows = [" ".join(f"{value:.12f}" for value in row) for row in lattice]
    sites = [" ".join(f"{value:.12f}" for value in row) for row in frac]
    header = ["slab", "1.0", *rows, "X", str(len(frac)), "Direct"]
    return "\n".join([*header, *sites, ""])


def test_cli_slab_reports_a_valid_transform(tmp_path) -> None:
    structure = _slab_structure()
    path = tmp_path / "POSCAR"
    path.write_text(_poscar(structure.lattice, structure.frac_coords))
    out = tmp_path / "slab.json"

    result = runner.invoke(
        app,
        [
            "slab",
            "--structure",
            str(path),
            "--matrix",
            "2 0 0 0 2 0 0 0 1",
            "--json",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["valid"] is True
    assert payload["non_periodic_axes"] == [2]
    assert payload["vacuum_gaps"][2] == pytest.approx(12.0)


def test_cli_slab_rejects_vacuum_folding(tmp_path) -> None:
    structure = _slab_structure()
    path = tmp_path / "POSCAR"
    path.write_text(_poscar(structure.lattice, structure.frac_coords))

    result = runner.invoke(
        app,
        ["slab", "--structure", str(path), "--matrix", "2 0 0 0 2 0 0 0 2"],
    )

    assert result.exit_code == 0, result.output
    assert "INVALID slab transform" in result.output
