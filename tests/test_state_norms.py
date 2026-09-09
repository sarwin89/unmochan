"""Norms of the stored plane-wave states, and what they bound.

The unfolding weight is a ratio of the matched norm to the total norm, so it
does not depend on the normalization of the stored state at all
(``UnfoldLab.weight_smul``).  What the stored norm does say is how *complete*
the file is, and the two bounds proved in ``RequestProject/Unfolding/
Normalization.lean`` -- for a truncated and for a perturbed expansion -- are
checked numerically here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from synthetic_wavecar import write_synthetic_wavecar
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.core.plane_waves import (
    matching_g_mask,
    state_norms_from_coefficients,
    weights_from_coefficients,
)
from unmochan.core.unfolding import (
    StateNormDiagnostics,
    diagnose_state_norms,
    truncation_weight_error_bound,
)
from unmochan.io.qe_wfc import state_norms_from_qe_save
from unmochan.io.vasp_wfc import (
    generate_vasp_g_vectors,
    state_norms_from_wavecar,
)

runner = CliRunner()

LATTICE = np.diag([4.0, 3.0, 5.0])
ENCUT = 120.0
KPOINTS = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
N_BANDS = 4
TRANSFORM = np.diag([2, 1, 1])


def test_state_norms_are_the_sum_of_the_squared_moduli():
    coefficients = np.array([[[1.0 + 0.0j, 0.0 + 2.0j]], [[3.0 + 4.0j, 0.0 + 0.0j]]])

    norms = state_norms_from_coefficients(coefficients)

    assert np.allclose(norms, [5.0, 25.0])


def test_state_norms_sum_over_spinor_components():
    coefficients = np.array([[[1.0 + 0.0j], [0.0 + 1.0j]]])

    assert np.allclose(state_norms_from_coefficients(coefficients), [2.0])


def test_state_norms_reject_a_wrongly_shaped_array():
    with pytest.raises(ValueError, match="n_components"):
        state_norms_from_coefficients(np.zeros((2, 3)))


def test_diagnostics_summarize_a_norm_table():
    report = diagnose_state_norms(np.array([[1.0, 0.9], [1.0, 0.0]]))

    assert isinstance(report, StateNormDiagnostics)
    assert report.n_kpoints == 2
    assert report.n_bands == 2
    assert report.min_norm == 0.0
    assert report.max_norm == 1.0
    assert report.n_zero_states == 1
    assert report.max_missing_fraction == 1.0
    assert report.weight_error_bound == float("inf")
    assert report.to_dict()["mean_norm"] == pytest.approx(0.725)


def test_diagnostics_reject_a_wrongly_shaped_or_negative_table():
    with pytest.raises(ValueError, match="n_kpoints"):
        diagnose_state_norms(np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="at least one state"):
        diagnose_state_norms(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        diagnose_state_norms(np.array([[-1.0]]))


def test_truncation_bound_is_delta_over_one_minus_delta():
    assert truncation_weight_error_bound(0.0) == 0.0
    assert truncation_weight_error_bound(0.5) == pytest.approx(1.0)
    assert truncation_weight_error_bound(0.02) == pytest.approx(0.02 / 0.98)
    assert truncation_weight_error_bound(1.0) == float("inf")
    with pytest.raises(ValueError):
        truncation_weight_error_bound(-0.1)


def _random_state(rng, n_g: int) -> np.ndarray:
    block = rng.normal(size=(1, 1, n_g)) + 1j * rng.normal(size=(1, 1, n_g))
    return block / np.sqrt(state_norms_from_coefficients(block)[0])


def test_weight_is_invariant_under_rescaling_the_state():
    """``UnfoldLab.weight_smul``: only the ray of the state matters."""

    rng = np.random.default_rng(11)
    g_vectors = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    coefficients = _random_state(rng, len(g_vectors))
    k_pc = np.array([0.25, 0.0, 0.0])
    k_sc = np.array([0.5, 0.0, 0.0])

    reference = weights_from_coefficients(g_vectors, coefficients, k_pc, k_sc, TRANSFORM)
    for scale in (1e-8, 3.0, -2.0, 1.0j, 5.0 - 1.0j):
        scaled = weights_from_coefficients(g_vectors, scale * coefficients, k_pc, k_sc, TRANSFORM)
        assert np.allclose(scaled, reference)


def test_truncated_expansion_stays_inside_the_proved_bound():
    """``UnfoldLab.weight_truncation_bound`` checked on random states."""

    rng = np.random.default_rng(20240918)
    g_vectors = np.array([[i, 0, 0] for i in range(12)])
    k_pc = np.array([0.25, 0.0, 0.0])
    k_sc = np.array([0.5, 0.0, 0.0])
    worst_ratio = 0.0
    for _ in range(50):
        coefficients = _random_state(rng, len(g_vectors))
        keep = rng.random(len(g_vectors)) > 0.25
        if not keep.any():
            continue
        full = weights_from_coefficients(g_vectors, coefficients, k_pc, k_sc, TRANSFORM)[0]
        truncated = weights_from_coefficients(
            g_vectors[keep], coefficients[:, :, keep], k_pc, k_sc, TRANSFORM
        )[0]
        delta = 1.0 - float(state_norms_from_coefficients(coefficients[:, :, keep])[0])
        bound = truncation_weight_error_bound(max(delta, 0.0))
        assert abs(truncated - full) <= bound + 1e-12
        if bound > 0:
            worst_ratio = max(worst_ratio, abs(truncated - full) / bound)
    # The bound is a worst case, so it must not be vacuous either: some draw
    # should come within an order of magnitude of it.
    assert worst_ratio > 0.05


def test_perturbed_state_stays_inside_the_proved_bound():
    """``UnfoldLab.weight_perturbation_bound``: ``|dw| <= 2 eps``."""

    rng = np.random.default_rng(777)
    g_vectors = np.array([[i, 0, 0] for i in range(10)])
    k_pc = np.array([0.25, 0.0, 0.0])
    k_sc = np.array([0.5, 0.0, 0.0])
    for _ in range(50):
        first = _random_state(rng, len(g_vectors))
        noise = rng.normal(size=first.shape) + 1j * rng.normal(size=first.shape)
        second = first + 0.1 * noise
        second = second / np.sqrt(state_norms_from_coefficients(second)[0])
        eps = float(np.sqrt(np.sum(np.abs(first - second) ** 2)))
        w1 = weights_from_coefficients(g_vectors, first, k_pc, k_sc, TRANSFORM)[0]
        w2 = weights_from_coefficients(g_vectors, second, k_pc, k_sc, TRANSFORM)[0]
        assert abs(w1 - w2) <= 2.0 * eps + 1e-12


def test_a_state_with_no_plane_wave_content_gets_a_zero_weight():
    g_vectors = np.array([[0, 0, 0], [1, 0, 0]])
    coefficients = np.zeros((1, 1, 2), dtype=complex)

    weights = weights_from_coefficients(
        g_vectors,
        coefficients,
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        TRANSFORM,
    )

    assert np.allclose(weights, 0.0)
    assert diagnose_state_norms(np.zeros((1, 1))).n_zero_states == 1


def test_mask_and_norms_reproduce_the_weight_by_hand():
    g_vectors = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    coefficients = np.array([[[1.0 + 0.0j, 2.0 + 0.0j, 0.0 + 1.0j]]])
    k_pc = np.array([0.0, 0.0, 0.0])
    k_sc = np.array([0.0, 0.0, 0.0])

    mask = matching_g_mask(g_vectors, k_pc, k_sc, TRANSFORM)
    matched = float(np.sum(np.abs(coefficients[0, 0, mask]) ** 2))
    total = float(state_norms_from_coefficients(coefficients)[0])

    weights = weights_from_coefficients(g_vectors, coefficients, k_pc, k_sc, TRANSFORM)
    assert weights[0] == pytest.approx(matched / total)


def _write_wavecar(path: Path, *, scale: float = 1.0) -> np.ndarray:
    """A two-k-point WAVECAR of normalized states scaled by ``scale``."""

    rng = np.random.default_rng(4242)
    blocks = []
    for kpoint in KPOINTS:
        g_vectors = generate_vasp_g_vectors(LATTICE, kpoint, ENCUT)
        shape = (N_BANDS, 1, len(g_vectors))
        block = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        block /= np.sqrt(state_norms_from_coefficients(block))[:, None, None]
        blocks.append((scale * block).astype(np.complex64))
    write_synthetic_wavecar(
        path,
        lattice=LATTICE,
        encut=ENCUT,
        rtag=45200,
        kpoints=KPOINTS,
        energies=np.zeros((KPOINTS.shape[0], N_BANDS)),
        occupations=np.ones((KPOINTS.shape[0], N_BANDS)),
        coefficients=blocks,
        record_length=4096,
    )
    return np.stack([state_norms_from_coefficients(block) for block in blocks])


def test_wavecar_state_norms_match_the_written_states(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR"
    expected = _write_wavecar(wavecar)

    norms = state_norms_from_wavecar(wavecar)

    assert norms.shape == (len(KPOINTS), N_BANDS)
    assert np.allclose(norms, expected, atol=1e-6)
    assert np.allclose(norms, 1.0, atol=1e-6)


@pytest.mark.parametrize("band_chunk", [1, 2, N_BANDS])
def test_wavecar_state_norms_do_not_depend_on_the_band_chunk(tmp_path: Path, band_chunk: int):
    wavecar = tmp_path / "WAVECAR"
    _write_wavecar(wavecar)

    whole = state_norms_from_wavecar(wavecar)
    chunked = state_norms_from_wavecar(wavecar, band_chunk=band_chunk)

    assert np.allclose(whole, chunked)


def test_wavecar_state_norms_reject_a_non_positive_band_chunk(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR"
    _write_wavecar(wavecar)

    with pytest.raises(ValueError, match="band_chunk"):
        state_norms_from_wavecar(wavecar, band_chunk=0)


def test_an_unnormalized_file_is_flagged_but_gives_the_same_weights(tmp_path: Path):
    """A different normalization convention changes the norms, not the weights."""

    plain = tmp_path / "WAVECAR.plain"
    scaled = tmp_path / "WAVECAR.scaled"
    _write_wavecar(plain)
    _write_wavecar(scaled, scale=0.5)

    from unmochan.io.vasp_wfc import compute_weights_from_wavecar

    primitive = np.array([[0.0, 0.0, 0.0], [0.125, 0.0, 0.0]])
    args = (primitive, KPOINTS, TRANSFORM)
    _k1, _e1, w_plain = compute_weights_from_wavecar(plain, *args)
    _k2, _e2, w_scaled = compute_weights_from_wavecar(scaled, *args)

    assert np.allclose(w_plain, w_scaled, atol=1e-6)
    assert diagnose_state_norms(state_norms_from_wavecar(plain)).min_norm == pytest.approx(
        1.0, abs=1e-6
    )
    assert diagnose_state_norms(state_norms_from_wavecar(scaled)).min_norm == pytest.approx(
        0.25, abs=1e-6
    )


def _write_qe_hdf5(save_dir: Path) -> None:
    h5py = pytest.importorskip("h5py")
    save_dir.mkdir()
    with h5py.File(save_dir / "wfc1.hdf5", "w") as handle:
        handle.attrs["ik"] = 1
        handle.attrs["ispin"] = 1
        handle.attrs["gamma_only"] = ".FALSE."
        handle.attrs["npol"] = 1
        handle.create_dataset("MillerIndices", data=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
        handle.create_dataset(
            "evc",
            data=np.array(
                [
                    [1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
        )


def test_qe_state_norms_from_a_save_directory(tmp_path: Path):
    save_dir = tmp_path / "qe.save"
    _write_qe_hdf5(save_dir)

    norms = state_norms_from_qe_save(save_dir, 2, file_format="hdf5")

    assert norms.shape == (1, 2)
    assert np.allclose(norms, [[6.0, 1.0]])
    report = diagnose_state_norms(norms)
    assert report.n_zero_states == 0
    assert report.max_missing_fraction == 0.0


def test_qe_state_norms_reject_a_non_positive_band_chunk(tmp_path: Path):
    save_dir = tmp_path / "qe.save"
    _write_qe_hdf5(save_dir)

    with pytest.raises(ValueError, match="band_chunk"):
        state_norms_from_qe_save(save_dir, 2, file_format="hdf5", band_chunk=0)


def test_norms_cli_reports_a_wavecar_and_writes_json(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR"
    _write_wavecar(wavecar)
    report = tmp_path / "norms.json"

    result = runner.invoke(app, ["norms", "--wavecar", str(wavecar), "--report", str(report)])

    assert result.exit_code == 0, result.output
    assert "Stored plane-wave norms" in result.output
    payload = json.loads(report.read_text())
    assert payload["summary"]["n_bands"] == N_BANDS
    assert payload["summary"]["min_norm"] == pytest.approx(1.0, abs=1e-6)
    assert len(payload["norms"]) == len(KPOINTS)
    # The PAW bound of `UnfoldLab.abs_augWeight_sub_weight_le` is reported next
    # to the truncation bound: for these norm-one states it is negligible.
    assert payload["augmentation"]["negligible"] is True
    assert payload["augmentation"]["weight_error_bound"] == pytest.approx(0.0, abs=1e-6)
    assert "PAW weight error bound" in result.output


def test_norms_cli_needs_exactly_one_source(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR"
    _write_wavecar(wavecar)

    empty = runner.invoke(app, ["norms"])
    both = runner.invoke(app, ["norms", "--wavecar", str(wavecar), "--qe-save-dir", str(tmp_path)])

    assert empty.exit_code != 0
    assert both.exit_code != 0


def test_norms_cli_rejects_too_many_bands(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR"
    _write_wavecar(wavecar)

    result = runner.invoke(app, ["norms", "--wavecar", str(wavecar), "--nbnd", str(N_BANDS + 3)])

    assert result.exit_code != 0
