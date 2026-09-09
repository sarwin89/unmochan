import numpy as np
import pytest

from unmochan.core.unfolding import (
    BandUnfoldingData,
    PlaneWaveKPointData,
    compare_effective_band_structures,
    compute_plane_wave_unfolding_weights,
    diagnose_weights,
)

pytestmark = [pytest.mark.unit, pytest.mark.parity]


def test_backend_neutral_plane_wave_data_gives_identical_vasp_qe_weights():
    transform = np.diag([2, 1, 1])
    g_vectors = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    coefficients = np.array(
        [
            [[1.0, 1.0, 2.0, 0.0]],
            [[0.0, 1.0, 0.0, 1.0]],
        ],
        dtype=np.complex128,
    )
    primitive = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    folded = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    qe_like = [
        PlaneWaveKPointData(k_pc, k_sc, g_vectors, coefficients)
        for k_pc, k_sc in zip(primitive, folded, strict=True)
    ]
    vasp_like = [
        PlaneWaveKPointData(k_pc, k_sc, g_vectors.copy(), coefficients.copy())
        for k_pc, k_sc in zip(primitive, folded, strict=True)
    ]

    qe_weights = compute_plane_wave_unfolding_weights(qe_like, transform)
    vasp_weights = compute_plane_wave_unfolding_weights(vasp_like, transform)

    assert np.allclose(qe_weights, vasp_weights)
    assert np.allclose(qe_weights, [[5.0 / 6.0, 0.0], [1.0 / 6.0, 1.0]])


def test_spinor_components_are_summed_before_normalization():
    transform = np.diag([2, 1, 1])
    g_vectors = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    coefficients = np.array([[[1.0, 1.0, 1.0], [2.0, 0.0, 2.0]]], dtype=np.complex128)
    data = [
        PlaneWaveKPointData(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            g_vectors,
            coefficients,
        )
    ]

    weights = compute_plane_wave_unfolding_weights(data, transform)

    assert np.allclose(weights, [[10.0 / 11.0]])


def test_zero_norm_band_is_reported_as_zero_weight_not_nan():
    data = [
        PlaneWaveKPointData(
            np.zeros(3),
            np.zeros(3),
            np.array([[0, 0, 0], [1, 0, 0]]),
            np.zeros((1, 1, 2), dtype=np.complex128),
        )
    ]

    weights = compute_plane_wave_unfolding_weights(data, np.diag([2, 1, 1]))

    assert np.allclose(weights, [[0.0]])
    assert np.isfinite(weights).all()


def test_effective_band_structure_comparison_is_backend_parity_contract():
    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    distances = np.array([0.0, 0.5])
    energies = np.array([[-1.0, 1.0], [-0.5, 1.5]])
    weights = np.array([[1.0, 0.25], [0.5, 0.75]])

    qe_ebs = BandUnfoldingData(
        kpoints=kpoints,
        distances=distances,
        energies=energies,
        weights=weights,
        source_code="qe",
        metadata={"weight_mode": "synthetic"},
    ).to_effective_band_structure()
    vasp_ebs = BandUnfoldingData(
        kpoints=kpoints.copy(),
        distances=distances.copy(),
        energies=energies.copy(),
        weights=weights.copy(),
        source_code="vasp",
        metadata={"weight_mode": "synthetic"},
    ).to_effective_band_structure()

    report = compare_effective_band_structures(qe_ebs, vasp_ebs)

    assert report["shape_match"] is True
    assert report["kpoints_close"] is True
    assert report["energies_close"] is True
    assert report["weights_close"] is True
    assert report["max_weight_delta"] == 0.0


def test_weight_diagnostics_catches_bounds_and_shape():
    diagnostics = diagnose_weights(np.array([[0.0, 1.0], [0.25, 0.75]]))

    assert diagnostics.to_dict() == {
        "min_weight": 0.0,
        "max_weight": 1.0,
        "out_of_bounds_count": 0,
        "max_band_sum": 1.0,
        "n_kpoints": 2,
        "n_bands": 2,
    }

    with pytest.raises(ValueError, match="shape"):
        diagnose_weights(np.array([0.0, 1.0]))
