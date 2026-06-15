import numpy as np

from unfoldlab.core.unfolding import (
    BandUnfoldingData,
    PlaneWaveKPointData,
    compute_plane_wave_unfolding_weights,
)


def test_backend_neutral_plane_wave_weights_match_for_equivalent_inputs():
    transform = np.diag([2, 1, 1])
    common = {
        "primitive_kpoint": np.array([0.0, 0.0, 0.0]),
        "folded_supercell_kpoint": np.array([0.0, 0.0, 0.0]),
        "g_supercell": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        "coefficients": np.array([[[1.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j]]]),
    }
    qe_like = PlaneWaveKPointData(**common)
    vasp_like = PlaneWaveKPointData(**common)

    qe_weights = compute_plane_wave_unfolding_weights([qe_like], transform)
    vasp_weights = compute_plane_wave_unfolding_weights([vasp_like], transform)

    assert np.allclose(qe_weights, vasp_weights)
    assert np.allclose(qe_weights, [[5.0 / 6.0]])


def test_band_unfolding_data_builds_common_ebs_container():
    data = BandUnfoldingData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        energies=np.array([[-1.0, 1.0], [-0.5, 1.5]]),
        weights=np.array([[1.0, 0.25], [0.5, 0.75]]),
        distances=np.array([0.0, 0.5]),
        source_code="qe",
        metadata={"backend": "synthetic"},
    )

    ebs = data.to_effective_band_structure()

    assert ebs.n_kpoints == 2
    assert ebs.metadata["source_code"] == "qe"
    assert ebs.metadata["backend"] == "synthetic"
