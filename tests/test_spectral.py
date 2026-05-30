import numpy as np

from unfoldlab.core.spectral import EffectiveBandStructure


def test_spectral_function_shape_and_positive_intensity():
    ebs = EffectiveBandStructure(
        kpoints=[[0, 0, 0], [0.5, 0, 0]],
        energies=[[0.0, 1.0], [0.2, 1.2]],
        weights=[[1.0, 0.5], [0.8, 0.2]],
    )
    grid = np.linspace(-1, 2, 50)

    spectral = ebs.spectral_function(grid, broadening=0.1)

    assert spectral.shape == (2, 50)
    assert np.all(spectral >= 0)
    assert spectral[0].max() > spectral[1].min()


def test_effective_band_structure_defaults_unit_weights_and_distances():
    ebs = EffectiveBandStructure(
        kpoints=[[0, 0, 0]],
        energies=[[-1.0, 1.0]],
    )

    assert np.allclose(ebs.weights, [[1.0, 1.0]])
    assert np.allclose(ebs.distances, [0.0])
