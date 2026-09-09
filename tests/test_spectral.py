import numpy as np

from unmochan.core.spectral import EffectiveBandStructure


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


def test_gaussian_spectral_function_conserves_weight():
    """``∫ A(k, E) dE == Σ_n w_kn`` (``UnfoldLab.integral_spectralFunction``)."""

    ebs = EffectiveBandStructure(
        kpoints=[[0, 0, 0], [0.5, 0, 0]],
        energies=[[0.0, 1.0], [0.2, 1.2]],
        weights=[[1.0, 0.5], [0.8, 0.2]],
    )
    grid = np.linspace(-20.0, 20.0, 400001)

    spectral = ebs.spectral_function(grid, broadening=0.1)
    integrals = np.trapezoid(spectral, grid, axis=1)

    np.testing.assert_allclose(integrals, ebs.weights.sum(axis=1), rtol=1e-6)


def test_lorentzian_spectral_function_conserves_weight_up_to_tail_truncation():
    ebs = EffectiveBandStructure(
        kpoints=[[0, 0, 0]],
        energies=[[0.0, 1.0]],
        weights=[[1.0, 0.5]],
    )
    grid = np.linspace(-4000.0, 4000.0, 800001)

    spectral = ebs.spectral_function(grid, broadening=0.05, kind="lorentzian")
    integral = float(np.trapezoid(spectral[0], grid))

    # The Lorentzian has 1/E^2 tails, so a finite grid loses O(gamma/E_max).
    assert abs(integral - 1.5) < 1e-3


def test_spectral_function_rejects_unknown_kernel():
    ebs = EffectiveBandStructure(kpoints=[[0, 0, 0]], energies=[[0.0]])

    with np.testing.assert_raises(ValueError):
        ebs.spectral_function(np.linspace(-1, 1, 5), broadening=0.1, kind="voigt")


def test_adaptive_widths_track_band_velocity():
    """A flat band keeps the floor width; a steep band gets a wider kernel."""

    distances = np.linspace(0.0, 1.0, 11)
    flat = np.zeros_like(distances)
    steep = 5.0 * distances
    ebs = EffectiveBandStructure(
        kpoints=np.column_stack([distances, np.zeros_like(distances), np.zeros_like(distances)]),
        energies=np.column_stack([flat, steep]),
        distances=distances,
    )

    widths = ebs.adaptive_widths(scale=1.0, minimum=1e-3)

    assert widths.shape == ebs.energies.shape
    assert np.allclose(widths[:, 0], 1e-3)
    # |dE/ds| * ds = 5 * 0.1 for the steep band.
    assert np.allclose(widths[:, 1], 0.5)


def test_adaptive_widths_respect_bounds_and_degenerate_paths():
    ebs = EffectiveBandStructure(
        kpoints=[[0, 0, 0], [0.5, 0, 0]],
        energies=[[0.0, 1.0], [3.0, 1.0]],
        distances=[0.0, 1.0],
    )

    capped = ebs.adaptive_widths(scale=1.0, minimum=0.01, maximum=0.5)
    assert np.all(capped <= 0.5)
    assert np.all(capped >= 0.01)

    single = EffectiveBandStructure(kpoints=[[0, 0, 0]], energies=[[0.0, 1.0]])
    assert np.allclose(single.adaptive_widths(minimum=0.05), 0.05)

    with np.testing.assert_raises(ValueError):
        ebs.adaptive_widths(minimum=0.0)
    with np.testing.assert_raises(ValueError):
        ebs.adaptive_widths(minimum=0.1, maximum=0.05)


def test_adaptive_broadening_conserves_weight():
    """``UnfoldLab.integral_adaptiveSpectralFunction``: per-state widths keep the area."""

    distances = np.linspace(0.0, 1.0, 5)
    ebs = EffectiveBandStructure(
        kpoints=np.column_stack([distances, np.zeros_like(distances), np.zeros_like(distances)]),
        energies=np.column_stack([np.zeros_like(distances), 2.0 * distances]),
        weights=np.column_stack([np.full_like(distances, 0.7), np.full_like(distances, 0.3)]),
        distances=distances,
    )
    widths = ebs.adaptive_widths(scale=1.0, minimum=0.05)
    grid = np.linspace(-40.0, 40.0, 400001)

    spectral = ebs.spectral_function(grid, broadening=widths)
    integrals = np.trapezoid(spectral, grid, axis=1)

    np.testing.assert_allclose(integrals, ebs.weights.sum(axis=1), rtol=1e-6)


def test_scalar_broadening_is_the_constant_adaptive_case():
    ebs = EffectiveBandStructure(
        kpoints=[[0, 0, 0], [0.5, 0, 0]],
        energies=[[0.0, 1.0], [0.2, 1.2]],
        weights=[[1.0, 0.5], [0.8, 0.2]],
    )
    grid = np.linspace(-2.0, 3.0, 101)

    scalar = ebs.spectral_function(grid, broadening=0.1)
    table = ebs.spectral_function(grid, broadening=np.full(ebs.energies.shape, 0.1))

    np.testing.assert_allclose(scalar, table)


def test_spectral_function_rejects_bad_width_tables():
    ebs = EffectiveBandStructure(
        kpoints=[[0, 0, 0], [0.5, 0, 0]],
        energies=[[0.0, 1.0], [0.2, 1.2]],
    )
    grid = np.linspace(-1.0, 2.0, 21)

    with np.testing.assert_raises(ValueError):
        ebs.spectral_function(grid, broadening=np.array([0.1, 0.2, 0.3]))
    with np.testing.assert_raises(ValueError):
        ebs.spectral_function(grid, broadening=np.zeros(ebs.energies.shape))
    with np.testing.assert_raises(ValueError):
        ebs.spectral_function(grid, broadening=-0.1)
