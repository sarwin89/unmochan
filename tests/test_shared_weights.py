"""The batched fiber-weight kernel against a first-principles reference.

One supercell wavefunction usually serves several primitive k-points -- a
complete fiber, or a band path that revisits an irreducible k-point under
symmetry.  ``shared_weights_from_coefficients`` does that in one pass;
everything here checks that it agrees exactly with the definition, since a
performance rewrite that quietly changes a number is the worst possible kind.
"""

from __future__ import annotations

import numpy as np
import pytest

from unmochan.core.plane_waves import (
    matching_g_mask,
    shared_weights_from_coefficients,
    weights_from_coefficients,
)
from unmochan.core.unfolding import (
    SharedWavefunctionGroup,
    compute_plane_wave_unfolding_weights,
    compute_plane_wave_unfolding_weights_shared,
)

TRANSFORM = np.diag([2, 1, 1])
FOLDED = np.zeros(3)
FIBER = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])


def g_vectors() -> np.ndarray:
    return np.array(
        [[i, j, 0] for i in range(-3, 4) for j in range(-2, 3)],
        dtype=int,
    )


def coefficients(n_bands: int = 5, n_components: int = 2) -> np.ndarray:
    rng = np.random.default_rng(20240917)
    n_g = len(g_vectors())
    real = rng.normal(size=(n_bands, n_components, n_g))
    imag = rng.normal(size=(n_bands, n_components, n_g))
    return real + 1j * imag


def reference_weights(
    g_supercell: np.ndarray,
    coeffs: np.ndarray,
    kpoints: np.ndarray,
    *,
    component_resolved: bool = False,
) -> np.ndarray:
    """Straight from the definition: a Python loop over every index."""

    n_bands, n_components, _ = coeffs.shape
    shape = (len(kpoints), n_bands, n_components) if component_resolved else (len(kpoints), n_bands)
    out = np.zeros(shape)
    for ik, kpoint in enumerate(kpoints):
        mask = matching_g_mask(g_supercell, kpoint, FOLDED, TRANSFORM)
        for band in range(n_bands):
            total = float(np.sum(np.abs(coeffs[band]) ** 2))
            if total <= 0.0:
                continue
            for component in range(n_components):
                partial = float(np.sum(np.abs(coeffs[band, component][mask]) ** 2))
                if component_resolved:
                    out[ik, band, component] = partial / total
                else:
                    out[ik, band] += partial / total
    return out


@pytest.mark.parametrize("component_resolved", [False, True])
def test_the_batched_kernel_reproduces_the_definition(component_resolved: bool) -> None:
    g_supercell = g_vectors()
    coeffs = coeffs_with_a_dark_band()
    got = shared_weights_from_coefficients(
        g_supercell,
        coeffs,
        FIBER,
        FOLDED,
        TRANSFORM,
        component_resolved=component_resolved,
    )
    expected = reference_weights(g_supercell, coeffs, FIBER, component_resolved=component_resolved)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-14)


def coeffs_with_a_dark_band() -> np.ndarray:
    """A band of exactly zero norm, which is the division-by-zero edge case."""

    coeffs = coefficients()
    coeffs[3] = 0.0
    return coeffs


@pytest.mark.parametrize("component_resolved", [False, True])
def test_a_zero_norm_band_gets_weight_zero(component_resolved: bool) -> None:
    got = shared_weights_from_coefficients(
        g_vectors(),
        coeffs_with_a_dark_band(),
        FIBER,
        FOLDED,
        TRANSFORM,
        component_resolved=component_resolved,
    )
    assert np.all(got[:, 3] == 0.0)
    assert np.isfinite(got).all()


@pytest.mark.parametrize("component_resolved", [False, True])
def test_one_row_at_a_time_gives_the_same_numbers(component_resolved: bool) -> None:
    g_supercell = g_vectors()
    coeffs = coeffs_with_a_dark_band()
    batched = shared_weights_from_coefficients(
        g_supercell,
        coeffs,
        FIBER,
        FOLDED,
        TRANSFORM,
        component_resolved=component_resolved,
    )
    for row, kpoint in enumerate(FIBER):
        single = weights_from_coefficients(
            g_supercell,
            coeffs,
            kpoint,
            FOLDED,
            TRANSFORM,
            component_resolved=component_resolved,
        )
        # Not bit-for-bit: contracting a ``(n_g, m)`` mask matrix and a
        # ``(n_g, 1)`` one take different BLAS paths, so the two differ in the
        # last bit or two.  Agreement to a few ulp is the honest claim.
        assert np.allclose(batched[row], single, rtol=0.0, atol=1e-15)


def test_the_fiber_sum_rule_survives_the_batching() -> None:
    weights = shared_weights_from_coefficients(
        g_vectors(), coefficients(), FIBER, FOLDED, TRANSFORM
    )
    assert np.allclose(weights.sum(axis=0), 1.0)


def test_the_component_split_sums_to_the_plain_weight() -> None:
    g_supercell = g_vectors()
    coeffs = coeffs_with_a_dark_band()
    plain = shared_weights_from_coefficients(g_supercell, coeffs, FIBER, FOLDED, TRANSFORM)
    split = shared_weights_from_coefficients(
        g_supercell, coeffs, FIBER, FOLDED, TRANSFORM, component_resolved=True
    )
    assert np.allclose(split.sum(axis=2), plain, atol=1e-15)


def test_no_kpoints_gives_an_empty_result_rather_than_an_error() -> None:
    empty = shared_weights_from_coefficients(
        g_vectors(), coefficients(), np.zeros((0, 3)), FOLDED, TRANSFORM
    )
    assert empty.shape == (0, 5)


def test_a_wrongly_shaped_kpoint_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="primitive_kpoints must have shape"):
        shared_weights_from_coefficients(
            g_vectors(), coefficients(), np.zeros(3), FOLDED, TRANSFORM
        )


def test_a_wrongly_shaped_coefficient_array_is_rejected() -> None:
    with pytest.raises(ValueError, match="coefficients must have shape"):
        shared_weights_from_coefficients(g_vectors(), np.zeros((4, 5)), FIBER, FOLDED, TRANSFORM)


def test_a_mismatched_g_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="coefficient G dimension"):
        shared_weights_from_coefficients(
            g_vectors(), np.zeros((2, 1, 3), dtype=complex), FIBER, FOLDED, TRANSFORM
        )


def group() -> SharedWavefunctionGroup:
    return SharedWavefunctionGroup(
        primitive_kpoints=FIBER,
        folded_supercell_kpoint=FOLDED,
        g_supercell=g_vectors(),
        coefficients=coeffs_with_a_dark_band(),
    )


@pytest.mark.parametrize("component_resolved", [False, True])
def test_the_grouped_driver_matches_the_expanded_stream(component_resolved: bool) -> None:
    shared = compute_plane_wave_unfolding_weights_shared(
        [group()], TRANSFORM, component_resolved=component_resolved
    )
    expanded = compute_plane_wave_unfolding_weights(
        group().expand(), TRANSFORM, component_resolved=component_resolved
    )
    assert np.allclose(shared, expanded, rtol=0.0, atol=1e-15)


def test_groups_are_concatenated_in_order() -> None:
    coeffs = coeffs_with_a_dark_band()
    first = SharedWavefunctionGroup(
        primitive_kpoints=FIBER[:1],
        folded_supercell_kpoint=FOLDED,
        g_supercell=g_vectors(),
        coefficients=coeffs,
    )
    second = SharedWavefunctionGroup(
        primitive_kpoints=FIBER[1:],
        folded_supercell_kpoint=FOLDED,
        g_supercell=g_vectors(),
        coefficients=coeffs,
    )
    split = compute_plane_wave_unfolding_weights_shared([first, second], TRANSFORM)
    whole = compute_plane_wave_unfolding_weights_shared([group()], TRANSFORM)
    assert np.allclose(split, whole, rtol=0.0, atol=1e-15)


def test_an_empty_group_contributes_no_rows() -> None:
    empty = SharedWavefunctionGroup(
        primitive_kpoints=np.zeros((0, 3)),
        folded_supercell_kpoint=FOLDED,
        g_supercell=g_vectors(),
        coefficients=coefficients(),
    )
    weights = compute_plane_wave_unfolding_weights_shared([empty, group()], TRANSFORM)
    assert weights.shape == (2, 5)


def test_a_stream_with_no_rows_at_all_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one k-point"):
        compute_plane_wave_unfolding_weights_shared([], TRANSFORM)


def test_groups_must_agree_on_the_band_count() -> None:
    other = SharedWavefunctionGroup(
        primitive_kpoints=FIBER,
        folded_supercell_kpoint=FOLDED,
        g_supercell=g_vectors(),
        coefficients=coefficients(n_bands=4),
    )
    with pytest.raises(ValueError, match="same number of bands"):
        compute_plane_wave_unfolding_weights_shared([group(), other], TRANSFORM)


def test_component_resolved_groups_must_agree_on_the_component_count() -> None:
    other = SharedWavefunctionGroup(
        primitive_kpoints=FIBER,
        folded_supercell_kpoint=FOLDED,
        g_supercell=g_vectors(),
        coefficients=coefficients(n_components=1),
    )
    with pytest.raises(ValueError, match="spinor/polarization components"):
        compute_plane_wave_unfolding_weights_shared(
            [group(), other], TRANSFORM, component_resolved=True
        )


def test_the_group_validates_its_own_shapes() -> None:
    with pytest.raises(ValueError, match="primitive_kpoints must have shape"):
        SharedWavefunctionGroup(
            primitive_kpoints=np.zeros(3),
            folded_supercell_kpoint=FOLDED,
            g_supercell=g_vectors(),
            coefficients=coefficients(),
        )
    with pytest.raises(ValueError, match="folded_supercell_kpoint must have shape"):
        SharedWavefunctionGroup(
            primitive_kpoints=FIBER,
            folded_supercell_kpoint=np.zeros((1, 3)),
            g_supercell=g_vectors(),
            coefficients=coefficients(),
        )
    with pytest.raises(ValueError, match="g_supercell must have shape"):
        SharedWavefunctionGroup(
            primitive_kpoints=FIBER,
            folded_supercell_kpoint=FOLDED,
            g_supercell=np.zeros(3),
            coefficients=coefficients(),
        )
    with pytest.raises(ValueError, match="coefficients must have shape"):
        SharedWavefunctionGroup(
            primitive_kpoints=FIBER,
            folded_supercell_kpoint=FOLDED,
            g_supercell=g_vectors(),
            coefficients=np.zeros((2, 3), dtype=complex),
        )


def test_the_batched_kernel_is_exported_at_the_top_level() -> None:
    import unmochan

    assert unmochan.shared_weights_from_coefficients is shared_weights_from_coefficients
    assert unmochan.SharedWavefunctionGroup is SharedWavefunctionGroup
    assert (
        unmochan.compute_plane_wave_unfolding_weights_shared
        is compute_plane_wave_unfolding_weights_shared
    )
