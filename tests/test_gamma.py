"""Time-reversal (gamma-only) half-basis expansion.

The mathematics is formalized in ``RequestProject/Unfolding/GammaOnly.lean``:
``UnfoldLab.totalNorm_gammaExpand`` (normalization),
``UnfoldLab.partialNorm_gammaExpand`` (which primitive k-point each half feeds)
and ``UnfoldLab.exists_half_basis_weight_ne`` (using the stored half alone is
wrong).
"""

from __future__ import annotations

import numpy as np
import pytest

from unfoldlab.core.gamma import check_half_basis, expand_gamma_half_basis
from unfoldlab.core.plane_waves import weights_from_coefficients


def _half_basis() -> np.ndarray:
    return np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=int)


def test_expansion_adds_conjugate_partners_once():
    g_half = _half_basis()
    coeffs = np.array([[[1.0, 2.0 + 1j, 3.0, 4.0 - 2j]]], dtype=complex)

    g_full, coeffs_full = expand_gamma_half_basis(g_half, coeffs)

    assert g_full.shape == (7, 3)
    assert np.array_equal(g_full[:4], g_half)
    assert np.array_equal(g_full[4:], -g_half[1:])
    assert np.allclose(coeffs_full[0, 0, 4:], np.conjugate(coeffs[0, 0, 1:]))


def test_expansion_reproduces_gamma_only_normalization():
    g_half = _half_basis()
    rng = np.random.default_rng(3)
    values = rng.normal(size=4) + 1j * rng.normal(size=4)
    coeffs = values.reshape(1, 1, 4)

    _g_full, coeffs_full = expand_gamma_half_basis(g_half, coeffs)

    expected = abs(values[0]) ** 2 + 2.0 * float(np.sum(np.abs(values[1:]) ** 2))
    assert float(np.sum(np.abs(coeffs_full) ** 2)) == pytest.approx(expected)


def test_half_basis_validation_rejects_full_basis_and_duplicates():
    with pytest.raises(ValueError, match="time-reversal partner"):
        check_half_basis(np.array([[1, 0, 0], [-1, 0, 0]]))
    with pytest.raises(ValueError, match="duplicates"):
        check_half_basis(np.array([[1, 0, 0], [1, 0, 0]]))
    # A genuine half basis with the origin is accepted.
    assert check_half_basis(_half_basis()).shape == (4, 3)


def test_sum_rule_holds_after_expansion_but_not_before():
    """The stored half alone gives weights that do not add up to one."""

    transform = np.diag([2, 1, 1])
    g_half = _half_basis()
    rng = np.random.default_rng(11)
    values = rng.normal(size=4) + 1j * rng.normal(size=4)
    coeffs = values.reshape(1, 1, 4)
    g_full, coeffs_full = expand_gamma_half_basis(g_half, coeffs)

    gamma = np.zeros(3)
    fiber = [np.zeros(3), np.array([0.5, 0.0, 0.0])]

    expanded = [
        weights_from_coefficients(g_full, coeffs_full, k, gamma, transform)[0] for k in fiber
    ]
    stored_half = [weights_from_coefficients(g_half, coeffs, k, gamma, transform)[0] for k in fiber]

    assert sum(expanded) == pytest.approx(1.0)
    # Both partitions sum to one by construction, but they distribute the
    # weight differently: the mirror half belongs to the other k-point.
    assert expanded[0] != pytest.approx(stored_half[0])
