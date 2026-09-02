"""PAW / ultrasoft augmentation: the error the plane-wave weight cannot see.

`unfoldlab.core.augmentation` states, in numbers, the caveat every plane-wave
unfolding of a PAW or ultrasoft calculation carries: the stored coefficients are
the pseudo wavefunction, so the weights are the pseudo weights.  The statements
it implements are proved in ``RequestProject/Unfolding/Augmentation.lean``, and
they are checked here against explicit matrices:

* the structure factor of an ideal supercell vanishes off the primitive
  reciprocal lattice and equals `|det T|` on it, and a single unreplicated site
  destroys that;
* a fiber-block-diagonal augmentation leaves the fiber sum rule intact -- which
  is exactly why the sum rule cannot detect the error -- and a cross term
  between fiber members breaks it;
* the difference between the augmented and the pseudo weight is bounded by the
  augmentation fraction, i.e. by the norm deficit the readers already report.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from unfoldlab.core.augmentation import (
    NEGLIGIBLE_FRACTION,
    SEVERE_FRACTION,
    augmentation_fraction,
    augmented_weights,
    diagnose_augmentation,
    is_fiber_block_diagonal,
    structure_factor,
    weight_error_bound,
)
from unfoldlab.core.plane_waves import matching_g_mask

TRANSFORM = np.diag([2, 1, 1])

#: Four plane waves of a doubled cell: even x belongs to one fiber member,
#: odd x to the other.
GVECTORS = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])

FIBER = (np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]))


def _fiber_masks() -> list[np.ndarray]:
    return [matching_g_mask(GVECTORS, kpoint, np.zeros(3), TRANSFORM) for kpoint in FIBER]


def _block_diagonal_augmentation() -> np.ndarray:
    """A Hermitian positive semidefinite `A` coupling only same-parity waves."""

    matrix = np.zeros((4, 4), dtype=complex)
    even, odd = [0, 2], [1, 3]
    matrix[np.ix_(even, even)] = np.array([[0.20, 0.05], [0.05, 0.20]])
    matrix[np.ix_(odd, odd)] = np.array([[0.10, 0.03], [0.03, 0.10]])
    return matrix


def test_masks_of_the_fiber_partition_the_plane_waves() -> None:
    masks = _fiber_masks()
    assert np.array_equal(masks[0], [True, False, True, False])
    assert np.array_equal(masks[1], [False, True, False, True])
    assert np.all(masks[0] | masks[1])
    assert not np.any(masks[0] & masks[1])


def test_structure_factor_of_an_ideal_supercell() -> None:
    """`structureFactor_eq`: `|det T|` on the primitive lattice, zero off it."""

    cells = [[0, 0, 0], [1, 0, 0]]
    on_lattice = structure_factor(TRANSFORM, cells, [[2, 0, 0], [0, 0, 0], [4, 0, 0]])
    assert np.allclose(on_lattice, 2.0)
    off_lattice = structure_factor(TRANSFORM, cells, [[1, 0, 0], [3, 0, 0]])
    assert np.allclose(off_lattice, 0.0, atol=1e-12)


def test_an_unreplicated_site_has_structure_factor_one_everywhere() -> None:
    """`exists_structureFactor_ne_zero_of_not_mem`: a defect couples the fiber."""

    single = structure_factor(TRANSFORM, [[0, 0, 0]], [[1, 0, 0], [2, 0, 0]])
    assert np.allclose(single, 1.0)


def test_structure_factor_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="invertible"):
        structure_factor(np.zeros((3, 3)), [[0, 0, 0]], [[1, 0, 0]])
    with pytest.raises(ValueError, match="cells"):
        structure_factor(TRANSFORM, [[0, 0]], [[1, 0, 0]])
    with pytest.raises(ValueError, match="transfer"):
        structure_factor(TRANSFORM, [[0, 0, 0]], [[1, 0]])


def test_block_diagonality_is_decided_exactly() -> None:
    good = _block_diagonal_augmentation()
    assert is_fiber_block_diagonal(good, GVECTORS, TRANSFORM)

    bad = good.copy()
    bad[0, 1] = bad[1, 0] = 0.05
    assert not is_fiber_block_diagonal(bad, GVECTORS, TRANSFORM)

    # A coupling below the tolerance is treated as absent.
    tiny = good.copy()
    tiny[0, 1] = tiny[1, 0] = 1e-12
    assert is_fiber_block_diagonal(tiny, GVECTORS, TRANSFORM, atol=1e-10)


def test_block_diagonality_validates_its_inputs() -> None:
    good = _block_diagonal_augmentation()
    with pytest.raises(ValueError, match="invertible"):
        is_fiber_block_diagonal(good, GVECTORS, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="gvectors"):
        is_fiber_block_diagonal(good, [[0, 0]], TRANSFORM)
    with pytest.raises(ValueError, match="square"):
        is_fiber_block_diagonal(good[:3], GVECTORS, TRANSFORM)


def test_without_augmentation_the_weight_is_the_plane_wave_weight() -> None:
    coefficients = np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 1.0]], dtype=complex)
    mask = _fiber_masks()[0]
    plain = augmented_weights(coefficients, mask)
    expected = (np.abs(coefficients[:, mask]) ** 2).sum(axis=1) / (np.abs(coefficients) ** 2).sum(
        axis=1
    )
    assert np.allclose(plain, expected)


def test_an_empty_state_gets_weight_zero_rather_than_a_division_by_zero() -> None:
    coefficients = np.zeros((1, 4), dtype=complex)
    assert augmented_weights(coefficients, _fiber_masks()[0]) == pytest.approx(0.0)


def test_augmented_weights_keep_the_fiber_sum_rule_when_block_diagonal() -> None:
    """`IsFiberRepr.sum_augWeight_eq_one`: the sum rule cannot see the error."""

    rng = np.random.default_rng(7)
    coefficients = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
    augmentation = _block_diagonal_augmentation()
    total = sum(augmented_weights(coefficients, mask, augmentation) for mask in _fiber_masks())
    assert np.allclose(total, 1.0)

    # ... and so do the pseudo weights, which is the point: the diagnostic is
    # blind to the difference between the two.
    pseudo = sum(augmented_weights(coefficients, mask) for mask in _fiber_masks())
    assert np.allclose(pseudo, 1.0)


def test_a_cross_term_between_fiber_members_breaks_the_sum_rule() -> None:
    """The block-diagonality hypothesis is not decorative."""

    rng = np.random.default_rng(11)
    coefficients = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    augmentation = _block_diagonal_augmentation()
    augmentation[0, 1] = 0.05
    augmentation[1, 0] = 0.05
    assert not is_fiber_block_diagonal(augmentation, GVECTORS, TRANSFORM)
    total = sum(augmented_weights(coefficients, mask, augmentation) for mask in _fiber_masks())
    assert not np.allclose(total, 1.0)


def test_the_augmented_and_pseudo_weights_differ_by_at_most_the_fraction() -> None:
    """`abs_augWeight_sub_weight_le`: the error is bounded by `q / (N + q)`."""

    rng = np.random.default_rng(3)
    coefficients = rng.normal(size=(32, 4)) + 1j * rng.normal(size=(32, 4))
    augmentation = _block_diagonal_augmentation()
    mask = _fiber_masks()[0]
    exact = augmented_weights(coefficients, mask, augmentation)
    pseudo = augmented_weights(coefficients, mask)

    plane_wave_norm = (np.abs(coefficients) ** 2).sum(axis=1)
    augmentation_part = np.einsum(
        "si,ij,sj->s", coefficients.conj(), augmentation, coefficients
    ).real
    bound = augmentation_part / (plane_wave_norm + augmentation_part)
    assert np.all(np.abs(exact - pseudo) <= bound + 1e-12)
    # The bound is not vacuous: this augmentation really does move the weights.
    assert np.max(np.abs(exact - pseudo)) > 1e-3


def test_augmented_weights_validate_their_inputs() -> None:
    coefficients = np.zeros((2, 4), dtype=complex)
    with pytest.raises(ValueError, match="one entry per plane wave"):
        augmented_weights(coefficients, np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="square"):
        augmented_weights(coefficients, np.ones(4, dtype=bool), np.zeros((3, 3)))


def test_the_augmentation_fraction_is_the_norm_deficit() -> None:
    norms = np.array([[1.0, 0.97], [0.90, 1.02]])
    fraction = augmentation_fraction(norms)
    assert fraction == pytest.approx(np.array([[0.0, 0.03], [0.10, 0.0]]))
    unclipped = augmentation_fraction(norms, clip=False)
    assert unclipped[1, 1] == pytest.approx(-0.02)
    assert weight_error_bound(norms) == pytest.approx(fraction)


def test_the_report_summarizes_a_paw_run() -> None:
    norms = np.array([[0.995, 0.86], [0.99, 0.97]])
    report = diagnose_augmentation(norms)
    assert report.n_states == 4
    assert report.min_norm == pytest.approx(0.86)
    assert report.max_fraction == pytest.approx(0.14)
    assert report.weight_error_bound == pytest.approx(0.14)
    assert report.n_severe == 1
    assert not report.negligible
    assert not report.negative_deficit
    assert "sum rule does NOT detect this" in report.summary()
    payload = report.to_dict()
    assert json.loads(json.dumps(payload))["max_fraction"] == pytest.approx(0.14)


def test_a_norm_conserving_run_is_reported_as_negligible() -> None:
    report = diagnose_augmentation(np.full((2, 3), 1.0 - NEGLIGIBLE_FRACTION / 2))
    assert report.negligible
    assert "negligible" in report.summary()
    assert report.n_severe == 0


def test_a_norm_above_the_reference_is_flagged() -> None:
    report = diagnose_augmentation(np.array([[1.0, 1.05]]))
    assert report.negative_deficit
    assert "normalization convention" in report.summary()


def test_the_severity_threshold_is_the_documented_one() -> None:
    below = diagnose_augmentation(np.array([[1.0 - SEVERE_FRACTION]]))
    above = diagnose_augmentation(np.array([[1.0 - SEVERE_FRACTION - 1e-6]]))
    assert below.n_severe == 0
    assert above.n_severe == 1


def test_the_report_needs_a_state() -> None:
    with pytest.raises(ValueError, match="at least one state"):
        diagnose_augmentation(np.zeros((0, 3)))


def test_the_module_is_exported() -> None:
    import unfoldlab

    assert unfoldlab.diagnose_augmentation is diagnose_augmentation
    assert unfoldlab.augmented_weights is augmented_weights
