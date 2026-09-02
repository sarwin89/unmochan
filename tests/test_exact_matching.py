"""Tests for the exact integer plane-wave matching kernel and the sum rule.

These mirror the machine-checked statements in ``RequestProject/Unfolding``:

* the matching mask partitions the supercell plane-wave basis over the fiber
  (``UnfoldLab.IsFiberRepr.existsUnique_pwMatches``),
* the fiber has ``|det T|`` members (``UnfoldLab.IsFiberRepr.card_eq_natAbs_det``),
* the weights of a fiber sum to one (``UnfoldLab.IsFiberRepr.sum_weight_eq_one``),
* the mask depends only on ``K + G`` (``UnfoldLab.pwMatches_of_add_eq``).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from unfoldlab.core.numerics import (
    check_atol,
    integer_adjugate3,
    integer_det3,
    wrap_fractional,
)
from unfoldlab.core.plane_waves import matching_g_mask, weights_from_coefficients
from unfoldlab.core.structures import Structure
from unfoldlab.core.transformations import (
    TransformationMatrix,
    check_supercell_consistency,
)
from unfoldlab.core.unfolding import (
    PlaneWaveKPointData,
    compute_plane_wave_unfolding_weights,
    diagnose_fiber_sum_rule,
)

DIAGONAL = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
NON_DIAGONAL = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]])
SHEARED = np.array([[2, 1, 0], [0, 1, 1], [1, 0, 3]])
MATRICES = [DIAGONAL, NON_DIAGONAL, SHEARED]


def primitive_fiber(matrix: np.ndarray, folded: np.ndarray, span: int = 4) -> list[np.ndarray]:
    """All primitive k-points that fold onto ``folded``, one per class."""

    inverse_t = np.linalg.inv(matrix).T
    classes: dict[tuple[float, float, float], np.ndarray] = {}
    for shift in itertools.product(range(-span, span + 1), repeat=3):
        candidate = (folded + np.array(shift, dtype=float)) @ inverse_t
        key = tuple(np.round(np.mod(np.round(candidate, 9), 1.0), 6))
        classes.setdefault(key, candidate)
    return list(classes.values())


@pytest.mark.unit
@pytest.mark.parametrize("matrix", MATRICES)
def test_integer_adjugate_and_determinant_are_exact(matrix):
    determinant = integer_det3(matrix)
    adjugate = integer_adjugate3(matrix)

    assert determinant == round(float(np.linalg.det(matrix)))
    np.testing.assert_array_equal(adjugate @ matrix, determinant * np.eye(3, dtype=int))
    np.testing.assert_array_equal(matrix @ adjugate, determinant * np.eye(3, dtype=int))


@pytest.mark.unit
@pytest.mark.parametrize("matrix", MATRICES)
def test_fiber_size_equals_multiplicity(matrix):
    transform = TransformationMatrix(matrix)
    fiber = primitive_fiber(matrix, np.zeros(3))

    assert len(fiber) == transform.multiplicity


@pytest.mark.unit
@pytest.mark.parametrize("matrix", MATRICES)
def test_mask_partitions_plane_waves_over_the_fiber(matrix):
    folded = np.array([0.25, 0.5, 0.0])
    fiber = primitive_fiber(matrix, folded)
    g_vectors = np.array(list(itertools.product(range(-3, 4), repeat=3)), dtype=int)

    masks = np.array(
        [matching_g_mask(g_vectors, k, folded, matrix) for k in fiber],
        dtype=int,
    )

    np.testing.assert_array_equal(masks.sum(axis=0), np.ones(g_vectors.shape[0], dtype=int))


@pytest.mark.unit
@pytest.mark.parametrize("matrix", MATRICES)
def test_mask_matches_floating_point_reference(matrix):
    folded = np.array([0.125, 0.0, 0.5])
    primitive = (folded + np.array([1.0, -2.0, 0.0])) @ np.linalg.inv(matrix).T
    g_vectors = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=int)

    reference_delta = (folded + g_vectors) @ np.linalg.inv(matrix).T - primitive
    reference = np.all(np.abs(reference_delta - np.rint(reference_delta)) < 1e-6, axis=1)

    np.testing.assert_array_equal(matching_g_mask(g_vectors, primitive, folded, matrix), reference)


@pytest.mark.unit
def test_mask_depends_only_on_the_physical_plane_wave():
    matrix = NON_DIAGONAL
    folded = np.array([0.25, 0.5, 0.0])
    primitive = folded @ np.linalg.inv(matrix).T
    g_vectors = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=int)
    shift = np.array([1, -2, 3])

    direct = matching_g_mask(g_vectors, primitive, folded, matrix)
    shifted = matching_g_mask(g_vectors - shift, primitive, folded + shift, matrix)

    np.testing.assert_array_equal(direct, shifted)


@pytest.mark.unit
def test_mask_rejects_inconsistent_kpoint_pair():
    matrix = DIAGONAL
    g_vectors = np.zeros((1, 3), dtype=int)

    with pytest.raises(ValueError, match="not the fold of"):
        matching_g_mask(g_vectors, np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), matrix)


@pytest.mark.unit
@pytest.mark.parametrize("matrix", MATRICES)
def test_weights_of_a_fiber_sum_to_one(matrix):
    folded = np.array([0.0, 0.25, 0.5])
    fiber = primitive_fiber(matrix, folded)
    g_vectors = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=int)
    rng = np.random.default_rng(20240517)
    coefficients = rng.normal(size=(3, 2, g_vectors.shape[0])) + 1j * rng.normal(
        size=(3, 2, g_vectors.shape[0])
    )

    total = np.zeros(3)
    for kpoint in fiber:
        weights = weights_from_coefficients(g_vectors, coefficients, kpoint, folded, matrix)
        assert np.all(weights >= -1e-12)
        assert np.all(weights <= 1.0 + 1e-12)
        total += weights

    np.testing.assert_allclose(total, np.ones(3), atol=1e-12)


@pytest.mark.unit
def test_diagnose_fiber_sum_rule_reports_complete_and_incomplete_fibers():
    matrix = DIAGONAL
    folded = np.array([0.0, 0.0, 0.0])
    fiber = primitive_fiber(matrix, folded)
    kpoints = np.array(fiber)
    weights = np.full((len(fiber), 2), 1.0 / len(fiber))

    diagnostics = diagnose_fiber_sum_rule(weights, kpoints, matrix)
    assert diagnostics.multiplicity == len(fiber)
    assert diagnostics.n_complete_fibers == 1
    assert diagnostics.satisfied

    partial = diagnose_fiber_sum_rule(weights[:1], kpoints[:1], matrix)
    assert partial.n_complete_fibers == 0
    assert partial.max_deviation is None
    assert not partial.satisfied
    # A partial fiber below one is fine: the sum rule is an inequality there.
    assert partial.max_partial_excess == pytest.approx(0.0)
    assert not partial.violated()


@pytest.mark.unit
def test_partial_fibers_may_not_exceed_one():
    """An incomplete fiber above one is a violation, and a path can see it.

    Formal statement: ``UnfoldLab.IsFiberRepr.sum_weight_subset_le_one``.
    """

    matrix = DIAGONAL
    fiber = primitive_fiber(matrix, np.array([0.0, 0.0, 0.0]))
    kpoints = np.array(fiber[:2])
    weights = np.array([[0.7, 0.1], [0.8, 0.2]])

    diagnostics = diagnose_fiber_sum_rule(weights, kpoints, matrix)

    assert diagnostics.n_complete_fibers == 0
    assert diagnostics.max_partial_excess == pytest.approx(0.5)
    assert diagnostics.worst_partial_fiber == (0.0, 0.0, 0.0)
    assert diagnostics.violated()
    assert diagnostics.to_dict()["worst_partial_fiber"] == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_check_atol_rejects_meaningless_tolerances():
    assert check_atol(1e-6) == 1e-6
    with pytest.raises(ValueError, match="meaningless"):
        check_atol(0.5)
    with pytest.raises(ValueError):
        check_atol(-1.0)


@pytest.mark.unit
def test_wrap_fractional_is_idempotent_and_in_unit_cell():
    values = np.array([-1.25, 0.75, 2.0])
    wrapped = wrap_fractional(values)

    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < 1.0)
    np.testing.assert_allclose(wrap_fractional(wrapped), wrapped)
    np.testing.assert_allclose(np.rint(wrapped - values), wrapped - values, atol=1e-12)


@pytest.mark.unit
def test_check_supercell_consistency_flags_missing_atoms():
    lattice = np.eye(3)
    primitive = Structure(lattice=lattice, species=("A",), frac_coords=[[0.0, 0.0, 0.0]])
    transform = TransformationMatrix(DIAGONAL)
    good = Structure(
        lattice=DIAGONAL @ lattice,
        species=("A",) * 4,
        frac_coords=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]],
    )
    defective = Structure(
        lattice=DIAGONAL @ lattice,
        species=("A",) * 3,
        frac_coords=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
    )

    assert check_supercell_consistency(primitive, good, transform).consistent
    report = check_supercell_consistency(primitive, defective, transform)
    assert report.volume_consistent
    assert report.sites_consistent is False
    assert not report.consistent


def test_component_resolved_weights_sum_to_the_total_weight():
    """Spinor components can be resolved without breaking the sum rule."""

    g_vectors = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    coefficients = np.array([[[1.0 + 0j, 2.0, 0.0, 1.0], [0.0, 1.0, 3.0, 0.0]]], dtype=complex)
    transform = np.diag([2, 1, 1])
    kpoint = np.zeros(3)

    total = weights_from_coefficients(g_vectors, coefficients, kpoint, kpoint, transform)
    resolved = weights_from_coefficients(
        g_vectors, coefficients, kpoint, kpoint, transform, component_resolved=True
    )

    assert resolved.shape == (1, 2)
    assert np.allclose(resolved.sum(axis=1), total)
    # G = 0 and G = 2 match k = 0; only the second component carries G = 2.
    norms = np.abs(coefficients[0]) ** 2
    expected_first = (norms[0, 0] + norms[0, 2]) / norms.sum()
    assert resolved[0, 0] == pytest.approx(expected_first)


def test_component_resolved_weights_at_the_kpoint_list_level():
    """`compute_plane_wave_unfolding_weights` can split spinor components."""

    transform = np.diag([2, 1, 1])
    g_vectors = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    coefficients = np.array(
        [[[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]]],
        dtype=np.complex128,
    )
    data = [
        PlaneWaveKPointData(
            primitive_kpoint=np.zeros(3),
            folded_supercell_kpoint=np.zeros(3),
            g_supercell=g_vectors,
            coefficients=coefficients,
        )
    ]

    total = compute_plane_wave_unfolding_weights(data, transform)
    resolved = compute_plane_wave_unfolding_weights(data, transform, component_resolved=True)

    assert resolved.shape == (1, 1, 2)
    assert np.allclose(resolved.sum(axis=2), total)
    # Even G-vectors match k = 0: |1|^2 + |2|^2 from the first component and
    # nothing from the second, out of a total norm of 30.
    assert np.allclose(resolved[0, 0], [5.0 / 30.0, 0.0])
