"""Unfolding against a symmetry-reduced set of wavefunctions.

The mathematics is ``UnfoldLab.weight_symmetry`` in
``RequestProject/Unfolding/Symmetry.lean``: rotating the primitive k-point and
the supercell k-point together, and re-indexing the plane waves by the same
operation, leaves the unfolding weight unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from unfoldlab.core.plane_waves import weights_from_coefficients
from unfoldlab.core.symmetry import (
    map_kpoints_to_stored,
    primitive_operation,
    supercell_operation,
)

INVERSION = -np.eye(3, dtype=int)
MIRROR_X = np.diag([-1, 1, 1])
TRANSFORM = np.diag([2, 1, 1])


def test_supercell_and_primitive_operations_are_inverse_to_each_other():
    s_sc = supercell_operation(MIRROR_X, TRANSFORM)

    assert np.array_equal(s_sc, MIRROR_X)  # diagonal T commutes with a diagonal op
    assert np.array_equal(primitive_operation(s_sc, TRANSFORM), MIRROR_X)


def test_supercell_operation_of_a_non_diagonal_transform():
    transform = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    s_sc = supercell_operation(INVERSION, transform)

    assert np.array_equal(s_sc, INVERSION)
    # A rotation that does not preserve the supercell is rejected.
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="not a symmetry of the supercell"):
        supercell_operation(swap, np.diag([2, 1, 1]))


def test_unimodularity_is_required():
    with pytest.raises(ValueError, match="unimodular"):
        map_kpoints_to_stored(np.zeros((1, 3)), np.zeros((1, 3)), TRANSFORM, [np.diag([2, 1, 1])])


def test_symmetry_image_reproduces_the_weight_of_the_stored_kpoint():
    """A file stored at K reproduces the weights at -K without rotating it."""

    rng = np.random.default_rng(23)
    g_stored = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [2, 0, 0], [0, 1, 0]])
    coeffs = (rng.normal(size=(2, 1, 5)) + 1j * rng.normal(size=(2, 1, 5))).astype(complex)

    k_file = np.array([0.25, 0.0, 0.0])
    k_primitive = np.array([0.125, 0.0, 0.0])
    stored_supercell = k_primitive @ TRANSFORM.T
    assert np.allclose(stored_supercell, k_file)

    # The rotated data: primitive and supercell k-points and G-vectors all
    # mapped by the inversion.
    s_sc = supercell_operation(INVERSION, TRANSFORM)
    rotated = weights_from_coefficients(
        g_stored @ s_sc.T,
        coeffs,
        k_primitive @ INVERSION.T,
        k_file @ s_sc.T,
        TRANSFORM,
    )
    original = weights_from_coefficients(g_stored, coeffs, k_primitive, k_file, TRANSFORM)

    assert np.allclose(rotated, original)


def test_map_kpoints_to_stored_finds_the_rotated_representative():
    stored = np.array([[0.25, 0.0, 0.0]])
    requested = np.array([[0.125, 0.0, 0.0], [-0.125, 0.0, 0.0]])

    matches = map_kpoints_to_stored(requested, stored, TRANSFORM, [INVERSION])

    assert [match.index for match in matches] == [0, 0]
    assert np.array_equal(matches[0].operation, np.eye(3, dtype=int))
    assert np.array_equal(matches[1].operation, INVERSION)
    # The second k-point is served by the same file, unfolded at +k.
    assert np.allclose(matches[1].effective_primitive_kpoint, [0.125, 0.0, 0.0])


def test_map_kpoints_to_stored_rejects_uncovered_kpoints():
    stored = np.array([[0.25, 0.0, 0.0]])
    with pytest.raises(ValueError, match="does not cover"):
        map_kpoints_to_stored(np.array([[0.3, 0.0, 0.0]]), stored, TRANSFORM, [INVERSION])


def test_symmetry_reduced_weights_match_a_full_mesh():
    """Weights from a reduced file set equal those from the full mesh."""

    rng = np.random.default_rng(5)
    g_stored = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [2, 0, 0], [-2, 0, 0]])
    coeffs = (rng.normal(size=(1, 1, 5)) + 1j * rng.normal(size=(1, 1, 5))).astype(complex)
    k_file = np.array([0.25, 0.0, 0.0])

    requested = np.array([[0.125, 0.0, 0.0], [-0.125, 0.0, 0.0]])
    matches = map_kpoints_to_stored(requested, np.array([k_file]), TRANSFORM, [INVERSION])

    s_sc = supercell_operation(INVERSION, TRANSFORM)
    for kpoint, match in zip(requested, matches, strict=True):
        # Reference: build the rotated wavefunction explicitly and unfold it.
        rotated_g = g_stored @ supercell_operation(match.operation, TRANSFORM).T
        reference = weights_from_coefficients(
            rotated_g,
            coeffs,
            kpoint,
            k_file @ supercell_operation(match.operation, TRANSFORM).T,
            TRANSFORM,
        )
        # What the library does instead: use the stored file untouched.
        actual = weights_from_coefficients(
            g_stored,
            coeffs,
            match.effective_primitive_kpoint,
            k_file,
            TRANSFORM,
        )
        assert np.allclose(actual, reference)
    assert np.array_equal(s_sc, INVERSION)


def _reference_matches(kpoints, stored, transform, operations, atol=1e-6):
    """The straightforward nested-loop resolution, kept as an oracle.

    ``map_kpoints_to_stored`` compares every requested k-point against all
    images at once; this is the readable version it has to agree with,
    tie-breaking included: the earliest stored k-point, and for it the earliest
    operation.
    """

    identity = np.eye(3, dtype=np.int64)
    ops = [identity]
    for candidate in np.asarray(operations, dtype=np.int64).reshape(-1, 3, 3):
        if not any(np.array_equal(candidate, known) for known in ops):
            ops.append(candidate)
    resolved = []
    for kpoint in np.asarray(kpoints, dtype=float):
        folded = kpoint @ np.asarray(transform, dtype=float).T
        for index, k_file in enumerate(np.asarray(stored, dtype=float)):
            hit = None
            for position, op in enumerate(ops):
                delta = folded - k_file @ supercell_operation(op, transform).T
                delta -= np.rint(delta)
                if np.all(np.abs(delta) <= atol):
                    hit = (index, position)
                    break
            if hit is not None:
                resolved.append(hit)
                break
        else:  # pragma: no cover - the fixtures below always match
            raise AssertionError("no match")
    return resolved, ops


def test_vectorized_resolution_agrees_with_the_nested_loops():
    transform = np.diag([2, 2, 1]).astype(np.int64)
    # A point group with several operations that act the same way on some
    # k-points, so the tie-breaking is actually exercised.
    operations = np.stack(
        [
            np.diag([1, 1, 1]),
            np.diag([-1, 1, 1]),
            np.diag([1, -1, 1]),
            np.diag([-1, -1, 1]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
        ]
    ).astype(np.int64)

    rng = np.random.default_rng(17)
    stored = np.round(rng.random((12, 3)) * 4.0) / 4.0
    requested = stored[rng.integers(0, 12, 40)] @ np.linalg.inv(transform).T

    expected, ops = _reference_matches(requested, stored, transform, operations)
    matches = map_kpoints_to_stored(requested, stored, transform, operations)

    assert [match.index for match in matches] == [index for index, _ in expected]
    for match, (_, position) in zip(matches, expected, strict=True):
        assert np.array_equal(match.operation, ops[position])
