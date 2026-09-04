"""Benchmarks against closed-form unfolded band structures.

Every case here has an answer that can be written down by hand, so a
disagreement is a bug in the code rather than a difference of convention.  The
cases and their references are collected in ``docs/benchmarks.md``.

The underlying statements are ``UnfoldLab.IsFiberRepr.weight_ideal`` (a
supercell of a perfect crystal unfolds to weight one on a single fiber member),
``UnfoldLab.IsFiberRepr.sum_weight_eq_one`` (the fiber sum rule) and
``UnfoldLab.sum_tbWeight_states`` (the tight-binding completeness sum).
"""

from __future__ import annotations

import numpy as np
import pytest

from unmochan.core.plane_waves import matching_g_mask, weights_from_coefficients
from unmochan.core.tight_binding import (
    TightBindingModel,
    unfold_tight_binding_model,
    unfold_tight_binding_path,
)
from unmochan.core.transformations import TransformationMatrix

TOL = 1e-10


def chain_model(hopping: float = -1.0, onsite: float = 0.0) -> TightBindingModel:
    """Single-orbital chain: ``E(k) = onsite + 2 * hopping * cos(2 pi k)``."""

    return TightBindingModel(
        1,
        {
            (0, 0, 0): [[onsite]],
            (1, 0, 0): [[hopping]],
            (-1, 0, 0): [[hopping]],
        },
    )


def square_model(tx: float = -1.0, ty: float = -0.7) -> TightBindingModel:
    """Square lattice: ``E(k) = 2 tx cos(2 pi kx) + 2 ty cos(2 pi ky)``."""

    return TightBindingModel(
        1,
        {
            (1, 0, 0): [[tx]],
            (-1, 0, 0): [[tx]],
            (0, 1, 0): [[ty]],
            (0, -1, 0): [[ty]],
        },
    )


def _assert_single_primitive_band(
    weights: np.ndarray, energies: np.ndarray, expected_energy: float
) -> None:
    """The unfolded spectrum at one k-point is a single primitive band.

    The whole weight (one, for a one-orbital model) sits at the primitive band
    energy.  It need not sit on a *single* eigenvector: two members of the
    fiber can fold onto degenerate supercell states, and the diagonalizer is
    then free to return any basis of that eigenspace.  What is basis
    independent -- and what the closed form predicts -- is the total weight and
    the energy of every state that carries any of it.
    """

    assert abs(weights.sum() - 1.0) < 1e-9
    carrying = np.flatnonzero(weights > 1e-8)
    assert carrying.size >= 1
    assert np.max(np.abs(energies[carrying] - expected_energy)) < 1e-9


@pytest.mark.parametrize("multiplicity", [2, 3, 4, 5])
def test_perfect_chain_unfolds_to_the_primitive_band(multiplicity: int):
    """A perfect supercell must give back the primitive cosine band."""

    hopping = -1.0
    transform = TransformationMatrix(np.diag([multiplicity, 1, 1]))
    kpoints = np.stack([np.linspace(0.0, 0.5, 11), np.zeros(11), np.zeros(11)], axis=1)
    energies, weights = unfold_tight_binding_path(chain_model(hopping), transform, kpoints)

    assert energies.shape == (11, multiplicity)
    for row, k in enumerate(kpoints[:, 0]):
        expected = 2.0 * hopping * np.cos(2.0 * np.pi * k)
        _assert_single_primitive_band(weights[row], energies[row], expected)


def test_perfect_square_lattice_unfolds_to_the_primitive_band():
    tx, ty = -1.0, -0.7
    transform = TransformationMatrix(np.diag([2, 2, 1]))
    kpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.25, 0.0], [0.5, 0.5, 0.0]])
    energies, weights = unfold_tight_binding_path(square_model(tx, ty), transform, kpoints)

    for row, k in enumerate(kpoints):
        expected = 2.0 * tx * np.cos(2.0 * np.pi * k[0]) + 2.0 * ty * np.cos(2.0 * np.pi * k[1])
        _assert_single_primitive_band(weights[row], energies[row], expected)


def test_dimerized_chain_opens_the_expected_gap():
    """A bond-alternating chain: the gap at the zone boundary is ``2|t1 - t2|``.

    Doubling the cell and alternating the hoppings ``t1, t2`` gives the
    Su-Schrieffer-Heeger two-band model, whose spectrum is
    ``±|t1 + t2 exp(2 pi i k)``.  Unfolded onto the primitive zone, both
    branches carry weight one half at the zone boundary -- the point where the
    two primitive states are degenerate and mix.
    """

    t1, t2 = -1.0, -0.6
    # A 2x1x1 supercell of the chain with the two bonds made inequivalent.
    model = TightBindingModel(
        2,
        {
            (0, 0, 0): [[0.0, t1], [t1, 0.0]],
            (1, 0, 0): [[0.0, 0.0], [t2, 0.0]],
            (-1, 0, 0): [[0.0, t2], [0.0, 0.0]],
        },
    )
    identity = TransformationMatrix(np.eye(3, dtype=int))
    _kpoints, energies, weights = unfold_tight_binding_model(
        model, identity, np.array([[0.5, 0.0, 0.0]])
    )
    gap = float(np.ptp(energies[0]))
    assert abs(gap - 2.0 * abs(t1 - t2)) < 1e-9
    # A 1x1x1 "supercell" is no supercell: every state keeps its full weight.
    assert np.allclose(weights, 1.0)


def test_single_plane_wave_state_is_the_ideal_supercell_case():
    """The closed form of ``UnfoldLab.IsFiberRepr.weight_ideal``.

    A supercell state occupying only plane waves that unfold to one primitive
    k-point has weight one there and zero at the other fiber members.
    """

    transform = np.diag([3, 1, 1])
    g_vectors = np.array([[m, 0, 0] for m in range(-4, 5)])
    folded = np.array([0.0, 0.0, 0.0])
    fiber = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0], [2.0 / 3.0, 0.0, 0.0]])

    for index, k in enumerate(fiber):
        # Occupy exactly the plane waves that belong to this fiber member.
        mask = matching_g_mask(g_vectors, k, folded, transform)
        coefficients = np.zeros((1, 1, g_vectors.shape[0]), dtype=complex)
        coefficients[0, 0, mask] = 1.0 + 0.0j

        weights = np.array(
            [
                weights_from_coefficients(g_vectors, coefficients, other, folded, transform)[0]
                for other in fiber
            ]
        )
        expected = np.zeros(3)
        expected[index] = 1.0
        assert np.max(np.abs(weights - expected)) < TOL
        # And the fiber sum rule holds trivially.
        assert abs(weights.sum() - 1.0) < TOL


def test_defect_chain_conserves_the_fiber_sum_rule():
    """A defect breaks the ideal case but not the sum rule."""

    transform = TransformationMatrix(np.diag([4, 1, 1]))
    supercell_kpoints = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.25, 0.0, 0.0]])
    _kpoints, _energies, weights = unfold_tight_binding_model(
        chain_model(), transform, supercell_kpoints, perturbations=[(0, 0, 1.5)]
    )

    multiplicity = 4
    n_states = weights.shape[1]
    assert weights.shape == (supercell_kpoints.shape[0] * multiplicity, n_states)
    # Fiber sum rule at fixed state index: the |det T| members share the weight.
    for block in range(supercell_kpoints.shape[0]):
        fiber_weights = weights[block * multiplicity : (block + 1) * multiplicity]
        assert np.max(np.abs(fiber_weights.sum(axis=0) - 1.0)) < 1e-9
    # A defect really does spread the weight: the ideal case is now violated.
    assert np.min(weights) < 0.99


def test_perturbation_must_be_a_scalar_shift():
    """A matrix-valued on-site term is a user error with a clear message."""

    transform = TransformationMatrix(np.diag([2, 1, 1]))
    with pytest.raises(ValueError, match="single on-site energy shift"):
        unfold_tight_binding_path(
            chain_model(),
            transform,
            np.zeros((1, 3)),
            perturbations=[(0, 0, [[1.0, 0.0], [0.0, 1.0]])],
        )
