"""Orbital-projected ("fat band") tight-binding unfolding weights.

The properties checked here are the numerical face of the Lean statements
``UnfoldLab.tbOrbitalWeight_union_of_disjoint``,
``UnfoldLab.tbOrbitalWeight_le_tbWeight`` and
``UnfoldLab.sum_tbOrbitalWeight_eq_norm_share``.
"""

from __future__ import annotations

import numpy as np
import pytest

from unmochan.core.tight_binding import (
    TightBindingModel,
    supercell_cells,
    tight_binding_orbital_weights,
    tight_binding_weights,
    unfold_tight_binding,
    unfold_tight_binding_path,
    unfold_tight_binding_path_projected,
)
from unmochan.core.transformations import TransformationMatrix


def _transform() -> TransformationMatrix:
    return TransformationMatrix(np.diag([3, 1, 1]))


def _two_orbital_chain() -> TightBindingModel:
    """A two-site chain: orbital 0 low, orbital 1 high, weakly coupled."""

    return TightBindingModel(
        2,
        {
            (0, 0, 0): np.array([[-1.5, 0.15], [0.15, 1.5]]),
            (1, 0, 0): np.array([[-0.4, 0.05], [0.05, 0.3]]),
            (-1, 0, 0): np.array([[-0.4, 0.05], [0.05, 0.3]]),
        },
    )


def _random_state(rng: np.random.Generator, n_orbitals: int, n_cells: int):
    real = rng.normal(size=(4, n_orbitals, n_cells))
    imaginary = rng.normal(size=(4, n_orbitals, n_cells))
    return real + 1j * imaginary


def test_a_partition_of_the_orbitals_reproduces_the_total_weight() -> None:
    transform = _transform()
    cells = supercell_cells(transform)
    rng = np.random.default_rng(7)
    coefficients = _random_state(rng, 4, cells.shape[0])
    kpoints = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0], [0.2, 0.1, 0.4]])

    total = tight_binding_weights(cells, coefficients, kpoints)
    parts = tight_binding_orbital_weights(cells, coefficients, kpoints, [[0, 2], [1], [3]])

    assert parts.shape == (3, kpoints.shape[0], coefficients.shape[0])
    assert parts.sum(axis=0) == pytest.approx(total)


def test_every_group_weight_is_bounded_by_the_total() -> None:
    transform = _transform()
    cells = supercell_cells(transform)
    rng = np.random.default_rng(11)
    coefficients = _random_state(rng, 3, cells.shape[0])
    kpoints = np.array([[0.1, 0.0, 0.0], [0.5, 0.25, 0.0]])

    total = tight_binding_weights(cells, coefficients, kpoints)
    parts = tight_binding_orbital_weights(cells, coefficients, kpoints, [[0], [0, 1]])

    assert np.all(parts >= 0.0)
    assert np.all(parts <= total[np.newaxis, :, :] + 1e-12)
    # Monotone in the group: {0} is contained in {0, 1}.
    assert np.all(parts[0] <= parts[1] + 1e-12)


def test_the_fiber_sum_of_a_group_is_its_share_of_the_norm() -> None:
    transform = _transform()
    cells = supercell_cells(transform)
    rng = np.random.default_rng(3)
    coefficients = _random_state(rng, 4, cells.shape[0])
    fiber, _ = unfold_tight_binding(cells, coefficients, transform, np.array([0.0, 0.0, 0.0]))

    group = [1, 3]
    parts = tight_binding_orbital_weights(
        cells, coefficients, fiber, [group], multiplicity=transform.multiplicity
    )
    density = (coefficients.real**2 + coefficients.imag**2).sum(axis=2)
    share = density[:, group].sum(axis=1) / density.sum(axis=1)

    assert parts[0].sum(axis=0) == pytest.approx(share)


def test_an_empty_group_carries_no_weight() -> None:
    cells = supercell_cells(_transform())
    rng = np.random.default_rng(5)
    coefficients = _random_state(rng, 2, cells.shape[0])
    kpoints = np.array([[0.0, 0.0, 0.0]])

    parts = tight_binding_orbital_weights(cells, coefficients, kpoints, [[]])

    assert parts == pytest.approx(np.zeros_like(parts))


def test_out_of_range_and_repeated_orbitals_are_rejected() -> None:
    cells = supercell_cells(_transform())
    coefficients = np.ones((1, 2, cells.shape[0]), dtype=complex)
    kpoints = np.zeros((1, 3))

    with pytest.raises(ValueError, match="outside"):
        tight_binding_orbital_weights(cells, coefficients, kpoints, [[2]])
    with pytest.raises(ValueError, match="repeats"):
        tight_binding_orbital_weights(cells, coefficients, kpoints, [[0, 0]])


def test_projected_path_agrees_with_the_plain_path() -> None:
    model = _two_orbital_chain()
    transform = _transform()
    path = np.stack([np.linspace(0.0, 0.5, 9), np.zeros(9), np.zeros(9)], axis=1)

    energies, weights = unfold_tight_binding_path(model, transform, path)
    projected_energies, projected_weights, parts = unfold_tight_binding_path_projected(
        model, transform, path, [[0], [1]]
    )

    assert projected_energies == pytest.approx(energies)
    assert projected_weights == pytest.approx(weights)
    assert parts.sum(axis=0) == pytest.approx(weights)


def test_the_low_band_of_a_perfect_crystal_is_the_low_orbital() -> None:
    """With a weak inter-orbital coupling the lowest band is orbital 0.

    A perfect-crystal supercell folds each primitive band onto exactly one
    fiber member with weight one; the fat band then says *which* orbital that
    unit of weight sits on, which is the whole point of the projection.
    """

    model = _two_orbital_chain()
    transform = _transform()
    path = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])

    energies, weights, parts = unfold_tight_binding_path_projected(
        model, transform, path, [[0], [1]]
    )

    for index in range(path.shape[0]):
        visible = np.flatnonzero(weights[index] > 0.5)
        assert visible.size == 2  # two primitive bands per k-point
        low, high = visible[np.argsort(energies[index][visible])]
        assert parts[0, index, low] > parts[1, index, low]
        assert parts[1, index, high] > parts[0, index, high]
