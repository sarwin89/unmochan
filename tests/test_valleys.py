import numpy as np
import pytest

from unfoldlab.core.valleys import (
    ValleyDefinition,
    lattice_growth_constant,
    minimum_image_distance,
)


def test_valley_definition_contains_points_modulo_reciprocal_lattice():
    valley = ValleyDefinition(label="user_defined", center_frac=[0.95, 0.0, 0.0], radius=0.1)

    assert valley.contains([0.01, 0.0, 0.0])
    assert not valley.contains([0.5, 0.0, 0.0])


def test_valley_definition_from_config():
    valley = ValleyDefinition.from_config(
        "example",
        {"center_frac": [1 / 3, 1 / 3, 0.0], "radius": 0.05, "reference_bz": "layer_1"},
    )

    assert valley.label == "example"
    assert valley.reference_bz == "layer_1"


def _hexagonal_reciprocal_lattice(a: float = 2.46, c: float = 10.0) -> np.ndarray:
    lattice = np.array([[a, 0.0, 0.0], [-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0], [0.0, 0.0, c]])
    return 2.0 * np.pi * np.linalg.inv(lattice).T


def test_minimum_image_distance_uses_cartesian_metric():
    recip = _hexagonal_reciprocal_lattice()
    delta = np.array([0.1, 0.0, 0.0])

    assert minimum_image_distance(delta, recip) == pytest.approx(
        float(np.linalg.norm(delta @ recip))
    )
    # Without a lattice the fractional norm is returned, which is a different
    # number entirely for a hexagonal cell.
    assert minimum_image_distance(delta) == pytest.approx(0.1)


def test_minimum_image_beats_naive_rint_reduction_for_hexagonal_cell():
    recip = _hexagonal_reciprocal_lattice()
    delta = np.array([0.4, 0.4, 0.0])

    naive = float(np.linalg.norm((delta - np.rint(delta)) @ recip))
    best = minimum_image_distance(delta, recip)

    assert best < naive - 1e-9

    brute = min(
        float(np.linalg.norm((delta + shift) @ recip))
        for shift in np.array(
            [(i, j, k) for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5)],
            dtype=float,
        )
    )
    assert best == pytest.approx(brute)


def test_valley_contains_is_a_true_sphere_when_lattice_is_given():
    recip = _hexagonal_reciprocal_lattice()
    b = float(np.linalg.norm(recip[0]))
    valley = ValleyDefinition(label="K", center_frac=[1 / 3, 1 / 3, 0.0], radius=0.2 * b)

    assert valley.contains([1 / 3, 1 / 3, 0.0], recip)
    # A point displaced along b1 by 0.1 fractional units is 0.1*b away.
    assert valley.contains([1 / 3 + 0.1, 1 / 3, 0.0], recip)
    assert not valley.contains([1 / 3 + 0.3, 1 / 3, 0.0], recip)
    assert valley.distance_to([1 / 3 + 0.1, 1 / 3, 0.0], recip) == pytest.approx(0.1 * b)


def test_valley_contains_defaults_to_fractional_metric():
    valley = ValleyDefinition(label="gamma", center_frac=[0.0, 0.0, 0.0], radius=0.1)

    assert valley.contains([0.05, 0.0, 0.0])
    assert not valley.contains([0.2, 0.0, 0.0])


def _brute_force_minimum_image(delta: np.ndarray, recip: np.ndarray, radius: int = 6) -> float:
    span = range(-radius, radius + 1)
    shifts = np.array([(i, j, k) for i in span for j in span for k in span], dtype=float)
    return float(np.min(np.linalg.norm((delta + shifts) @ recip, axis=1)))


def test_growth_constant_bounds_every_lattice_translation():
    recip = _hexagonal_reciprocal_lattice()
    sigma = lattice_growth_constant(recip)
    rng = np.random.default_rng(5)

    assert sigma > 0.0
    for _ in range(200):
        m = rng.integers(-4, 5, size=3).astype(float)
        assert float(np.linalg.norm(m @ recip)) >= sigma * float(np.linalg.norm(m)) - 1e-12


def test_certified_search_matches_brute_force_for_skewed_cells():
    rng = np.random.default_rng(17)
    lattices = [
        _hexagonal_reciprocal_lattice(),
        np.array([[1.0, 0.0, 0.0], [0.9, 0.2, 0.0], [0.0, 0.0, 1.3]]),
        np.array([[2.0, 0.0, 0.0], [1.0, 1.8, 0.0], [0.5, 0.4, 0.3]]),
    ]
    for recip in lattices:
        for _ in range(25):
            delta = rng.uniform(-1.0, 1.0, size=3)
            expected = _brute_force_minimum_image(delta - np.rint(delta), recip)
            assert minimum_image_distance(delta, recip) == pytest.approx(expected)


def test_certified_search_reports_failure_instead_of_guessing():
    # A nearly degenerate cell needs a large box; with the box capped the search
    # refuses to return an uncertified value.
    skewed = np.array([[1.0, 0.0, 0.0], [0.999, 0.01, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(RuntimeError, match="did not certify"):
        minimum_image_distance([0.3, 0.3, 0.0], skewed, max_radius=1)
