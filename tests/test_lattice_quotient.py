"""Exact enumeration of ``Z^3 / T Z^3`` and of the unfolding fiber."""

from __future__ import annotations

import numpy as np
import pytest

from unmochan.core.kpoints import (
    KPoint,
    fiber_kpoint_mappings,
    fiber_kpoints,
    fold_kpoint_to_supercell,
)
from unmochan.core.lattice_quotient import coset_representatives, smith_normal_form
from unmochan.core.numerics import wrap_fractional
from unmochan.core.plane_waves import weights_from_coefficients
from unmochan.core.transformations import TransformationMatrix

MATRICES = [
    np.diag([1, 1, 1]),
    np.diag([2, 2, 1]),
    np.diag([3, 1, 1]),
    np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]]),
    np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]),
    np.array([[2, 0, 0], [0, 1, 0], [0, 0, -3]]),
    np.array([[3, 1, 2], [1, 2, 1], [0, 1, 3]]),
]


def _random_nonsingular(rng: np.random.Generator) -> np.ndarray:
    while True:
        matrix = rng.integers(-4, 5, size=(3, 3))
        if abs(round(float(np.linalg.det(matrix)))) > 0:
            return matrix


@pytest.mark.parametrize("matrix", MATRICES)
def test_smith_normal_form_factorizes_exactly(matrix: np.ndarray) -> None:
    snf = smith_normal_form(matrix)
    assert np.array_equal(snf.left @ matrix @ snf.right, snf.diagonal)
    assert abs(round(float(np.linalg.det(snf.left)))) == 1
    assert abs(round(float(np.linalg.det(snf.right)))) == 1
    assert np.array_equal(snf.left @ snf.left_inverse, np.eye(3, dtype=int))


@pytest.mark.parametrize("matrix", MATRICES)
def test_invariant_factors_are_a_divisibility_chain(matrix: np.ndarray) -> None:
    factors = smith_normal_form(matrix).invariant_factors
    assert all(factor > 0 for factor in factors)
    assert all(factors[i] % factors[i - 1] == 0 for i in range(1, len(factors)))
    assert int(np.prod(factors)) == abs(round(float(np.linalg.det(matrix))))


def test_smith_normal_form_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        smith_normal_form(np.ones((2, 3), dtype=int))


def test_coset_representatives_rejects_singular_matrix() -> None:
    with pytest.raises(ValueError, match="singular"):
        coset_representatives(np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1]]))


@pytest.mark.parametrize("matrix", MATRICES)
def test_coset_representatives_are_a_transversal(matrix: np.ndarray) -> None:
    reps = coset_representatives(matrix)
    determinant = abs(round(float(np.linalg.det(matrix))))
    assert reps.shape == (determinant, 3)
    assert np.array_equal(reps[0], np.zeros(3, dtype=int))

    inverse = np.linalg.inv(matrix)
    classes = set()
    for rep in reps:
        residue = inverse @ rep
        classes.add(tuple(np.round((residue - np.floor(residue)) * 1e7).astype(np.int64) % 10**7))
    assert len(classes) == determinant


def test_coset_representatives_cover_every_class() -> None:
    matrix = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    reps = coset_representatives(matrix)
    inverse = np.linalg.inv(matrix)
    # Every small integer vector must be congruent to exactly one representative.
    for x in range(-3, 4):
        for y in range(-3, 4):
            for z in range(-2, 3):
                target = np.array([x, y, z], dtype=float)
                hits = 0
                for rep in reps:
                    residue = inverse @ (target - rep)
                    if np.max(np.abs(residue - np.rint(residue))) < 1e-9:
                        hits += 1
                assert hits == 1


@pytest.mark.parametrize("matrix", MATRICES)
def test_fiber_kpoints_all_fold_back(matrix: np.ndarray) -> None:
    transform = TransformationMatrix(matrix)
    folded = np.array([0.3, 0.15, 0.4])
    members = fiber_kpoints(folded, transform)
    assert members.shape == (transform.multiplicity, 3)
    for point in members:
        back = fold_kpoint_to_supercell(point, transform).fractional
        delta = back - folded
        assert np.max(np.abs(delta - np.rint(delta))) < 1e-9


@pytest.mark.parametrize("matrix", MATRICES)
def test_fiber_members_are_pairwise_inequivalent(matrix: np.ndarray) -> None:
    transform = TransformationMatrix(matrix)
    members = fiber_kpoints(np.array([0.1, 0.7, 0.25]), transform)
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            delta = members[i] - members[j]
            assert np.max(np.abs(delta - np.rint(delta))) > 1e-9


def test_fiber_kpoints_is_reproducible_and_wrapped() -> None:
    transform = TransformationMatrix(np.diag([2, 2, 1]))
    first = fiber_kpoints([0.5, 0.0, 0.0], transform)
    second = fiber_kpoints([0.5, 0.0, 0.0], transform)
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first < 1.0))
    unwrapped = fiber_kpoints([0.5, 0.0, 0.0], transform, wrap=False)
    assert unwrapped.shape == first.shape
    reduced = np.sort(wrap_fractional(unwrapped), axis=0)
    assert np.allclose(reduced, np.sort(first, axis=0))


def test_fiber_kpoints_matches_brute_force_on_a_diagonal_cell() -> None:
    transform = TransformationMatrix(np.diag([2, 3, 1]))
    folded = np.array([0.25, 0.5, 0.0])
    expected = {
        (round((folded[0] + i) / 2, 9), round((folded[1] + j) / 3, 9), 0.0)
        for i in range(2)
        for j in range(3)
    }
    got = {tuple(round(float(x), 9) for x in point) for point in fiber_kpoints(folded, transform)}
    assert got == expected


def test_fiber_kpoint_mappings_split_the_weight() -> None:
    transform = TransformationMatrix(np.diag([2, 2, 1]))
    mappings = fiber_kpoint_mappings(
        [KPoint([0.0, 0.0, 0.0], weight=2.0, label="G"), KPoint([0.5, 0.0, 0.0], weight=1.0)],
        transform,
    )
    assert len(mappings) == 8
    assert sum(mapping.primitive.weight for mapping in mappings) == pytest.approx(3.0)
    assert mappings[0].primitive.label == "G"
    for mapping in mappings:
        back = fold_kpoint_to_supercell(mapping.primitive, transform).fractional
        delta = back - mapping.supercell.fractional
        assert np.max(np.abs(delta - np.rint(delta))) < 1e-9


@pytest.mark.parametrize("matrix", MATRICES)
def test_a_complete_fiber_makes_the_sum_rule_exact(matrix: np.ndarray) -> None:
    """The point of enumerating the fiber: the weights then add up to one."""

    transform = TransformationMatrix(matrix)
    rng = np.random.default_rng(20240905)
    g_vectors = rng.integers(-3, 4, size=(60, 3))
    coefficients = (rng.normal(size=(4, 1, 60)) + 1j * rng.normal(size=(4, 1, 60))).astype(
        np.complex128
    )
    folded = np.array([0.2, 0.35, 0.1])

    total = np.zeros(4)
    for point in fiber_kpoints(folded, transform):
        total += weights_from_coefficients(g_vectors, coefficients, point, folded, transform.matrix)
    assert np.allclose(total, 1.0, atol=1e-12)


def test_random_transforms_give_complete_fibers() -> None:
    rng = np.random.default_rng(7)
    for _ in range(25):
        matrix = _random_nonsingular(rng)
        transform = TransformationMatrix(matrix)
        folded = rng.random(3)
        members = fiber_kpoints(folded, transform)
        assert len(members) == transform.multiplicity
        for point in members:
            delta = point @ matrix.T - folded
            assert np.max(np.abs(delta - np.rint(delta))) < 1e-9


def test_fiber_cli_lists_the_fiber_and_writes_a_kmap(tmp_path, capsys) -> None:
    import json as _json

    from unmochan.cli.main import main
    from unmochan.io.qe import read_kmap

    out = tmp_path / "fiber.kmap"
    code = main(
        [
            "fiber",
            "--matrix",
            "2 0 0 0 2 0 0 0 1",
            "--kpoint",
            "0.5,0,0",
            "--kmap",
            str(out),
        ]
    )
    assert code == 0
    payload = _json.loads(capsys.readouterr().out.split("Wrote")[0])
    assert payload["multiplicity"] == 4
    assert len(payload["fibers"]) == 1
    assert len(payload["fibers"][0]["primitive"]) == 4

    kmap = read_kmap(out)
    assert kmap.n_kpoints == 4
    assert np.allclose(kmap.supercell_folded_kpoints, np.array([0.5, 0.0, 0.0]))


def test_fiber_cli_requires_a_transform(tmp_path) -> None:
    from unmochan.cli.main import main

    assert main(["fiber", "--kpoint", "0,0,0"]) != 0
