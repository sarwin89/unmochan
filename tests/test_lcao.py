"""Unfolding in a non-orthogonal (LCAO) basis.

Every claim checked here has a counterpart in
``RequestProject/Unfolding/Overlap.lean``; the names are quoted in the tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from unmochan.core.lcao import (
    bloch_overlap_kernel,
    diagnose_overlap_neglect,
    lcao_norms,
    lcao_weights,
    orthonormal_overlap_kernel,
    overlap_kernel_from_matrix,
    overlap_neglect_from_model,
    reciprocal_overlap,
    solve_generalized_eigenproblem,
    supercell_overlap_matrix,
    unfold_lcao,
    unfold_lcao_model,
    unfold_lcao_path,
    validate_overlap_kernel,
)
from unmochan.core.tight_binding import (
    TightBindingModel,
    supercell_cells,
    tight_binding_weights,
)
from unmochan.core.transformations import TransformationMatrix

pytestmark = pytest.mark.unit

MATRICES = [
    np.diag([3, 1, 1]),
    np.diag([2, 2, 1]),
    np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]]),
    np.array([[2, 1, 0], [-1, 1, 0], [0, 0, 2]]),
]

KPOINTS = [
    np.zeros(3),
    np.array([0.1, 0.2, 0.0]),
    np.array([-0.37, 0.11, 0.25]),
]


def random_coefficients(n_states: int, n_orbitals: int, n_cells: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_states, n_orbitals, n_cells)) + 1j * rng.normal(
        size=(n_states, n_orbitals, n_cells)
    )


def overlap_model(n_orbitals: int, strength: float, seed: int) -> TightBindingModel:
    """A short-ranged, diagonally dominant primitive overlap table."""

    rng = np.random.default_rng(seed)
    neighbours = ((1, 0, 0), (0, 1, 0), (1, 1, 0))
    blocks = {(0, 0, 0): np.eye(n_orbitals, dtype=complex)}
    for cell in neighbours:
        block = rng.normal(size=(n_orbitals, n_orbitals)) + 1j * rng.normal(
            size=(n_orbitals, n_orbitals)
        )
        # Scale so that Gershgorin keeps the supercell overlap positive
        # definite: every off-diagonal row sum stays below 2 * strength.
        block /= np.max(np.abs(block)) * n_orbitals * len(neighbours)
        blocks[cell] = strength * block
    return TightBindingModel(n_orbitals, blocks).hermitized()


@pytest.mark.parametrize("matrix", MATRICES)
def test_orthonormal_kernel_reproduces_tight_binding(matrix: np.ndarray) -> None:
    """``UnfoldLab.ovWeight_orthonormal``: an orthonormal kernel changes nothing."""

    transform = TransformationMatrix(matrix)
    cells = supercell_cells(transform)
    coefficients = random_coefficients(4, 3, cells.shape[0], seed=11)
    kernel = orthonormal_overlap_kernel(3, cells)
    points, weights = unfold_lcao(
        cells, coefficients, kernel, transform, np.array([0.13, -0.2, 0.05])
    )
    reference = tight_binding_weights(
        cells, coefficients, points, multiplicity=transform.multiplicity
    )
    assert np.allclose(weights, reference, atol=1e-12)


def test_orthonormal_reciprocal_overlap_is_the_identity() -> None:
    transform = TransformationMatrix(np.diag([2, 2, 1]))
    cells = supercell_cells(transform)
    kernel = orthonormal_overlap_kernel(2, cells)
    reciprocal = reciprocal_overlap(cells, kernel, np.array([[0.1, 0.3, 0.0]]))
    assert np.allclose(reciprocal[0], np.eye(2), atol=1e-12)


@pytest.mark.parametrize("matrix", MATRICES)
@pytest.mark.parametrize("kpoint", KPOINTS)
def test_fiber_sum_rule_with_overlap(matrix: np.ndarray, kpoint: np.ndarray) -> None:
    """``UnfoldLab.sum_ovWeight_eq_one``: the fiber weights add up to one."""

    transform = TransformationMatrix(matrix)
    overlap = overlap_model(2, 0.15, seed=5)
    kernel, cells = bloch_overlap_kernel(overlap.hoppings, transform, kpoint)
    validate_overlap_kernel(cells, kernel, transform, kpoint, require_positive_definite=True)
    coefficients = random_coefficients(5, 2, cells.shape[0], seed=7)
    _, weights = unfold_lcao(cells, coefficients, kernel, transform, kpoint)
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-10)


def test_weights_are_invariant_under_rescaling() -> None:
    """``UnfoldLab.ovWeight_smul``: a normalization convention cannot matter."""

    transform = TransformationMatrix(np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]]))
    kpoint = np.array([0.2, -0.1, 0.0])
    overlap = overlap_model(2, 0.2, seed=3)
    kernel, cells = bloch_overlap_kernel(overlap.hoppings, transform, kpoint)
    coefficients = random_coefficients(3, 2, cells.shape[0], seed=13)
    _, weights = unfold_lcao(cells, coefficients, kernel, transform, kpoint)
    _, scaled = unfold_lcao(cells, (0.5 - 2.5j) * coefficients, kernel, transform, kpoint)
    assert np.allclose(weights, scaled, atol=1e-12)


def test_zero_state_gets_zero_weight() -> None:
    transform = TransformationMatrix(np.diag([2, 1, 1]))
    kpoint = np.array([0.25, 0.0, 0.0])
    overlap = overlap_model(2, 0.1, seed=2)
    kernel, cells = bloch_overlap_kernel(overlap.hoppings, transform, kpoint)
    coefficients = np.zeros((1, 2, cells.shape[0]), dtype=complex)
    _, weights = unfold_lcao(cells, coefficients, kernel, transform, kpoint)
    assert np.array_equal(weights, np.zeros_like(weights))


def test_norm_is_the_quadratic_form_of_the_dense_overlap() -> None:
    transform = TransformationMatrix(np.array([[2, 1, 0], [-1, 1, 0], [0, 0, 2]]))
    kpoint = np.array([0.05, 0.35, -0.15])
    overlap = overlap_model(3, 0.12, seed=17)
    kernel, cells = bloch_overlap_kernel(overlap.hoppings, transform, kpoint)
    matrix = supercell_overlap_matrix(cells, kernel, transform, kpoint)
    coefficients = random_coefficients(2, 3, cells.shape[0], seed=19)
    norms = lcao_norms(cells, coefficients, kernel, transform, kpoint)
    flat = np.swapaxes(coefficients, 1, 2).reshape(coefficients.shape[0], -1)
    expected = np.einsum("si,ij,sj->s", flat.conj(), matrix, flat).real
    assert np.allclose(norms, expected, atol=1e-12)
    assert np.all(norms > 0.0)


def test_dense_overlap_is_hermitian_and_positive_definite() -> None:
    transform = TransformationMatrix(np.diag([3, 1, 1]))
    kpoint = np.array([0.1, 0.0, 0.0])
    overlap = overlap_model(2, 0.15, seed=23)
    kernel, cells = bloch_overlap_kernel(overlap.hoppings, transform, kpoint)
    matrix = validate_overlap_kernel(
        cells, kernel, transform, kpoint, require_positive_definite=True
    )
    assert np.allclose(matrix, matrix.conj().T, atol=1e-12)


def test_non_hermitian_kernel_is_rejected() -> None:
    transform = TransformationMatrix(np.diag([2, 1, 1]))
    kpoint = np.zeros(3)
    cells = supercell_cells(transform)
    kernel = orthonormal_overlap_kernel(2, cells).astype(complex)
    kernel[0, 1, 0] = 0.5  # breaks S_{ab}(0)* = S_{ba}(0)
    with pytest.raises(ValueError, match="not Hermitian"):
        validate_overlap_kernel(cells, kernel, transform, kpoint)


def test_kernel_round_trips_through_the_dense_matrix() -> None:
    transform = TransformationMatrix(np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]]))
    kpoint = np.array([0.3, 0.1, 0.0])
    overlap = overlap_model(2, 0.2, seed=29)
    kernel, cells = bloch_overlap_kernel(overlap.hoppings, transform, kpoint)
    matrix = supercell_overlap_matrix(cells, kernel, transform, kpoint)
    recovered = overlap_kernel_from_matrix(matrix, cells.shape[0], 2)
    assert np.allclose(recovered, kernel, atol=1e-12)


def test_generalized_eigenproblem_gives_s_orthonormal_states() -> None:
    rng = np.random.default_rng(31)
    size = 6
    a = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    hamiltonian = a + a.conj().T
    b = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    overlap = b @ b.conj().T + size * np.eye(size)
    energies, vectors = solve_generalized_eigenproblem(hamiltonian, overlap)
    assert np.allclose(vectors.conj().T @ overlap @ vectors, np.eye(size), atol=1e-10)
    residual = hamiltonian @ vectors - overlap @ vectors * energies[np.newaxis, :]
    assert np.max(np.abs(residual)) < 1e-9


def test_generalized_eigenproblem_rejects_a_singular_metric() -> None:
    hamiltonian = np.eye(2, dtype=complex)
    overlap = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
    with pytest.raises(ValueError, match="positive definite"):
        solve_generalized_eigenproblem(hamiltonian, overlap)


def test_perfect_non_orthogonal_chain_unfolds_to_one_kpoint() -> None:
    """A perfect crystal in an LCAO basis still folds back exactly.

    The primitive dispersion of the non-orthogonal chain is the closed form
    ``E(k) = 2 t cos(2 pi k) / (1 + 2 s cos(2 pi k))``, which the unfolded
    energies must reproduce at the k-point that carries the weight.
    """

    hop, ovl = -1.0, 0.2
    model = TightBindingModel(
        1,
        {
            (0, 0, 0): np.array([[0.0]], dtype=complex),
            (1, 0, 0): np.array([[hop]], dtype=complex),
            (-1, 0, 0): np.array([[hop]], dtype=complex),
        },
    )
    overlap = TightBindingModel(
        1,
        {
            (0, 0, 0): np.array([[1.0]], dtype=complex),
            (1, 0, 0): np.array([[ovl]], dtype=complex),
            (-1, 0, 0): np.array([[ovl]], dtype=complex),
        },
    )
    transform = TransformationMatrix(np.diag([3, 1, 1]))
    kpoints, energies, weights = unfold_lcao_model(
        model, overlap, transform, [np.array([0.07, 0.0, 0.0])]
    )
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-10)
    # Every band sits entirely on one primitive k-point.
    assert np.allclose(np.sort(weights, axis=0)[-1], 1.0, atol=1e-10)
    carrier = np.argmax(weights, axis=0)
    cosine = np.cos(2.0 * np.pi * kpoints[carrier, 0])
    exact = 2.0 * hop * cosine / (1.0 + 2.0 * ovl * cosine)
    assert np.allclose(energies[carrier, np.arange(energies.shape[1])], exact, atol=1e-10)


def test_defective_chain_shows_the_error_of_the_orthonormal_formula() -> None:
    """``UnfoldLab.ovNumer_sub_tbNumer``: ignoring the overlap misassigns weight.

    A defect-free supercell folds back exactly either way, so the discrepancy is
    exhibited on a state that genuinely spreads over the fiber.
    """

    transform = TransformationMatrix(np.diag([4, 1, 1]))
    kpoint = np.array([0.11, 0.0, 0.0])
    overlap = TightBindingModel(
        1,
        {
            (0, 0, 0): np.array([[1.0]], dtype=complex),
            (1, 0, 0): np.array([[0.3]], dtype=complex),
            (-1, 0, 0): np.array([[0.3]], dtype=complex),
        },
    )
    kernel, cells = bloch_overlap_kernel(overlap.hoppings, transform, kpoint)
    # A state localized on one cell: maximally spread over the fiber.
    coefficients = np.zeros((1, 1, cells.shape[0]), dtype=complex)
    coefficients[0, 0, 0] = 1.0
    report = diagnose_overlap_neglect(cells, coefficients, kernel, transform, kpoint)
    assert report.significant()
    assert report.max_kernel_deviation > 0.1
    assert report.max_weight_difference > 1e-3
    assert 0.0 < report.max_fiber_total_variation <= 1.0
    assert report.n_kpoints == transform.multiplicity
    assert set(report.to_dict()) == {
        "max_kernel_deviation",
        "max_norm_ratio_deviation",
        "max_weight_difference",
        "max_fiber_total_variation",
        "n_kpoints",
        "n_states",
    }


def test_orthonormal_case_reports_no_neglect() -> None:
    transform = TransformationMatrix(np.diag([2, 2, 1]))
    kpoint = np.array([0.25, 0.25, 0.0])
    cells = supercell_cells(transform)
    kernel = orthonormal_overlap_kernel(2, cells)
    coefficients = random_coefficients(3, 2, cells.shape[0], seed=41)
    report = diagnose_overlap_neglect(cells, coefficients, kernel, transform, kpoint)
    assert not report.significant()
    assert report.max_weight_difference < 1e-12


def test_input_validation() -> None:
    transform = TransformationMatrix(np.diag([2, 1, 1]))
    cells = supercell_cells(transform)
    kernel = orthonormal_overlap_kernel(2, cells)
    coefficients = random_coefficients(1, 2, cells.shape[0], seed=43)
    with pytest.raises(ValueError, match="one entry per orbital"):
        lcao_weights(
            cells,
            coefficients[:, :1, :],
            kernel,
            np.zeros((1, 3)),
            transform,
            np.zeros(3),
        )
    with pytest.raises(ValueError, match="one entry per cell"):
        lcao_weights(
            cells,
            coefficients,
            kernel[:, :, :1],
            np.zeros((1, 3)),
            transform,
            np.zeros(3),
        )
    with pytest.raises(ValueError, match="multiplicity must be positive"):
        lcao_weights(
            cells,
            coefficients,
            kernel,
            np.zeros((1, 3)),
            transform,
            np.zeros(3),
            multiplicity=0,
        )
    with pytest.raises(ValueError, match="at least one block"):
        bloch_overlap_kernel({}, transform, np.zeros(3))
    with pytest.raises(ValueError, match="same number of orbitals"):
        unfold_lcao_model(
            TightBindingModel(1, {(0, 0, 0): np.array([[0.0]], dtype=complex)}),
            TightBindingModel(2, {(0, 0, 0): np.eye(2, dtype=complex)}),
            transform,
            [np.zeros(3)],
        )


def test_path_unfolding_reproduces_the_non_orthogonal_dispersion() -> None:
    """The band-path counterpart: a perfect chain keeps weight one on its band."""

    hop, ovl = -1.0, 0.25
    model = TightBindingModel(
        1,
        {
            (1, 0, 0): np.array([[hop]], dtype=complex),
            (-1, 0, 0): np.array([[hop]], dtype=complex),
        },
    )
    overlap = TightBindingModel(
        1,
        {
            (0, 0, 0): np.array([[1.0]], dtype=complex),
            (1, 0, 0): np.array([[ovl]], dtype=complex),
            (-1, 0, 0): np.array([[ovl]], dtype=complex),
        },
    )
    transform = TransformationMatrix(np.diag([3, 1, 1]))
    grid = np.stack([np.linspace(0.0, 0.5, 6), np.zeros(6), np.zeros(6)], axis=1)
    energies, weights = unfold_lcao_path(model, overlap, transform, grid)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)
    carrier = np.argmax(weights, axis=1)
    picked = energies[np.arange(grid.shape[0]), carrier]
    cosine = np.cos(2.0 * np.pi * grid[:, 0])
    exact = 2.0 * hop * cosine / (1.0 + 2.0 * ovl * cosine)
    assert np.allclose(picked, exact, atol=1e-10)


def test_band_sum_rule_survives_the_overlap() -> None:
    """A complete S-orthonormal set still carries `n_orbitals` per k-point.

    The across-bands sum rule is not obvious once the basis overlaps -- the
    completeness relation is ``sum_s c_s c_s* = S^-1`` rather than the identity
    -- but the two overlaps cancel and the total is unchanged.  Proved in
    ``UnfoldLab.sum_ovWeight_states_of_inverse``
    (``RequestProject/Unfolding/OverlapCompleteness.lean``).
    """

    transform = TransformationMatrix(np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]]))
    model = TightBindingModel(
        2,
        {
            (0, 0, 0): np.diag([0.2, -0.3]).astype(complex),
            (1, 0, 0): np.array([[-1.0, 0.2], [0.1, -0.8]], dtype=complex),
            (0, 1, 0): np.array([[-0.7, 0.05], [0.3, -0.6]], dtype=complex),
        },
    ).hermitized()
    overlap = overlap_model(2, 0.2, seed=53)
    grid = np.array([[0.0, 0.0, 0.0], [0.13, -0.2, 0.0], [0.4, 0.25, 0.1]])
    _, weights = unfold_lcao_path(model, overlap, transform, grid)
    assert np.allclose(weights.sum(axis=1), 2.0, atol=1e-9)


def test_on_site_defect_spreads_weight_over_the_fiber() -> None:
    transform = TransformationMatrix(np.diag([4, 1, 1]))
    model = TightBindingModel(
        1,
        {
            (1, 0, 0): np.array([[-1.0]], dtype=complex),
            (-1, 0, 0): np.array([[-1.0]], dtype=complex),
        },
    )
    overlap = TightBindingModel(
        1,
        {
            (0, 0, 0): np.array([[1.0]], dtype=complex),
            (1, 0, 0): np.array([[0.25]], dtype=complex),
            (-1, 0, 0): np.array([[0.25]], dtype=complex),
        },
    )
    kpoint = np.array([0.1, 0.0, 0.0])
    _, _, weights = unfold_lcao_model(
        model, overlap, transform, [kpoint], perturbations=[(1, 0, 1.5)]
    )
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-10)
    # No band is a pure primitive state any more.
    assert float(weights.max()) < 1.0 - 1e-6
    report = overlap_neglect_from_model(
        model, overlap, transform, kpoint, perturbations=[(1, 0, 1.5)]
    )
    assert report.significant()
    assert report.max_weight_difference > 1e-3
