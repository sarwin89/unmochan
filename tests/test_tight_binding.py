"""Tight-binding / Wannier / model-Hamiltonian unfolding."""

from __future__ import annotations

import numpy as np
import pytest

from unfoldlab.core.kpoints import fiber_kpoints
from unfoldlab.core.tight_binding import (
    TightBindingModel,
    diagnose_tight_binding_weights,
    supercell_bloch_hamiltonian,
    supercell_cells,
    tight_binding_weights,
    unfold_tight_binding,
    unfold_tight_binding_model,
    unfold_tight_binding_path,
    validate_supercell_cells,
)
from unfoldlab.core.transformations import TransformationMatrix

MATRICES = [
    np.diag([1, 1, 1]),
    np.diag([3, 1, 1]),
    np.diag([2, 2, 1]),
    np.array([[2, 1, 0], [-1, 1, 0], [0, 0, 1]]),
    np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]),
    np.array([[2, 0, 0], [0, 1, 0], [0, 0, -3]]),
    np.array([[3, 1, 2], [1, 2, 1], [0, 1, 3]]),
]


def _chain_model(hopping: float = -1.0, onsite: float = 0.0) -> TightBindingModel:
    return TightBindingModel(
        1,
        {
            (0, 0, 0): [[onsite]],
            (1, 0, 0): [[hopping]],
            (-1, 0, 0): [[hopping]],
        },
    )


def _square_model() -> TightBindingModel:
    return TightBindingModel(
        1,
        {
            (0, 0, 0): [[0.3]],
            (1, 0, 0): [[-1.0]],
            (-1, 0, 0): [[-1.0]],
            (0, 1, 0): [[-0.7]],
            (0, -1, 0): [[-0.7]],
        },
    )


@pytest.mark.parametrize("matrix", MATRICES)
def test_supercell_cells_are_a_valid_transversal(matrix: np.ndarray) -> None:
    transform = TransformationMatrix(matrix)
    cells = supercell_cells(transform)
    assert cells.shape == (transform.multiplicity, 3)
    assert np.array_equal(cells[0], np.zeros(3, dtype=int))
    # Validation accepts exactly this list.
    assert np.array_equal(validate_supercell_cells(cells, transform), cells)


def test_validate_supercell_cells_rejects_wrong_count_and_duplicates() -> None:
    transform = TransformationMatrix(np.diag([3, 1, 1]))
    with pytest.raises(ValueError, match="3 = \\|det T\\|"):
        validate_supercell_cells([[0, 0, 0], [1, 0, 0]], transform)
    with pytest.raises(ValueError, match="pairwise inequivalent"):
        # (3, 0, 0) is the same cell as the origin for this supercell.
        validate_supercell_cells([[0, 0, 0], [1, 0, 0], [3, 0, 0]], transform)


@pytest.mark.parametrize("matrix", MATRICES)
def test_sum_rule_over_a_complete_fiber(matrix: np.ndarray) -> None:
    transform = TransformationMatrix(matrix)
    cells = supercell_cells(transform)
    rng = np.random.default_rng(11)
    coefficients = rng.normal(size=(4, 2, cells.shape[0])) + 1j * rng.normal(
        size=(4, 2, cells.shape[0])
    )
    supercell_kpoint = np.array([0.13, -0.27, 0.41])
    points, weights = unfold_tight_binding(cells, coefficients, transform, supercell_kpoint)
    assert points.shape == (transform.multiplicity, 3)
    assert weights.shape == (transform.multiplicity, 4)
    assert np.all(weights >= -1e-12)
    assert np.all(weights <= 1.0 + 1e-12)
    assert np.allclose(weights.sum(axis=0), 1.0)


def test_weights_are_invariant_under_wrapping_the_kpoints() -> None:
    transform = TransformationMatrix(np.array([[2, 1, 0], [-1, 1, 0], [0, 0, 1]]))
    cells = supercell_cells(transform)
    rng = np.random.default_rng(5)
    coefficients = rng.normal(size=(3, 2, cells.shape[0])) + 1j * rng.normal(
        size=(3, 2, cells.shape[0])
    )
    points = fiber_kpoints([0.2, 0.3, 0.0], transform)
    shifted = points + np.array([1.0, -2.0, 3.0])
    assert np.allclose(
        tight_binding_weights(cells, coefficients, points),
        tight_binding_weights(cells, coefficients, shifted),
    )


def test_zero_state_has_zero_weight() -> None:
    transform = TransformationMatrix(np.diag([2, 1, 1]))
    cells = supercell_cells(transform)
    coefficients = np.zeros((1, 1, cells.shape[0]), dtype=complex)
    weights = tight_binding_weights(cells, coefficients, fiber_kpoints(0.0 * cells[0], transform))
    assert np.allclose(weights, 0.0)


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([3, 1, 1]),
        np.diag([2, 2, 1]),
        np.array([[2, 1, 0], [-1, 1, 0], [0, 0, 1]]),
        np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]),
    ],
)
def test_perfect_crystal_unfolds_to_the_primitive_bands(matrix: np.ndarray) -> None:
    """A supercell of a perfect crystal must undo the folding exactly.

    Every primitive k-point of the fiber receives a total weight equal to the
    number of primitive bands, and the weighted energy at that k-point is the
    primitive band energy.  This is stated in a degeneracy-proof way: when two
    fiber members are degenerate the eigenvectors returned by a diagonalizer are
    an arbitrary basis of the degenerate subspace, so individual states need not
    carry unit weight even though the unfolded spectral function is exact.
    """

    transform = TransformationMatrix(matrix)
    model = _square_model()
    rng = np.random.default_rng(3)
    for q in rng.random((3, 3)):
        points, energies, weights = unfold_tight_binding_model(model, transform, [q])
        assert np.allclose(weights.sum(axis=0), 1.0)
        # One primitive band per k-point, and it sits at the primitive energy.
        assert np.allclose(weights.sum(axis=1), model.n_orbitals)
        primitive = model.bands(points)[:, 0]
        assert np.allclose((weights * energies).sum(axis=1), primitive, atol=1e-9)


def test_nondegenerate_perfect_crystal_gives_delta_weights() -> None:
    """Away from degeneracies each supercell state belongs to a single k-point."""

    transform = TransformationMatrix(np.array([[2, 1, 0], [-1, 1, 0], [0, 0, 1]]))
    model = _square_model()
    points, energies, weights = unfold_tight_binding_model(model, transform, [[0.11, 0.23, 0.0]])
    assert np.allclose(weights.max(axis=0), 1.0, atol=1e-9)
    selected = np.argmax(weights, axis=0)
    primitive = model.bands(points)[:, 0]
    assert np.allclose(energies[0], primitive[selected], atol=1e-9)


def test_perturbed_supercell_spreads_weight_but_conserves_it() -> None:
    transform = TransformationMatrix(np.diag([4, 1, 1]))
    model = _chain_model()
    points, energies, weights = unfold_tight_binding_model(
        model, transform, [[0.1, 0.0, 0.0]], perturbations=[(0, 0, 1.5)]
    )
    assert energies.shape == weights.shape
    assert np.allclose(weights.sum(axis=0), 1.0)
    # A single-site perturbation genuinely mixes the fiber.
    assert weights.max() < 0.999


def test_supercell_hamiltonian_is_hermitian_and_reproduces_folded_spectrum() -> None:
    transform = TransformationMatrix(np.diag([3, 1, 1]))
    model = _chain_model(hopping=-1.0, onsite=0.4)
    q = np.array([0.07, 0.0, 0.0])
    matrix, cells = supercell_bloch_hamiltonian(model, transform, q)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.conj().T)
    supercell_spectrum = np.linalg.eigvalsh(matrix)
    primitive_spectrum = np.sort(
        model.bands(fiber_kpoints(q @ transform.matrix.T, transform))[:, 0]
    )
    assert np.allclose(supercell_spectrum, primitive_spectrum)
    assert cells.shape == (3, 3)


def test_hermitized_completes_the_conjugate_hoppings() -> None:
    model = TightBindingModel(2, {(1, 0, 0): [[0.0, 1.0], [2.0, 0.0]]}).hermitized()
    assert set(model.hoppings) == {(1, 0, 0), (-1, 0, 0)}
    assert np.allclose(model.hoppings[(-1, 0, 0)], model.hoppings[(1, 0, 0)].conj().T)
    matrix = model.bloch_hamiltonian([0.13, 0.0, 0.0])
    assert np.allclose(matrix, matrix.conj().T)


def test_hermitized_preserves_amplitudes_and_is_idempotent() -> None:
    """Completing the conjugate hoppings must not halve the physical amplitude."""

    model = TightBindingModel(1, {(0, 0, 0): [[0.2]], (1, 0, 0): [[-1.0]]}).hermitized()
    assert model.hoppings[(0, 0, 0)] == pytest.approx(np.array([[0.2]]))
    assert model.hoppings[(-1, 0, 0)] == pytest.approx(np.array([[-1.0]]))
    # Band edges of the chain: onsite -+ 2|t|.
    assert model.bands([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])[:, 0] == pytest.approx([-1.8, 2.2])
    again = model.hermitized()
    assert set(again.hoppings) == set(model.hoppings)
    for cell, block in model.hoppings.items():
        assert again.hoppings[cell] == pytest.approx(block)


def test_sum_rule_diagnostic_sees_both_rules() -> None:
    transform = TransformationMatrix(np.diag([3, 1, 1]))
    model = _chain_model()
    _, _, weights = unfold_tight_binding_model(
        model, transform, [[0.05, 0.0, 0.0]], perturbations=[(1, 0, 0.8)]
    )
    report = diagnose_tight_binding_weights(weights, n_orbitals=model.n_orbitals)
    assert report.n_kpoints == 3
    assert report.n_states == 3
    assert report.max_fiber_deviation < 1e-10
    assert report.max_band_deviation is not None
    assert report.max_band_deviation < 1e-10
    assert not report.violated()
    assert report.to_dict()["n_orbitals"] == 1


def test_sum_rule_diagnostic_reports_a_violation() -> None:
    broken = np.array([[0.5, 0.1], [0.2, 0.3]])
    report = diagnose_tight_binding_weights(broken, n_orbitals=1)
    assert report.max_fiber_deviation == pytest.approx(0.6)
    assert report.max_band_deviation == pytest.approx(0.5)
    assert report.violated()


def test_sum_rule_diagnostic_skips_the_band_rule_for_a_partial_band_set() -> None:
    weights = np.array([[0.4], [0.6]])
    report = diagnose_tight_binding_weights(weights, n_orbitals=1)
    assert report.max_band_deviation is None
    assert not report.violated()
    with pytest.raises(ValueError, match="weights must have shape"):
        diagnose_tight_binding_weights(np.zeros(3))


def test_shape_errors_are_reported() -> None:
    transform = TransformationMatrix(np.diag([2, 1, 1]))
    cells = supercell_cells(transform)
    with pytest.raises(ValueError, match="coefficients must have shape"):
        tight_binding_weights(cells, np.zeros((2, 2)), [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="one entry per cell"):
        tight_binding_weights(cells, np.zeros((1, 1, 5), dtype=complex), [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="kpoints must have shape"):
        tight_binding_weights(cells, np.zeros((1, 1, 2), dtype=complex), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="multiplicity must be positive"):
        tight_binding_weights(
            cells, np.zeros((1, 1, 2), dtype=complex), [[0.0, 0.0, 0.0]], multiplicity=0
        )


def _random_model(n_orbitals: int, seed: int = 0) -> TightBindingModel:
    rng = np.random.default_rng(seed)
    cells = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (-1, 0, 1), (2, 0, 0)]
    hoppings = {
        cell: rng.normal(size=(n_orbitals, n_orbitals))
        + 1j * rng.normal(size=(n_orbitals, n_orbitals))
        for cell in cells
    }
    return TightBindingModel(n_orbitals, hoppings).hermitized()


def _reference_supercell_hamiltonian(
    model: TightBindingModel, transform: TransformationMatrix, kpoint: np.ndarray
) -> np.ndarray:
    """The definition, written out as a triple loop over cells and hoppings."""

    cells = supercell_cells(transform)
    n_cells = cells.shape[0]
    n_orbitals = model.n_orbitals
    matrix = np.zeros((n_cells * n_orbitals, n_cells * n_orbitals), dtype=complex)
    for row, cell_r in enumerate(cells):
        for column, cell_rp in enumerate(cells):
            for cell_d, block in model.hoppings.items():
                translation = np.asarray(cell_d) + cell_r - cell_rp
                if np.all((translation @ transform.adjugate) % transform.determinant == 0):
                    phase = np.exp(2j * np.pi * float(translation @ kpoint))
                    matrix[
                        row * n_orbitals : (row + 1) * n_orbitals,
                        column * n_orbitals : (column + 1) * n_orbitals,
                    ] += phase * block
    return matrix


@pytest.mark.parametrize("matrix", MATRICES)
def test_fast_supercell_hamiltonian_matches_the_definition(matrix: np.ndarray) -> None:
    """The sparse (hopping, cell) assembly must equal the naive double loop."""

    transform = TransformationMatrix(matrix.astype(float))
    model = _random_model(2, seed=3)
    kpoint = np.array([0.13, -0.07, 0.21])

    fast, cells = supercell_bloch_hamiltonian(model, transform, kpoint)
    assert cells.shape[0] == transform.multiplicity
    assert fast == pytest.approx(_reference_supercell_hamiltonian(model, transform, kpoint))
    assert np.allclose(fast, fast.conj().T)


def test_batched_bloch_hamiltonians_match_the_single_kpoint_form() -> None:
    model = _random_model(3, seed=11)
    points = np.array([[0.0, 0.0, 0.0], [0.31, -0.2, 0.05], [0.5, 0.5, 0.5]])

    stacked = model.bloch_hamiltonians(points)
    assert stacked.shape == (3, 3, 3)
    for index, point in enumerate(points):
        assert stacked[index] == pytest.approx(model.bloch_hamiltonian(point))
    assert model.bands(points) == pytest.approx(
        np.array([np.linalg.eigvalsh(matrix) for matrix in stacked])
    )


def test_an_empty_model_has_a_zero_hamiltonian() -> None:
    model = TightBindingModel(2, {})
    assert model.hopping_cells.shape == (0, 3)
    assert model.bloch_hamiltonians([[0.1, 0.2, 0.3]]) == pytest.approx(np.zeros((1, 2, 2)))
    matrix, cells = supercell_bloch_hamiltonian(
        model, TransformationMatrix(np.diag([2.0, 1.0, 1.0])), [0.0, 0.0, 0.0]
    )
    assert cells.shape == (2, 3)
    assert matrix == pytest.approx(np.zeros((4, 4)))


def test_out_of_range_perturbations_are_rejected() -> None:
    transform = TransformationMatrix(np.diag([2.0, 1.0, 1.0]))
    model = _random_model(2, seed=5)
    with pytest.raises(ValueError, match="cell index"):
        supercell_bloch_hamiltonian(model, transform, [0.0, 0.0, 0.0], perturbations=[(5, 0, 1.0)])
    with pytest.raises(ValueError, match="orbital"):
        supercell_bloch_hamiltonian(model, transform, [0.0, 0.0, 0.0], perturbations=[(0, 7, 1.0)])


def test_a_large_supercell_still_obeys_both_sum_rules() -> None:
    """A 48-cell supercell of a two-orbital model, on several k-points."""

    transform = TransformationMatrix(np.array([[4.0, 2.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 3.0]]))
    model = _random_model(2, seed=7)
    points = [[0.0, 0.0, 0.0], [0.03, 0.11, -0.07]]

    _, _, weights = unfold_tight_binding_model(model, transform, points)
    multiplicity = transform.multiplicity
    assert weights.shape == (2 * multiplicity, multiplicity * model.n_orbitals)
    for start in (0, multiplicity):
        block = weights[start : start + multiplicity]
        report = diagnose_tight_binding_weights(block, n_orbitals=model.n_orbitals)
        assert not report.violated(1e-9)


@pytest.mark.parametrize("matrix", MATRICES)
def test_path_unfolding_recovers_the_primitive_bands(matrix: np.ndarray) -> None:
    """Along a primitive path a perfect supercell shows its primitive bands."""

    transform = TransformationMatrix(matrix.astype(float))
    model = _random_model(2, seed=13)
    path = np.array([[t, 0.17, -0.09] for t in np.linspace(0.0, 0.5, 7)])

    energies, weights = unfold_tight_binding_path(model, transform, path)
    assert energies.shape == (7, transform.multiplicity * model.n_orbitals)
    # Every k-point carries exactly n_orbitals worth of weight ...
    assert weights.sum(axis=1) == pytest.approx(np.full(7, float(model.n_orbitals)))
    # ... and it sits on the primitive eigenvalues at that k-point.
    primitive = model.bands(path)
    for index in range(path.shape[0]):
        heavy = np.flatnonzero(weights[index] > 1e-8)
        assert np.allclose(np.sort(energies[index][heavy]), primitive[index], atol=1e-9)
        assert weights[index][heavy] == pytest.approx(np.ones(heavy.size))


def test_path_unfolding_agrees_with_the_fiber_unfolding() -> None:
    """The path weight at k equals the fiber weight of the k-point it folds to."""

    transform = TransformationMatrix(np.array([[2.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 2.0]]))
    model = _random_model(2, seed=17)
    kpoint = np.array([0.13, -0.07, 0.21])

    energies, weights = unfold_tight_binding_path(model, transform, kpoint[np.newaxis, :])
    fiber, fiber_energies, fiber_weights = unfold_tight_binding_model(model, transform, [kpoint])
    # The fiber is returned wrapped into the first zone, so the member matching
    # the requested k-point differs from it by an integer reciprocal vector.
    offsets = fiber - kpoint
    row = int(np.argmin(np.linalg.norm(offsets - np.round(offsets), axis=1)))
    assert np.allclose(offsets[row], np.round(offsets[row]))
    assert energies[0] == pytest.approx(fiber_energies[row])
    assert weights[0] == pytest.approx(fiber_weights[row])


def test_path_unfolding_smears_weight_around_a_defect() -> None:
    transform = TransformationMatrix(np.diag([4.0, 1.0, 1.0]))
    model = TightBindingModel(
        1, {(0, 0, 0): np.array([[0.0]]), (1, 0, 0): np.array([[-1.0]])}
    ).hermitized()
    path = np.array([[t, 0.0, 0.0] for t in np.linspace(0.0, 0.5, 9)])

    clean, clean_weights = unfold_tight_binding_path(model, transform, path)
    _, defect_weights = unfold_tight_binding_path(
        model, transform, path, perturbations=[(1, 0, 1.5)]
    )

    assert np.max(clean_weights) == pytest.approx(1.0)
    assert np.max(defect_weights) < 1.0
    # The sum rule survives the defect even though the delta peaks do not.
    assert defect_weights.sum(axis=1) == pytest.approx(np.ones(path.shape[0]))


def test_weights_are_invariant_under_a_per_orbital_phase() -> None:
    """Where a Wannier function sits inside the cell cannot change the weights.

    Changing the convention for the orbital positions multiplies every amplitude
    of orbital ``a`` by one phase, constant along the cell sum, so it drops out
    of the modulus (``UnfoldLab.tbWeight_orbital_phase``).
    """

    transform = TransformationMatrix(np.array([[2.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 2.0]]))
    cells = supercell_cells(transform)
    rng = np.random.default_rng(23)
    coefficients = rng.normal(size=(5, 3, cells.shape[0])) + 1j * rng.normal(
        size=(5, 3, cells.shape[0])
    )
    fiber = fiber_kpoints(np.array([0.1, 0.2, 0.3]), transform)

    phases = np.exp(2j * np.pi * rng.uniform(size=3))
    regauged = coefficients * phases[np.newaxis, :, np.newaxis]

    assert tight_binding_weights(cells, regauged, fiber) == pytest.approx(
        tight_binding_weights(cells, coefficients, fiber)
    )
