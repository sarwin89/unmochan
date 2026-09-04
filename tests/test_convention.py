"""Bloch-sum conventions: which phase choices change an unfolded band structure.

The claims checked here are the numerical counterparts of
``RequestProject/Unfolding/Convention.lean``:

* a per-orbital phase (the atomic-versus-cell convention) is a *gauge* of the
  orthonormal weight and of every fat band
  (``UnfoldLab.tbWeight_gaugeOrbital``, ``UnfoldLab.tbOrbitalWeight_gaugeOrbital``);
* a per-cell phase translates the whole distribution in k
  (``UnfoldLab.tbWeight_gaugeShift``) and permutes the fiber when the shift is a
  difference of fiber members (``UnfoldLab.tbWeight_gaugeCell``), while leaving
  the fiber sum rule intact -- so no runtime check catches it;
* in a non-orthogonal basis the per-orbital phase is a gauge only if the overlap
  kernel is regauged with the state (``UnfoldLab.ovWeight_gauge``).
"""

from __future__ import annotations

import numpy as np
import pytest

from unmochan.core.convention import (
    apply_cell_phase,
    apply_orbital_phase,
    cell_phase_factors,
    convert_coefficients,
    diagnose_bloch_convention,
    fiber_shift_permutation,
    gauge_overlap_kernel,
    orbital_phase_factors,
)
from unmochan.core.kpoints import fiber_kpoints
from unmochan.core.lcao import lcao_weights, orthonormal_overlap_kernel
from unmochan.core.tight_binding import (
    TightBindingModel,
    diagnose_tight_binding_weights,
    supercell_bloch_hamiltonian,
    supercell_cells,
    tight_binding_orbital_weights,
    tight_binding_weights,
)
from unmochan.core.transformations import TransformationMatrix

pytestmark = pytest.mark.unit


def _transform() -> TransformationMatrix:
    return TransformationMatrix(np.array([[3, 1, 0], [0, 2, 0], [0, 0, 1]]))


def _random_state(n_states: int, n_orbitals: int, n_cells: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_states, n_orbitals, n_cells)) + 1j * rng.normal(
        size=(n_states, n_orbitals, n_cells)
    )


def test_orbital_phase_factors_are_unimodular() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.3, 0.25, 0.1], [-0.4, 0.5, 0.2]])
    factors = orbital_phase_factors(positions, [0.13, -0.2, 0.4])
    assert np.allclose(np.abs(factors), 1.0)


def test_orbital_gauge_leaves_the_weights_unchanged() -> None:
    """``UnfoldLab.tbWeight_gaugeOrbital``: an atomic-convention phase is a gauge."""

    transform = _transform()
    cells = supercell_cells(transform)
    q = np.array([0.1, 0.2, 0.0])
    points = fiber_kpoints(q, transform)
    coefficients = _random_state(4, 3, cells.shape[0])
    positions = np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.0], [0.5, 0.1, 0.3]])

    reference = tight_binding_weights(
        cells, coefficients, points, multiplicity=transform.multiplicity
    )
    regauged = tight_binding_weights(
        cells,
        apply_orbital_phase(coefficients, orbital_phase_factors(positions, q)),
        points,
        multiplicity=transform.multiplicity,
    )
    assert np.allclose(reference, regauged, atol=1e-12)


def test_orbital_gauge_leaves_fat_bands_unchanged() -> None:
    """``UnfoldLab.tbOrbitalWeight_gaugeOrbital``: every projected weight too."""

    transform = _transform()
    cells = supercell_cells(transform)
    q = np.array([0.0, 0.35, 0.0])
    points = fiber_kpoints(q, transform)
    coefficients = _random_state(3, 4, cells.shape[0], seed=11)
    positions = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.1, 0.1, 0.4]])
    groups = [[0, 2], [1], [3]]

    reference = tight_binding_orbital_weights(
        cells, coefficients, points, groups, multiplicity=transform.multiplicity
    )
    regauged = tight_binding_orbital_weights(
        cells,
        apply_orbital_phase(coefficients, orbital_phase_factors(positions, q)),
        points,
        groups,
        multiplicity=transform.multiplicity,
    )
    assert np.allclose(reference, regauged, atol=1e-12)


def test_cell_phase_translates_the_distribution_in_k() -> None:
    """``UnfoldLab.tbWeight_gaugeShift``: the weight at k becomes the weight at k + x."""

    transform = _transform()
    cells = supercell_cells(transform)
    q = np.array([0.1, 0.2, 0.0])
    points = fiber_kpoints(q, transform)
    coefficients = _random_state(3, 2, cells.shape[0], seed=3)

    mistaken = tight_binding_weights(
        cells,
        apply_cell_phase(coefficients, cell_phase_factors(cells, q)),
        points,
        multiplicity=transform.multiplicity,
    )
    shifted = tight_binding_weights(
        cells, coefficients, points - q[np.newaxis, :], multiplicity=transform.multiplicity
    )
    assert np.allclose(mistaken, shifted, atol=1e-12)


def test_cell_phase_survives_the_sum_rule_check() -> None:
    """The mistake is invisible to both runtime sum rules -- hence this module."""

    transform = _transform()
    cells = supercell_cells(transform)
    q = np.array([0.1, 0.2, 0.0])
    points = fiber_kpoints(q, transform)
    coefficients = _random_state(3, 2, cells.shape[0], seed=5)

    mistaken = tight_binding_weights(
        cells,
        apply_cell_phase(coefficients, cell_phase_factors(cells, q)),
        points,
        multiplicity=transform.multiplicity,
    )
    rules = diagnose_tight_binding_weights(mistaken)
    assert not rules.violated()
    assert rules.max_fiber_deviation < 1e-12


def test_cell_phase_permutes_the_fiber_at_the_supercell_gamma_point() -> None:
    """``UnfoldLab.tbWeight_gaugeCell``: a fiber-member shift is a permutation."""

    transform = _transform()
    cells = supercell_cells(transform)
    points = fiber_kpoints(np.zeros(3), transform)
    coefficients = _random_state(2, 2, cells.shape[0], seed=13)

    shift = points[2]
    permutation = fiber_shift_permutation(points, shift)
    reference = tight_binding_weights(
        cells, coefficients, points, multiplicity=transform.multiplicity
    )
    mistaken = tight_binding_weights(
        cells,
        apply_cell_phase(coefficients, cell_phase_factors(cells, shift)),
        points,
        multiplicity=transform.multiplicity,
    )
    assert np.allclose(mistaken[permutation], reference, atol=1e-12)


def test_fiber_shift_permutation_rejects_a_shift_off_the_fiber() -> None:
    transform = _transform()
    points = fiber_kpoints(np.array([0.1, 0.2, 0.0]), transform)
    with pytest.raises(ValueError, match="does not map the fiber"):
        fiber_shift_permutation(points, [0.1, 0.2, 0.0])


def test_convert_coefficients_round_trips() -> None:
    transform = _transform()
    cells = supercell_cells(transform)
    q = np.array([0.1, 0.2, 0.3])
    coefficients = _random_state(2, 3, cells.shape[0], seed=17)
    positions = np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.0], [0.5, 0.1, 0.3]])

    for other in ("atomic", "wavefunction"):
        away = convert_coefficients(
            coefficients,
            source="cell",
            target=other,
            supercell_kpoint=q,
            cells=cells,
            orbital_positions=positions,
        )
        back = convert_coefficients(
            away,
            source=other,
            target="cell",
            supercell_kpoint=q,
            cells=cells,
            orbital_positions=positions,
        )
        assert np.allclose(back, coefficients, atol=1e-12)


def test_convert_coefficients_validates_its_arguments() -> None:
    coefficients = _random_state(1, 2, 3)
    with pytest.raises(ValueError, match="source must be one of"):
        convert_coefficients(
            coefficients, source="bloch", target="cell", supercell_kpoint=np.zeros(3)
        )
    with pytest.raises(ValueError, match="needs orbital_positions"):
        convert_coefficients(
            coefficients, source="atomic", target="cell", supercell_kpoint=np.zeros(3)
        )
    with pytest.raises(ValueError, match="needs cells"):
        convert_coefficients(
            coefficients,
            source="cell",
            target="wavefunction",
            supercell_kpoint=np.zeros(3),
        )


def test_diagnose_bloch_convention_reports_the_misplaced_weight() -> None:
    transform = _transform()
    cells = supercell_cells(transform)
    q = np.array([0.1, 0.2, 0.0])
    coefficients = _random_state(3, 2, cells.shape[0], seed=19)

    report = diagnose_bloch_convention(cells, coefficients, transform, q)
    assert report.matters()
    assert 0.0 < report.max_total_variation <= 1.0
    assert report.permutation is None  # the shift leaves the fiber
    assert report.orbital_gauge_matters is False
    assert report.to_dict()["n_states"] == 3


def test_diagnose_bloch_convention_at_gamma_reports_a_permutation() -> None:
    transform = _transform()
    cells = supercell_cells(transform)
    coefficients = _random_state(2, 2, cells.shape[0], seed=23)

    report = diagnose_bloch_convention(cells, coefficients, transform, np.zeros(3))
    assert report.permutation == tuple(range(transform.multiplicity))
    assert report.max_total_variation == pytest.approx(0.0, abs=1e-12)


def test_lcao_weight_needs_the_kernel_regauged_with_the_state() -> None:
    """``UnfoldLab.ovWeight_gauge`` and ``ovNumer_gaugeOrbital_untransformed``."""

    transform = TransformationMatrix(np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]))
    cells = supercell_cells(transform)
    n_cells = cells.shape[0]
    q = np.array([0.15, 0.0, 0.0])
    points = fiber_kpoints(q, transform)
    coefficients = _random_state(2, 2, n_cells, seed=29)

    # A genuinely non-orthogonal kernel: identity on site plus a small
    # inter-orbital overlap inside the origin cell.
    origin = int(np.flatnonzero(np.all(cells == 0, axis=1))[0])
    kernel = orthonormal_overlap_kernel(2, cells).astype(np.complex128)
    kernel[0, 1, origin] += 0.2
    kernel[1, 0, origin] += 0.2

    factors = orbital_phase_factors(np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]), q)
    reference = lcao_weights(cells, coefficients, kernel, points, transform, q)
    consistent = lcao_weights(
        cells,
        apply_orbital_phase(coefficients, factors),
        gauge_overlap_kernel(kernel, factors),
        points,
        transform,
        q,
    )
    inconsistent = lcao_weights(
        cells,
        apply_orbital_phase(coefficients, factors),
        kernel,
        points,
        transform,
        q,
    )

    assert np.allclose(reference, consistent, atol=1e-12)
    assert not np.allclose(reference, inconsistent, atol=1e-6)


def test_gauge_overlap_kernel_preserves_an_orthonormal_kernel() -> None:
    kernel = orthonormal_overlap_kernel(3, supercell_cells(_transform()))
    factors = orbital_phase_factors(
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.4, 0.0]]), [0.3, 0.1, 0.0]
    )
    assert np.allclose(gauge_overlap_kernel(kernel, factors), kernel, atol=1e-12)


def test_gauge_overlap_kernel_validates_shapes() -> None:
    with pytest.raises(ValueError, match="kernel must have shape"):
        gauge_overlap_kernel(np.zeros((2, 3, 4)), np.ones(2))
    with pytest.raises(ValueError, match="one entry per orbital"):
        gauge_overlap_kernel(np.zeros((2, 2, 4)), np.ones(3))


def test_perfect_chain_lands_on_the_wrong_k_in_the_wavefunction_convention() -> None:
    """The end-to-end symptom: a folded band reappears at the wrong k-point.

    A three-cell supercell of the perfect chain ``E(k) = 2 t cos(2 pi k)`` is
    diagonalized at one supercell k-point and unfolded.  Read in the package's
    convention every band carries weight one at a single primitive k-point and
    its energy matches the primitive dispersion there.  Read as full-wavefunction
    amplitudes the fiber sum rule still holds exactly -- so no runtime check
    complains -- but the distribution has been translated off the fiber
    (``UnfoldLab.tbWeight_gaugeShift``): a sharp band turns into a smear over
    the whole fiber whose maximum sits at the wrong k-point.
    """

    hopping = 0.5
    model = TightBindingModel(
        n_orbitals=1,
        hoppings={
            (1, 0, 0): np.array([[hopping]], dtype=complex),
            (-1, 0, 0): np.array([[hopping]], dtype=complex),
        },
    )
    transform = TransformationMatrix(np.diag([3, 1, 1]))
    # The shift must exceed half the fiber spacing (1/6 here) for the maximum
    # to move to another fiber member; below that the band is merely smeared.
    q = np.array([0.2, 0.0, 0.0])

    matrix, cells = supercell_bloch_hamiltonian(model, transform, q)
    energies, vectors = np.linalg.eigh(matrix)
    coefficients = np.swapaxes(vectors.T.reshape(-1, cells.shape[0], 1), 1, 2)
    points = fiber_kpoints(q @ transform.matrix.T, transform)

    def dispersion(kpoints: np.ndarray) -> np.ndarray:
        return 2.0 * hopping * np.cos(2.0 * np.pi * kpoints[:, 0])

    correct = tight_binding_weights(
        cells, coefficients, points, multiplicity=transform.multiplicity
    )
    mistaken = tight_binding_weights(
        cells,
        apply_cell_phase(coefficients, cell_phase_factors(cells, q)),
        points,
        multiplicity=transform.multiplicity,
    )

    # Both obey the fiber sum rule exactly: the mistake is invisible to it.
    assert np.allclose(correct.sum(axis=0), 1.0)
    assert np.allclose(mistaken.sum(axis=0), 1.0)

    # The correct reading is a sharp band at the k-point of the dispersion.
    assert np.allclose(np.max(correct, axis=0), 1.0, atol=1e-10)
    correct_k = points[np.argmax(correct, axis=0)]
    assert np.allclose(dispersion(correct_k), energies, atol=1e-10)

    # The mistaken reading smears every band over the whole fiber and puts its
    # maximum on a k-point whose primitive energy is a different number.
    assert float(np.max(mistaken)) < 0.9
    assert float(np.min(np.max(mistaken, axis=0))) > 0.5
    mistaken_k = points[np.argmax(mistaken, axis=0)]
    assert not np.allclose(dispersion(mistaken_k), energies, atol=1e-6)

    # And the repair puts them back.
    repaired = tight_binding_weights(
        cells,
        convert_coefficients(
            apply_cell_phase(coefficients, cell_phase_factors(cells, q)),
            source="wavefunction",
            target="cell",
            supercell_kpoint=q,
            cells=cells,
        ),
        points,
        multiplicity=transform.multiplicity,
    )
    assert np.allclose(repaired, correct, atol=1e-12)
