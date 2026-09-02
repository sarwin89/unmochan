"""Conditioning of the LCAO overlap and canonical orthogonalization.

The theorems behind these tests are in `RequestProject/Unfolding/Conditioning.lean`:
`sNorm_ge_of_diagonally_dominant` (the cheap positive-definiteness certificate),
`standard_of_generalized` / `generalized_of_standard` (the transformation is an
equivalence for any `X` with `X* S X = 1`) and `sNorm_transformed` (the pulled
back states are `S`-orthonormal).
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.core.conditioning import (
    DEFAULT_OVERLAP_THRESHOLD,
    canonical_orthogonalization,
    diagnose_overlap_conditioning,
    diagonal_dominance_bound,
    hermitian_part,
    solve_generalized_eigenproblem_truncated,
)
from unfoldlab.core.lcao import solve_generalized_eigenproblem


def chain_overlap(n: int, off: float) -> np.ndarray:
    """Unit-diagonal nearest-neighbour overlap of an open chain."""

    matrix = np.eye(n, dtype=complex)
    for index in range(n - 1):
        matrix[index, index + 1] = off
        matrix[index + 1, index] = off
    return matrix


def rank_deficient_overlap() -> np.ndarray:
    """Gram matrix of three vectors spanning only two dimensions."""

    basis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return basis @ basis.T


def test_hermitian_part_symmetrizes() -> None:
    matrix = np.array([[1.0, 0.4 + 0.1j], [0.2 - 0.3j, 2.0]])
    part = hermitian_part(matrix)
    assert np.allclose(part, part.conj().T)
    assert np.isclose(part[0, 1], 0.5 * (matrix[0, 1] + matrix[1, 0].conjugate()))


def test_hermitian_part_rejects_a_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        hermitian_part(np.zeros((2, 3)))


def test_dominance_bound_matches_the_row_sum() -> None:
    """An interior row of the chain has two neighbours, so `r = 2 |off|`."""

    overlap = chain_overlap(4, 0.2)
    assert diagonal_dominance_bound(overlap) == pytest.approx(1.0 - 0.4)


def test_dominance_bound_is_a_true_lower_bound_on_the_smallest_eigenvalue() -> None:
    for off in (0.05, 0.15, 0.3):
        overlap = chain_overlap(6, off)
        smallest = float(np.linalg.eigvalsh(overlap)[0])
        assert diagonal_dominance_bound(overlap) <= smallest + 1e-12


def test_dominance_bound_is_invariant_under_diagonal_scaling() -> None:
    """The criterion is applied to `D^-1/2 S D^-1/2`, so a rescaled basis agrees."""

    overlap = chain_overlap(4, 0.2)
    scale = np.diag([1.0, 4.0, 0.25, 9.0])
    scaled = scale @ overlap @ scale
    assert diagonal_dominance_bound(scaled) == pytest.approx(diagonal_dominance_bound(overlap))


def test_dominance_bound_rejects_a_non_positive_diagonal() -> None:
    with pytest.raises(ValueError, match="not a Gram matrix"):
        diagonal_dominance_bound(np.array([[0.0, 0.1], [0.1, 1.0]]))


def test_dominance_bound_can_be_negative_for_a_positive_definite_overlap() -> None:
    """The criterion is sufficient, never necessary: it certifies nothing here."""

    overlap = chain_overlap(4, 0.6)
    assert diagonal_dominance_bound(overlap) < 0.0
    assert float(np.linalg.eigvalsh(overlap)[0]) > 0.0


def test_diagnosis_of_a_well_conditioned_overlap() -> None:
    report = diagnose_overlap_conditioning(chain_overlap(5, 0.2))
    assert report.positive_definite
    assert report.certified_by_dominance
    assert report.n_below_threshold == 0
    assert report.condition_number == pytest.approx(report.largest / report.smallest)
    assert "positive definite" in report.summary()
    assert report.to_dict()["threshold"] == DEFAULT_OVERLAP_THRESHOLD


def test_diagnosis_finds_the_null_direction_of_a_rank_deficient_overlap() -> None:
    report = diagnose_overlap_conditioning(rank_deficient_overlap())
    assert report.n_below_threshold == 1
    assert not report.certified_by_dominance
    assert "canonical orthogonalization" in report.summary()


def test_diagnosis_reports_an_infinite_condition_number_when_singular() -> None:
    report = diagnose_overlap_conditioning(np.zeros((2, 2)))
    assert not report.positive_definite
    assert not report.certified_by_dominance
    assert report.condition_number == float("inf")
    assert report.to_dict()["positive_definite"] is False


def test_diagnosis_rejects_a_negative_threshold() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        diagnose_overlap_conditioning(np.eye(2), threshold=-1.0)


def test_canonical_orthogonalization_is_an_s_isometry() -> None:
    """`X* S X = 1` on the retained subspace: the hypothesis of the equivalence."""

    overlap = chain_overlap(5, 0.3)
    transform, retained = canonical_orthogonalization(overlap)
    assert transform.shape == (5, 5)
    assert retained.size == 5
    assert np.allclose(transform.conj().T @ overlap @ transform, np.eye(5), atol=1e-12)


def test_canonical_orthogonalization_drops_the_null_direction() -> None:
    overlap = rank_deficient_overlap()
    transform, retained = canonical_orthogonalization(overlap)
    assert transform.shape == (3, 2)
    assert np.all(retained >= DEFAULT_OVERLAP_THRESHOLD)
    assert np.allclose(transform.conj().T @ overlap @ transform, np.eye(2), atol=1e-10)


def test_canonical_orthogonalization_rejects_a_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        canonical_orthogonalization(np.eye(2), threshold=0.0)


def test_canonical_orthogonalization_rejects_a_wholly_singular_overlap() -> None:
    with pytest.raises(ValueError, match="no well-conditioned direction"):
        canonical_orthogonalization(np.zeros((3, 3)))


def test_truncated_solver_agrees_with_cholesky_when_the_overlap_is_healthy() -> None:
    overlap = chain_overlap(4, 0.2)
    hamiltonian = np.diag([-1.0, 0.0, 0.5, 2.0]) + 0.3 * chain_overlap(4, 1.0) - 0.3 * np.eye(4)
    reference, _ = solve_generalized_eigenproblem(hamiltonian, overlap)
    energies, vectors = solve_generalized_eigenproblem_truncated(hamiltonian, overlap)
    assert np.allclose(energies, reference, atol=1e-10)
    assert np.allclose(vectors.conj().T @ overlap @ vectors, np.eye(4), atol=1e-10)


def test_truncated_solver_returns_fewer_states_than_basis_functions() -> None:
    """The rank-deficient case: Cholesky fails, truncation returns a real answer."""

    overlap = rank_deficient_overlap()
    hamiltonian = np.diag([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        solve_generalized_eigenproblem(hamiltonian, overlap)
    energies, vectors = solve_generalized_eigenproblem_truncated(hamiltonian, overlap)
    assert energies.size == 2
    assert vectors.shape == (3, 2)
    assert np.allclose(vectors.conj().T @ overlap @ vectors, np.eye(2), atol=1e-10)


def test_truncated_states_solve_the_generalized_problem_in_the_retained_subspace() -> None:
    """`generalized_of_standard`: `X* (H c - E S c) = 0` for every returned state."""

    overlap = rank_deficient_overlap()
    hamiltonian = np.array([[1.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 3.0]])
    transform, _ = canonical_orthogonalization(overlap)
    energies, vectors = solve_generalized_eigenproblem_truncated(hamiltonian, overlap)
    for index, energy in enumerate(energies):
        residual = hamiltonian @ vectors[:, index] - energy * (overlap @ vectors[:, index])
        assert np.allclose(transform.conj().T @ residual, 0.0, atol=1e-10)


def test_truncated_solver_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        solve_generalized_eigenproblem_truncated(np.eye(3), np.eye(2))


def test_lcao_solver_threshold_delegates_to_the_truncated_solver() -> None:
    overlap = rank_deficient_overlap()
    hamiltonian = np.diag([1.0, 2.0, 3.0])
    energies, vectors = solve_generalized_eigenproblem(hamiltonian, overlap, threshold=1e-8)
    assert energies.size == 2
    assert np.allclose(vectors.conj().T @ overlap @ vectors, np.eye(2), atol=1e-10)


def test_cli_overlap_reports_a_healthy_matrix(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "overlap.txt"
    np.savetxt(path, np.real(chain_overlap(4, 0.2)))
    result = CliRunner().invoke(app, ["overlap", "--overlap", str(path)])
    assert result.exit_code == 0, result.output
    assert "positive definite" in result.output


def test_cli_overlap_solves_and_writes_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    overlap_path = tmp_path / "overlap.npy"
    hamiltonian_path = tmp_path / "hamiltonian.npy"
    json_path = tmp_path / "report.json"
    np.save(overlap_path, rank_deficient_overlap())
    np.save(hamiltonian_path, np.diag([1.0, 2.0, 3.0]))
    result = CliRunner().invoke(
        app,
        [
            "overlap",
            "--overlap",
            str(overlap_path),
            "--hamiltonian",
            str(hamiltonian_path),
            "--json",
            str(json_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(json_path.read_text())
    assert payload["n_retained"] == 2
    assert payload["n_below_threshold"] == 1
    assert len(payload["energies"]) == 2


def test_cli_overlap_rejects_a_non_square_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "overlap.txt"
    np.savetxt(path, np.zeros((2, 3)))
    result = CliRunner().invoke(app, ["overlap", "--overlap", str(path)])
    assert result.exit_code != 0
