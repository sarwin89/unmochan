"""Mass (isotope) defects in a phonon supercell.

A mass defect is a *congruence* of the dynamical matrix, ``D -> S D S`` with
``S = diag(sqrt(m_ref / m_site))``, not an on-site term; that is
``UnfoldLab.dynMatrix_massScale`` and ``UnfoldLab.mass_change_not_diagonal_shift``
in ``RequestProject/Unfolding/Phonons.lean``.  These tests pin the numerical
side of both statements, plus the two invariants a mass defect must not break:
the sum rules and the acoustic zero mode at Gamma.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.core.phonons import (
    ForceConstantModel,
    mass_site_scaling,
    supercell_dynamical_matrix,
    supercell_site_masses,
    unfold_phonon_fiber,
    unfold_phonon_path,
)
from unfoldlab.core.tight_binding import supercell_bloch_hamiltonian
from unfoldlab.core.transformations import TransformationMatrix
from unfoldlab.io.force_constants import read_force_constant_problem

runner = CliRunner()

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TRIPLE = TransformationMatrix(np.diag([3.0, 1.0, 1.0]))


def _spring_chain(spring: float = 1.0, mass: float = 2.0) -> ForceConstantModel:
    onsite = np.zeros((3, 3))
    onsite[0, 0] = 2.0 * spring
    hop = np.zeros((3, 3))
    hop[0, 0] = -spring
    return ForceConstantModel(
        np.array([mass]),
        {(0, 0, 0): onsite, (1, 0, 0): hop, (-1, 0, 0): hop},
    )


def _longitudinal(matrix: np.ndarray, n_cells: int, n_atoms: int = 1) -> np.ndarray:
    """The ``x`` block of a supercell dynamical matrix of a chain."""

    indices = [3 * n_atoms * cell for cell in range(n_cells)]
    return matrix[np.ix_(indices, indices)]


def test_site_masses_default_to_the_primitive_ones() -> None:
    model = _spring_chain()
    masses = supercell_site_masses(model, TRIPLE)
    assert masses.shape == (3, 1)
    assert masses == pytest.approx(2.0)
    assert mass_site_scaling(model, TRIPLE, masses) == pytest.approx(1.0)


def test_a_uniform_substitution_only_rescales_the_frequencies() -> None:
    model = _spring_chain()
    heavy = supercell_site_masses(model, TRIPLE, [(cell, 0, 8.0) for cell in range(3)])
    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.5, 0.0, 0.0]])

    plain, plain_weights = unfold_phonon_path(model, TRIPLE, points)
    scaled, scaled_weights = unfold_phonon_path(model, TRIPLE, points, site_masses=heavy)

    assert scaled == pytest.approx(plain * np.sqrt(2.0 / 8.0))
    assert scaled_weights == pytest.approx(plain_weights)


def test_one_isotope_matches_the_explicit_three_site_chain() -> None:
    model = _spring_chain()
    masses = supercell_site_masses(model, TRIPLE, [(1, 0, 8.0)])
    matrix = supercell_dynamical_matrix(model, TRIPLE, np.zeros(3), site_masses=masses)

    site_masses = np.array([2.0, 8.0, 2.0])
    force_constants = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])
    expected = force_constants / np.sqrt(np.outer(site_masses, site_masses))

    assert _longitudinal(matrix, 3) == pytest.approx(expected)
    assert matrix == pytest.approx(matrix.conj().T)


def test_a_mass_defect_is_not_an_on_site_term() -> None:
    """The off-diagonal entries move too, which no on-site shift can do."""

    model = _spring_chain()
    masses = supercell_site_masses(model, TRIPLE, [(1, 0, 8.0)])
    reference, _ = supercell_bloch_hamiltonian(model.mass_weighted_model(), TRIPLE, np.zeros(3))
    defect = supercell_dynamical_matrix(model, TRIPLE, np.zeros(3), site_masses=masses)

    difference = defect - reference
    off_diagonal = difference - np.diag(np.diag(difference))
    assert np.max(np.abs(off_diagonal)) > 0.1

    # And no diagonal shift can reproduce the spectrum either: the best on-site
    # imitation matches the diagonal exactly and still gets the modes wrong.
    imitation = reference + np.diag(np.diag(difference))
    assert not np.allclose(np.linalg.eigvalsh(imitation), np.linalg.eigvalsh(defect), atol=1e-6)


def test_a_heavy_isotope_softens_the_optical_modes_and_keeps_the_sum_rules() -> None:
    model = _spring_chain()
    masses = supercell_site_masses(model, TRIPLE, [(1, 0, 8.0)])
    kpoints, frequencies, weights = unfold_phonon_fiber(
        model, TRIPLE, np.zeros((1, 3)), site_masses=masses
    )

    assert kpoints.shape == (3, 3)
    assert weights.sum(axis=0) == pytest.approx(np.ones(weights.shape[1]))
    assert weights.sum(axis=1) == pytest.approx(3.0 * np.ones(weights.shape[0]))

    # The highest mode lives on the light atoms and does not move; the branch
    # that the heavy atom participates in is pulled down.
    plain = unfold_phonon_fiber(model, TRIPLE, np.zeros((1, 3)))[1]
    assert np.sort(frequencies[0])[-2] < np.sort(plain[0])[-2] - 1e-6


def test_the_acoustic_mode_survives_a_mass_defect() -> None:
    """Rigid translation stays a zero mode: ``UnfoldLab.acoustic_zero_mode``."""

    model = _spring_chain()
    masses = supercell_site_masses(model, TRIPLE, [(1, 0, 8.0), (2, 0, 5.0)])
    matrix = supercell_dynamical_matrix(model, TRIPLE, np.zeros(3), site_masses=masses)
    block = _longitudinal(matrix, 3)

    eigenvalues, eigenvectors = np.linalg.eigh(block)
    assert eigenvalues[0] == pytest.approx(0.0, abs=1e-12)
    displacement = eigenvectors[:, 0] / np.sqrt(masses.reshape(-1))
    assert displacement / displacement[0] == pytest.approx(np.ones(3))


def test_mass_defects_are_rejected_when_they_do_not_make_sense() -> None:
    model = _spring_chain()
    with pytest.raises(ValueError, match="cell index"):
        supercell_site_masses(model, TRIPLE, [(3, 0, 8.0)])
    with pytest.raises(ValueError, match="atom index"):
        supercell_site_masses(model, TRIPLE, [(0, 1, 8.0)])
    with pytest.raises(ValueError, match="positive"):
        supercell_site_masses(model, TRIPLE, [(0, 0, -1.0)])
    with pytest.raises(ValueError, match="shape"):
        mass_site_scaling(model, TRIPLE, np.ones((2, 1)))
    with pytest.raises(ValueError, match="positive"):
        mass_site_scaling(model, TRIPLE, np.zeros((3, 1)))


def test_the_reader_keeps_mass_defects_apart_from_on_site_terms() -> None:
    problem = read_force_constant_problem(EXAMPLES / "toy_1d_phonon_isotope.json")
    assert problem.perturbations == []
    assert problem.substitutions == [(1, 0, 8.0)]

    stiffened = read_force_constant_problem(EXAMPLES / "toy_1d_phonon_defect.json")
    assert stiffened.substitutions == []
    assert stiffened.perturbations == [(1, 0, 0.75)]


def test_the_reader_rejects_an_impossible_site_mass(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "masses": [2.0],
                "force_constants": [
                    {"cell": [0, 0, 0], "matrix": np.diag([2.0, 0.0, 0.0]).tolist()}
                ],
                "site_masses": [{"cell": 0, "atom": 0, "mass": 0.0}],
            }
        )
    )
    with pytest.raises(ValueError, match="positive"):
        read_force_constant_problem(path)


def test_the_cli_applies_mass_defects_from_the_file_and_the_command_line() -> None:
    common = [
        "phonon",
        "bands",
        "--matrix",
        "3 0 0 0 1 0 0 0 1",
        "--path",
        "0,0,0:0.5,0,0",
        "--points",
        "3",
    ]
    from_file = runner.invoke(
        app, [*common, "--model", str(EXAMPLES / "toy_1d_phonon_isotope.json")]
    )
    assert from_file.exit_code == 0, from_file.output
    payload = json.loads(from_file.output)
    assert payload["n_mass_defects"] == 1
    assert payload["max_band_deviation"] < 1e-10

    from_flag = runner.invoke(
        app,
        [
            *common,
            "--model",
            str(EXAMPLES / "toy_1d_phonon_chain.json"),
            "--site-mass",
            "1,0,8.0",
        ],
    )
    assert from_flag.exit_code == 0, from_flag.output
    assert json.loads(from_flag.output) == payload


def test_the_cli_rejects_a_malformed_site_mass() -> None:
    result = runner.invoke(
        app,
        [
            "phonon",
            "bands",
            "--model",
            str(EXAMPLES / "toy_1d_phonon_chain.json"),
            "--matrix",
            "3 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0",
            "--site-mass",
            "1,0,-8.0",
        ],
    )
    assert result.exit_code != 0
    assert "positive" in result.output
