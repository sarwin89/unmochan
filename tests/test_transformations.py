import numpy as np
import pytest

from unfoldlab.core.kpoints import KPoint, fold_kpoint_to_supercell
from unfoldlab.core.structures import Structure
from unfoldlab.core.transformations import detect_transformation


def test_detects_diagonal_supercell_transform():
    primitive = Structure(np.eye(3), ("Si",), [[0, 0, 0]])
    supercell = Structure(np.diag([2, 2, 1]), ("Si",), [[0, 0, 0]])

    transform = detect_transformation(primitive, supercell)

    assert transform.matrix.tolist() == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    assert transform.multiplicity == 4


def test_detects_non_diagonal_transform():
    primitive = Structure(np.eye(3), ("X",), [[0, 0, 0]])
    supercell_lattice = np.array([[2, 1, 0], [0, 3, 0], [0, 0, 1]], dtype=float)
    supercell = Structure(supercell_lattice, ("X",), [[0, 0, 0]])

    transform = detect_transformation(primitive, supercell)

    assert transform.matrix.tolist() == [[2, 1, 0], [0, 3, 0], [0, 0, 1]]
    assert transform.multiplicity == 6


def test_rejects_non_integer_transform():
    primitive = Structure(np.eye(3), ("X",), [[0, 0, 0]])
    supercell = Structure(np.diag([1.5, 2, 1]), ("X",), [[0, 0, 0]])

    with pytest.raises(ValueError, match="not related by an integer"):
        detect_transformation(primitive, supercell)


def test_folds_primitive_kpoint_to_supercell_bz():
    transform = detect_transformation(np.eye(3), np.diag([2, 2, 1]))

    folded = fold_kpoint_to_supercell(KPoint([0.5, 0.25, 0], label="X"), transform)

    assert folded.label == "X"
    assert np.allclose(folded.fractional, [0.0, 0.5, 0.0])
