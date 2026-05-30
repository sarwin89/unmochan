import numpy as np

from unfoldlab.core.structures import Structure
from unfoldlab.twist import TwistedUnfoldingProblem, assign_layers_by_axis


def test_assign_layers_supports_more_than_two_layers():
    structure = Structure(
        lattice=np.eye(3),
        species=("A", "A", "A", "A"),
        frac_coords=[[0, 0, 0.1], [0, 0, 0.35], [0, 0, 0.6], [0, 0, 0.9]],
    )

    layers = assign_layers_by_axis(structure, n_layers=4)

    assert layers.tolist() == [0, 1, 2, 3]


def test_twisted_problem_accepts_arbitrary_reference_labels():
    reference = Structure(np.eye(3), ("A",), [[0, 0, 0]])
    substrate = Structure(np.diag([2, 2, 1]), ("B",), [[0, 0, 0]])
    supercell = Structure(np.diag([3, 3, 1]), ("A", "B"), [[0, 0, 0.2], [0, 0, 0.8]])

    problem = TwistedUnfoldingProblem.from_structures(
        references={"layer_1": reference, "substrate": substrate},
        supercell=supercell,
    )

    assert sorted(problem.references) == ["layer_1", "substrate"]
    assert problem.detect_layers().tolist() == [0, 1]
    assert set(problem.detect_relative_rotations()) == {"layer_1", "substrate"}
    assert problem.build_moire_reciprocal_lattice().shape == (3, 3)
