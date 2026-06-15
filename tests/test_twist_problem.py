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


def test_twisted_problem_loads_qe_inputs(tmp_path):
    reference = tmp_path / "reference.in"
    supercell = tmp_path / "supercell.in"
    reference.write_text(
        "\n".join(
            [
                "&SYSTEM",
                "  ibrav = 0,",
                "/",
                "CELL_PARAMETERS angstrom",
                "1 0 0",
                "0 1 0",
                "0 0 10",
                "ATOMIC_POSITIONS crystal",
                "A 0 0 0.25",
            ]
        )
    )
    supercell.write_text(
        "\n".join(
            [
                "&SYSTEM",
                "  ibrav = 0,",
                "/",
                "CELL_PARAMETERS angstrom",
                "2 0 0",
                "0 2 0",
                "0 0 10",
                "ATOMIC_POSITIONS crystal",
                "A 0 0 0.25",
                "B 0 0 0.75",
            ]
        )
    )

    problem = TwistedUnfoldingProblem.from_qe(
        references={"layer": reference},
        supercell=supercell,
    )

    assert problem.supercell_structure.n_sites == 2
    assert problem.references["layer"].n_sites == 1
    assert problem.detect_layers().tolist() == [0, 1]


def test_twisted_problem_loads_qe_alat_units(tmp_path):
    reference = tmp_path / "reference_alat.in"
    supercell = tmp_path / "supercell_alat.in"
    qe_text = "\n".join(
        [
            "&SYSTEM",
            "  ibrav = 0,",
            "  celldm(1) = 10.0,",
            "/",
            "CELL_PARAMETERS alat",
            "1 0 0",
            "0 1 0",
            "0 0 2",
            "ATOMIC_POSITIONS alat",
            "A 0 0 0.25",
        ]
    )
    reference.write_text(qe_text)
    supercell.write_text(
        qe_text.replace("A 0 0 0.25", "A 0 0 0.25\nB 0 0 1.50")
    )

    problem = TwistedUnfoldingProblem.from_structures(
        references={"layer": reference},
        supercell=supercell,
        code="auto",
    )

    assert problem.supercell_structure.n_sites == 2
