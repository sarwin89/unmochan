import numpy as np
import pytest
from synthetic_wavecar import write_synthetic_wavecar

from unmochan.core.kpoints import KPoint
from unmochan.core.structures import Structure
from unmochan.io.vasp_wfc import generate_vasp_g_vectors
from unmochan.twist import TwistedUnfoldingProblem, assign_layers_by_axis


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
    supercell.write_text(qe_text.replace("A 0 0 0.25", "A 0 0 0.25\nB 0 0 1.50"))

    problem = TwistedUnfoldingProblem.from_structures(
        references={"layer": reference},
        supercell=supercell,
        code="auto",
    )

    assert problem.supercell_structure.n_sites == 2


def test_moire_lattice_inverts_only_the_in_plane_block():
    """B_a - B_b is singular for a layered stack; the in-plane block is not."""

    bottom = Structure(np.diag([1.0, 1.0, 10.0]), ("A",), [[0, 0, 0]])
    top = Structure(np.diag([0.5, 0.5, 10.0]), ("B",), [[0, 0, 0]])
    supercell = Structure(np.diag([1.0, 1.0, 10.0]), ("A", "B"), [[0, 0, 0.2], [0, 0, 0.8]])
    problem = TwistedUnfoldingProblem.from_structures(
        references={"bottom": bottom, "top": top},
        supercell=supercell,
    )

    difference = problem.build_moire_reciprocal_lattice()
    assert abs(float(np.linalg.det(difference))) < 1e-12

    moire = problem.moire_lattice()
    # Commensurate 1x1 and 0.5x0.5 layers repeat with the period of the larger
    # cell, and the stacking vector is inherited from the supercell.
    assert np.allclose(np.abs(np.diag(moire)), [1.0, 1.0, 10.0])
    assert np.allclose(moire[2], supercell.lattice[2])


def test_moire_lattice_rejects_identical_references():
    reference = Structure(np.diag([1.0, 1.0, 10.0]), ("A",), [[0, 0, 0]])
    supercell = Structure(np.diag([1.0, 1.0, 10.0]), ("A", "A"), [[0, 0, 0.2], [0, 0, 0.8]])
    problem = TwistedUnfoldingProblem.from_structures(
        references={"bottom": reference, "top": reference},
        supercell=supercell,
    )

    with pytest.raises(ValueError, match="no moire cell"):
        problem.moire_lattice()


def test_unfold_to_reference_uses_that_references_transform(tmp_path):
    lattice = np.diag([2.0, 1.0, 1.0])
    encut = 200.0
    g_vectors = generate_vasp_g_vectors(lattice, np.zeros(3), encut)
    rng = np.random.default_rng(5)
    values = (rng.normal(size=len(g_vectors)) + 1j * rng.normal(size=len(g_vectors))).astype(
        np.complex64
    )
    wavecar = tmp_path / "WAVECAR"
    write_synthetic_wavecar(
        wavecar,
        lattice=lattice,
        encut=encut,
        rtag=45200,
        kpoints=np.zeros((1, 3)),
        energies=np.array([[-0.5]]),
        occupations=np.array([[1.0]]),
        coefficients=values.reshape(1, 1, 1, len(g_vectors)),
    )

    problem = TwistedUnfoldingProblem.from_structures(
        references={"layer": Structure(np.eye(3), ("A",), [[0, 0, 0]])},
        supercell=Structure(lattice, ("A", "A"), [[0, 0, 0], [0.5, 0, 0]]),
        outputs=tmp_path,
    )
    ebs = problem.unfold_to_reference(
        "layer",
        kpath=[KPoint([0.0, 0.0, 0.0]), KPoint([0.5, 0.0, 0.0])],
        operations=np.zeros((0, 3, 3), dtype=int),
    )

    assert ebs.metadata["reference"] == "layer"
    assert ebs.metadata["transformation"]["multiplicity"] == 2
    assert np.allclose(ebs.energies, -0.5)
    assert ebs.weights[:, 0].sum() == pytest.approx(1.0)


def test_unfold_to_reference_rejects_an_incommensurate_reference(tmp_path):
    problem = TwistedUnfoldingProblem.from_structures(
        references={"layer": Structure(np.diag([0.7, 1.0, 1.0]), ("A",), [[0, 0, 0]])},
        supercell=Structure(np.diag([2.0, 1.0, 1.0]), ("A",), [[0, 0, 0]]),
        outputs=tmp_path,
    )

    with pytest.raises(ValueError, match="integer transformation"):
        problem.unfold_to_reference(
            "layer",
            kpath=[KPoint([0.0, 0.0, 0.0])],
            wavecar=tmp_path / "missing",
        )
