from pathlib import Path

import numpy as np

from unmochan.io.vasp import read_eigenval, read_poscar


def test_read_poscar_with_species_names(tmp_path: Path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "\n".join(
            [
                "Si primitive",
                "1.0",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "Si",
                "2",
                "Direct",
                "0 0 0",
                "0.25 0.25 0.25",
            ]
        )
    )

    structure = read_poscar(poscar)

    assert structure.name == "Si primitive"
    assert structure.species == ("Si", "Si")
    assert np.allclose(structure.frac_coords[1], [0.25, 0.25, 0.25])


def test_read_eigenval_basic_non_spin_file(tmp_path: Path):
    eigenval = tmp_path / "EIGENVAL"
    eigenval.write_text(
        "\n".join(
            [
                " header 1",
                " header 2",
                " header 3",
                " header 4",
                " header 5",
                " 2 1 2",
                "",
                "0.0 0.0 0.0 1.0",
                "1 -1.0 1.0",
                "2  2.0 0.0",
            ]
        )
    )

    data = read_eigenval(eigenval)

    assert data.nelect == 2
    assert np.allclose(data.kpoints, [[0.0, 0.0, 0.0]])
    assert np.allclose(data.energies, [[-1.0, 2.0]])
    assert np.allclose(data.occupations, [[1.0, 0.0]])
