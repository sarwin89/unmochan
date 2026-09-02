"""Write a minimal but realistic QE ``data-file-schema.xml`` for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

NAMESPACE = "http://www.quantum-espresso.org/ns/qes/qes-1.0"


def _vector(tag: str, values: NDArray[np.float64]) -> str:
    body = " ".join(f"{value:.12e}" for value in np.asarray(values, dtype=float))
    return f"<{tag}>{body}</{tag}>"


def write_data_file_schema(
    path: str | Path,
    *,
    lattice_bohr: NDArray[np.float64],
    alat_bohr: float,
    kpoints_cart_alat: NDArray[np.float64],
    eigenvalues_hartree: NDArray[np.float64] | None = None,
    weights: NDArray[np.float64] | None = None,
    uspp: bool = False,
    paw: bool = False,
    lsda: bool = False,
    noncolin: bool = False,
    gamma_only: bool = False,
    species: tuple[tuple[str, str], ...] = (("Si", "Si.pz-vbc.UPF"),),
    nat: int = 2,
    fermi_hartree: float | None = 0.2,
) -> Path:
    """Write a QE-like XML document and return its path."""

    lattice = np.asarray(lattice_bohr, dtype=float).reshape(3, 3)
    kpoints = np.asarray(kpoints_cart_alat, dtype=float).reshape(-1, 3)
    n_kpoints = kpoints.shape[0]
    if eigenvalues_hartree is None:
        energies = np.zeros((n_kpoints, 0), dtype=float)
    else:
        energies = np.asarray(eigenvalues_hartree, dtype=float).reshape(n_kpoints, -1)
    if weights is None:
        weights = np.full(n_kpoints, 1.0 / max(n_kpoints, 1), dtype=float)

    if abs(float(np.linalg.det(lattice))) > 1e-12:
        reciprocal = np.linalg.inv(lattice / alat_bohr).T
    else:  # deliberately degenerate cell, for the rejection tests
        reciprocal = None

    species_xml = "".join(
        f'<species name="{name}"><mass>28.0</mass><pseudo_file>{pseudo}</pseudo_file></species>'
        for name, pseudo in species
    )
    structure_xml = (
        f'<atomic_structure nat="{nat}" alat="{alat_bohr:.12e}">'
        "<atomic_positions>"
        f'<atom name="{species[0][0]}" index="1">0.0 0.0 0.0</atom>'
        "</atomic_positions>"
        "<cell>"
        + _vector("a1", lattice[0])
        + _vector("a2", lattice[1])
        + _vector("a3", lattice[2])
        + "</cell></atomic_structure>"
    )
    reciprocal_xml = (
        ""
        if reciprocal is None
        else "<reciprocal_lattice>"
        + _vector("b1", reciprocal[0])
        + _vector("b2", reciprocal[1])
        + _vector("b3", reciprocal[2])
        + "</reciprocal_lattice>"
    )
    basis_xml = (
        "<basis_set>"
        f"<gamma_only>{'true' if gamma_only else 'false'}</gamma_only>"
        "<ecutwfc>1.8e1</ecutwfc>"
        f"{reciprocal_xml}</basis_set>"
    )

    blocks = []
    for index in range(n_kpoints):
        eigen = energies[index]
        blocks.append(
            f'<ks_energies><k_point weight="{weights[index]:.12e}">'
            + " ".join(f"{value:.12e}" for value in kpoints[index])
            + "</k_point>"
            + f"<npw>{100 + index}</npw>"
            + f'<eigenvalues size="{eigen.size}">'
            + " ".join(f"{value:.12e}" for value in eigen)
            + "</eigenvalues>"
            + f'<occupations size="{eigen.size}">'
            + " ".join("1.0" for _ in eigen)
            + "</occupations></ks_energies>"
        )

    band_xml = (
        "<band_structure>"
        f"<lsda>{'true' if lsda else 'false'}</lsda>"
        f"<noncolin>{'true' if noncolin else 'false'}</noncolin>"
        "<spinorbit>false</spinorbit>"
        f"<nbnd>{energies.shape[1]}</nbnd>"
        "<nelec>8.0</nelec>"
        + ("" if fermi_hartree is None else f"<fermi_energy>{fermi_hartree:.12e}</fermi_energy>")
        + f"<nks>{n_kpoints}</nks>"
        + "".join(blocks)
        + "</band_structure>"
    )

    algorithmic_xml = (
        "<algorithmic_info><real_space_q>false</real_space_q>"
        f"<uspp>{'true' if uspp else 'false'}</uspp>"
        f"<paw>{'true' if paw else 'false'}</paw></algorithmic_info>"
    )

    document = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<qes:espresso xmlns:qes="{NAMESPACE}">'
        '<general_info><creator NAME="PWSCF" VERSION="7.2"/></general_info>'
        "<input>"
        f'<atomic_species ntyp="{len(species)}">{species_xml}</atomic_species>'
        f"{structure_xml}</input>"
        "<output>"
        f'<atomic_species ntyp="{len(species)}">{species_xml}</atomic_species>'
        f"{structure_xml}{basis_xml}{band_xml}{algorithmic_xml}"
        "</output></qes:espresso>"
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out
