"""End-to-end unfolding through the high-level :class:`UnfoldingProblem` API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic_procar import write_synthetic_procar
from synthetic_wavecar import write_synthetic_wavecar

from unmochan.core.kpoints import KPoint, cartesian_path_distances
from unmochan.io.vasp_wfc import generate_vasp_g_vectors
from unmochan.workflows.problem import UnfoldingProblem

ENCUT = 200.0
SUPERCELL_LATTICE = np.diag([2.0, 1.0, 1.0])


def test_problem_unfold_from_wavecar(tmp_path: Path):
    primitive = tmp_path / "POSCAR.prim"
    supercell = tmp_path / "POSCAR.sc"
    primitive.write_text(_poscar("primitive", ["1 0 0", "0 1 0", "0 0 1"], ["0 0 0"]))
    supercell.write_text(_poscar("supercell", ["2 0 0", "0 1 0", "0 0 1"], ["0 0 0", "0.5 0 0"]))
    wavecar = _write_gamma_wavecar(tmp_path / "WAVECAR", seed=3)

    problem = UnfoldingProblem.from_vasp(
        primitive=primitive,
        supercell=supercell,
        outputs=tmp_path,
        primitive_kpath=[KPoint([0.0, 0.0, 0.0]), KPoint([0.5, 0.0, 0.0])],
    )
    # Both primitive k-points fold onto the single stored supercell k-point.
    ebs = problem.unfold(operations=np.zeros((0, 3, 3), dtype=int))

    assert problem.transformation is not None
    assert np.allclose(problem.transformation.matrix, np.diag([2, 1, 1]))
    assert ebs.metadata["wavecar"] == str(wavecar)
    assert ebs.energies.shape == (2, 1)
    assert np.allclose(ebs.energies, -1.0)
    # The two members of the fiber carry the whole spectral weight of the band.
    assert ebs.weights[:, 0].sum() == pytest.approx(1.0)
    assert np.all(ebs.weights > 0.0)
    # The abscissa is the Cartesian reciprocal-space length of the path.
    assert np.allclose(ebs.distances, [0.0, np.pi])


def test_problem_unfold_applies_procar_projections(tmp_path: Path):
    problem, _ = _projection_problem(tmp_path)
    plain = problem.unfold(operations=np.zeros((0, 3, 3), dtype=int))
    first = problem.unfold(projections=["atom:0"], operations=np.zeros((0, 3, 3), dtype=int))
    second = problem.unfold(projections=["atom:1"], operations=np.zeros((0, 3, 3), dtype=int))

    # The synthetic PROCAR puts a quarter of the character on the first site.
    assert np.allclose(first.weights, 0.25 * plain.weights)
    assert np.allclose(second.weights, 0.75 * plain.weights)
    # A projection redistributes the unfolded weight; it never creates any.
    assert np.allclose(first.weights + second.weights, plain.weights)
    assert first.metadata["projection"] == "atom:0"


def test_problem_unfold_projection_selects_orbitals(tmp_path: Path):
    problem, _ = _projection_problem(tmp_path)
    plain = problem.unfold(operations=np.zeros((0, 3, 3), dtype=int))
    s_only = problem.unfold(projections=["orbital:s"], operations=np.zeros((0, 3, 3), dtype=int))
    p_only = problem.unfold(projections=["orbital:p"], operations=np.zeros((0, 3, 3), dtype=int))

    assert np.allclose(s_only.weights, 0.5 * plain.weights)
    assert np.allclose(s_only.weights + p_only.weights, plain.weights)


def test_problem_unfold_reports_a_missing_procar(tmp_path: Path):
    problem, procar = _projection_problem(tmp_path)
    procar.unlink()

    with pytest.raises(FileNotFoundError, match="no PROCAR found"):
        problem.unfold(projections=["atom:0"], operations=np.zeros((0, 3, 3), dtype=int))


def _projection_problem(tmp_path: Path) -> tuple[UnfoldingProblem, Path]:
    primitive = tmp_path / "POSCAR.prim"
    supercell = tmp_path / "POSCAR.sc"
    primitive.write_text(_poscar("primitive", ["1 0 0", "0 1 0", "0 0 1"], ["0 0 0"]))
    supercell.write_text(_poscar("supercell", ["2 0 0", "0 1 0", "0 0 1"], ["0 0 0", "0.5 0 0"]))
    _write_gamma_wavecar(tmp_path / "WAVECAR", seed=3)
    # One Gamma point, one band, two ions, s and px character.
    projections = np.array([[[[[0.1, 0.1], [0.3, 0.3]]]]], dtype=float)
    procar = write_synthetic_procar(
        tmp_path / "PROCAR",
        kpoints=np.zeros((1, 3)),
        kpoint_weights=np.ones(1),
        energies=np.array([[[-1.0]]]),
        occupations=np.array([[[1.0]]]),
        projections=projections,
        orbital_labels=("s", "px"),
    )
    problem = UnfoldingProblem.from_vasp(
        primitive=primitive,
        supercell=supercell,
        outputs=tmp_path,
        primitive_kpath=[KPoint([0.0, 0.0, 0.0]), KPoint([0.5, 0.0, 0.0])],
    )
    return problem, procar


def test_problem_unfold_reports_a_missing_wavecar(tmp_path: Path):
    primitive = tmp_path / "POSCAR.prim"
    supercell = tmp_path / "POSCAR.sc"
    primitive.write_text(_poscar("primitive", ["1 0 0", "0 1 0", "0 0 1"], ["0 0 0"]))
    supercell.write_text(_poscar("supercell", ["2 0 0", "0 1 0", "0 0 1"], ["0 0 0", "0.5 0 0"]))
    problem = UnfoldingProblem.from_vasp(
        primitive=primitive,
        supercell=supercell,
        outputs=tmp_path,
        primitive_kpath=[KPoint([0.0, 0.0, 0.0])],
    )

    with pytest.raises(FileNotFoundError, match="no WAVECAR found"):
        problem.unfold()


def test_cartesian_path_distances_uses_the_reciprocal_metric():
    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]])
    reciprocal = np.diag([1.0, 4.0, 1.0])

    fractional = cartesian_path_distances(kpoints)
    cartesian = cartesian_path_distances(kpoints, reciprocal)

    assert np.allclose(fractional, [0.0, 0.5, 1.0])
    # The second segment is four times longer in Cartesian reciprocal space.
    assert np.allclose(cartesian, [0.0, 0.5, 2.5])


def _write_gamma_wavecar(path: Path, *, seed: int) -> Path:
    g_vectors = generate_vasp_g_vectors(SUPERCELL_LATTICE, np.zeros(3), ENCUT)
    rng = np.random.default_rng(seed)
    values = (rng.normal(size=len(g_vectors)) + 1j * rng.normal(size=len(g_vectors))).astype(
        np.complex64
    )
    write_synthetic_wavecar(
        path,
        lattice=SUPERCELL_LATTICE,
        encut=ENCUT,
        rtag=45200,
        kpoints=np.zeros((1, 3)),
        energies=np.array([[-1.0]]),
        occupations=np.array([[1.0]]),
        coefficients=values.reshape(1, 1, 1, len(g_vectors)),
    )
    return path


def _poscar(name: str, lattice: list[str], positions: list[str]) -> str:
    return "\n".join([name, "1.0", *lattice, "X", str(len(positions)), "Direct", *positions])
