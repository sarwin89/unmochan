from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.io.qe import read_kmap
from unfoldlab.io.vasp import write_vasp_path_files
from unfoldlab.io.vasp_wfc import (
    WavecarReader,
    compute_weights_from_wavecar,
    generate_vasp_g_vectors,
    read_wavecar_header,
)

runner = CliRunner()


def test_read_synthetic_collinear_wavecar_and_weights(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR"
    lattice = np.eye(3)
    encut = 200.0
    g_vectors = generate_vasp_g_vectors(lattice, np.array([0.0, 0.0, 0.0]), encut)
    coeffs = np.ones((1, 1, len(g_vectors)), dtype=np.complex64)
    _write_synthetic_wavecar(
        wavecar,
        lattice=lattice,
        encut=encut,
        rtag=45200,
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        energies=np.array([[-1.0]]),
        occupations=np.array([[1.0]]),
        coefficients=coeffs[np.newaxis, ...],
    )

    header = read_wavecar_header(wavecar)
    with WavecarReader(wavecar) as reader:
        k_header, read_coeffs = reader.read_kpoint_wavefunction(1, 1)

    _kpoints, energies, weights = compute_weights_from_wavecar(
        wavecar,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
        np.diag([2, 1, 1]),
    )

    expected = np.count_nonzero(g_vectors[:, 0] % 2 == 0) / len(g_vectors)
    assert header.n_kpoints == 1
    assert header.n_bands == 1
    assert k_header.n_plane_waves == len(g_vectors)
    assert read_coeffs.shape == (1, 1, len(g_vectors))
    assert np.allclose(energies, [[-1.0]])
    assert np.allclose(weights, [[expected]])


def test_read_synthetic_spinor_wavecar_sums_components(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR.spinor"
    lattice = np.eye(3)
    encut = 200.0
    g_vectors = generate_vasp_g_vectors(lattice, np.array([0.0, 0.0, 0.0]), encut)
    coeffs = np.zeros((1, 1, 2, len(g_vectors)), dtype=np.complex64)
    coeffs[0, 0, 0, :] = 1.0
    coeffs[0, 0, 1, :] = 2.0
    _write_synthetic_wavecar(
        wavecar,
        lattice=lattice,
        encut=encut,
        rtag=53300,
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        energies=np.array([[0.5]]),
        occupations=np.array([[1.0]]),
        coefficients=coeffs,
    )

    _kpoints, _energies, weights = compute_weights_from_wavecar(
        wavecar,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
        np.diag([2, 1, 1]),
    )

    expected = np.count_nonzero(g_vectors[:, 0] % 2 == 0) / len(g_vectors)
    assert np.allclose(weights, [[expected]])


def test_write_vasp_path_files_matches_shared_kmap(tmp_path: Path):
    path_json = tmp_path / "path.json"
    kpoints = tmp_path / "KPOINTS"
    kmap = tmp_path / "kmap.tsv"
    ticks = tmp_path / "ticks.tsv"
    path_json.write_text(
        '{"transformation_matrix": [[2,0,0],[0,1,0],[0,0,1]], '
        '"path": [{"label":"G","k":[0,0,0],"n":2},{"label":"X","k":[0.5,0,0]}]}'
    )

    write_vasp_path_files(path_json, kpoints=kpoints, kmap=kmap, ticks=ticks)
    mapping = read_kmap(kmap)

    assert "Reciprocal" in kpoints.read_text()
    assert np.allclose(mapping.supercell_folded_kpoints[-1], [0.0, 0.0, 0.0])


def test_unified_vasp_weights_and_unfold_cli_from_wavecar(tmp_path: Path):
    path_json, kmap, _ticks = _write_path_files(tmp_path)
    wavecar = tmp_path / "WAVECAR"
    lattice = np.eye(3)
    encut = 200.0
    g_vectors = generate_vasp_g_vectors(lattice, np.array([0.0, 0.0, 0.0]), encut)
    coeffs = np.ones((2, 1, 1, len(g_vectors)), dtype=np.complex64)
    _write_synthetic_wavecar(
        wavecar,
        lattice=lattice,
        encut=encut,
        rtag=45200,
        kpoints=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        energies=np.array([[-1.0], [-0.5]]),
        occupations=np.array([[1.0], [1.0]]),
        coefficients=coeffs,
    )
    weights = tmp_path / "weights.dat"
    unfolded = tmp_path / "unfolded.dat"
    plot = tmp_path / "plot.svg"

    weights_result = runner.invoke(
        app,
        [
            "weights",
            "--code",
            "vasp",
            "--kmap",
            str(kmap),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--wavecar",
            str(wavecar),
            "--out",
            str(weights),
        ],
    )
    unfold_result = runner.invoke(
        app,
        [
            "unfold",
            "--code",
            "vasp",
            "--kmap",
            str(kmap),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--wavecar",
            str(wavecar),
            "--out",
            str(unfolded),
            "--plot",
            str(plot),
        ],
    )

    assert path_json.exists()
    assert weights_result.exit_code == 0, weights_result.output
    assert unfold_result.exit_code == 0, unfold_result.output
    assert weights.exists()
    assert unfolded.exists()
    assert plot.exists()


def _write_synthetic_wavecar(
    path: Path,
    *,
    lattice: np.ndarray,
    encut: float,
    rtag: int,
    kpoints: np.ndarray,
    energies: np.ndarray,
    occupations: np.ndarray,
    coefficients: np.ndarray,
) -> None:
    record_length = 512
    n_spin = 1
    n_kpoints = kpoints.shape[0]
    n_bands = energies.shape[1]
    coeff_dtype = np.complex128 if rtag == 45210 else np.complex64
    with path.open("wb") as handle:
        first_record = np.array([record_length, n_spin, rtag], dtype=np.float64)
        _write_record(handle, record_length, first_record)
        header = np.concatenate(
            [
                np.array([n_kpoints, n_bands, encut], dtype=np.float64),
                lattice.reshape(-1),
                np.array([0.0], dtype=np.float64),
            ]
        )
        _write_record(handle, record_length, header)
        for ik, kpoint in enumerate(kpoints):
            g_vectors = generate_vasp_g_vectors(lattice, kpoint, encut)
            band_rows = []
            for ib in range(n_bands):
                band_rows.extend([energies[ik, ib], 0.0, occupations[ik, ib]])
            k_header = np.array([len(g_vectors), *kpoint, *band_rows], dtype=np.float64)
            _write_record(handle, record_length, k_header)
            for ib in range(n_bands):
                values = coefficients[ik, ib].reshape(-1).astype(coeff_dtype)
                _write_record(handle, record_length, values)


def _write_record(handle, record_length: int, values: np.ndarray) -> None:
    payload = values.tobytes()
    if len(payload) > record_length:
        raise ValueError("synthetic record is too large")
    handle.write(payload)
    handle.write(b"\x00" * (record_length - len(payload)))


def _write_path_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    path_json = tmp_path / "path.json"
    kpoints = tmp_path / "KPOINTS"
    kmap = tmp_path / "kmap.tsv"
    ticks = tmp_path / "ticks.tsv"
    path_json.write_text(
        '{"transformation_matrix": [[2,0,0],[0,1,0],[0,0,1]], '
        '"path": [{"label":"G","k":[0,0,0],"n":2},{"label":"G","k":[0,0,0]}]}'
    )
    write_vasp_path_files(path_json, kpoints=kpoints, kmap=kmap, ticks=ticks)
    return path_json, kmap, ticks
