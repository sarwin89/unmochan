"""Synthetic VASP WAVECAR writer shared by the tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from unmochan.io.vasp_wfc import generate_vasp_g_vectors


def write_synthetic_wavecar(
    path: Path,
    *,
    lattice: np.ndarray,
    encut: float,
    rtag: int,
    kpoints: np.ndarray,
    energies: np.ndarray,
    occupations: np.ndarray,
    coefficients: np.ndarray,
    n_plane_waves: int | None = None,
    record_length: int = 512,
) -> None:
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
            stored = len(g_vectors) if n_plane_waves is None else n_plane_waves
            k_header = np.array([stored, *kpoint, *band_rows], dtype=np.float64)
            _write_record(handle, record_length, k_header)
            for ib in range(n_bands):
                values = np.asarray(coefficients[ik][ib]).reshape(-1).astype(coeff_dtype)
                _write_record(handle, record_length, values)


def _write_record(handle, record_length: int, values: np.ndarray) -> None:
    payload = values.tobytes()
    if len(payload) > record_length:
        raise ValueError("synthetic record is too large")
    handle.write(payload)
    handle.write(b"\x00" * (record_length - len(payload)))
