"""Reading a WAVECAR must not regenerate the plane-wave sphere per band.

Deriving a k-point header means enumerating every G-vector inside the cutoff
sphere, which dominates the cost of reading a band.  The reader caches the
derived header, so a k-point is resolved once however many bands are read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.synthetic_wavecar import write_synthetic_wavecar
from unfoldlab.io import vasp_wfc
from unfoldlab.io.vasp_wfc import WavecarReader, generate_vasp_g_vectors

LATTICE = np.eye(3) * 4.0
ENCUT = 30.0
KPOINTS = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
N_BANDS = 4


def _write(path: Path) -> int:
    n_g = generate_vasp_g_vectors(LATTICE, KPOINTS[0], ENCUT).shape[0]
    rng = np.random.default_rng(0)
    coefficients = rng.normal(size=(len(KPOINTS), N_BANDS, n_g)) + 0j
    write_synthetic_wavecar(
        path,
        lattice=LATTICE,
        encut=ENCUT,
        rtag=45210,
        kpoints=KPOINTS,
        energies=np.zeros((len(KPOINTS), N_BANDS)),
        occupations=np.ones((len(KPOINTS), N_BANDS)),
        coefficients=coefficients,
    )
    return n_g


def _counting_generator(monkeypatch) -> list[int]:
    calls: list[int] = []
    original = vasp_wfc.generate_vasp_g_vectors

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(vasp_wfc, "generate_vasp_g_vectors", counted)
    return calls


def test_a_kpoint_basis_is_generated_once_per_kpoint(tmp_path: Path, monkeypatch):
    path = tmp_path / "WAVECAR"
    _write(path)
    calls = _counting_generator(monkeypatch)

    with WavecarReader(path) as reader:
        _header, coefficients = reader.read_kpoint_wavefunction(1, 1)

    assert coefficients.shape[0] == N_BANDS
    # One sphere for the k-point, not one per band.
    assert len(calls) == 1


def test_the_cache_is_bounded_and_still_correct(tmp_path: Path, monkeypatch):
    path = tmp_path / "WAVECAR"
    _write(path)

    with WavecarReader(path, cache_size=1) as reader:
        first = reader.read_kpoint_header(1, 1)
        second = reader.read_kpoint_header(1, 2)
        again = reader.read_kpoint_header(1, 1)

    assert np.allclose(first.kpoint, KPOINTS[0])
    assert np.allclose(second.kpoint, KPOINTS[1])
    assert np.allclose(again.g_vectors, first.g_vectors)

    calls = _counting_generator(monkeypatch)
    with WavecarReader(path, cache_size=1) as reader:
        reader.read_kpoint_header(1, 1)
        reader.read_kpoint_header(1, 2)
        reader.read_kpoint_header(1, 1)
    # With room for a single entry the alternation misses on the third read.
    assert len(calls) == 3


def test_repeated_reads_of_the_same_kpoint_are_served_from_the_cache(tmp_path: Path, monkeypatch):
    path = tmp_path / "WAVECAR"
    _write(path)
    calls = _counting_generator(monkeypatch)

    with WavecarReader(path) as reader:
        for _ in range(3):
            reader.read_kpoint_header(1, 2)
            reader.read_coefficients(1, 2, 1)

    assert len(calls) == 1
