"""Band-chunked reading must reproduce the unchunked weights exactly.

The mathematical content is ``UnfoldLab.bandWeights_flatten``: the weight of a
band is a function of that band's coefficients alone, so splitting the bands of
a k-point into blocks and concatenating the results cannot change the answer.
These tests check the claim on real file reads, and also that the low-memory
path really does hold only one block at a time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic_wavecar import write_synthetic_wavecar
from test_qe_wfc_kpoints import write_synthetic_wfc_dat
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.core.unfolding import (
    PlaneWaveKPointData,
    compute_plane_wave_unfolding_weights,
    compute_plane_wave_unfolding_weights_chunked,
)
from unmochan.io.qe_wfc import compute_weights_from_qe_save, read_wfc_hdf5
from unmochan.io.vasp_wfc import (
    WavecarReader,
    compute_weights_from_wavecar,
    generate_vasp_g_vectors,
)

LATTICE = np.diag([4.0, 3.0, 5.0])
ENCUT = 120.0
KPOINTS = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
#: Primitive k-points folding onto ``KPOINTS`` under ``diag(2, 1, 1)``.
PRIMITIVE = np.array([[0.0, 0.0, 0.0], [0.125, 0.0, 0.0]])
N_BANDS = 5


def _write_multiband_wavecar(path: Path) -> None:
    rng = np.random.default_rng(20240617)
    blocks = []
    for kpoint in KPOINTS:
        g_vectors = generate_vasp_g_vectors(LATTICE, kpoint, ENCUT)
        shape = (N_BANDS, 1, len(g_vectors))
        block = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        blocks.append(block.astype(np.complex64))
    # The two k-points hold different numbers of G-vectors, so the blocks are
    # kept as a list rather than stacked.
    coefficients = blocks
    write_synthetic_wavecar(
        path,
        lattice=LATTICE,
        encut=ENCUT,
        rtag=45200,
        kpoints=KPOINTS,
        energies=np.linspace(-3.0, 3.0, KPOINTS.shape[0] * N_BANDS).reshape(
            KPOINTS.shape[0], N_BANDS
        ),
        occupations=np.ones((KPOINTS.shape[0], N_BANDS)),
        coefficients=coefficients,
        record_length=4096,
    )


@pytest.mark.parametrize("band_chunk", [1, 2, 3, N_BANDS, N_BANDS + 4])
def test_band_chunked_weights_match_whole_kpoint(tmp_path: Path, band_chunk: int):
    wavecar = tmp_path / "WAVECAR"
    _write_multiband_wavecar(wavecar)
    transform = np.diag([2, 1, 1])

    kpoints, energies, weights = compute_weights_from_wavecar(
        wavecar, PRIMITIVE, KPOINTS, transform
    )
    chunked_kpoints, chunked_energies, chunked_weights = compute_weights_from_wavecar(
        wavecar, PRIMITIVE, KPOINTS, transform, band_chunk=band_chunk
    )

    assert chunked_weights.shape == (KPOINTS.shape[0], N_BANDS)
    assert np.array_equal(chunked_kpoints, kpoints)
    assert np.array_equal(chunked_energies, energies)
    # Mathematically identical (``UnfoldLab.bandWeights_flatten``); in floating
    # point the block sums are contracted by a different BLAS call, so the last
    # bits may differ.
    assert np.max(np.abs(chunked_weights - weights)) < 1e-12


def test_band_chunk_must_be_positive(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR"
    _write_multiband_wavecar(wavecar)
    with pytest.raises(ValueError, match="band_chunk"):
        compute_weights_from_wavecar(wavecar, PRIMITIVE, KPOINTS, np.diag([2, 1, 1]), band_chunk=0)


def test_reader_band_block_is_a_slice_of_the_whole_kpoint(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR"
    _write_multiband_wavecar(wavecar)
    with WavecarReader(wavecar) as reader:
        _, whole = reader.read_kpoint_wavefunction(1, 1)
        _, block = reader.read_kpoint_wavefunction(1, 1, band_start=3, band_count=2)
        assert block.shape == (2, whole.shape[1], whole.shape[2])
        assert np.array_equal(block, whole[2:4])
        with pytest.raises(ValueError, match="band_start"):
            reader.read_kpoint_wavefunction(1, 1, band_start=0)
        with pytest.raises(ValueError, match="band_count"):
            reader.read_kpoint_wavefunction(1, 1, band_start=4, band_count=3)


def test_chunked_reader_holds_one_block_at_a_time(tmp_path: Path):
    """The blocks must be produced lazily, not all read up front."""

    wavecar = tmp_path / "WAVECAR"
    _write_multiband_wavecar(wavecar)
    resident: list[int] = []

    with WavecarReader(wavecar) as reader:
        original = reader.read_kpoint_wavefunction

        def counting(*args, **kwargs):
            k_header, coeffs = original(*args, **kwargs)
            resident.append(int(coeffs.shape[0]))
            return k_header, coeffs

        reader.read_kpoint_wavefunction = counting  # type: ignore[method-assign]

        def blocks(ik: int):
            for start in range(1, N_BANDS + 1, 2):
                count = min(2, N_BANDS - start + 1)
                k_header, coeffs = reader.read_kpoint_wavefunction(
                    1, ik, band_start=start, band_count=count
                )
                yield PlaneWaveKPointData(
                    primitive_kpoint=PRIMITIVE[ik - 1],
                    folded_supercell_kpoint=k_header.kpoint,
                    g_supercell=k_header.g_vectors,
                    coefficients=coeffs,
                )

        weights = compute_plane_wave_unfolding_weights_chunked(
            (blocks(ik) for ik in (1, 2)), np.diag([2, 1, 1])
        )

    assert weights.shape == (2, N_BANDS)
    assert max(resident) == 2  # never the whole five-band k-point


def test_chunked_and_flat_apis_agree_on_the_same_data(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR"
    _write_multiband_wavecar(wavecar)
    transform = np.diag([2, 1, 1])
    with WavecarReader(wavecar) as reader:
        items = [
            reader.plane_wave_kpoint_data(
                PRIMITIVE[ik - 1], KPOINTS[ik - 1], spin=1, kpoint_index=ik
            )
            for ik in (1, 2)
        ]
        chunks = [
            [
                reader.plane_wave_kpoint_data(
                    PRIMITIVE[ik - 1],
                    KPOINTS[ik - 1],
                    spin=1,
                    kpoint_index=ik,
                    band_start=start,
                    band_count=count,
                )
                for start, count in ((1, 2), (3, 3))
            ]
            for ik in (1, 2)
        ]
    flat = compute_plane_wave_unfolding_weights(items, transform)
    chunked = compute_plane_wave_unfolding_weights_chunked(chunks, transform)
    assert chunked.shape == flat.shape
    assert np.max(np.abs(flat - chunked)) < 1e-12

    resolved_flat = compute_plane_wave_unfolding_weights(items, transform, component_resolved=True)
    resolved_chunked = compute_plane_wave_unfolding_weights_chunked(
        chunks, transform, component_resolved=True
    )
    assert resolved_chunked.shape == resolved_flat.shape
    assert np.max(np.abs(resolved_flat - resolved_chunked)) < 1e-12


def test_chunks_must_describe_the_same_kpoint(tmp_path: Path):
    wavecar = tmp_path / "WAVECAR"
    _write_multiband_wavecar(wavecar)
    with WavecarReader(wavecar) as reader:
        first = reader.plane_wave_kpoint_data(
            PRIMITIVE[0], KPOINTS[0], spin=1, kpoint_index=1, band_start=1, band_count=2
        )
        other = reader.plane_wave_kpoint_data(
            PRIMITIVE[1], KPOINTS[1], spin=1, kpoint_index=2, band_start=3, band_count=3
        )
    with pytest.raises(ValueError, match="share the k-point"):
        compute_plane_wave_unfolding_weights_chunked([[first, other]], np.diag([2, 1, 1]))


def test_cli_band_chunk_matches_default(tmp_path: Path):
    """``unmochan weights --band-chunk`` writes the same table."""

    runner = CliRunner()
    wavecar = tmp_path / "WAVECAR"
    _write_multiband_wavecar(wavecar)
    kmap = tmp_path / "kmap.dat"
    rows = ["ik\ts_pc\tk1\tk2\tk3\tK1\tK2\tK3\tKf1\tKf2\tKf3\tlabel"]
    for index, (primitive, folded) in enumerate(zip(PRIMITIVE, KPOINTS, strict=True), start=1):
        columns = [str(index), f"{float(index - 1):.6f}"]
        columns += [f"{value:.6f}" for value in primitive]
        columns += [f"{value:.6f}" for value in folded]
        columns += [f"{value:.6f}" for value in folded]
        columns += [""]
        rows.append("\t".join(columns))
    kmap.write_text("\n".join(rows) + "\n")
    common = [
        "weights",
        "--code",
        "vasp",
        "--kmap",
        str(kmap),
        "--matrix",
        "2 0 0 0 1 0 0 0 1",
        "--wavecar",
        str(wavecar),
    ]
    whole = tmp_path / "whole.dat"
    chunked = tmp_path / "chunked.dat"
    first = runner.invoke(app, [*common, "--out", str(whole)])
    second = runner.invoke(app, [*common, "--out", str(chunked), "--band-chunk", "2"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert np.allclose(
        np.loadtxt(whole, ndmin=2), np.loadtxt(chunked, ndmin=2), rtol=0.0, atol=1e-12
    )


QE_MILLER = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
QE_COEFFS = np.array(
    [
        [1.0 + 0.5j, 1.0, 2.0, 0.5j],
        [0.0, 1.0, 0.0, 2.0 - 1.0j],
        [1.0, -1.0, 0.5, 0.25],
        [0.5, 0.5j, -2.0, 1.0],
    ],
    dtype=complex,
)


@pytest.mark.parametrize("band_chunk", [1, 2, 3, 4])
def test_qe_dat_band_chunking_matches_whole_file(tmp_path: Path, band_chunk: int):
    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), QE_MILLER, QE_COEFFS)

    common = {
        "file_format": "dat",
        "lattice_alat": np.eye(3),
    }
    whole = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.25, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0]]),
        np.diag([2, 1, 1]),
        4,
        **common,
    )
    chunked = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.25, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0]]),
        np.diag([2, 1, 1]),
        4,
        band_chunk=band_chunk,
        **common,
    )
    assert np.max(np.abs(whole - chunked)) < 1e-12


@pytest.mark.parametrize("band_chunk", [1, 3])
def test_qe_hdf5_band_chunking_matches_whole_file(tmp_path: Path, band_chunk: int):
    h5py = pytest.importorskip("h5py")
    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    with h5py.File(save_dir / "wfc1.hdf5", "w") as handle:
        handle.attrs["ik"] = 1
        handle.attrs["ispin"] = 1
        handle.attrs["gamma_only"] = ".FALSE."
        handle.attrs["npol"] = 1
        handle.create_dataset("MillerIndices", data=QE_MILLER)
        interleaved = np.empty((QE_COEFFS.shape[0], 2 * QE_COEFFS.shape[1]))
        interleaved[:, 0::2] = QE_COEFFS.real
        interleaved[:, 1::2] = QE_COEFFS.imag
        handle.create_dataset("evc", data=interleaved)

    whole = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
        np.diag([2, 1, 1]),
        4,
        file_format="hdf5",
    )
    chunked = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
        np.diag([2, 1, 1]),
        4,
        file_format="hdf5",
        band_chunk=band_chunk,
    )
    assert np.max(np.abs(whole - chunked)) < 1e-12

    # A block read is a slice of the whole read, not a re-indexed one.
    _, all_bands = read_wfc_hdf5(save_dir / "wfc1.hdf5", 4)
    _, block = read_wfc_hdf5(save_dir / "wfc1.hdf5", 4, band_start=2, band_count=2)
    assert np.array_equal(block, all_bands[1:3])
