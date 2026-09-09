"""The k-point recorded inside a QE wavefunction file must match the k-map.

The Miller indices stored in ``wfc*.dat`` / ``wfc*.hdf5`` are indexed against
the k-point of that file, so unfolding against a different representative of
the same k-point selects a shifted set of plane waves.  These tests cover the
cross-check and the ``xk`` plumbing that makes it possible.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from unmochan.io.qe_wfc import (
    compute_weights_from_qe_save,
    fractional_kpoint_from_xk,
    parse_binary_header,
    read_xk_dat,
)


def _record(payload: bytes) -> bytes:
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def write_synthetic_wfc_dat(
    path: Path,
    xk: tuple[float, float, float],
    miller: np.ndarray,
    coefficients: np.ndarray,
    *,
    gamma_only: bool = False,
) -> None:
    """Write a minimal QE unformatted ``wfc*.dat`` file.

    ``coefficients`` has shape ``(nbnd, igwx)`` and ``miller`` shape
    ``(igwx, 3)``.
    """

    igwx = int(miller.shape[0])
    nbnd = int(coefficients.shape[0])
    chunks = [
        _record(struct.pack("<i3diid", 1, xk[0], xk[1], xk[2], 1, int(gamma_only), 1.0)),
        _record(struct.pack("<4i", igwx, igwx, 1, nbnd)),
        _record(struct.pack("<9d", *np.eye(3).reshape(-1))),
        # QE writes ``mill(3, igwx)`` in Fortran order, i.e. component-fastest,
        # which is exactly the C-order layout of an ``(igwx, 3)`` array.
        _record(np.ascontiguousarray(miller, dtype="<i4").tobytes()),
    ]
    for band in range(nbnd):
        chunks.append(_record(np.asarray(coefficients[band], dtype="<c16").tobytes()))
    path.write_bytes(b"".join(chunks))


MILLER = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
COEFFS = np.array([[1.0, 1.0, 2.0], [0.0, 1.0, 0.0]], dtype=complex)
TRANSFORM = np.diag([2, 1, 1])


def test_binary_header_exposes_xk(tmp_path: Path):
    path = tmp_path / "wfc1.dat"
    write_synthetic_wfc_dat(path, (0.5, 0.0, 0.0), MILLER, COEFFS)

    header = parse_binary_header(path)

    assert header.ik == 1
    assert header.nbnd == 2
    assert header.xk == (0.5, 0.0, 0.0)
    assert np.allclose(read_xk_dat(path), [0.5, 0.0, 0.0])


def test_fractional_kpoint_from_xk_uses_the_direct_lattice():
    # Cubic cell: at = identity, so xk already is the fractional coordinate.
    assert np.allclose(fractional_kpoint_from_xk((0.5, 0.0, 0.0), np.eye(3)), [0.5, 0.0, 0.0])
    # Tetragonal cell with c = 2a: a Cartesian 0.25 along z is 0.5 fractional.
    tetragonal = np.diag([1.0, 1.0, 2.0])
    assert np.allclose(fractional_kpoint_from_xk((0.0, 0.0, 0.25), tetragonal), [0.0, 0.0, 0.5])


def test_weights_from_dat_with_consistent_kpoint(tmp_path: Path):
    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    weights = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.25, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0]]),
        TRANSFORM,
        2,
        file_format="dat",
        lattice_alat=np.eye(3),
    )

    # G = (0,0,0) and (2,0,0) belong to the primitive k-point, (1,0,0) does not.
    assert np.allclose(weights, [[5.0 / 6.0, 0.0]])


def test_weights_from_dat_rejects_inconsistent_kpoint(tmp_path: Path):
    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    with pytest.raises(ValueError, match="k-map expects"):
        compute_weights_from_qe_save(
            save_dir,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            TRANSFORM,
            2,
            file_format="dat",
            lattice_alat=np.eye(3),
        )


def test_kpoint_check_is_skipped_without_a_lattice(tmp_path: Path):
    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    weights = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.25, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0]]),
        TRANSFORM,
        2,
        file_format="dat",
    )

    assert np.allclose(weights, [[5.0 / 6.0, 0.0]])


def test_equivalent_representative_of_the_same_kpoint_is_accepted(tmp_path: Path):
    """A k-map entry shifted by a reciprocal lattice vector is the same point."""

    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    weights = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.25, 0.0, 0.0]]),
        np.array([[-0.5, 0.0, 0.0]]),
        TRANSFORM,
        2,
        file_format="dat",
        lattice_alat=np.eye(3),
    )

    # The file's own representative is used, so the weight is unchanged.
    assert np.allclose(weights, [[5.0 / 6.0, 0.0]])


def test_hdf5_kpoint_cross_check(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    with h5py.File(save_dir / "wfc1.hdf5", "w") as handle:
        handle.attrs["ik"] = 1
        handle.attrs["ispin"] = 1
        handle.attrs["gamma_only"] = ".FALSE."
        handle.attrs["npol"] = 1
        handle.attrs["xk"] = np.array([0.5, 0.0, 0.0])
        handle.create_dataset("MillerIndices", data=MILLER)
        handle.create_dataset(
            "evc",
            data=np.array([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        )

    weights = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.25, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0]]),
        TRANSFORM,
        2,
        file_format="hdf5",
        lattice_alat=np.eye(3),
    )
    assert np.allclose(weights, [[5.0 / 6.0, 0.0]])

    with pytest.raises(ValueError, match="k-map expects"):
        compute_weights_from_qe_save(
            save_dir,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            TRANSFORM,
            2,
            file_format="hdf5",
            lattice_alat=np.eye(3),
        )


GAMMA_HALF = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def test_gamma_only_dat_is_expanded_and_satisfies_the_sum_rule(tmp_path: Path):
    """A gamma-only file stores half the basis; unfolding must expand it."""

    from unmochan.io.qe_wfc import read_wfc_dat

    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    coefficients = np.array([[1.0 + 0.0j, 2.0 - 1.0j, 0.5j]])
    write_synthetic_wfc_dat(
        save_dir / "wfc1.dat", (0.0, 0.0, 0.0), GAMMA_HALF, coefficients, gamma_only=True
    )

    g_vectors, coeffs = read_wfc_dat(save_dir / "wfc1.dat", 1)

    assert g_vectors.shape == (5, 3)
    assert np.array_equal(g_vectors[3:], -GAMMA_HALF[1:])
    assert np.allclose(coeffs[0, 0, 3:], np.conjugate(coefficients[0, 1:]))

    # Both members of the fiber over Gamma, weights must add up to one.
    weights = [
        compute_weights_from_qe_save(
            save_dir,
            np.array([[kx, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            TRANSFORM,
            1,
        )[0, 0]
        for kx in (0.0, 0.5)
    ]
    assert sum(weights) == pytest.approx(1.0)
