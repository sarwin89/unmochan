"""The QE ``.save`` XML supplies what the caller used to have to pass by hand.

With a ``data-file-schema.xml`` present, the ``at`` matrix, the k-point list
and the pseudopotential kind are read from the calculation itself, so the
k-point cross-check is on by default and symmetry-reduced unfolding no longer
needs ``lattice_alat``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.synthetic_qe_xml import write_data_file_schema
from tests.test_qe_wfc_kpoints import COEFFS, MILLER, TRANSFORM, write_synthetic_wfc_dat
from unfoldlab.io.qe_wfc import (
    compute_symmetry_weights_from_qe_save,
    compute_weights_from_qe_save,
    resolve_qe_metadata,
)

ALAT = 4.0
CUBIC = np.eye(3) * ALAT


def _save_with_xml(tmp_path: Path, kpoints_cart_alat, **kwargs) -> Path:
    save_dir = tmp_path / "qe.save"
    save_dir.mkdir(exist_ok=True)
    write_data_file_schema(
        save_dir / "data-file-schema.xml",
        lattice_bohr=CUBIC,
        alat_bohr=ALAT,
        kpoints_cart_alat=np.asarray(kpoints_cart_alat, dtype=float),
        **kwargs,
    )
    return save_dir


def test_lattice_alat_is_taken_from_the_xml(tmp_path: Path):
    save_dir = _save_with_xml(tmp_path, [[0.5, 0.0, 0.0]])
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    resolved, metadata = resolve_qe_metadata(save_dir, None)

    assert metadata is not None
    assert np.allclose(resolved, np.eye(3))

    weights = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.25, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0]]),
        TRANSFORM,
        2,
        file_format="dat",
    )
    assert np.allclose(weights, [[5.0 / 6.0, 0.0]])


def test_the_cross_check_is_active_without_an_explicit_lattice(tmp_path: Path):
    save_dir = _save_with_xml(tmp_path, [[0.5, 0.0, 0.0]])
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    with pytest.raises(ValueError, match="k-map expects"):
        compute_weights_from_qe_save(
            save_dir,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            TRANSFORM,
            2,
            file_format="dat",
        )


def test_an_explicit_lattice_still_wins(tmp_path: Path):
    save_dir = _save_with_xml(tmp_path, [[0.5, 0.0, 0.0]])
    resolved, metadata = resolve_qe_metadata(save_dir, np.diag([1.0, 1.0, 2.0]))
    assert metadata is not None
    assert np.allclose(resolved, np.diag([1.0, 1.0, 2.0]))


def test_xml_use_can_be_switched_off(tmp_path: Path):
    save_dir = _save_with_xml(tmp_path, [[0.5, 0.0, 0.0]])
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    resolved, metadata = resolve_qe_metadata(save_dir, None, use_xml_metadata=False)
    assert resolved is None
    assert metadata is None

    # Without the XML there is nothing to check against, so a wrong k-map passes.
    weights = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.25, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0]]),
        TRANSFORM,
        2,
        file_format="dat",
        use_xml_metadata=False,
    )
    assert np.allclose(weights, [[5.0 / 6.0, 0.0]])


def test_a_missing_xml_is_not_an_error(tmp_path: Path):
    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    resolved, metadata = resolve_qe_metadata(save_dir, None)
    assert resolved is None
    assert metadata is None

    weights = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.25, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0]]),
        TRANSFORM,
        2,
        file_format="dat",
    )
    assert np.allclose(weights, [[5.0 / 6.0, 0.0]])


def test_xml_and_wavefunction_kpoints_must_agree(tmp_path: Path):
    save_dir = _save_with_xml(tmp_path, [[0.25, 0.0, 0.0]])
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    with pytest.raises(ValueError, match="same calculation"):
        compute_weights_from_qe_save(
            save_dir,
            np.array([[0.25, 0.0, 0.0]]),
            np.array([[0.5, 0.0, 0.0]]),
            TRANSFORM,
            2,
            file_format="dat",
        )


def test_paw_calculations_warn_about_pseudo_weights(tmp_path: Path):
    save_dir = _save_with_xml(tmp_path, [[0.5, 0.0, 0.0]], paw=True)
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    with pytest.warns(UserWarning, match="PAW"):
        compute_weights_from_qe_save(
            save_dir,
            np.array([[0.25, 0.0, 0.0]]),
            np.array([[0.5, 0.0, 0.0]]),
            TRANSFORM,
            2,
            file_format="dat",
        )


def test_symmetry_reduced_unfolding_no_longer_needs_lattice_alat(tmp_path: Path):
    save_dir = _save_with_xml(tmp_path, [[0.5, 0.0, 0.0]])
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    result = compute_symmetry_weights_from_qe_save(
        save_dir,
        np.array([[0.25, 0.0, 0.0]]),
        TRANSFORM,
        2,
        operations=np.zeros((0, 3, 3), dtype=int),
        file_format="dat",
    )

    assert np.allclose(result.weights, [[5.0 / 6.0, 0.0]])
    assert result.n_stored == 1
    assert np.allclose(result.stored_kpoints, [[0.5, 0.0, 0.0]])


def test_symmetry_reduced_unfolding_still_reports_a_missing_lattice(tmp_path: Path):
    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.5, 0.0, 0.0), MILLER, COEFFS)

    with pytest.raises(ValueError, match="requires lattice_alat"):
        compute_symmetry_weights_from_qe_save(
            save_dir,
            np.array([[0.25, 0.0, 0.0]]),
            TRANSFORM,
            2,
            operations=np.zeros((0, 3, 3), dtype=int),
            file_format="dat",
        )


def test_hdf5_without_xk_is_checked_against_the_xml(tmp_path: Path):
    """HDF5 wavefunctions need not record ``xk``; the XML then supplies it."""

    h5py = pytest.importorskip("h5py")
    save_dir = _save_with_xml(tmp_path, [[0.5, 0.0, 0.0]])
    with h5py.File(save_dir / "wfc1.hdf5", "w") as handle:
        handle.attrs["ik"] = 1
        handle.attrs["ispin"] = 1
        handle.attrs["gamma_only"] = ".FALSE."
        handle.attrs["npol"] = 1
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
    )
    assert np.allclose(weights, [[5.0 / 6.0, 0.0]])

    # And the check is real: a k-map at the wrong k-point is now rejected even
    # though the file itself says nothing about its k-point.
    with pytest.raises(ValueError, match="k-map expects"):
        compute_weights_from_qe_save(
            save_dir,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            TRANSFORM,
            2,
            file_format="hdf5",
        )
