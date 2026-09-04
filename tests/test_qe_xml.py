"""Tests for the QE ``data-file-schema.xml`` reader."""

from __future__ import annotations

import numpy as np
import pytest

from tests.synthetic_qe_xml import write_data_file_schema
from unmochan.io.qe_xml import (
    BOHR_ANGSTROM,
    HARTREE_EV,
    QEXMLError,
    check_xml_kpoint_order,
    find_qe_xml,
    read_qe_lattice_alat,
    read_qe_xml,
    try_read_qe_metadata,
)

ALAT = 10.26
LATTICE = (
    np.array(
        [
            [-0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [-0.5, 0.5, 0.0],
        ]
    )
    * ALAT
)
KPOINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.25, -0.25, 0.25],
        [0.5, -0.5, 0.5],
    ]
)


def _write(tmp_path, **kwargs):
    save = tmp_path / "pwscf.save"
    return write_data_file_schema(
        save / "data-file-schema.xml",
        lattice_bohr=LATTICE,
        alat_bohr=ALAT,
        kpoints_cart_alat=KPOINTS,
        eigenvalues_hartree=np.arange(3 * 4, dtype=float).reshape(3, 4) * 0.01,
        **kwargs,
    )


def test_reads_cell_alat_and_at_matrix(tmp_path):
    _write(tmp_path)
    meta = read_qe_xml(tmp_path / "pwscf.save")
    assert meta.alat_bohr == pytest.approx(ALAT)
    assert np.allclose(meta.lattice_bohr, LATTICE)
    assert np.allclose(meta.lattice_alat, LATTICE / ALAT)
    assert np.allclose(meta.lattice_angstrom, LATTICE * BOHR_ANGSTROM)


def test_at_matrix_is_unit_independent(tmp_path):
    """``at`` is dimensionless: writing the cell in a different unit is a no-op.

    Formal statement: ``UnfoldLab.latticeAlat_unit_invariant``.
    """

    _write(tmp_path)
    bohr = read_qe_lattice_alat(tmp_path / "pwscf.save")
    other = tmp_path / "other.save"
    write_data_file_schema(
        other / "data-file-schema.xml",
        lattice_bohr=LATTICE * BOHR_ANGSTROM,
        alat_bohr=ALAT * BOHR_ANGSTROM,
        kpoints_cart_alat=KPOINTS,
    )
    assert np.allclose(bohr, read_qe_lattice_alat(other))


def test_fractional_kpoints_match_the_manual_conversion(tmp_path):
    _write(tmp_path)
    meta = read_qe_xml(tmp_path / "pwscf.save")
    expected = KPOINTS @ (LATTICE / ALAT).T
    assert np.allclose(meta.kpoints_frac, expected)
    # xk = f @ bg with bg dual to at, so the round trip returns f.
    reciprocal = np.linalg.inv(LATTICE / ALAT).T
    assert np.allclose(meta.kpoints_frac @ reciprocal, KPOINTS)
    assert meta.reciprocal_alat is not None
    assert np.allclose(meta.reciprocal_alat, reciprocal)


def test_eigenvalues_and_fermi_are_converted_to_ev(tmp_path):
    _write(tmp_path)
    meta = read_qe_xml(tmp_path / "pwscf.save")
    expected = np.arange(12, dtype=float).reshape(3, 4) * 0.01 * HARTREE_EV
    assert np.allclose(meta.eigenvalues_ev, expected)
    assert meta.fermi_energy_ev == pytest.approx(0.2 * HARTREE_EV)
    assert meta.nbnd == 4
    assert meta.n_kpoints == 3
    assert meta.eigenvalues_by_spin() == (pytest.approx(meta.eigenvalues_ev),)


def test_pseudo_warning_only_for_uspp_or_paw(tmp_path):
    _write(tmp_path)
    assert read_qe_xml(tmp_path / "pwscf.save").pseudo_warning() is None
    paw_dir = tmp_path / "paw"
    write_data_file_schema(
        paw_dir / "pwscf.save" / "data-file-schema.xml",
        lattice_bohr=LATTICE,
        alat_bohr=ALAT,
        kpoints_cart_alat=KPOINTS,
        paw=True,
    )
    message = read_qe_xml(paw_dir).pseudo_warning()
    assert message is not None
    assert "PAW" in message


def test_find_xml_accepts_file_save_dir_and_outdir(tmp_path):
    path = _write(tmp_path)
    assert find_qe_xml(path) == path
    assert find_qe_xml(tmp_path / "pwscf.save") == path
    assert find_qe_xml(tmp_path) == path
    with pytest.raises(FileNotFoundError):
        find_qe_xml(tmp_path / "missing")


def test_species_and_flags(tmp_path):
    _write(tmp_path, uspp=True, gamma_only=True)
    meta = read_qe_xml(tmp_path / "pwscf.save")
    assert meta.species == (("Si", "Si.pz-vbc.UPF"),)
    assert meta.uspp is True
    assert meta.paw is False
    assert meta.gamma_only is True
    assert meta.n_atoms == 2


def test_kpoint_order_check(tmp_path):
    _write(tmp_path)
    meta = read_qe_xml(tmp_path / "pwscf.save")
    check_xml_kpoint_order(meta, KPOINTS)
    check_xml_kpoint_order(meta, KPOINTS[:2])
    # A NaN row means "the reader could not recover xk" and is skipped.
    partial = KPOINTS.copy()
    partial[1] = np.nan
    check_xml_kpoint_order(meta, partial)
    with pytest.raises(ValueError, match="does not match"):
        check_xml_kpoint_order(meta, KPOINTS[::-1])
    with pytest.raises(ValueError, match="wavefunction files"):
        check_xml_kpoint_order(meta, np.vstack([KPOINTS, KPOINTS]))


def test_malformed_documents_are_rejected(tmp_path):
    broken = tmp_path / "broken.xml"
    broken.write_text("<qes:espresso xmlns:qes='x'><output/></qes:espresso>", encoding="utf-8")
    with pytest.raises(QEXMLError):
        read_qe_xml(broken)
    assert try_read_qe_metadata(broken) is None
    assert try_read_qe_metadata(tmp_path / "nothing-here") is None

    singular = tmp_path / "singular.save"
    write_data_file_schema(
        singular / "data-file-schema.xml",
        lattice_bohr=np.zeros((3, 3)),
        alat_bohr=ALAT,
        kpoints_cart_alat=KPOINTS,
    )
    with pytest.raises(QEXMLError, match="singular"):
        read_qe_xml(singular)
