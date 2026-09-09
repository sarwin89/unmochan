"""End-to-end: a fiber kmap makes the unfolding sum rule an exact equality.

A supercell calculation on a mesh knows only its own k-points.  ``unmochan
fiber`` turns one of them into the ``|det T|`` primitive k-points it carries and
writes them as a kmap; feeding that kmap to the ordinary unfolding must then
reproduce a weight sum of one per band, and the sum-rule diagnostic must see a
complete fiber rather than a partial one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic_wavecar import write_synthetic_wavecar

from unmochan.cli.main import main
from unmochan.core.unfolding import diagnose_fiber_sum_rule
from unmochan.io.qe import read_kmap
from unmochan.io.vasp_wfc import generate_vasp_g_vectors
from unmochan.workflows.vasp import build_vasp_effective_band_structure

ENCUT = 200.0
LATTICE = np.diag([2.0, 1.0, 1.0])
TRANSFORM = np.diag([2.0, 1.0, 1.0])
N_BANDS = 3


def _write_wavecar(path: Path) -> None:
    rng = np.random.default_rng(4242)
    g_vectors = generate_vasp_g_vectors(LATTICE, np.zeros(3), ENCUT)
    coefficients = (
        rng.normal(size=(1, N_BANDS, 1, len(g_vectors)))
        + 1j * rng.normal(size=(1, N_BANDS, 1, len(g_vectors)))
    ).astype(np.complex64)
    write_synthetic_wavecar(
        path,
        lattice=LATTICE,
        encut=ENCUT,
        rtag=45200,
        kpoints=np.zeros((1, 3)),
        energies=np.array([[-2.0, -1.0, 0.5]]),
        occupations=np.ones((1, N_BANDS)),
        coefficients=coefficients,
    )


def test_fiber_kmap_drives_a_complete_unfolding(tmp_path: Path) -> None:
    kmap_path = tmp_path / "fiber.kmap"
    assert (
        main(
            [
                "fiber",
                "--matrix",
                "2 0 0 0 1 0 0 0 1",
                "--kpoint",
                "0,0,0",
                "--kmap",
                str(kmap_path),
            ]
        )
        == 0
    )

    mapping = read_kmap(kmap_path)
    assert mapping.n_kpoints == 2
    assert np.allclose(np.sort(mapping.primitive_kpoints[:, 0]), [0.0, 0.5])

    wavecar = tmp_path / "WAVECAR"
    _write_wavecar(wavecar)

    # The whole fiber shares one supercell wavefunction, so the reader has to
    # be allowed to serve several k-map rows from the same stored k-point.
    ebs, _mode = build_vasp_effective_band_structure(
        kmap=kmap_path,
        wavecar=wavecar,
        transform=TRANSFORM,
        operations=np.zeros((0, 3, 3), dtype=int),
    )
    assert ebs.weights is not None
    assert ebs.weights.shape == (2, N_BANDS)
    # The two rows are the whole fiber over Gamma, so the weights add to one.
    assert np.allclose(ebs.weights.sum(axis=0), 1.0, atol=1e-6)

    diagnostics = diagnose_fiber_sum_rule(ebs.weights, mapping.primitive_kpoints, TRANSFORM)
    assert diagnostics.n_fibers == 1
    assert diagnostics.n_complete_fibers == 1
    assert diagnostics.max_deviation < 1e-6
    assert not diagnostics.violated()


def test_strict_mode_rejects_a_fiber_kmap(tmp_path: Path) -> None:
    """Without the reuse flag the k-point count check fires, as it should."""

    kmap_path = tmp_path / "fiber.kmap"
    main(["fiber", "--matrix", "2 0 0 0 1 0 0 0 1", "--kpoint", "0,0,0", "--kmap", str(kmap_path)])
    wavecar = tmp_path / "WAVECAR"
    _write_wavecar(wavecar)

    with pytest.raises(ValueError, match="WAVECAR has"):
        build_vasp_effective_band_structure(kmap=kmap_path, wavecar=wavecar, transform=TRANSFORM)


def test_weights_cli_reuse_kpoints_accepts_a_fiber_kmap(tmp_path: Path) -> None:
    kmap_path = tmp_path / "fiber.kmap"
    main(["fiber", "--matrix", "2 0 0 0 1 0 0 0 1", "--kpoint", "0,0,0", "--kmap", str(kmap_path)])
    wavecar = tmp_path / "WAVECAR"
    _write_wavecar(wavecar)
    out = tmp_path / "weights.dat"

    assert (
        main(
            [
                "weights",
                "--code",
                "vasp",
                "--kmap",
                str(kmap_path),
                "--matrix",
                "2 0 0 0 1 0 0 0 1",
                "--wavecar",
                str(wavecar),
                "--reuse-kpoints",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    table = np.loadtxt(out)
    assert table.shape[0] == 2 * N_BANDS
    assert np.isclose(table[:, -1].sum(), N_BANDS, atol=1e-6)
