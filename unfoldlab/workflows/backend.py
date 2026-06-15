"""Backend-neutral workflow dispatchers for VASP and QE."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.unfolding import WeightDiagnostics, diagnose_weights
from unfoldlab.io.qe import read_gnu_blocks, read_kmap, write_qe_path_files, write_weight_table
from unfoldlab.io.qe_wfc import compute_weights_from_qe_save
from unfoldlab.io.vasp import write_vasp_path_files
from unfoldlab.io.vasp_wfc import compute_weights_from_wavecar
from unfoldlab.workflows.qe import QEUnfoldResult, unfold_qe_bands
from unfoldlab.workflows.vasp import VaspUnfoldResult, unfold_vasp_bands

BackendCode = Literal["vasp", "qe"]


def write_backend_path_files(
    code: BackendCode,
    path_json: str | Path,
    *,
    kpoints: str | Path,
    kmap: str | Path,
    ticks: str | Path,
):
    if code == "qe":
        return write_qe_path_files(path_json, qe_kpoints=kpoints, kmap=kmap, ticks=ticks)
    if code == "vasp":
        return write_vasp_path_files(path_json, kpoints=kpoints, kmap=kmap, ticks=ticks)
    raise ValueError("code must be vasp or qe")


def compute_backend_weights(
    code: BackendCode,
    *,
    kmap: str | Path,
    transform: NDArray[np.float64],
    out: str | Path,
    bands: str | Path | None = None,
    nbnd: int | None = None,
    coefficients: str | Path | None = None,
    qe_save_dir: str | Path | None = None,
    wavecar: str | Path | None = None,
    spin: int = 1,
    wfc_format: str = "auto",
    wfc_pattern: str | None = None,
    tol: float = 1e-6,
) -> WeightDiagnostics:
    mapping = read_kmap(kmap)
    if code == "qe":
        if bands is not None:
            _, energies = read_gnu_blocks(bands)
            n_bands = energies.shape[1]
        elif nbnd is not None:
            n_bands = nbnd
        else:
            raise ValueError("QE weights require bands or nbnd")
        if coefficients is not None:
            from unfoldlab.io.qe import weights_from_coefficient_table

            weights = weights_from_coefficient_table(
                coefficients,
                mapping,
                transform,
                n_bands,
                tol=tol,
            )
        elif qe_save_dir is not None:
            weights = compute_weights_from_qe_save(
                qe_save_dir,
                mapping.primitive_kpoints,
                mapping.supercell_folded_kpoints,
                transform,
                n_bands,
                tol=tol,
                spin=spin,
                pattern=wfc_pattern,
                file_format=wfc_format,
            )
        else:
            raise ValueError("QE weights require qe_save_dir or coefficients")
    elif code == "vasp":
        if wavecar is None:
            raise ValueError("VASP weights require wavecar")
        _energies, weights = compute_weights_from_wavecar(
            wavecar,
            mapping.primitive_kpoints,
            mapping.supercell_folded_kpoints,
            transform,
            spin=spin,
            tol=tol,
        )[1:]
    else:
        raise ValueError("code must be vasp or qe")

    write_weight_table(out, weights)
    return diagnose_weights(weights)


def unfold_backend_bands(
    code: BackendCode,
    *,
    kmap: str | Path,
    transform: NDArray[np.float64] | None = None,
    ticks: str | Path | None = None,
    bands: str | Path | None = None,
    weights: str | Path | None = None,
    qe_save_dir: str | Path | None = None,
    coefficients: str | Path | None = None,
    wavecar: str | Path | None = None,
    eigenval: str | Path | None = None,
    fermi: float = 0.0,
    out: str | Path | None = None,
    plot: str | Path | None = None,
    write_weights: str | Path | None = None,
    spin: int = 1,
    tol: float = 1e-6,
    wfc_format: str = "auto",
    wfc_pattern: str | None = None,
) -> QEUnfoldResult | VaspUnfoldResult:
    if code == "qe":
        if bands is None:
            raise ValueError("QE unfold requires bands")
        return unfold_qe_bands(
            bands=bands,
            kmap=kmap,
            ticks=ticks,
            weights=weights,
            qe_save_dir=qe_save_dir,
            coefficients=coefficients,
            transform=transform,
            fermi=fermi,
            out=out,
            plot=plot,
            write_weights=write_weights,
            tol=tol,
            spin=spin,
            wfc_pattern=wfc_pattern,
            wfc_format=wfc_format,
        )
    if code == "vasp":
        return unfold_vasp_bands(
            kmap=kmap,
            ticks=ticks,
            wavecar=wavecar,
            eigenval=eigenval,
            weights=weights,
            transform=transform,
            fermi=fermi,
            out=out,
            plot=plot,
            write_weights=write_weights,
            spin=spin,
            tol=tol,
        )
    raise ValueError("code must be vasp or qe")
