"""High-level Quantum ESPRESSO unfolding workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.spectral import EffectiveBandStructure
from unfoldlab.io.qe import (
    plot_unfolded,
    qe_effective_band_structure,
    read_gnu_blocks,
    read_kmap,
    read_ticks,
    read_weight_table,
    weights_from_coefficient_table,
    write_unfolded,
    write_weight_table,
)
from unfoldlab.io.qe_wfc import compute_weights_from_qe_save


@dataclass(frozen=True)
class QEUnfoldResult:
    ebs: EffectiveBandStructure
    mode: str
    output_path: Path | None = None
    plot_path: Path | None = None
    weights_path: Path | None = None


def compute_qe_weights(
    *,
    save_dir: str | Path,
    kmap: str | Path,
    transform: NDArray[np.float64],
    n_bands: int,
    tol: float = 1e-6,
    spin: int | None = None,
    wfc_pattern: str | None = None,
    wfc_format: str = "auto",
) -> NDArray[np.float64]:
    mapping = read_kmap(kmap)
    return compute_weights_from_qe_save(
        save_dir,
        mapping.primitive_kpoints,
        mapping.supercell_folded_kpoints,
        transform,
        n_bands,
        tol=tol,
        spin=spin,
        pattern=wfc_pattern,
        file_format=wfc_format,
    )


def build_qe_effective_band_structure(
    *,
    bands: str | Path,
    kmap: str | Path,
    weights: str | Path | None = None,
    qe_save_dir: str | Path | None = None,
    coefficients: str | Path | None = None,
    transform: NDArray[np.float64] | None = None,
    reference_energy: float = 0.0,
    tol: float = 1e-6,
    spin: int | None = None,
    wfc_pattern: str | None = None,
    wfc_format: str = "auto",
) -> tuple[EffectiveBandStructure, str]:
    _, energies = read_gnu_blocks(bands)
    mapping = read_kmap(kmap)
    if mapping.n_kpoints != energies.shape[0]:
        raise ValueError(
            f"kmap has {mapping.n_kpoints} k-points but bands file has {energies.shape[0]}"
        )

    spectral_weights: NDArray[np.float64]
    mode: str
    if qe_save_dir is not None:
        if transform is None:
            raise ValueError("transform is required when qe_save_dir is used")
        spectral_weights = compute_weights_from_qe_save(
            qe_save_dir,
            mapping.primitive_kpoints,
            mapping.supercell_folded_kpoints,
            transform,
            energies.shape[1],
            tol=tol,
            spin=spin,
            pattern=wfc_pattern,
            file_format=wfc_format,
        )
        mode = f"computed spectral weights from QE {wfc_format} wavefunctions"
    elif coefficients is not None:
        if transform is None:
            raise ValueError("transform is required when coefficients are used")
        spectral_weights = weights_from_coefficient_table(
            coefficients,
            mapping,
            transform,
            energies.shape[1],
            tol=tol,
        )
        mode = "computed spectral weights from plane-wave coefficient table"
    elif weights is not None:
        spectral_weights = read_weight_table(weights, *energies.shape)
        mode = "read spectral weights"
    else:
        spectral_weights = np.ones_like(energies)
        mode = "unit weights only; folded supercell data on a primitive path"

    return (
        qe_effective_band_structure(
            mapping,
            energies,
            spectral_weights,
            reference_energy=reference_energy,
            metadata={"bands": str(bands), "kmap": str(kmap), "weight_mode": mode},
        ),
        mode,
    )


def unfold_qe_bands(
    *,
    bands: str | Path,
    kmap: str | Path,
    ticks: str | Path | None = None,
    weights: str | Path | None = None,
    qe_save_dir: str | Path | None = None,
    coefficients: str | Path | None = None,
    transform: NDArray[np.float64] | None = None,
    fermi: float = 0.0,
    emin: float | None = None,
    emax: float | None = None,
    marker_scale: float = 28.0,
    out: str | Path | None = None,
    plot: str | Path | None = None,
    write_weights: str | Path | None = None,
    tol: float = 1e-6,
    spin: int | None = None,
    wfc_pattern: str | None = None,
    wfc_format: str = "auto",
) -> QEUnfoldResult:
    ebs, mode = build_qe_effective_band_structure(
        bands=bands,
        kmap=kmap,
        weights=weights,
        qe_save_dir=qe_save_dir,
        coefficients=coefficients,
        transform=transform,
        reference_energy=fermi,
        tol=tol,
        spin=spin,
        wfc_pattern=wfc_pattern,
        wfc_format=wfc_format,
    )
    mapping = read_kmap(kmap)

    output_path = Path(out) if out is not None else None
    if output_path is not None:
        write_unfolded(output_path, mapping, ebs.energies, ebs.weights)

    weights_path = Path(write_weights) if write_weights is not None else None
    if weights_path is not None:
        write_weight_table(weights_path, ebs.weights)

    plot_path = None
    if plot is not None:
        plot_path = plot_unfolded(
            plot,
            mapping,
            ebs.energies,
            ebs.weights,
            read_ticks(ticks, mapping),
            fermi=fermi,
            emin=emin,
            emax=emax,
            marker_scale=marker_scale,
        )

    return QEUnfoldResult(
        ebs=ebs,
        mode=mode,
        output_path=output_path,
        plot_path=plot_path,
        weights_path=weights_path,
    )
