"""High-level VASP unfolding workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.spectral import EffectiveBandStructure
from unfoldlab.core.unfolding import BandUnfoldingData
from unfoldlab.io.qe import (
    plot_unfolded,
    read_kmap,
    read_ticks,
    read_weight_table,
    write_unfolded,
    write_weight_table,
)
from unfoldlab.io.vasp import read_eigenval
from unfoldlab.io.vasp_wfc import compute_weights_from_wavecar


@dataclass(frozen=True)
class VaspUnfoldResult:
    ebs: EffectiveBandStructure
    mode: str
    output_path: Path | None = None
    plot_path: Path | None = None
    weights_path: Path | None = None


def compute_vasp_weights(
    *,
    wavecar: str | Path,
    kmap: str | Path,
    transform: NDArray[np.float64],
    spin: int = 1,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mapping = read_kmap(kmap)
    _kpoints, energies, weights = compute_weights_from_wavecar(
        wavecar,
        mapping.primitive_kpoints,
        mapping.supercell_folded_kpoints,
        transform,
        spin=spin,
        tol=tol,
    )
    return energies, weights


def build_vasp_effective_band_structure(
    *,
    kmap: str | Path,
    wavecar: str | Path | None = None,
    eigenval: str | Path | None = None,
    weights: str | Path | None = None,
    transform: NDArray[np.float64] | None = None,
    reference_energy: float = 0.0,
    spin: int = 1,
    tol: float = 1e-6,
) -> tuple[EffectiveBandStructure, str]:
    mapping = read_kmap(kmap)
    metadata = {"kmap": str(kmap), "spin_channel": spin}
    if wavecar is not None:
        if transform is None:
            raise ValueError("transform is required when wavecar is used")
        _wave_kpoints, energies, spectral_weights = compute_weights_from_wavecar(
            wavecar,
            mapping.primitive_kpoints,
            mapping.supercell_folded_kpoints,
            transform,
            spin=spin,
            tol=tol,
        )
        mode = "computed spectral weights from VASP WAVECAR"
        metadata.update({"wavecar": str(wavecar), "weight_mode": mode})
    elif eigenval is not None:
        parsed = read_eigenval(eigenval)
        if parsed.energies.ndim != 2:
            raise ValueError("select one spin channel before using spin-resolved EIGENVAL data")
        energies = parsed.energies
        if weights is None:
            spectral_weights = np.ones_like(energies)
            mode = "unit weights only; VASP EIGENVAL folded-band data"
        else:
            spectral_weights = read_weight_table(weights, *energies.shape)
            mode = "read spectral weights"
        metadata.update({"eigenval": str(eigenval), "weight_mode": mode})
    else:
        raise ValueError("provide either wavecar or eigenval")

    if energies.shape[0] != mapping.n_kpoints:
        raise ValueError(
            f"kmap has {mapping.n_kpoints} k-points but VASP energies have {energies.shape[0]}"
        )
    return (
        BandUnfoldingData(
            kpoints=mapping.primitive_kpoints,
            energies=energies,
            weights=spectral_weights,
            distances=mapping.distances,
            reference_energy=reference_energy,
            source_code="vasp",
            metadata=metadata,
        ).to_effective_band_structure(),
        mode,
    )


def unfold_vasp_bands(
    *,
    kmap: str | Path,
    ticks: str | Path | None = None,
    wavecar: str | Path | None = None,
    eigenval: str | Path | None = None,
    weights: str | Path | None = None,
    transform: NDArray[np.float64] | None = None,
    fermi: float = 0.0,
    emin: float | None = None,
    emax: float | None = None,
    marker_scale: float = 28.0,
    out: str | Path | None = None,
    plot: str | Path | None = None,
    write_weights: str | Path | None = None,
    spin: int = 1,
    tol: float = 1e-6,
) -> VaspUnfoldResult:
    ebs, mode = build_vasp_effective_band_structure(
        kmap=kmap,
        wavecar=wavecar,
        eigenval=eigenval,
        weights=weights,
        transform=transform,
        reference_energy=fermi,
        spin=spin,
        tol=tol,
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
    return VaspUnfoldResult(
        ebs=ebs,
        mode=mode,
        output_path=output_path,
        plot_path=plot_path,
        weights_path=weights_path,
    )
