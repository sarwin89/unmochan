"""Backend-neutral unfolding data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.plane_waves import weights_from_coefficients
from unfoldlab.core.spectral import EffectiveBandStructure


@dataclass(frozen=True)
class PlaneWaveKPointData:
    """Plane-wave coefficients for one supercell k-point.

    This container is intentionally code-agnostic. QE readers, future VASP
    WAVECAR readers, and external coefficient exporters should all normalize
    into this shape before weights are calculated.
    """

    primitive_kpoint: NDArray[np.float64]
    folded_supercell_kpoint: NDArray[np.float64]
    g_supercell: NDArray[np.int64]
    coefficients: NDArray[np.complex128]

    def __post_init__(self) -> None:
        primitive = np.asarray(self.primitive_kpoint, dtype=float)
        folded = np.asarray(self.folded_supercell_kpoint, dtype=float)
        g_vectors = np.asarray(self.g_supercell, dtype=int)
        coefficients = np.asarray(self.coefficients, dtype=np.complex128)
        if primitive.shape != (3,):
            raise ValueError("primitive_kpoint must have shape (3,)")
        if folded.shape != (3,):
            raise ValueError("folded_supercell_kpoint must have shape (3,)")
        if g_vectors.ndim != 2 or g_vectors.shape[1] != 3:
            raise ValueError("g_supercell must have shape (n_g, 3)")
        if coefficients.ndim != 3 or coefficients.shape[2] != g_vectors.shape[0]:
            raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
        object.__setattr__(self, "primitive_kpoint", primitive)
        object.__setattr__(self, "folded_supercell_kpoint", folded)
        object.__setattr__(self, "g_supercell", g_vectors)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def n_bands(self) -> int:
        return int(self.coefficients.shape[0])


@dataclass(frozen=True)
class BandUnfoldingData:
    """Backend-neutral band energies plus optional spectral weights."""

    kpoints: NDArray[np.float64]
    energies: NDArray[np.float64]
    weights: NDArray[np.float64] | None = None
    distances: NDArray[np.float64] | None = None
    reference_energy: float = 0.0
    source_code: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_effective_band_structure(self) -> EffectiveBandStructure:
        return EffectiveBandStructure(
            kpoints=self.kpoints,
            energies=self.energies,
            weights=self.weights,
            distances=self.distances,
            reference_energy=self.reference_energy,
            metadata={"source_code": self.source_code, **self.metadata},
        )


@dataclass(frozen=True)
class WeightDiagnostics:
    min_weight: float
    max_weight: float
    out_of_bounds_count: int
    max_band_sum: float
    n_kpoints: int
    n_bands: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "out_of_bounds_count": self.out_of_bounds_count,
            "max_band_sum": self.max_band_sum,
            "n_kpoints": self.n_kpoints,
            "n_bands": self.n_bands,
        }


def compute_plane_wave_unfolding_weights(
    kpoint_data: list[PlaneWaveKPointData] | tuple[PlaneWaveKPointData, ...],
    transform: NDArray[np.float64],
    *,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute backend-independent plane-wave unfolding weights.

    Identical systems represented with equivalent plane-wave coefficients,
    reciprocal G-vector indexing, transform matrix, and primitive k-path should
    produce identical weights regardless of whether the source reader is QE,
    VASP, or an external exporter.
    """

    if not kpoint_data:
        raise ValueError("at least one k-point wavefunction is required")
    n_bands = kpoint_data[0].n_bands
    weights = np.zeros((len(kpoint_data), n_bands), dtype=float)
    for index, item in enumerate(kpoint_data):
        if item.n_bands != n_bands:
            raise ValueError("all k-points must have the same number of bands")
        weights[index] = weights_from_coefficients(
            item.g_supercell,
            item.coefficients,
            item.primitive_kpoint,
            item.folded_supercell_kpoint,
            transform,
            tol=tol,
        )
    return weights


def diagnose_weights(
    weights: NDArray[np.float64],
    *,
    tol: float = 1e-8,
) -> WeightDiagnostics:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 2:
        raise ValueError("weights must have shape (n_kpoints, n_bands)")
    out_of_bounds = np.logical_or(arr < -tol, arr > 1.0 + tol)
    return WeightDiagnostics(
        min_weight=float(np.min(arr)) if arr.size else 0.0,
        max_weight=float(np.max(arr)) if arr.size else 0.0,
        out_of_bounds_count=int(np.count_nonzero(out_of_bounds)),
        max_band_sum=float(np.max(np.sum(arr, axis=1))) if arr.size else 0.0,
        n_kpoints=int(arr.shape[0]),
        n_bands=int(arr.shape[1]),
    )


def compare_effective_band_structures(
    first: EffectiveBandStructure,
    second: EffectiveBandStructure,
    *,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Compare two backend outputs at the common EBS level."""

    shape_match = (
        first.kpoints.shape == second.kpoints.shape
        and first.energies.shape == second.energies.shape
        and first.weights.shape == second.weights.shape
    )
    return {
        "shape_match": shape_match,
        "kpoints_close": shape_match
        and bool(np.allclose(first.kpoints, second.kpoints, atol=atol)),
        "energies_close": shape_match
        and bool(np.allclose(first.energies, second.energies, atol=atol)),
        "weights_close": shape_match
        and bool(np.allclose(first.weights, second.weights, atol=atol)),
        "max_weight_delta": float(np.max(np.abs(first.weights - second.weights)))
        if first.weights.shape == second.weights.shape
        else None,
    }
