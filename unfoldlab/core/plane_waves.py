"""Plane-wave unfolding kernels shared by electronic-structure backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def matching_g_mask(
    g_supercell: NDArray[np.integer] | NDArray[np.floating],
    primitive_kpoint: NDArray[np.float64],
    folded_supercell_kpoint: NDArray[np.float64],
    transform: NDArray[np.float64],
    *,
    tol: float = 1e-6,
) -> NDArray[np.bool_]:
    """Return the supercell G-vectors compatible with a primitive k-point.

    The direct-lattice convention is ``A_sc = T @ A_pc``. Fractional reciprocal
    coordinates therefore satisfy ``K_sc = k_pc @ T.T``. A supercell plane-wave
    component contributes to primitive k-point ``k`` when
    ``(K_sc + G_sc) @ inv(T).T - k`` is an integer vector.
    """

    g_arr = np.asarray(g_supercell, dtype=float)
    if g_arr.ndim != 2 or g_arr.shape[1] != 3:
        raise ValueError("g_supercell must have shape (n_g, 3)")
    primitive = np.asarray(primitive_kpoint, dtype=float)
    folded = np.asarray(folded_supercell_kpoint, dtype=float)
    matrix = np.asarray(transform, dtype=float)
    if primitive.shape != (3,) or folded.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("primitive k-point, folded k-point, and transform have invalid shapes")

    primitive_fractional = (folded[np.newaxis, :] + g_arr) @ np.linalg.inv(matrix).T
    delta = primitive_fractional - primitive[np.newaxis, :]
    return np.all(np.abs(delta - np.rint(delta)) < tol, axis=1)


def weights_from_coefficients(
    g_supercell: NDArray[np.integer] | NDArray[np.floating],
    coefficients: NDArray[np.complexfloating],
    primitive_kpoint: NDArray[np.float64],
    folded_supercell_kpoint: NDArray[np.float64],
    transform: NDArray[np.float64],
    *,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute unfolded weights for all bands in one k-point wavefunction.

    ``coefficients`` must have shape ``(n_bands, n_components, n_g)``. Spinor or
    polarization components are summed in the norm, which preserves the usual
    plane-wave unfolding sum rule for noncollinear files.
    """

    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
    if coeffs.shape[2] != len(g_supercell):
        raise ValueError("coefficient G dimension must match number of G-vectors")

    mask = matching_g_mask(
        g_supercell,
        primitive_kpoint,
        folded_supercell_kpoint,
        transform,
        tol=tol,
    )
    norms = np.abs(coeffs) ** 2
    total = norms.sum(axis=(1, 2))
    partial = norms[:, :, mask].sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0.0, partial / total, 0.0).astype(float)


def compute_weights_from_coefficient_table(
    coeff_path: str | Path,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: NDArray[np.float64],
    n_bands: int,
    *,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute unfolding weights from rows ``ik ib G1 G2 G3 Re Im``."""

    primitive = np.asarray(primitive_kpoints, dtype=float)
    folded = np.asarray(folded_supercell_kpoints, dtype=float)
    if primitive.shape != folded.shape or primitive.ndim != 2 or primitive.shape[1] != 3:
        raise ValueError("primitive and folded k-points must both have shape (n_kpoints, 3)")
    if n_bands <= 0:
        raise ValueError("n_bands must be positive")

    partial = np.zeros((primitive.shape[0], n_bands), dtype=float)
    total = np.zeros((primitive.shape[0], n_bands), dtype=float)
    path = Path(coeff_path)

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 7:
            raise ValueError("coefficient rows must be: ik ib G1 G2 G3 Re Im")
        ik = int(parts[0]) - 1
        ib = int(parts[1]) - 1
        if ik < 0 or ik >= primitive.shape[0] or ib < 0 or ib >= n_bands:
            raise ValueError(f"coefficient index outside band grid: {line}")

        g_sc = np.array([[int(parts[2]), int(parts[3]), int(parts[4])]], dtype=float)
        coeff_norm = abs(complex(float(parts[5]), float(parts[6]))) ** 2
        total[ik, ib] += coeff_norm
        if matching_g_mask(g_sc, primitive[ik], folded[ik], transform, tol=tol)[0]:
            partial[ik, ib] += coeff_norm

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0.0, partial / total, 0.0)
