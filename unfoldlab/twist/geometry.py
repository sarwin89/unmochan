"""Small geometry utilities for the twist workflow."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import as_matrix3
from unfoldlab.core.structures import Structure


def in_plane_twist_angle(
    reference_lattice: ArrayLike,
    rotated_lattice: ArrayLike,
    *,
    vector_index: int = 0,
) -> float:
    """Estimate the in-plane twist angle in degrees from two lattice vectors."""

    ref = as_matrix3(reference_lattice, name="reference_lattice")[vector_index, :2]
    rot = as_matrix3(rotated_lattice, name="rotated_lattice")[vector_index, :2]
    if np.linalg.norm(ref) == 0 or np.linalg.norm(rot) == 0:
        raise ValueError("selected in-plane lattice vector has zero length")
    ref_angle = np.arctan2(ref[1], ref[0])
    rot_angle = np.arctan2(rot[1], rot[0])
    angle = np.degrees(rot_angle - ref_angle)
    return float((angle + 180.0) % 360.0 - 180.0)


def assign_layers_by_axis(
    structure: Structure,
    *,
    axis: int = 2,
    n_layers: int = 2,
    threshold: float | None = None,
) -> NDArray[np.int64]:
    """Assign atoms to layer-like groups by Cartesian coordinate along one axis."""

    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    coords = structure.cart_coords[:, axis]
    if n_layers == 1:
        return np.zeros(structure.n_sites, dtype=int)
    if threshold is not None:
        if n_layers != 2:
            raise ValueError("threshold is only valid for two-layer assignment")
        return (coords > float(threshold)).astype(int)
    quantiles = np.linspace(0.0, 1.0, n_layers + 1)[1:-1]
    thresholds = np.quantile(coords, quantiles)
    return np.searchsorted(thresholds, coords, side="right").astype(int)
