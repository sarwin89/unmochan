"""Shared numerical helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_array3(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Return a 3-vector as a float array."""

    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    return arr


def as_matrix3(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Return a 3x3 matrix as a float array."""

    arr = np.asarray(values, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}")
    return arr


def wrap_fractional(values: ArrayLike, *, atol: float = 1e-12) -> NDArray[np.float64]:
    """Wrap fractional coordinates into the half-open interval [0, 1)."""

    wrapped = np.mod(np.asarray(values, dtype=float), 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=atol)] = 0.0
    wrapped[np.isclose(wrapped, 0.0, atol=atol)] = 0.0
    return wrapped


def allclose_mod1(a: ArrayLike, b: ArrayLike, *, atol: float = 1e-8) -> bool:
    """Compare fractional coordinates modulo reciprocal lattice vectors."""

    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta -= np.rint(delta)
    return bool(np.all(np.abs(delta) <= atol))
