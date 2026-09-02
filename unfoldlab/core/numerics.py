"""Shared numerical helpers.

The integer helpers in this module exist so that lattice-commensurability
questions are answered by exact integer arithmetic instead of floating point
comparisons.  See ``docs/formal-model.md`` for the formal statements that
justify the exact tests.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Largest tolerance that still makes the ``rint`` integrality test meaningful.
#: For ``atol >= 0.5`` every real vector passes ``|x - rint(x)| <= atol``, so the
#: test carries no information.
MAX_MEANINGFUL_ATOL = 0.5

#: The primitive-to-supercell transform as it travels through the API.  It is an
#: *integer* matrix mathematically, and every consumer runs it through
#: :func:`as_int_array`, but a caller that read it from a file or computed it
#: from two lattices legitimately holds it as floats with integral entries, so
#: both are accepted.
TransformLike: TypeAlias = "NDArray[np.int64] | NDArray[np.float64]"


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


def check_atol(atol: float, *, name: str = "atol") -> float:
    """Validate a tolerance used with an ``|x - rint(x)| <= atol`` test.

    Tolerances of one half or more make the test vacuous, which would silently
    accept any input, so they are rejected.
    """

    value = float(atol)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {atol!r}")
    if value >= MAX_MEANINGFUL_ATOL:
        raise ValueError(
            f"{name}={value:g} is meaningless: every vector satisfies "
            "|x - rint(x)| <= 0.5, so the tolerance must be smaller than 0.5"
        )
    return value


def as_int_array(values: ArrayLike, *, name: str, atol: float = 1e-6) -> NDArray[np.int64]:
    """Round an array to integers, rejecting non-integral input.

    Unlike ``np.rint(...).astype(int)`` this refuses to silently discard a
    non-integral part.
    """

    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64)
    arr = np.asarray(arr, dtype=float)
    rounded = np.rint(arr)
    if arr.size and float(np.max(np.abs(arr - rounded))) > check_atol(atol):
        raise ValueError(
            f"{name} must contain integers within atol={atol:g}; "
            f"max residual={float(np.max(np.abs(arr - rounded))):g}"
        )
    return rounded.astype(np.int64)


def integer_det3(matrix: NDArray[np.int64]) -> int:
    """Exact determinant of a 3x3 integer matrix.

    ``np.linalg.det`` works in floating point and can return values such as
    ``1.9999999999999998`` that round incorrectly for ill-conditioned integer
    matrices; the cofactor expansion below is exact.
    """

    m = np.asarray(matrix, dtype=np.int64)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3), got {m.shape}")
    a, b, c = (int(m[0, 0]), int(m[0, 1]), int(m[0, 2]))
    d, e, f = (int(m[1, 0]), int(m[1, 1]), int(m[1, 2]))
    g, h, i = (int(m[2, 0]), int(m[2, 1]), int(m[2, 2]))
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def integer_adjugate3(matrix: NDArray[np.int64]) -> NDArray[np.int64]:
    """Exact adjugate of a 3x3 integer matrix.

    Satisfies ``adjugate(T) @ T == T @ adjugate(T) == det(T) * I`` exactly, so
    ``T @ v == w`` is solvable over the integers exactly when ``det(T)`` divides
    every component of ``adjugate(T) @ w``.
    """

    m = np.asarray(matrix, dtype=np.int64)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3), got {m.shape}")
    adj = np.empty((3, 3), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            rows = [r for r in range(3) if r != j]
            cols = [c for c in range(3) if c != i]
            minor = m[np.ix_(rows, cols)]
            cofactor = int(minor[0, 0]) * int(minor[1, 1]) - int(minor[0, 1]) * int(minor[1, 0])
            adj[i, j] = cofactor if (i + j) % 2 == 0 else -cofactor
    return adj


def wrap_fractional(values: ArrayLike, *, atol: float = 1e-12) -> NDArray[np.float64]:
    """Wrap fractional coordinates into the half-open interval [0, 1)."""

    wrapped = np.mod(np.asarray(values, dtype=float), 1.0)
    wrapped = np.atleast_1d(wrapped)
    wrapped[np.isclose(wrapped, 1.0, atol=atol)] = 0.0
    wrapped[np.isclose(wrapped, 0.0, atol=atol)] = 0.0
    return wrapped


def allclose_mod1(a: ArrayLike, b: ArrayLike, *, atol: float = 1e-8) -> bool:
    """Compare fractional coordinates modulo reciprocal lattice vectors."""

    check_atol(atol)
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta -= np.rint(delta)
    return bool(np.all(np.abs(delta) <= atol))
