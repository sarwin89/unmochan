"""Timing of the plane-wave fiber-weight kernel against the naive recipes.

Run with ``python examples/benchmark_weights.py``.  Nothing here is a test --
timings are machine dependent -- but the three numbers it prints are the ones
quoted in ``docs/findings.md``, and it asserts that the three routes agree to
machine precision, which *is* the point: the fast route must not change a
single weight.

The three routes are

``per k-point``
    one call to ``weights_from_coefficients`` per primitive k-point, which
    squares the whole coefficient array and rebuilds the integer matching mask
    ``|det T|`` times over;
``masked batch``
    the squared moduli once and a dense ``(n_g, n_kpoints)`` mask matrix, the
    shape of the kernel before the G-vector classes were introduced;
``class batch``
    the current kernel: one class label per G-vector, one segmented sum, and a
    lookup per k-point.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from unfoldlab.core.kpoints import fiber_kpoints
from unfoldlab.core.plane_waves import (
    matching_g_mask,
    shared_weights_from_coefficients,
    weights_from_coefficients,
)
from unfoldlab.core.transformations import TransformationMatrix

TRANSFORM = TransformationMatrix.from_values(np.diag([4, 4, 4]))
N_BANDS = 60
G_LIMIT = 16


def inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12345)
    axis = np.arange(-G_LIMIT, G_LIMIT + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    g_supercell = grid.reshape(-1, 3)
    shape = (N_BANDS, 1, g_supercell.shape[0])
    coefficients = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    fiber = fiber_kpoints(np.zeros(3), TRANSFORM)
    return g_supercell, coefficients, fiber


def per_kpoint(g_supercell: np.ndarray, coefficients: np.ndarray, fiber: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            weights_from_coefficients(
                g_supercell, coefficients, kpoint, np.zeros(3), TRANSFORM.matrix
            )
            for kpoint in fiber
        ]
    )


def masked_batch(
    g_supercell: np.ndarray, coefficients: np.ndarray, fiber: np.ndarray
) -> np.ndarray:
    masks = np.stack(
        [matching_g_mask(g_supercell, kpoint, np.zeros(3), TRANSFORM.matrix) for kpoint in fiber],
        axis=1,
    ).astype(float)
    norms = (coefficients.real**2 + coefficients.imag**2).sum(axis=1)
    return (norms @ masks).T / norms.sum(axis=1)[np.newaxis, :]


def class_batch(g_supercell: np.ndarray, coefficients: np.ndarray, fiber: np.ndarray) -> np.ndarray:
    return shared_weights_from_coefficients(
        g_supercell, coefficients, fiber, np.zeros(3), TRANSFORM.matrix
    )


def timed(
    label: str,
    routine: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    start = time.perf_counter()
    result = routine(*arrays)
    print(f"{label:<14} {time.perf_counter() - start:8.3f} s")
    return result


def main() -> None:
    arrays = inputs()
    g_supercell, coefficients, fiber = arrays
    print(
        f"{g_supercell.shape[0]} G-vectors, {coefficients.shape[0]} bands, "
        f"{fiber.shape[0]} k-points in the fiber"
    )
    naive = timed("per k-point", per_kpoint, arrays)
    masked = timed("masked batch", masked_batch, arrays)
    classes = timed("class batch", class_batch, arrays)
    assert np.allclose(naive, masked, atol=1e-15)
    assert np.allclose(naive, classes, atol=1e-15)
    assert np.allclose(classes.sum(axis=0), 1.0)
    print(f"largest disagreement: {np.abs(naive - classes).max():.2e}")


if __name__ == "__main__":
    main()
