"""The class decomposition of the supercell G-vectors.

The matching test ``(G + m0) @ adj(T).T ≡ 0 (mod det T)`` splits the G-vectors
into ``|det T|`` classes that do not depend on the primitive k-point: the
k-point only chooses *which* class it collects.  The weight kernels exploit
that to touch the coefficient array once instead of once per k-point, so what
matters is that the classes reproduce the mask exactly -- for a non-diagonal
transform, for repeated k-points, for a class no stored G-vector occupies, and
on the fallback labelling used when ``|det T|`` is too large to pack three
residues into one integer.
"""

from __future__ import annotations

import numpy as np
import pytest

from unmochan.core import plane_waves
from unmochan.core.plane_waves import (
    compute_weights_from_coefficient_table,
    matching_g_mask,
    shared_weights_from_coefficients,
)

FOLDED = np.zeros(3)


def g_grid(limit: int = 3) -> np.ndarray:
    return np.array(
        [
            [i, j, k]
            for i in range(-limit, limit + 1)
            for j in range(-limit, limit + 1)
            for k in range(-limit, limit + 1)
        ],
        dtype=int,
    )


def random_coefficients(n_g: int, n_bands: int = 4, n_components: int = 2) -> np.ndarray:
    rng = np.random.default_rng(4242)
    return rng.normal(size=(n_bands, n_components, n_g)) + 1j * rng.normal(
        size=(n_bands, n_components, n_g)
    )


def masked_reference(
    g_supercell: np.ndarray,
    coeffs: np.ndarray,
    kpoints: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """The weights one k-point at a time, straight from :func:`matching_g_mask`."""

    norms = (coeffs.real**2 + coeffs.imag**2).sum(axis=1)
    total = norms.sum(axis=1)
    out = np.zeros((len(kpoints), coeffs.shape[0]))
    for index, kpoint in enumerate(kpoints):
        mask = matching_g_mask(g_supercell, kpoint, FOLDED, transform)
        out[index] = np.where(total > 0.0, norms[:, mask].sum(axis=1) / total, 0.0)
    return out


def test_a_non_diagonal_transform_still_matches_the_mask() -> None:
    transform = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 2]])
    g_supercell = g_grid()
    coeffs = random_coefficients(len(g_supercell))
    fiber = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
        ]
    )
    got = shared_weights_from_coefficients(g_supercell, coeffs, fiber, FOLDED, transform)
    assert np.allclose(got, masked_reference(g_supercell, coeffs, fiber, transform))
    assert np.allclose(got.sum(axis=0), 1.0)


def test_a_repeated_kpoint_gives_a_repeated_row() -> None:
    transform = np.diag([3, 1, 1])
    g_supercell = g_grid(2)
    coeffs = random_coefficients(len(g_supercell))
    kpoints = np.array([[1.0 / 3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0]])
    got = shared_weights_from_coefficients(g_supercell, coeffs, kpoints, FOLDED, transform)
    assert np.allclose(got[0], got[2])
    assert np.allclose(got, masked_reference(g_supercell, coeffs, kpoints, transform))


def test_a_class_no_stored_g_vector_occupies_gets_weight_zero() -> None:
    transform = np.diag([2, 1, 1])
    # Only even G_x is stored, so the k = 1/2 class is empty.
    g_supercell = np.array([[0, 0, 0], [2, 0, 0], [-2, 0, 0]])
    coeffs = random_coefficients(len(g_supercell))
    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    got = shared_weights_from_coefficients(g_supercell, coeffs, kpoints, FOLDED, transform)
    assert np.allclose(got[0], 1.0)
    assert np.allclose(got[1], 0.0)


def test_an_empty_g_list_gives_zero_weights_rather_than_an_error() -> None:
    transform = np.diag([2, 1, 1])
    g_supercell = np.zeros((0, 3), dtype=int)
    coeffs = np.zeros((3, 1, 0), dtype=complex)
    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    got = shared_weights_from_coefficients(g_supercell, coeffs, kpoints, FOLDED, transform)
    assert got.shape == (2, 3)
    assert np.all(got == 0.0)


def test_the_fallback_labelling_agrees_with_the_packed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transform = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 2]])
    g_supercell = g_grid(2)
    coeffs = random_coefficients(len(g_supercell))
    fiber = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]])
    packed = shared_weights_from_coefficients(g_supercell, coeffs, fiber, FOLDED, transform)
    monkeypatch.setattr(plane_waves, "_MAX_PACKED_MODULUS", 0)
    sorted_labels = shared_weights_from_coefficients(g_supercell, coeffs, fiber, FOLDED, transform)
    assert np.array_equal(packed, sorted_labels)


def test_a_determinant_too_large_to_pack_still_unfolds() -> None:
    """``|det T|**3`` overflows ``int64`` here, so the sorted labelling runs."""

    modulus = plane_waves._MAX_PACKED_MODULUS + 1
    transform = np.diag([1, 1, modulus])
    g_supercell = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [1, 0, 0]])
    coeffs = random_coefficients(len(g_supercell))
    kpoints = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 / modulus]])
    got = shared_weights_from_coefficients(g_supercell, coeffs, kpoints, FOLDED, transform)
    assert np.allclose(got, masked_reference(g_supercell, coeffs, kpoints, transform))


def test_the_coefficient_table_reader_agrees_with_the_mask(tmp_path) -> None:
    transform = np.diag([2, 1, 1])
    rng = np.random.default_rng(7)
    lines = ["# ik ib G1 G2 G3 Re Im"]
    n_kpoints, n_bands = 3, 2
    entries: list[tuple[int, int, np.ndarray, complex]] = []
    for ik in range(2):  # the third k-point deliberately carries no rows
        for ib in range(n_bands):
            for gx in (-2, -1, 0, 1, 2):
                value = complex(rng.normal(), rng.normal())
                g = np.array([gx, 0, 0])
                entries.append((ik, ib, g, value))
                lines.append(f"{ik + 1} {ib + 1} {gx} 0 0 {value.real!r} {value.imag!r}")
    path = tmp_path / "coefficients.txt"
    path.write_text("\n".join(lines) + "\n")

    primitive = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    folded = np.zeros((n_kpoints, 3))
    got = compute_weights_from_coefficient_table(path, primitive, folded, transform, n_bands)

    expected = np.zeros((n_kpoints, n_bands))
    for ik in range(n_kpoints):
        for ib in range(n_bands):
            rows = [item for item in entries if item[0] == ik and item[1] == ib]
            if not rows:
                continue
            g_rows = np.array([item[2] for item in rows])
            values = np.array([item[3] for item in rows])
            mask = matching_g_mask(g_rows, primitive[ik], folded[ik], transform)
            total = float(np.sum(np.abs(values) ** 2))
            expected[ik, ib] = float(np.sum(np.abs(values[mask]) ** 2)) / total
    assert np.allclose(got, expected)
    assert np.all(got[2] == 0.0)
