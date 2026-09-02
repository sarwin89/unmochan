"""Expansion of time-reversal reduced (``gamma_only``) plane-wave bases.

A Γ-only plane-wave calculation stores one member of every pair ``{G, -G}`` and
reconstructs the other from time reversal,

``c(-G) = conj(c(G))``.

The two members of a pair belong to *different* primitive k-points of an
unfolding fiber (``UnfoldLab.pwMatches_neg``), so the half basis must be
expanded before the matching mask is applied; using the stored half directly
gives different, wrong weights.  A machine-checked counterexample is
``UnfoldLab.exists_half_basis_weight_ne`` and the expansion itself is modelled
by ``UnfoldLab.gammaExpand`` / ``UnfoldLab.gammaCoeff`` in
``RequestProject/Unfolding/GammaOnly.lean``; the resulting norm is the familiar
``|c_0|^2 + 2 * sum_{G != 0} |c_G|^2``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.numerics import as_int_array

__all__ = ["expand_gamma_half_basis", "check_half_basis"]


def check_half_basis(g_vectors: NDArray[np.integer]) -> NDArray[np.int64]:
    """Validate a Γ-only half basis and return it as an integer array.

    A half basis must list each pair ``{G, -G}`` once.  ``G = 0`` is its own
    partner and may appear (at most once).  Duplicates, or both members of a
    pair, would double count plane waves after the expansion.
    """

    g_int = as_int_array(np.asarray(g_vectors), name="g_vectors")
    if g_int.ndim != 2 or g_int.shape[1] != 3:
        raise ValueError("g_vectors must have shape (n_g, 3)")

    keys = {tuple(int(x) for x in row) for row in g_int}
    if len(keys) != g_int.shape[0]:
        raise ValueError("gamma-only G-vector list contains duplicates")
    for key in keys:
        if key == (0, 0, 0):
            continue
        if tuple(-value for value in key) in keys:
            raise ValueError(
                f"G-vector list contains both {key} and its time-reversal partner; "
                "it is a full basis, not a gamma-only half basis"
            )
    return g_int


def expand_gamma_half_basis(
    g_half: NDArray[np.integer],
    coefficients_half: NDArray[np.complexfloating],
) -> tuple[NDArray[np.int64], NDArray[np.complex128]]:
    """Expand a Γ-only half basis to the full plane-wave basis.

    ``g_half`` has shape ``(n_g, 3)`` and ``coefficients_half`` shape
    ``(n_bands, n_components, n_g)``.  The returned pair appends the partners
    ``-G`` of every nonzero ``G`` together with the conjugated coefficients, so
    the result can be fed to the ordinary unfolding machinery.

    The expansion also restores the correct normalization: the total norm of
    the expanded expansion is ``|c_0|^2 + 2 * sum_{G != 0} |c_G|^2``.
    """

    g_int = check_half_basis(g_half)
    coeffs = np.asarray(coefficients_half, dtype=np.complex128)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
    if coeffs.shape[2] != g_int.shape[0]:
        raise ValueError("coefficient G dimension must match number of G-vectors")

    nonzero = np.any(g_int != 0, axis=1)
    g_full = np.concatenate([g_int, -g_int[nonzero]], axis=0)
    coeffs_full = np.concatenate([coeffs, np.conjugate(coeffs[:, :, nonzero])], axis=2)
    return g_full, coeffs_full
