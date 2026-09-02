"""Layer-resolved unfolding of a plane-wave slab.

An unfolded slab band structure answers "how much of this state belongs to the
primitive k-point k".  The next question a surface calculation asks is "and how
much of *that* sits on the surface rather than in the interior".  The two
questions are not equally well posed, and this module is built around the
difference, which is proved in ``RequestProject/Unfolding/Layers.lean``.

``sum_regionCharge_layers``
    **At a fixed k-point the layer split is exact.**  The layer charges of the
    matched component of a state add up to its matched norm, for any partition
    of space into layers.  So :func:`layer_resolved_weights` really does split
    the unfolding weight of each k-point, with no approximation.

``regionCharge_eq_blockSum_add_crossForm``
    **Across k-points it is not.**  The charge a state carries in a region is
    the sum of the per-k-point charges *plus* a cross term between the k-point
    blocks.  A layer projector is diagonal in real space and therefore a full
    matrix in the plane-wave basis, so that term is generally nonzero:
    layer-resolved weights may not be summed over k and compared with a
    real-space charge.  :func:`diagnose_layer_conservation` measures it.

``sum_regionCharge_blocks_of_diagonal``
    the cross term vanishes exactly when the projector is diagonal in the same
    basis as the matching -- a site or orbital projection in a tight-binding or
    LCAO basis.  That is why
    :func:`unfoldlab.core.tight_binding.tight_binding_orbital_weights` carries a
    fiber sum rule and the plane-wave layer charges do not.

The planar-averaged density is computed from its Fourier coefficients, so the
charge of a region is a closed-form integral rather than a grid sum: no
real-space mesh, and no discretization error at the layer boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import TransformLike, as_int_array, check_atol
from unfoldlab.core.plane_waves import matching_g_mask
from unfoldlab.core.site_projection import detect_layers
from unfoldlab.core.structures import Structure

__all__ = [
    "LayerConservation",
    "density_fourier_coefficients",
    "diagnose_layer_conservation",
    "layer_charges",
    "layer_edges_from_structure",
    "layer_resolved_weights",
    "planar_average_density",
]


def _prepare(
    g_supercell: ArrayLike,
    coefficients: ArrayLike,
    axis: int,
    atol: float,
) -> tuple[NDArray[np.int64], NDArray[np.complex128]]:
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    g_int = as_int_array(g_supercell, name="g_supercell", atol=check_atol(atol))
    if g_int.ndim != 2 or g_int.shape[1] != 3:
        raise ValueError("g_supercell must have shape (n_g, 3)")
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
    if coeffs.shape[2] != g_int.shape[0]:
        raise ValueError("coefficient G dimension must match number of G-vectors")
    return g_int, coeffs


def density_fourier_coefficients(
    g_supercell: ArrayLike,
    coefficients: ArrayLike,
    *,
    axis: int = 2,
    atol: float = 1e-6,
) -> tuple[NDArray[np.int64], NDArray[np.complex128]]:
    """Fourier coefficients of the planar-averaged density along ``axis``.

    The planar average of ``|psi(r)|^2`` over the two other directions is

    ``rho(z) = sum_m rho_m exp(2 pi i m z)``

    with ``z`` the fractional coordinate along ``axis``, and

    ``rho_m = sum_{G_par} sum_{G_z} conj(c[G_par, G_z]) c[G_par, G_z + m]``.

    Returns ``(orders, rho)`` with ``orders`` the integers ``m`` in increasing
    order and ``rho`` of shape ``(n_bands, len(orders))``.  ``rho_0`` is the
    squared norm of the state, so a normalized state integrates to one.
    """

    g_int, coeffs = _prepare(g_supercell, coefficients, axis, atol)
    parallel_axes = [i for i in (0, 1, 2) if i != axis]
    parallel = g_int[:, parallel_axes]
    perpendicular = g_int[:, axis]

    unique_par, par_index = np.unique(parallel, axis=0, return_inverse=True)
    par_index = np.asarray(par_index).ravel()
    lowest = int(perpendicular.min())
    highest = int(perpendicular.max())
    n_shells = highest - lowest + 1

    amplitudes = np.zeros(
        (coeffs.shape[0], coeffs.shape[1], unique_par.shape[0], n_shells),
        dtype=np.complex128,
    )
    amplitudes[:, :, par_index, perpendicular - lowest] = coeffs

    orders = np.arange(-(n_shells - 1), n_shells, dtype=np.int64)
    rho = np.zeros((coeffs.shape[0], orders.size), dtype=np.complex128)
    for position, order in enumerate(orders):
        if order >= 0:
            left = amplitudes[:, :, :, : n_shells - order]
            right = amplitudes[:, :, :, order:]
        else:
            left = amplitudes[:, :, :, -order:]
            right = amplitudes[:, :, :, : n_shells + order]
        rho[:, position] = (np.conjugate(left) * right).sum(axis=(1, 2, 3))
    return orders, rho


def planar_average_density(
    g_supercell: ArrayLike,
    coefficients: ArrayLike,
    *,
    axis: int = 2,
    n_points: int = 200,
    atol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the planar-averaged density on a grid of fractional coordinates.

    Returns ``(z, rho)`` with ``z`` of length ``n_points`` in ``[0, 1)`` and
    ``rho`` of shape ``(n_bands, n_points)``.  The density is nonnegative by
    construction; the grid is only for plotting, since
    :func:`layer_charges` integrates the Fourier series in closed form.
    """

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    orders, rho = density_fourier_coefficients(g_supercell, coefficients, axis=axis, atol=atol)
    z = np.linspace(0.0, 1.0, n_points, endpoint=False)
    phases = np.exp(2j * np.pi * np.outer(orders.astype(float), z))
    return z, np.real(rho @ phases)


def _segment_integrals(
    orders: NDArray[np.int64], lower: float, upper: float
) -> NDArray[np.complex128]:
    """``int_lower^upper exp(2 pi i m z) dz`` for each order ``m``."""

    out = np.empty(orders.size, dtype=np.complex128)
    nonzero = orders != 0
    out[~nonzero] = upper - lower
    m = orders[nonzero].astype(float)
    out[nonzero] = (np.exp(2j * np.pi * m * upper) - np.exp(2j * np.pi * m * lower)) / (
        2j * np.pi * m
    )
    return out


def layer_charges(
    g_supercell: ArrayLike,
    coefficients: ArrayLike,
    edges: ArrayLike,
    *,
    axis: int = 2,
    atol: float = 1e-6,
) -> NDArray[np.float64]:
    """Charge of each state in each layer, in closed form.

    ``edges`` are fractional coordinates along ``axis``; ``n`` edges define
    ``n`` layers cyclically, layer ``i`` running from ``edges[i]`` to
    ``edges[i + 1]`` and the last wrapping through the cell boundary.  The
    charges of a state add up to its squared norm exactly
    (``UnfoldLab.sum_regionCharge_layers``), whatever the edges are.

    Returns an array of shape ``(n_bands, n_layers)``.
    """

    boundaries = np.asarray(edges, dtype=float).ravel()
    if boundaries.size < 1:
        raise ValueError("edges must contain at least one boundary")
    if np.unique(np.mod(boundaries, 1.0)).size != boundaries.size:
        raise ValueError("edges must be distinct modulo one")
    boundaries = np.sort(np.mod(boundaries, 1.0))

    orders, rho = density_fourier_coefficients(g_supercell, coefficients, axis=axis, atol=atol)
    charges = np.empty((rho.shape[0], boundaries.size), dtype=np.float64)
    for index in range(boundaries.size):
        lower = boundaries[index]
        upper = boundaries[(index + 1) % boundaries.size]
        if index + 1 == boundaries.size:
            upper += 1.0
        charges[:, index] = np.real(rho @ _segment_integrals(orders, lower, upper))
    return charges


def layer_edges_from_structure(
    structure: Structure,
    *,
    axis: int = 2,
    tol: float = 0.5,
) -> NDArray[np.float64]:
    """Layer boundaries midway between the atomic planes of ``structure``.

    Uses :func:`unfoldlab.core.site_projection.detect_layers` to find the planes
    and puts a boundary halfway between neighbouring ones, cyclically, so that
    each layer contains exactly one plane.  The result is in fractional
    coordinates along ``axis`` and can be passed straight to
    :func:`layer_charges`.
    """

    assignment = detect_layers(structure, axis=axis, tol=tol)
    lattice = structure.lattice
    other = [i for i in (0, 1, 2) if i != axis]
    normal = np.cross(lattice[other[0]], lattice[other[1]])
    normal = normal / float(np.linalg.norm(normal))
    period = float(abs(np.dot(lattice[axis], normal)))

    planes = np.sort(
        np.array(
            [
                float(np.mod(assignment.heights[assignment.index == layer].mean(), period))
                for layer in range(assignment.n_layers)
            ]
        )
    )
    if planes.size == 1:
        return np.asarray([float(np.mod(planes[0] + 0.5 * period, period) / period)])
    following = np.roll(planes, -1)
    following[-1] += period
    midpoints = 0.5 * (planes + following)
    return np.asarray(np.sort(np.mod(midpoints / period, 1.0)), dtype=np.float64)


def layer_resolved_weights(
    g_supercell: ArrayLike,
    coefficients: ArrayLike,
    primitive_kpoint: ArrayLike,
    folded_supercell_kpoint: ArrayLike,
    transform: TransformLike,
    edges: ArrayLike,
    *,
    axis: int = 2,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Split the unfolding weight of one primitive k-point among the layers.

    The matched component of the state -- the plane waves the primitive k-point
    claims -- has a planar-averaged density of its own, and its charge in each
    layer, divided by the total norm of the state, is the layer's share of the
    weight.  Summing over the layers gives the ordinary weight exactly
    (``UnfoldLab.sum_regionCharge_layers``), so this is a decomposition and not a
    separate quantity.

    What it is *not* is a decomposition of the layer's charge: summing over the
    k-points of a fiber does not return the charge the layer holds, because the
    layer projector is not diagonal in the plane-wave basis
    (``UnfoldLab.regionCharge_eq_blockSum_add_crossForm``).  Use
    :func:`diagnose_layer_conservation` to see how large that gap is before
    reading anything into a k-summed layer weight.

    Returns an array of shape ``(n_bands, n_layers)``.
    """

    g_int, coeffs = _prepare(g_supercell, coefficients, axis, tol)
    mask = matching_g_mask(
        g_int,
        np.asarray(primitive_kpoint, dtype=float),
        np.asarray(folded_supercell_kpoint, dtype=float),
        transform,
        tol=tol,
    )
    matched = np.where(mask[np.newaxis, np.newaxis, :], coeffs, 0.0)
    total = (coeffs.real**2 + coeffs.imag**2).sum(axis=(1, 2))
    safe = np.where(total > 0.0, total, 1.0)
    charges = layer_charges(g_int, matched, edges, axis=axis, atol=tol)
    return np.where(total[:, np.newaxis] > 0.0, charges / safe[:, np.newaxis], 0.0)


@dataclass(frozen=True)
class LayerConservation:
    """How far the k-summed layer weights are from the layer charges."""

    layer_charges: NDArray[np.float64]
    summed_weights: NDArray[np.float64]
    fiber_complete: bool

    @property
    def cross_term(self) -> NDArray[np.float64]:
        """Per band and layer, the charge the k-point split fails to account for."""

        return np.asarray(self.layer_charges - self.summed_weights, dtype=np.float64)

    @property
    def max_abs_cross_term(self) -> float:
        return float(np.max(np.abs(self.cross_term))) if self.cross_term.size else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_charges": self.layer_charges.tolist(),
            "summed_weights": self.summed_weights.tolist(),
            "fiber_complete": self.fiber_complete,
            "cross_term": self.cross_term.tolist(),
            "max_abs_cross_term": self.max_abs_cross_term,
        }

    def summary(self) -> str:
        lines = [
            "Layer weights summed over the fiber against the layer charges:",
            f"  largest discrepancy: {self.max_abs_cross_term:.3g} of the state norm",
        ]
        if not self.fiber_complete:
            lines.append(
                "  the k-points given are not a complete fiber, so part of the "
                "discrepancy is simply missing weight"
            )
        lines.append(
            "  the rest is the cross term between k-point blocks: a layer is not "
            "diagonal in the plane-wave basis, so its charge does not split by "
            "k-point (UnfoldLab.regionCharge_eq_blockSum_add_crossForm)"
        )
        return "\n".join(lines)


def diagnose_layer_conservation(
    g_supercell: ArrayLike,
    coefficients: ArrayLike,
    primitive_kpoints: ArrayLike,
    folded_supercell_kpoint: ArrayLike,
    transform: TransformLike,
    edges: ArrayLike,
    *,
    axis: int = 2,
    tol: float = 1e-6,
) -> LayerConservation:
    """Compare the k-summed layer weights with the layer charges of the state.

    The difference is the cross term between the k-point blocks.  It is not a
    numerical error and it does not shrink with a better basis: it is the part
    of a layer's charge that lives in the interference between the components
    the fiber's k-points claim.  Reported so that a k-summed layer weight is
    read with the right expectations.
    """

    g_int, coeffs = _prepare(g_supercell, coefficients, axis, tol)
    points = np.atleast_2d(np.asarray(primitive_kpoints, dtype=float))
    if points.shape[-1] != 3:
        raise ValueError("primitive_kpoints must have shape (3,) or (n, 3)")

    total = (coeffs.real**2 + coeffs.imag**2).sum(axis=(1, 2))
    safe = np.where(total > 0.0, total, 1.0)
    charges = layer_charges(g_int, coeffs, edges, axis=axis, atol=tol) / safe[:, np.newaxis]

    summed = np.zeros_like(charges)
    covered = np.zeros(g_int.shape[0], dtype=bool)
    for point in points:
        summed += layer_resolved_weights(
            g_int,
            coeffs,
            point,
            folded_supercell_kpoint,
            transform,
            edges,
            axis=axis,
            tol=tol,
        )
        covered |= matching_g_mask(
            g_int,
            point,
            np.asarray(folded_supercell_kpoint, dtype=float),
            transform,
            tol=tol,
        )
    return LayerConservation(
        layer_charges=charges,
        summed_weights=summed,
        fiber_complete=bool(covered.all()),
    )
