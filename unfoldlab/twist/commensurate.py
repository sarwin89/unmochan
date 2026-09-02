"""Commensurate twisted stacks: angles, moire cells, and two-reference unfolding.

Why this exists
---------------

Before pass 31 the twist area of the package could *describe* a stack -- load
two references, estimate a rotation angle, unfold onto one of them -- but it
could not answer the two questions a twisted-bilayer user actually starts from:

1. **which twist angles are commensurate at all**, and how big is the cell for
   each of them?  A twisted bilayer only has a Brillouin zone (and therefore
   only admits exact unfolding) at the discrete commensurate angles;
2. **what does unfolding a stack onto one layer mean**, given that the stack
   has *two* primitive references and each of them is a legitimate unfolding
   target?

The mathematics is settled in ``RequestProject/Unfolding/Twist.lean``:

*a stack is two folding problems sharing one supercell*
    ``UnfoldLab.IsCommensurateStack`` bundles ``Am = Ta * Aa = Tb * Ab``.  The
    multiplicities are then fixed by the cell volumes
    (``IsCommensurateStack.det_relation``): ``|det Ta| |det Aa| = |det Tb|
    |det Ab|``, so for two layers of equal cell area the two multiplicities are
    equal.  :func:`diagnose_stack` checks this and reports both transforms;

*the transfer map between the two primitive zones is the identity in Cartesian
 reciprocal space*
    ``IsCommensurateStack.transfer_cartesian_eq``: the fractional coordinates
    differ, the physical k-vector does not.  :func:`transfer_kpoint` implements
    it with exact integer arithmetic, ``k_b = k_a @ Ta @ adj(Tb) / det Tb``;

*and the pitfall: unfolding onto one layer is blind to the other*
    ``IsCommensurateStack.sum_weight_both_layers``.  A state of the stack has
    total unfolding weight one in layer A's zone *and* total weight one in
    layer B's zone -- the same electron is counted twice, once per reference.
    Spectral weight in one layer's effective band structure therefore says
    nothing about which layer the state lives on.  To resolve that one needs a
    *layer-projected* weight, ``UnfoldLab.layerWeight``, whose sum rule
    (``IsFiberRepr.sum_layerWeight_eq_fraction``) is the layer's share of the
    state norm rather than one.

    That projection is only exact in a *site* basis: a real-space layer is not
    a diagonal subset of the plane-wave basis (see ``Layers.lean``), so there
    is deliberately no plane-wave "layer subset" kernel here.  The exact case
    is tight binding / LCAO, and :func:`layer_projected_weights` wires the
    layer labels straight into
    :func:`unfoldlab.core.tight_binding.tight_binding_orbital_weights`.

Commensurate hexagonal twists
-----------------------------

For a hexagonal lattice with basis vectors at 60 degrees, the vector
``m a1 + n a2`` has squared length ``a^2 (m^2 - m n + n^2)``
(``UnfoldLab.hexVec_normSq``), so ``(m, n)`` and ``(n, m)`` have the *same*
length and one is a rotation of the other.  Taking that rotation as the twist
gives (``UnfoldLab.cosAngle_hexVec_swap``)

``cos theta = (4 m n - m^2 - n^2) / (2 (m^2 - m n + n^2))``

and the moire cell contains ``Q = m^2 - m n + n^2`` primitive cells of each
layer (``UnfoldLab.det_moireBasis``).  The transform for layer A is
``moireBasis(m, n)``, for layer B ``moireBasis(n, m)``; the determinants agree,
as :func:`commensurate_stack_transforms` asserts.

``(m, n) = (2, 3)`` gives ``Q = 7`` and ``cos theta = 11/14``, i.e. 38.21
degrees, which is ``60 - 21.79`` -- the classic 21.79-degree twisted bilayer
graphene cell.  :func:`hexagonal_commensurate_twists` enumerates these.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import acos, degrees, gcd
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import (
    as_int_array,
    as_matrix3,
    check_atol,
    integer_adjugate3,
    integer_det3,
)
from unfoldlab.core.strain import CommensurabilityReport, diagnose_commensurability
from unfoldlab.core.structures import Structure
from unfoldlab.core.tight_binding import tight_binding_orbital_weights

__all__ = [
    "CommensurateTwist",
    "LayerStackInfo",
    "StackReport",
    "commensurate_stack_transforms",
    "diagnose_stack",
    "hexagonal_commensurate_twists",
    "layer_projected_weights",
    "moire_cell_size",
    "transfer_kpoint",
]

#: Hexagonal lattices are invariant under a 60-degree rotation, so two twists
#: differing by a multiple of 60 degrees describe the same stack.
HEXAGONAL_PERIOD_DEG = 60.0


def moire_cell_size(m: int, n: int) -> int:
    """``Q = m^2 - m n + n^2``, the number of primitive cells per layer.

    ``UnfoldLab.det_moireBasis``.  This is also the unfolding multiplicity of
    each layer, and it is zero only for ``m = n = 0``.
    """

    return int(m) ** 2 - int(m) * int(n) + int(n) ** 2


def _fold_hexagonal_angle(angle_deg: float) -> float:
    """Fold an angle into ``[0, 30]`` using the sixfold symmetry and mirror."""

    folded = float(angle_deg) % HEXAGONAL_PERIOD_DEG
    return min(folded, HEXAGONAL_PERIOD_DEG - folded)


@dataclass(frozen=True)
class CommensurateTwist:
    """One commensurate twist of a hexagonal bilayer, labelled by ``(m, n)``.

    ``cells`` is ``Q = m^2 - m n + n^2`` and ``cos_angle`` is exact in the
    sense that it is the rational ``(4 m n - m^2 - n^2) / (2 Q)`` evaluated in
    floating point, not a number read off a relaxed structure.
    ``equivalent_angle_deg`` folds the angle into ``[0, 30]`` degrees, which is
    the range twist angles are quoted in, because a hexagonal lattice is
    invariant under 60-degree rotations and under the mirror.
    """

    m: int
    n: int
    cells: int
    cos_angle: float
    angle_deg: float
    equivalent_angle_deg: float

    @property
    def is_aligned(self) -> bool:
        """Whether the twist is trivial (the layers are 60-degree equivalent)."""

        return self.equivalent_angle_deg <= 1e-9

    def to_dict(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "n": self.n,
            "cells": self.cells,
            "cos_angle": self.cos_angle,
            "angle_deg": self.angle_deg,
            "equivalent_angle_deg": self.equivalent_angle_deg,
        }


def _twist_from_indices(m: int, n: int) -> CommensurateTwist:
    cells = moire_cell_size(m, n)
    if cells == 0:
        raise ValueError("(m, n) = (0, 0) does not define a lattice vector")
    cosine = (4.0 * m * n - m * m - n * n) / (2.0 * cells)
    cosine = min(1.0, max(-1.0, cosine))
    angle = degrees(acos(cosine))
    return CommensurateTwist(
        m=int(m),
        n=int(n),
        cells=int(cells),
        cos_angle=float(cosine),
        angle_deg=float(angle),
        equivalent_angle_deg=_fold_hexagonal_angle(angle),
    )


def hexagonal_commensurate_twists(
    max_index: int, *, include_aligned: bool = False
) -> list[CommensurateTwist]:
    """Enumerate commensurate twists of a hexagonal bilayer up to ``max_index``.

    Only coprime ``(m, n)`` with ``0 < m < n <= max_index`` are generated: a
    common factor ``d`` rescales the moire vector by ``d`` and multiplies the
    cell by ``d^2`` without changing the angle, so it is the same stack in a
    needlessly large cell.  Aligned pairs -- those whose angle is a multiple of
    60 degrees, e.g. ``(1, 2)`` -- are dropped unless ``include_aligned``,
    since they describe an untwisted stack.

    Duplicates are removed as well, and this is the subtle part: several index
    pairs give the *same* twist in cells of different size.  ``(1, 3)`` and
    ``(2, 3)`` are both the 21.79-degree stack in 7 cells, but so is ``(1, 5)``
    in 21 cells -- a threefold-larger supercell of the same structure, and a
    threefold-more-expensive calculation for nothing.  Entries are therefore
    keyed by the folded angle and only the smallest cell found within
    ``max_index`` is kept.  The result is sorted by increasing cell size.
    """

    limit = int(max_index)
    if limit < 1:
        raise ValueError("max_index must be at least 1")
    seen: dict[float, CommensurateTwist] = {}
    for n in range(1, limit + 1):
        for m in range(1, n):
            if gcd(m, n) != 1:
                continue
            twist = _twist_from_indices(m, n)
            if twist.is_aligned and not include_aligned:
                continue
            key = round(twist.equivalent_angle_deg, 9)
            best = seen.get(key)
            if best is None or twist.cells < best.cells:
                seen[key] = twist
    return sorted(seen.values(), key=lambda t: (t.cells, t.m, t.n))


def _moire_basis(m: int, n: int) -> NDArray[np.int64]:
    """Row-convention moire transform: rows ``(m, n)`` and ``(m - n, m)``.

    The Lean ``UnfoldLab.moireBasis`` is column-convention (``!![m, m-p; p,
    m]``); the package stores transforms as rows of supercell vectors in
    primitive coordinates (``A_s = T @ A_p``), so this is its transpose,
    embedded in three dimensions with a trivial stacking axis.
    """

    matrix = np.eye(3, dtype=np.int64)
    matrix[0, 0] = int(m)
    matrix[0, 1] = int(n)
    matrix[1, 0] = int(m) - int(n)
    matrix[1, 1] = int(m)
    return matrix


def commensurate_stack_transforms(m: int, n: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The two layer transforms of the ``(m, n)`` hexagonal commensurate stack.

    Returns ``(T_a, T_b)`` with ``T_a`` built from ``(m, n)`` and ``T_b`` from
    ``(n, m)``: the same moire cell written in each layer's own basis.  Their
    determinants are both ``Q = m^2 - m n + n^2``, which is the equal-area case
    of ``UnfoldLab.IsCommensurateStack.det_relation``.
    """

    if moire_cell_size(m, n) == 0:
        raise ValueError("(m, n) = (0, 0) does not define a moire cell")
    return _moire_basis(m, n), _moire_basis(n, m)


def transfer_kpoint(
    kpoints: ArrayLike,
    transform_a: ArrayLike,
    transform_b: ArrayLike,
) -> NDArray[np.float64]:
    """Re-express layer-A primitive k-points in layer B's fractional coordinates.

    ``UnfoldLab.transferK`` in row convention.  Fractional k-points transform
    as ``k_s = k_p @ T.T`` (:func:`unfoldlab.core.kpoints.fold_kpoint_to_supercell`),
    which is the column-vector ``T *_v k`` of the Lean development, so the
    transfer ``T_b^-1 T_a`` becomes ``k_b = k_a @ (adj(T_b) @ T_a).T /
    det(T_b)``.  The adjugate is exact
    (:func:`unfoldlab.core.numerics.integer_adjugate3`), so the only rounding is
    the final division.  The Cartesian k-vector is unchanged
    (``IsCommensurateStack.transfer_cartesian_eq``) -- what changes is the
    lattice the coordinates refer to -- and the transferred point folds onto the
    same supercell k-point (``UnfoldLab.foldsTo_transferK``).
    """

    points = np.atleast_2d(np.asarray(kpoints, dtype=float))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("kpoints must have shape (3,) or (n_kpoints, 3)")
    matrix_a = as_int_array(as_matrix3(transform_a, name="transform_a"), name="transform_a")
    matrix_b = as_int_array(as_matrix3(transform_b, name="transform_b"), name="transform_b")
    determinant = integer_det3(matrix_b)
    if determinant == 0:
        raise ValueError("transform_b must be invertible")
    combined = (integer_adjugate3(matrix_b) @ matrix_a).T
    return np.asarray(points @ combined.astype(float) / float(determinant), dtype=np.float64)


@dataclass(frozen=True)
class LayerStackInfo:
    """How one layer of a stack relates to the moire cell."""

    label: str
    report: CommensurabilityReport
    twist_deg: float

    @property
    def transform(self) -> NDArray[np.int64]:
        return self.report.transform

    @property
    def multiplicity(self) -> int:
        """``|det T|``, the number of primitive cells of this layer per moire cell."""

        return abs(integer_det3(self.report.transform))

    @property
    def commensurate(self) -> bool:
        return self.report.commensurate

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "twist_deg": self.twist_deg,
            "multiplicity": self.multiplicity,
            **self.report.to_dict(),
        }


@dataclass(frozen=True)
class StackReport:
    """Diagnosis of a multi-layer stack against all of its primitive references."""

    layers: tuple[LayerStackInfo, ...]
    reference: str

    @property
    def commensurate(self) -> bool:
        """Whether *every* layer is an exact integer multiple of the moire cell."""

        return all(layer.commensurate for layer in self.layers)

    @property
    def multiplicities(self) -> dict[str, int]:
        return {layer.label: layer.multiplicity for layer in self.layers}

    @property
    def twist_angles(self) -> dict[str, float]:
        return {layer.label: layer.twist_deg for layer in self.layers}

    def layer(self, label: str) -> LayerStackInfo:
        for entry in self.layers:
            if entry.label == label:
                return entry
        raise KeyError(f"no layer named {label!r}")

    def transfer(self, kpoints: ArrayLike, source: str, target: str) -> NDArray[np.float64]:
        """Re-express ``kpoints`` given in layer ``source`` in layer ``target``."""

        return transfer_kpoint(kpoints, self.layer(source).transform, self.layer(target).transform)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "commensurate": self.commensurate,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def summary(self) -> str:
        lines = [
            f"stack of {len(self.layers)} references "
            f"({'commensurate' if self.commensurate else 'NOT commensurate'}), "
            f"twists measured against {self.reference!r}"
        ]
        for layer in self.layers:
            state = "exact" if layer.commensurate else f"residual {layer.report.residual:.3g}"
            lines.append(
                f"  {layer.label}: {layer.multiplicity} cells per moire cell, "
                f"twist {layer.twist_deg:+.3f} deg ({state})"
            )
        if len(self.layers) > 1:
            lines.append(
                "  caution: each reference carries a complete fiber, so a state has "
                "total unfolding weight one in EVERY layer's zone "
                "(UnfoldLab.IsCommensurateStack.sum_weight_both_layers); use "
                "layer_projected_weights to ask which layer a state lives on"
            )
        return "\n".join(lines)


def _lattice_of(value: Structure | ArrayLike, *, name: str) -> NDArray[np.float64]:
    lattice = value.lattice if isinstance(value, Structure) else value
    return as_matrix3(lattice, name=name)


def _in_plane_angle(lattice: NDArray[np.float64], *, vector_index: int = 0) -> float:
    vector = lattice[vector_index, :2]
    if float(np.linalg.norm(vector)) == 0.0:
        raise ValueError("selected in-plane lattice vector has zero length")
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def diagnose_stack(
    moire: Structure | ArrayLike,
    layers: Mapping[str, Structure | ArrayLike],
    *,
    reference: str | None = None,
    atol: float = 1e-6,
    vector_index: int = 0,
) -> StackReport:
    """Diagnose a moire cell against each of its layer references.

    Every layer gets a full :func:`unfoldlab.core.strain.diagnose_commensurability`
    run -- integer transform, residual, residual strain, worst-case k error --
    plus its in-plane rotation relative to ``reference`` (the first layer by
    default).  Nothing here is re-derived: the commensurability machinery is
    the same one the single-reference workflow uses, applied once per layer.
    """

    check_atol(atol)
    if not layers:
        raise ValueError("at least one layer reference is required")
    labels = list(layers)
    anchor = labels[0] if reference is None else reference
    if anchor not in layers:
        raise KeyError(f"reference layer {anchor!r} is not among {labels}")
    moire_lattice = _lattice_of(moire, name="moire_lattice")
    anchor_angle = _in_plane_angle(
        _lattice_of(layers[anchor], name="layer_lattice"), vector_index=vector_index
    )

    entries: list[LayerStackInfo] = []
    for label in labels:
        lattice = _lattice_of(layers[label], name="layer_lattice")
        report = diagnose_commensurability(lattice, moire_lattice, atol=atol)
        angle = _in_plane_angle(lattice, vector_index=vector_index) - anchor_angle
        entries.append(
            LayerStackInfo(
                label=label,
                report=report,
                twist_deg=float((angle + 180.0) % 360.0 - 180.0),
            )
        )
    return StackReport(layers=tuple(entries), reference=anchor)


def layer_projected_weights(
    cells: ArrayLike,
    coefficients: ArrayLike,
    kpoints: ArrayLike,
    layer_of_orbital: Sequence[int] | NDArray[np.int64],
    *,
    multiplicity: int | None = None,
) -> tuple[NDArray[np.float64], list[int]]:
    """Layer-resolved tight-binding unfolding weights.

    ``layer_of_orbital`` labels each orbital of the moire cell with the layer it
    belongs to; the orbitals of one layer form one group and the weights are
    those of :func:`unfoldlab.core.tight_binding.tight_binding_orbital_weights`,
    i.e. ``UnfoldLab.layerWeight``: numerator restricted to the layer,
    denominator the full state norm.  Over a complete fiber a layer's weights
    therefore add up to the *fraction* of the state living on that layer
    (``UnfoldLab.IsFiberRepr.sum_layerWeight_eq_fraction``), and the fractions
    of a partition into layers add to one.

    This is the honest answer to "which layer is this band on?", and it is
    exact only because the basis is a site basis.  The plane-wave kernel cannot
    do it: a real-space layer is not a subset of plane waves.

    Returns ``(weights, labels)`` with ``weights`` of shape
    ``(n_layers, n_kpoints, n_states)`` and ``labels`` the sorted layer labels
    in the same order as the first axis.
    """

    assignment = np.asarray(layer_of_orbital, dtype=np.int64).reshape(-1)
    if assignment.size == 0:
        raise ValueError("layer_of_orbital must not be empty")
    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_states, n_orbitals, n_cells)")
    if assignment.size != coeffs.shape[1]:
        raise ValueError("layer_of_orbital must have one entry per orbital")
    labels = sorted({int(value) for value in assignment})
    groups = [np.flatnonzero(assignment == label).astype(np.int64) for label in labels]
    weights = tight_binding_orbital_weights(
        cells, coefficients, kpoints, groups, multiplicity=multiplicity
    )
    return weights, labels
