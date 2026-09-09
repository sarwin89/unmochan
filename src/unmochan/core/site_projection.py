"""Resolution of generic projection selectors against a structure.

:mod:`unmochan.core.projections` defines the *syntax* of a selector.  This
module turns a set of selectors into a boolean mask over the ``(site, orbital)``
table of a projected-character file such as VASP's ``PROCAR``, and turns that
mask into per-band projection fractions.

Combination rule: selectors of the same namespace are combined with OR,
different namespaces with AND.  ``species:A species:B orbital:d`` therefore
selects the d orbitals of every A and B site.  A namespace that appears with no
selector is unconstrained, and ``ns:*`` is the same as omitting ``ns``.

Only ``species``, ``atom``, ``orbital``, ``layer`` and ``surface`` can be
resolved from geometry alone.  ``sublattice``, ``region``, ``interface``,
``valley``, ``defect_shell`` and the role tags are user conventions: they are
resolved from an explicit ``site_groups`` mapping and raise otherwise, rather
than being guessed.  ``distance_from_defect`` needs ``defect_center``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.projections import ProjectionSelector
from unmochan.core.structures import Structure
from unmochan.core.valleys import minimum_image_distance

__all__ = [
    "LayerAssignment",
    "detect_layers",
    "orbital_group_mask",
    "projection_fractions",
    "resolve_site_mask",
    "resolve_site_orbital_mask",
]

#: Namespaces that select sites (as opposed to orbitals or spin components).
_SITE_NAMESPACES = frozenset(
    {
        "species",
        "atom",
        "layer",
        "surface",
        "sublattice",
        "region",
        "defect_shell",
        "interface",
        "valley",
        "role",
        "distance_from_defect",
    }
)

#: Namespaces that can only come from a user-supplied grouping.
_USER_NAMESPACES = frozenset(
    {"sublattice", "region", "defect_shell", "interface", "valley", "role"}
)

#: Shell letters recognized as orbital *groups*.
_ORBITAL_GROUPS = ("s", "p", "d", "f")


@dataclass(frozen=True)
class LayerAssignment:
    """Per-site layer indices along a stacking direction.

    ``index[i]`` is the layer of site ``i``, counted from the layer that follows
    the largest gap along the stacking direction; for a slab with vacuum that is
    the bottom-most layer.  ``heights`` are the Cartesian coordinates of the
    sites along the outward normal of the plane spanned by the other two lattice
    vectors, which is the physical interlayer coordinate even for a lattice
    whose stacking vector is not orthogonal to the layers.
    """

    index: NDArray[np.int64]
    heights: NDArray[np.float64]
    n_layers: int

    def sites_in_layer(self, layer: int) -> NDArray[np.int64]:
        return np.flatnonzero(self.index == layer).astype(np.int64)


def detect_layers(
    structure: Structure,
    *,
    axis: int = 2,
    tol: float = 0.5,
) -> LayerAssignment:
    """Group the sites of ``structure`` into layers along a stacking axis.

    Two sites belong to the same layer when their heights differ by at most
    ``tol`` (in the length unit of the lattice, normally angstrom) from a site
    already in the layer, with the periodic image along the stacking direction
    taken into account.
    """

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    lattice = structure.lattice
    other = [i for i in (0, 1, 2) if i != axis]
    normal = np.cross(lattice[other[0]], lattice[other[1]])
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("the two lattice vectors transverse to axis are collinear")
    normal = normal / norm
    period = float(abs(np.dot(lattice[axis], normal)))
    if period <= 0.0:
        raise ValueError("the stacking vector lies in the plane of the other two")

    heights = np.asarray(structure.cart_coords @ normal, dtype=float)
    reduced = np.mod(heights, period)
    order = np.argsort(reduced, kind="stable")
    sorted_heights = reduced[order]
    n_sites = len(order)
    if n_sites == 0:
        return LayerAssignment(np.zeros(0, dtype=np.int64), heights, 0)

    gaps = np.empty(n_sites, dtype=float)
    gaps[:-1] = sorted_heights[1:] - sorted_heights[:-1]
    gaps[-1] = period - sorted_heights[-1] + sorted_heights[0]
    breaks = gaps > tol
    if not breaks.any():
        return LayerAssignment(np.zeros(n_sites, dtype=np.int64), heights, 1)

    start = int(np.argmax(gaps)) + 1  # first site after the largest gap
    index = np.zeros(n_sites, dtype=np.int64)
    layer = 0
    for step in range(n_sites):
        position = (start + step) % n_sites
        if step > 0 and breaks[(position - 1) % n_sites]:
            layer += 1
        index[order[position]] = layer
    return LayerAssignment(index, heights, layer + 1)


def orbital_group_mask(
    orbital_labels: Sequence[str],
    value: str | int | float,
) -> NDArray[np.bool_]:
    """Boolean mask over ``orbital_labels`` selected by one ``orbital:`` value."""

    labels = [str(label) for label in orbital_labels]
    text = str(value).strip()
    if text == "*":
        return np.ones(len(labels), dtype=bool)
    normalized = [_normalize_orbital(label) for label in labels]
    target = _normalize_orbital(text)
    exact = np.array([label == target for label in normalized], dtype=bool)
    if exact.any():
        return exact
    if text.lower() in _ORBITAL_GROUPS:
        shell = text.lower()
        grouped = np.array([label.startswith(shell) for label in normalized], dtype=bool)
        if grouped.any():
            return grouped
    raise ValueError(f"no orbital in {labels} matches selector orbital:{text}")


def resolve_site_mask(
    selectors: Iterable[ProjectionSelector],
    structure: Structure,
    *,
    site_groups: Mapping[str, Mapping[object, Sequence[int]]] | None = None,
    layer_axis: int = 2,
    layer_tol: float = 0.5,
    defect_center: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Boolean mask over the sites of ``structure`` selected by ``selectors``."""

    selectors = tuple(selectors)
    mask = np.ones(structure.n_sites, dtype=bool)
    grouped: dict[str, list[ProjectionSelector]] = {}
    for selector in selectors:
        if selector.namespace == "orbital":
            continue
        if selector.namespace == "spin":
            raise ValueError(
                "spin selectors do not select sites; pass the spin channel or component explicitly"
            )
        if selector.namespace not in _SITE_NAMESPACES:
            raise ValueError(f"cannot resolve namespace {selector.namespace!r} against a structure")
        grouped.setdefault(selector.namespace, []).append(selector)

    layers: LayerAssignment | None = None
    for namespace, items in grouped.items():
        if any(item.is_wildcard for item in items):
            continue
        if namespace in ("layer", "surface") and layers is None:
            layers = detect_layers(structure, axis=layer_axis, tol=layer_tol)
        union = np.zeros(structure.n_sites, dtype=bool)
        for item in items:
            union |= _single_site_mask(
                item,
                structure,
                layers=layers,
                site_groups=site_groups,
                defect_center=defect_center,
            )
        mask &= union
    return mask


def resolve_site_orbital_mask(
    selectors: Iterable[ProjectionSelector],
    structure: Structure,
    orbital_labels: Sequence[str],
    *,
    site_groups: Mapping[str, Mapping[object, Sequence[int]]] | None = None,
    layer_axis: int = 2,
    layer_tol: float = 0.5,
    defect_center: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Boolean ``(n_sites, n_orbitals)`` mask selected by ``selectors``."""

    selectors = tuple(selectors)
    sites = resolve_site_mask(
        selectors,
        structure,
        site_groups=site_groups,
        layer_axis=layer_axis,
        layer_tol=layer_tol,
        defect_center=defect_center,
    )
    orbital_selectors = [item for item in selectors if item.namespace == "orbital"]
    orbitals = np.ones(len(orbital_labels), dtype=bool)
    if orbital_selectors and not any(item.is_wildcard for item in orbital_selectors):
        orbitals = np.zeros(len(orbital_labels), dtype=bool)
        for item in orbital_selectors:
            orbitals |= orbital_group_mask(orbital_labels, item.value)
    return sites[:, None] & orbitals[None, :]


def projection_fractions(
    projections: ArrayLike,
    mask: ArrayLike,
    *,
    total_tol: float = 1e-12,
) -> NDArray[np.float64]:
    """Fraction of the projected character selected by ``mask``.

    ``projections`` has shape ``(..., n_sites, n_orbitals)`` and ``mask`` shape
    ``(n_sites, n_orbitals)``.  The result has shape ``(...)`` and equals the
    masked sum divided by the total over all sites and orbitals, which is the
    only normalization that makes the fractions of a partition of the table sum
    to one: the raw sphere projections of a plane-wave code do not sum to the
    norm of the band.

    Bands whose total projected character is below ``total_tol`` (an empty
    sphere set, or an all-interstitial state) get a fraction of zero, so a
    projected weight is never larger than the unprojected one.

    Negative entries -- which appear in the magnetization tables, not in the
    charge table -- are rejected, since a signed table has no fraction
    interpretation.
    """

    values = np.asarray(projections, dtype=float)
    selection = np.asarray(mask, dtype=bool)
    if values.ndim < 2:
        raise ValueError("projections must have shape (..., n_sites, n_orbitals)")
    if selection.shape != values.shape[-2:]:
        raise ValueError("mask shape must match the last two axes of projections")
    if np.any(values < 0.0):
        raise ValueError("projection fractions require a nonnegative table")
    total = values.sum(axis=(-2, -1))
    selected = (values * selection).sum(axis=(-2, -1))
    fractions = np.zeros_like(total)
    nonzero = total > total_tol
    np.divide(selected, total, out=fractions, where=nonzero)
    return np.clip(fractions, 0.0, 1.0)


def _single_site_mask(
    selector: ProjectionSelector,
    structure: Structure,
    *,
    layers: LayerAssignment | None,
    site_groups: Mapping[str, Mapping[object, Sequence[int]]] | None,
    defect_center: ArrayLike | None,
) -> NDArray[np.bool_]:
    namespace = selector.namespace
    value = selector.value
    n_sites = structure.n_sites

    if namespace == "species":
        target = str(value)
        return np.array([symbol == target for symbol in structure.species], dtype=bool)

    if namespace == "atom":
        if not isinstance(value, int):
            raise ValueError(f"atom selector needs an integer index, got {value!r}")
        if not 0 <= value < n_sites:
            raise ValueError(f"atom index {value} is out of range for {n_sites} sites")
        mask = np.zeros(n_sites, dtype=bool)
        mask[value] = True
        return mask

    if namespace == "layer":
        assert layers is not None
        if not isinstance(value, int):
            raise ValueError(f"layer selector needs an integer index, got {value!r}")
        if not 0 <= value < layers.n_layers:
            raise ValueError(f"layer {value} is out of range for {layers.n_layers} layers")
        return layers.index == value

    if namespace == "surface":
        assert layers is not None
        text = str(value).lower()
        if text == "bottom":
            return layers.index == 0
        if text == "top":
            return layers.index == layers.n_layers - 1
        raise ValueError("surface selector must be 'top' or 'bottom'")

    if namespace == "distance_from_defect":
        if defect_center is None:
            raise ValueError("distance_from_defect requires defect_center")
        radius = float(value)
        center = np.asarray(defect_center, dtype=float)
        if center.shape != (3,):
            raise ValueError("defect_center must be fractional coordinates of shape (3,)")
        distances = np.array(
            [
                minimum_image_distance(frac - center, structure.lattice)
                for frac in structure.frac_coords
            ],
            dtype=float,
        )
        return distances <= radius

    if namespace in _USER_NAMESPACES:
        groups = (site_groups or {}).get(namespace)
        if groups is None:
            raise ValueError(
                f"{namespace} is a user convention: pass site_groups[{namespace!r}] "
                "mapping each label to its site indices"
            )
        if value not in groups:
            available = sorted(str(key) for key in groups)
            raise ValueError(f"unknown {namespace} {value!r}; known labels: {available}")
        mask = np.zeros(n_sites, dtype=bool)
        indices = np.asarray(list(groups[value]), dtype=int)
        if indices.size and (indices.min() < 0 or indices.max() >= n_sites):
            raise ValueError(f"site index out of range in site_groups[{namespace!r}]")
        mask[indices] = True
        return mask

    raise ValueError(f"cannot resolve selector {selector.to_string()!r}")


def _normalize_orbital(label: str) -> str:
    text = label.strip().lower()
    for source in ("_", "-", "^"):
        text = text.replace(source, "")
    aliases = {
        "dx2y2": "dx2",
        "dx2-y2": "dx2",
        "dz2r2": "dz2",
        "x2y2": "dx2",
    }
    return aliases.get(text, text)
