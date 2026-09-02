"""Backend-neutral unfolding data contracts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.numerics import TransformLike, as_int_array, integer_det3, wrap_fractional
from unfoldlab.core.plane_waves import (
    shared_weights_from_coefficients,
    spin_texture_from_coefficients,
    weights_from_coefficients,
)
from unfoldlab.core.spectral import (
    EffectiveBandStructure,
    default_distances,
    default_weights,
)


@dataclass(frozen=True)
class PlaneWaveKPointData:
    """Plane-wave coefficients for one supercell k-point.

    This container is intentionally code-agnostic. QE readers, future VASP
    WAVECAR readers, and external coefficient exporters should all normalize
    into this shape before weights are calculated.
    """

    primitive_kpoint: NDArray[np.float64]
    folded_supercell_kpoint: NDArray[np.float64]
    g_supercell: NDArray[np.int64]
    coefficients: NDArray[np.complex128]

    def __post_init__(self) -> None:
        primitive = np.asarray(self.primitive_kpoint, dtype=float)
        folded = np.asarray(self.folded_supercell_kpoint, dtype=float)
        g_vectors = np.asarray(self.g_supercell, dtype=int)
        coefficients = np.asarray(self.coefficients, dtype=np.complex128)
        if primitive.shape != (3,):
            raise ValueError("primitive_kpoint must have shape (3,)")
        if folded.shape != (3,):
            raise ValueError("folded_supercell_kpoint must have shape (3,)")
        if g_vectors.ndim != 2 or g_vectors.shape[1] != 3:
            raise ValueError("g_supercell must have shape (n_g, 3)")
        if coefficients.ndim != 3 or coefficients.shape[2] != g_vectors.shape[0]:
            raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
        object.__setattr__(self, "primitive_kpoint", primitive)
        object.__setattr__(self, "folded_supercell_kpoint", folded)
        object.__setattr__(self, "g_supercell", g_vectors)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def n_bands(self) -> int:
        return int(self.coefficients.shape[0])


@dataclass(frozen=True)
class BandUnfoldingData:
    """Backend-neutral band energies plus optional spectral weights."""

    kpoints: NDArray[np.float64]
    energies: NDArray[np.float64]
    weights: NDArray[np.float64] | None = None
    distances: NDArray[np.float64] | None = None
    reference_energy: float = 0.0
    source_code: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_effective_band_structure(self) -> EffectiveBandStructure:
        return EffectiveBandStructure(
            kpoints=self.kpoints,
            energies=self.energies,
            weights=(default_weights(self.energies) if self.weights is None else self.weights),
            distances=(
                default_distances(self.kpoints.shape[0])
                if self.distances is None
                else self.distances
            ),
            reference_energy=self.reference_energy,
            metadata={"source_code": self.source_code, **self.metadata},
        )


@dataclass(frozen=True)
class WeightDiagnostics:
    """Range diagnostics for a ``(n_kpoints, n_bands)`` weight table.

    ``max_band_sum`` is the largest sum of weights *over bands* at a single
    k-point.  It is bounded by ``n_bands``, not by one: the unfolding sum rule
    constrains the sum over the ``|det T|`` primitive k-points that fold onto a
    common supercell k-point *at fixed band index*, which is checked by
    :func:`diagnose_fiber_sum_rule` instead.
    """

    min_weight: float
    max_weight: float
    out_of_bounds_count: int
    max_band_sum: float
    n_kpoints: int
    n_bands: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "out_of_bounds_count": self.out_of_bounds_count,
            "max_band_sum": self.max_band_sum,
            "n_kpoints": self.n_kpoints,
            "n_bands": self.n_bands,
        }


def compute_plane_wave_unfolding_weights(
    kpoint_data: Iterable[PlaneWaveKPointData],
    transform: TransformLike,
    *,
    tol: float = 1e-6,
    component_resolved: bool = False,
) -> NDArray[np.float64]:
    """Compute backend-independent plane-wave unfolding weights.

    Identical systems represented with equivalent plane-wave coefficients,
    reciprocal G-vector indexing, transform matrix, and primitive k-path should
    produce identical weights regardless of whether the source reader is QE,
    VASP, or an external exporter.

    With ``component_resolved=True`` the result has shape
    ``(n_kpoints, n_bands, n_components)`` and splits each weight into the
    contributions of the spinor/polarization components; summing over the last
    axis recovers the ordinary weights.  All k-points must then have the same
    number of components.

    ``kpoint_data`` may be any iterable, and each entry is used and released
    before the next is requested.  Passing a generator that reads one k-point at
    a time therefore keeps the peak memory at a single wavefunction rather than
    at the whole band path, which for a large supercell is the difference
    between gigabytes and terabytes.  The result itself is tiny: one float per
    ``(k-point, band)``.
    """

    rows: list[NDArray[np.float64]] = []
    n_bands: int | None = None
    n_components: int | None = None
    for item in kpoint_data:
        if n_bands is None:
            n_bands = item.n_bands
            n_components = int(item.coefficients.shape[1])
        elif item.n_bands != n_bands:
            raise ValueError("all k-points must have the same number of bands")
        if component_resolved and int(item.coefficients.shape[1]) != n_components:
            raise ValueError(
                "component-resolved weights require all k-points to have the same "
                "number of spinor/polarization components"
            )
        rows.append(
            weights_from_coefficients(
                item.g_supercell,
                item.coefficients,
                item.primitive_kpoint,
                item.folded_supercell_kpoint,
                transform,
                tol=tol,
                component_resolved=component_resolved,
            )
        )
    if not rows:
        raise ValueError("at least one k-point wavefunction is required")
    return np.stack(rows).astype(float, copy=False)


def compute_plane_wave_unfolding_weights_chunked(
    chunked_kpoint_data: Iterable[Iterable[PlaneWaveKPointData]],
    transform: TransformLike,
    *,
    tol: float = 1e-6,
    component_resolved: bool = False,
) -> NDArray[np.float64]:
    """Weights from a band-chunked stream, one k-point per outer element.

    Each outer element is an iterable of :class:`PlaneWaveKPointData` objects
    holding *consecutive blocks of bands* of the same k-point; the blocks are
    concatenated along the band axis.  A reader that materializes one block at
    a time keeps the peak memory at ``band_chunk`` bands rather than at the
    whole k-point, which for a large supercell is the dominant allocation.

    The result is the one
    :func:`compute_plane_wave_unfolding_weights` would return for the
    unchunked stream: a band's weight is a function of that band's
    coefficients alone (``UnfoldLab.bandWeights_flatten``).

    Every block of one k-point must describe the same k-point and the same
    G-vector list, which is checked, since silently unfolding two blocks
    against different plane-wave bases would be a hard bug to see in the
    output.
    """

    rows: list[NDArray[np.float64]] = []
    n_bands: int | None = None
    for blocks in chunked_kpoint_data:
        block_weights: list[NDArray[np.float64]] = []
        reference: PlaneWaveKPointData | None = None
        for block in blocks:
            if reference is None:
                reference = block
            else:
                if not np.array_equal(
                    reference.folded_supercell_kpoint, block.folded_supercell_kpoint
                ) or not np.array_equal(reference.primitive_kpoint, block.primitive_kpoint):
                    raise ValueError("band chunks of one k-point must share the k-point")
                if not np.array_equal(reference.g_supercell, block.g_supercell):
                    raise ValueError("band chunks of one k-point must share the G-vector list")
            block_weights.append(
                weights_from_coefficients(
                    block.g_supercell,
                    block.coefficients,
                    block.primitive_kpoint,
                    block.folded_supercell_kpoint,
                    transform,
                    tol=tol,
                    component_resolved=component_resolved,
                )
            )
        if not block_weights:
            raise ValueError("each k-point needs at least one band chunk")
        row = np.concatenate(block_weights, axis=0)
        if n_bands is None:
            n_bands = int(row.shape[0])
        elif int(row.shape[0]) != n_bands:
            raise ValueError("all k-points must have the same number of bands")
        rows.append(row)
    if not rows:
        raise ValueError("at least one k-point wavefunction is required")
    return np.stack(rows).astype(float, copy=False)


@dataclass(frozen=True)
class SharedWavefunctionGroup:
    """One supercell wavefunction serving several primitive k-points.

    This is the shape of the work whenever a stored wavefunction is reused: a
    complete fiber unfolded from a single supercell k-point, or a band path
    that revisits the same irreducible k-point under symmetry.  Splitting the
    group into one :class:`PlaneWaveKPointData` per k-point is mathematically
    the same calculation but recomputes ``|c|**2`` -- the dominant cost -- once
    per k-point instead of once per wavefunction.
    """

    primitive_kpoints: NDArray[np.float64]
    folded_supercell_kpoint: NDArray[np.float64]
    g_supercell: NDArray[np.int64]
    coefficients: NDArray[np.complex128]

    def __post_init__(self) -> None:
        primitive = np.asarray(self.primitive_kpoints, dtype=float)
        folded = np.asarray(self.folded_supercell_kpoint, dtype=float)
        g_vectors = np.asarray(self.g_supercell, dtype=int)
        coefficients = np.asarray(self.coefficients, dtype=np.complex128)
        if primitive.ndim != 2 or primitive.shape[1] != 3:
            raise ValueError("primitive_kpoints must have shape (n_kpoints, 3)")
        if folded.shape != (3,):
            raise ValueError("folded_supercell_kpoint must have shape (3,)")
        if g_vectors.ndim != 2 or g_vectors.shape[1] != 3:
            raise ValueError("g_supercell must have shape (n_g, 3)")
        if coefficients.ndim != 3 or coefficients.shape[2] != g_vectors.shape[0]:
            raise ValueError("coefficients must have shape (n_bands, n_components, n_g)")
        object.__setattr__(self, "primitive_kpoints", primitive)
        object.__setattr__(self, "folded_supercell_kpoint", folded)
        object.__setattr__(self, "g_supercell", g_vectors)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def n_bands(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def n_kpoints(self) -> int:
        return int(self.primitive_kpoints.shape[0])

    def expand(self) -> Iterator[PlaneWaveKPointData]:
        """The equivalent one-k-point-at-a-time stream, for reference."""

        for kpoint in self.primitive_kpoints:
            yield PlaneWaveKPointData(
                primitive_kpoint=kpoint,
                folded_supercell_kpoint=self.folded_supercell_kpoint,
                g_supercell=self.g_supercell,
                coefficients=self.coefficients,
            )


def compute_plane_wave_unfolding_weights_shared(
    groups: Iterable[SharedWavefunctionGroup],
    transform: TransformLike,
    *,
    tol: float = 1e-6,
    component_resolved: bool = False,
) -> NDArray[np.float64]:
    """Weights from a stream of shared wavefunctions.

    The rows come out in the order the k-points appear inside the groups, so
    the result is exactly what :func:`compute_plane_wave_unfolding_weights`
    returns for the concatenation of ``group.expand()``.  The difference is
    cost: a group of ``m`` k-points squares the coefficient array once rather
    than ``m`` times, which is the whole of the saving for a complete fiber.

    A group with no k-points contributes no rows; as for the unbatched
    function, at least one row must be produced overall.
    """

    rows: list[NDArray[np.float64]] = []
    n_bands: int | None = None
    n_components: int | None = None
    for group in groups:
        if n_bands is None:
            n_bands = group.n_bands
            n_components = int(group.coefficients.shape[1])
        elif group.n_bands != n_bands:
            raise ValueError("all k-points must have the same number of bands")
        if component_resolved and int(group.coefficients.shape[1]) != n_components:
            raise ValueError(
                "component-resolved weights require all k-points to have the same "
                "number of spinor/polarization components"
            )
        if group.n_kpoints == 0:
            continue
        rows.append(
            shared_weights_from_coefficients(
                group.g_supercell,
                group.coefficients,
                group.primitive_kpoints,
                group.folded_supercell_kpoint,
                transform,
                tol=tol,
                component_resolved=component_resolved,
            )
        )
    if not rows:
        raise ValueError("at least one k-point wavefunction is required")
    return np.concatenate(rows, axis=0).astype(float, copy=False)


def compute_plane_wave_spin_textures(
    kpoint_data: Iterable[PlaneWaveKPointData],
    transform: TransformLike,
    *,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Unfolded spin textures for a noncollinear band path.

    Returns an array of shape ``(n_kpoints, n_bands, 3)``.  Every k-point must
    carry exactly two spinor components; a collinear or spin-polarized file has
    no transverse spin and is rejected rather than silently reported as fully
    polarized along ``z``.

    As for the weights, the iterable is consumed one k-point at a time.
    """

    rows: list[NDArray[np.float64]] = []
    n_bands: int | None = None
    for item in kpoint_data:
        if n_bands is None:
            n_bands = item.n_bands
        elif item.n_bands != n_bands:
            raise ValueError("all k-points must have the same number of bands")
        rows.append(
            spin_texture_from_coefficients(
                item.g_supercell,
                item.coefficients,
                item.primitive_kpoint,
                item.folded_supercell_kpoint,
                transform,
                tol=tol,
            )
        )
    if not rows:
        raise ValueError("at least one k-point wavefunction is required")
    return np.stack(rows).astype(float, copy=False)


@dataclass(frozen=True)
class SpinTextureDiagnostics:
    """Consistency checks for a ``(n_kpoints, n_bands, 3)`` spin-texture table.

    ``max_texture_excess`` is the largest amount by which the length of a
    texture vector exceeds the corresponding unfolding weight.  It must be zero
    up to rounding: a two-component spinor is a pure state for every single
    plane wave, so the triangle inequality bounds the sum
    (``UnfoldLab.norm_spinTexture_le_pairWeight``).  A positive value means the
    weights and the textures were not computed from the same coefficients.
    """

    max_norm: float
    max_texture_excess: float
    n_kpoints: int
    n_bands: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "max_norm": self.max_norm,
            "max_texture_excess": self.max_texture_excess,
            "n_kpoints": self.n_kpoints,
            "n_bands": self.n_bands,
        }


def diagnose_spin_textures(
    textures: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> SpinTextureDiagnostics:
    """Check a spin-texture table against its weights."""

    arr = np.asarray(textures, dtype=float)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("textures must have shape (n_kpoints, n_bands, 3)")
    norms = np.linalg.norm(arr, axis=2)
    if weights is None:
        excess = 0.0
    else:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.shape != arr.shape[:2]:
            raise ValueError("weights must have shape (n_kpoints, n_bands)")
        excess = float(np.max(norms - weight_arr)) if norms.size else 0.0
    return SpinTextureDiagnostics(
        max_norm=float(np.max(norms)) if norms.size else 0.0,
        max_texture_excess=max(excess, 0.0),
        n_kpoints=int(arr.shape[0]),
        n_bands=int(arr.shape[1]),
    )


def truncation_weight_error_bound(missing_fraction: float) -> float:
    """Worst-case weight error of a state with a missing norm fraction.

    If a fraction ``delta`` of the squared norm of a normalized state is absent
    from the stored plane-wave expansion, every weight computed from what is
    stored is within ``delta / (1 - delta)`` of the weight the complete state
    would give.  This is exactly ``UnfoldLab.weight_truncation_bound``; it is a
    worst case over the possible missing components, and nothing sharper can be
    said without them.

    A ``missing_fraction`` of one or more leaves no information at all and
    returns ``inf``.  A negative one -- a stored norm above one, which happens
    for a state normalized to a different convention -- is meaningless as a
    truncation and raises.
    """

    delta = float(missing_fraction)
    if delta < 0.0:
        raise ValueError("missing_fraction must be non-negative")
    if delta >= 1.0:
        return float("inf")
    return delta / (1.0 - delta)


@dataclass(frozen=True)
class StateNormDiagnostics:
    """Norms of the stored plane-wave expansions of a wavefunction set.

    The weights themselves are ratios and do not depend on how the states are
    normalized (``UnfoldLab.weight_smul``), so these numbers are not used in the
    unfolding.  They answer a different question: *is the file complete?*  For a
    norm-conserving calculation the stored norms are one; a value below one
    means part of the state is missing, and
    :attr:`weight_error_bound` turns that into a bound on the weights
    (``UnfoldLab.weight_truncation_bound``).

    A physical shortfall of a few percent is expected for a PAW or ultrasoft
    calculation, whose plane-wave coefficients carry only the pseudo part of the
    state -- the augmentation charge inside the spheres is not in the file and
    cannot be unfolded from it.  A shortfall of tens of percent instead points
    at a reader or a cutoff problem.

    ``n_zero_states`` counts states whose stored norm is numerically zero; those
    get a weight of zero by convention rather than a division by zero, so they
    are invisible in the weight table and worth reporting separately.
    """

    min_norm: float
    max_norm: float
    mean_norm: float
    n_zero_states: int
    n_kpoints: int
    n_bands: int

    @property
    def max_missing_fraction(self) -> float:
        """Largest fraction of the squared norm missing from a stored state."""

        return max(0.0, 1.0 - self.min_norm)

    @property
    def weight_error_bound(self) -> float:
        """Worst-case error of any weight caused by the missing norm."""

        return truncation_weight_error_bound(self.max_missing_fraction)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_norm": self.min_norm,
            "max_norm": self.max_norm,
            "mean_norm": self.mean_norm,
            "n_zero_states": self.n_zero_states,
            "n_kpoints": self.n_kpoints,
            "n_bands": self.n_bands,
            "max_missing_fraction": self.max_missing_fraction,
            "weight_error_bound": self.weight_error_bound,
        }


def diagnose_state_norms(
    norms: NDArray[np.float64],
    *,
    zero_tol: float = 1e-12,
) -> StateNormDiagnostics:
    """Summarize a ``(n_kpoints, n_bands)`` table of stored state norms."""

    arr = np.asarray(norms, dtype=float)
    if arr.ndim != 2:
        raise ValueError("norms must have shape (n_kpoints, n_bands)")
    if arr.size == 0:
        raise ValueError("norms must contain at least one state")
    if float(np.min(arr)) < -zero_tol:
        raise ValueError("state norms must be non-negative")
    return StateNormDiagnostics(
        min_norm=float(np.min(arr)),
        max_norm=float(np.max(arr)),
        mean_norm=float(np.mean(arr)),
        n_zero_states=int(np.count_nonzero(arr <= zero_tol)),
        n_kpoints=int(arr.shape[0]),
        n_bands=int(arr.shape[1]),
    )


def diagnose_weights(
    weights: NDArray[np.float64],
    *,
    tol: float = 1e-8,
) -> WeightDiagnostics:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 2:
        raise ValueError("weights must have shape (n_kpoints, n_bands)")
    out_of_bounds = np.logical_or(arr < -tol, arr > 1.0 + tol)
    return WeightDiagnostics(
        min_weight=float(np.min(arr)) if arr.size else 0.0,
        max_weight=float(np.max(arr)) if arr.size else 0.0,
        out_of_bounds_count=int(np.count_nonzero(out_of_bounds)),
        max_band_sum=float(np.max(np.sum(arr, axis=1))) if arr.size else 0.0,
        n_kpoints=int(arr.shape[0]),
        n_bands=int(arr.shape[1]),
    )


def compare_effective_band_structures(
    first: EffectiveBandStructure,
    second: EffectiveBandStructure,
    *,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Compare two backend outputs at the common EBS level."""

    shape_match = (
        first.kpoints.shape == second.kpoints.shape
        and first.energies.shape == second.energies.shape
        and first.weights.shape == second.weights.shape
    )
    return {
        "shape_match": shape_match,
        "kpoints_close": shape_match
        and bool(np.allclose(first.kpoints, second.kpoints, atol=atol)),
        "energies_close": shape_match
        and bool(np.allclose(first.energies, second.energies, atol=atol)),
        "weights_close": shape_match
        and bool(np.allclose(first.weights, second.weights, atol=atol)),
        "max_weight_delta": float(np.max(np.abs(first.weights - second.weights)))
        if first.weights.shape == second.weights.shape
        else None,
    }


@dataclass(frozen=True)
class FiberSumRuleDiagnostics:
    """Check of the plane-wave unfolding sum rule on a primitive k-path.

    Every supercell plane wave belongs to exactly one of the ``|det T|``
    primitive k-points that fold onto a given supercell k-point, so for each
    band the weights of a complete fiber add up to one
    (``UnfoldLab.IsFiberRepr.sum_weight_eq_one``).

    A band-structure path rarely samples a complete fiber, but a *partial*
    fiber still obeys an inequality: its weights add up to at most one
    (``UnfoldLab.IsFiberRepr.sum_weight_subset_le_one``).  ``max_partial_excess``
    reports how far above one the worst incomplete fiber goes, which is a
    genuine violation and is what makes the diagnostic informative on a path.
    """

    multiplicity: int
    n_fibers: int
    n_complete_fibers: int
    max_deviation: float | None
    worst_fiber: tuple[float, float, float] | None
    #: Largest amount by which an *incomplete* fiber exceeds a weight sum of
    #: one, or ``None`` when the path samples no incomplete fiber.
    max_partial_excess: float | None = None
    worst_partial_fiber: tuple[float, float, float] | None = None

    @property
    def satisfied(self) -> bool:
        return self.max_deviation is not None and self.max_deviation <= 1e-6

    def violated(self, atol: float = 1e-6) -> bool:
        """Any complete fiber off one, or any partial fiber above one."""

        if self.max_deviation is not None and self.max_deviation > atol:
            return True
        return self.max_partial_excess is not None and self.max_partial_excess > atol

    def to_dict(self) -> dict[str, Any]:
        return {
            "multiplicity": self.multiplicity,
            "n_fibers": self.n_fibers,
            "n_complete_fibers": self.n_complete_fibers,
            "max_deviation": self.max_deviation,
            "worst_fiber": list(self.worst_fiber) if self.worst_fiber is not None else None,
            "max_partial_excess": self.max_partial_excess,
            "worst_partial_fiber": list(self.worst_partial_fiber)
            if self.worst_partial_fiber is not None
            else None,
        }


def diagnose_fiber_sum_rule(
    weights: NDArray[np.float64],
    primitive_kpoints: NDArray[np.float64],
    transform: TransformLike,
    *,
    decimals: int = 6,
) -> FiberSumRuleDiagnostics:
    """Verify the unfolding sum rule on the k-points that are actually present.

    Primitive k-points are grouped by their folded supercell image; within a
    group, k-points equal modulo a reciprocal-lattice vector are counted once.
    Only groups containing all ``|det T|`` distinct members can satisfy the sum
    rule with equality, and for those the deviation of the per-band weight sum
    from one is reported.  Incomplete groups are checked against the inequality
    instead: their weights may not add up to more than one.
    """

    weight_arr = np.asarray(weights, dtype=float)
    kpoints = np.asarray(primitive_kpoints, dtype=float)
    if weight_arr.ndim != 2:
        raise ValueError("weights must have shape (n_kpoints, n_bands)")
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError("primitive_kpoints must have shape (n_kpoints, 3)")
    if kpoints.shape[0] != weight_arr.shape[0]:
        raise ValueError("weights and primitive_kpoints must agree on n_kpoints")

    matrix = as_int_array(transform, name="transform")
    if matrix.shape != (3, 3):
        raise ValueError("transform must have shape (3, 3)")
    multiplicity = abs(integer_det3(matrix))

    def _key(vector: NDArray[np.float64]) -> tuple[float, float, float]:
        wrapped = np.mod(np.round(wrap_fractional(vector), decimals), 1.0)
        return (float(wrapped[0]), float(wrapped[1]), float(wrapped[2]))

    fibers: dict[tuple[float, float, float], dict[tuple[float, float, float], int]] = {}
    for index, kpoint in enumerate(kpoints):
        supercell_key = _key(kpoint @ matrix.T)
        members = fibers.setdefault(supercell_key, {})
        members.setdefault(_key(kpoint), index)

    max_deviation: float | None = None
    worst_fiber: tuple[float, float, float] | None = None
    max_partial_excess: float | None = None
    worst_partial_fiber: tuple[float, float, float] | None = None
    n_complete = 0
    for supercell_key, members in fibers.items():
        band_sums = weight_arr[list(members.values()), :].sum(axis=0)
        if len(members) == multiplicity:
            n_complete += 1
            deviation = float(np.max(np.abs(band_sums - 1.0))) if band_sums.size else 0.0
            if max_deviation is None or deviation > max_deviation:
                max_deviation = deviation
                worst_fiber = supercell_key
        else:
            excess = float(np.max(band_sums) - 1.0) if band_sums.size else 0.0
            excess = max(excess, 0.0)
            if max_partial_excess is None or excess > max_partial_excess:
                max_partial_excess = excess
                worst_partial_fiber = supercell_key

    return FiberSumRuleDiagnostics(
        multiplicity=multiplicity,
        n_fibers=len(fibers),
        n_complete_fibers=n_complete,
        max_deviation=max_deviation,
        worst_fiber=worst_fiber,
        max_partial_excess=max_partial_excess,
        worst_partial_fiber=worst_partial_fiber,
    )
