"""Effective band structure and spectral-function containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

BroadeningKind = Literal["gaussian", "lorentzian"]

#: Default for the two derived fields of :class:`EffectiveBandStructure`.  They
#: are optional *arguments* but never optional *attributes*: ``__post_init__``
#: fills both in, weights with ones and distances with the k-point index.  The
#: cast records that invariant, so every reader of ``ebs.weights`` or
#: ``ebs.distances`` gets an array rather than a union it has to re-check.
_DERIVED: NDArray[np.float64] = cast("NDArray[np.float64]", None)


def default_weights(energies: ArrayLike) -> NDArray[np.float64]:
    """The weights of a band structure that carries none: one per state."""

    return np.ones_like(np.asarray(energies, dtype=float))


def default_distances(n_kpoints: int) -> NDArray[np.float64]:
    """The path abscissa of a band structure that carries none: the index."""

    return np.arange(int(n_kpoints), dtype=float)


@dataclass(frozen=True)
class EffectiveBandStructure:
    """Band energies with unfolding or projection weights."""

    kpoints: NDArray[np.float64]
    energies: NDArray[np.float64]
    weights: NDArray[np.float64] = _DERIVED
    distances: NDArray[np.float64] = _DERIVED
    reference_energy: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kpoints = np.asarray(self.kpoints, dtype=float)
        energies = np.asarray(self.energies, dtype=float)
        if kpoints.ndim != 2 or kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (n_kpoints, 3)")
        if energies.ndim != 2 or energies.shape[0] != kpoints.shape[0]:
            raise ValueError("energies must have shape (n_kpoints, n_bands)")

        if self.weights is None:
            weights = default_weights(energies)
        else:
            weights = np.asarray(self.weights, dtype=float)
            if weights.shape != energies.shape:
                raise ValueError("weights must have the same shape as energies")

        if self.distances is None:
            distances = default_distances(kpoints.shape[0])
        else:
            distances = np.asarray(self.distances, dtype=float)
            if distances.shape != (kpoints.shape[0],):
                raise ValueError("distances must have shape (n_kpoints,)")

        object.__setattr__(self, "kpoints", kpoints)
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "reference_energy", float(self.reference_energy))

    @property
    def n_kpoints(self) -> int:
        return int(self.kpoints.shape[0])

    @property
    def n_bands(self) -> int:
        return int(self.energies.shape[1])

    def shifted_energies(self) -> NDArray[np.float64]:
        return self.energies - self.reference_energy

    def adaptive_widths(
        self,
        *,
        scale: float = 1.0,
        minimum: float,
        maximum: float | None = None,
    ) -> NDArray[np.float64]:
        """Per-state broadening widths proportional to the local band velocity.

        A band that sweeps a lot of energy between neighbouring k-points cannot
        be resolved more finely than that sweep, while a flat band can.  The
        width of state ``(k, n)`` is therefore

        ``w = scale * |dE_n/ds| * ds``,

        the energy the band covers over the local spacing ``ds`` of the path
        abscissa, clipped from below by ``minimum`` (which must be positive: a
        zero width is not a kernel) and optionally from above by ``maximum``.
        The derivative is a central difference in the path coordinate, so the
        result has the same units as the energies.

        A single k-point, or a path whose abscissae do not increase, has no
        meaningful velocity; ``minimum`` is then returned everywhere.
        """

        if minimum <= 0:
            raise ValueError("minimum broadening must be positive")
        if maximum is not None and maximum < minimum:
            raise ValueError("maximum broadening must not be below the minimum")
        if scale < 0:
            raise ValueError("scale must be non-negative")

        energies = self.energies
        distances = self.distances
        spacing = np.diff(distances)
        if self.n_kpoints < 2 or not np.all(spacing > 0.0):
            widths = np.full(energies.shape, float(minimum))
        else:
            derivative = np.gradient(energies, distances, axis=0)
            local_spacing = np.gradient(distances)
            widths = scale * np.abs(derivative) * local_spacing[:, None]
            widths = np.maximum(widths, float(minimum))
        if maximum is not None:
            widths = np.minimum(widths, float(maximum))
        return widths.astype(float, copy=False)

    def spectral_function(
        self,
        energy_grid: ArrayLike,
        *,
        broadening: float | ArrayLike,
        kind: BroadeningKind = "gaussian",
    ) -> NDArray[np.float64]:
        """Return A(k, E) on a ``(n_kpoints, n_energies)`` grid.

        ``broadening`` is either a single positive width or a per-state table
        broadcastable to ``(n_kpoints, n_bands)`` -- for instance the output of
        :meth:`adaptive_widths`.  Both kernels are normalized probability
        densities *for every individual state*, so integrating the result over
        the energy axis returns the total weight of each k-point,
        ``weights[ik].sum()``, whether the width is fixed or adaptive.  Those
        statements are proved as ``UnfoldLab.integral_spectralFunction`` and
        ``UnfoldLab.integral_adaptiveSpectralFunction``.
        """

        if kind not in ("gaussian", "lorentzian"):
            raise ValueError(f"unsupported broadening kind: {kind}")
        grid = np.asarray(energy_grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError("energy_grid must be one-dimensional")
        energies = self.shifted_energies()

        widths = np.asarray(broadening, dtype=float)
        if widths.ndim == 0:
            widths = np.broadcast_to(widths, energies.shape)
        else:
            try:
                widths = np.broadcast_to(widths, energies.shape)
            except ValueError as error:
                raise ValueError(
                    "broadening must be a scalar or broadcastable to "
                    f"(n_kpoints, n_bands) = {energies.shape}"
                ) from error
        if not np.all(widths > 0.0):
            raise ValueError("broadening must be positive")

        intensity = np.zeros((self.n_kpoints, grid.size), dtype=float)
        for ik in range(self.n_kpoints):
            width = widths[ik][:, None]
            delta = grid[None, :] - energies[ik, :, None]
            if kind == "gaussian":
                kernel = np.exp(-0.5 * (delta / width) ** 2)
                kernel /= width * np.sqrt(2.0 * np.pi)
            else:
                kernel = width / np.pi / (delta**2 + width**2)
            # One matrix-vector product instead of a broadcast multiply plus sum.
            intensity[ik] = self.weights[ik] @ kernel
        return intensity

    def to_dict(self) -> dict[str, Any]:
        return {
            "kpoints": self.kpoints.tolist(),
            "energies": self.energies.tolist(),
            "weights": self.weights.tolist(),
            "distances": self.distances.tolist(),
            "reference_energy": self.reference_energy,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EffectiveBandStructure:
        kpoints = np.asarray(data["kpoints"], dtype=float)
        energies = np.asarray(data["energies"], dtype=float)
        weights = data.get("weights")
        distances = data.get("distances")
        return cls(
            kpoints=kpoints,
            energies=energies,
            weights=(
                default_weights(energies) if weights is None else np.asarray(weights, dtype=float)
            ),
            distances=(
                default_distances(kpoints.shape[0])
                if distances is None
                else np.asarray(distances, dtype=float)
            ),
            reference_energy=data.get("reference_energy", 0.0),
            metadata=data.get("metadata", {}),
        )


def normalized_kpoint_weights(
    structure: EffectiveBandStructure,
    kpoint_weights: ArrayLike | None,
) -> NDArray[np.float64]:
    """Brillouin-zone weights of the k-points of ``structure``, summing to one.

    ``None`` means uniform.  Every zone average in the package -- the electron
    count of :mod:`unmochan.core.fermi`, the density of states of
    :mod:`unmochan.core.dos` -- goes through this function, so they all treat a
    weight list the same way and all reject the same malformed ones.  Note that
    uniform weights are only correct for a uniform mesh: a high-symmetry path is
    not a zone sample at all, whatever weights it is given.
    """

    if kpoint_weights is None:
        return np.full(structure.n_kpoints, 1.0 / structure.n_kpoints)
    weights = np.asarray(kpoint_weights, dtype=float)
    if weights.shape != (structure.n_kpoints,):
        raise ValueError("kpoint_weights must have one entry per k-point")
    if np.any(weights < 0.0):
        raise ValueError("kpoint_weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("kpoint_weights must not sum to zero")
    return np.asarray(weights / total, dtype=np.float64)
