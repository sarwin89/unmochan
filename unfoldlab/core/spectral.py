"""Effective band structure and spectral-function containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

BroadeningKind = Literal["gaussian", "lorentzian"]


@dataclass(frozen=True)
class EffectiveBandStructure:
    """Band energies with unfolding or projection weights."""

    kpoints: NDArray[np.float64]
    energies: NDArray[np.float64]
    weights: NDArray[np.float64] | None = None
    distances: NDArray[np.float64] | None = None
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
            weights = np.ones_like(energies)
        else:
            weights = np.asarray(self.weights, dtype=float)
            if weights.shape != energies.shape:
                raise ValueError("weights must have the same shape as energies")

        if self.distances is None:
            distances = np.arange(kpoints.shape[0], dtype=float)
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

    def spectral_function(
        self,
        energy_grid: ArrayLike,
        *,
        broadening: float,
        kind: BroadeningKind = "gaussian",
    ) -> NDArray[np.float64]:
        """Return A(k, E) on ``(n_kpoints, n_energies)`` grid."""

        if broadening <= 0:
            raise ValueError("broadening must be positive")
        grid = np.asarray(energy_grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError("energy_grid must be one-dimensional")
        energies = self.shifted_energies()
        intensity = np.zeros((self.n_kpoints, grid.size), dtype=float)

        for ik in range(self.n_kpoints):
            delta = grid[None, :] - energies[ik, :, None]
            if kind == "gaussian":
                kernel = np.exp(-0.5 * (delta / broadening) ** 2)
                kernel /= broadening * np.sqrt(2.0 * np.pi)
            elif kind == "lorentzian":
                kernel = broadening / np.pi / (delta**2 + broadening**2)
            else:
                raise ValueError(f"unsupported broadening kind: {kind}")
            intensity[ik] = np.sum(self.weights[ik, :, None] * kernel, axis=0)
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
        return cls(
            kpoints=data["kpoints"],
            energies=data["energies"],
            weights=data.get("weights"),
            distances=data.get("distances"),
            reference_energy=data.get("reference_energy", 0.0),
            metadata=data.get("metadata", {}),
        )
