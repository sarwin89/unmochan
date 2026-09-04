"""Structure and lattice containers.

The lattice convention is row-vector based: each row is one real-space lattice
vector in Cartesian coordinates. Fractional coordinates multiply the lattice
matrix from the left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.numerics import as_matrix3


@dataclass(frozen=True)
class Structure:
    """Minimal periodic structure representation used by the core algorithms."""

    lattice: NDArray[np.float64]
    species: tuple[str, ...]
    frac_coords: NDArray[np.float64]
    name: str | None = None

    def __post_init__(self) -> None:
        lattice = as_matrix3(self.lattice, name="lattice")
        frac_coords = np.asarray(self.frac_coords, dtype=float)
        if frac_coords.ndim != 2 or frac_coords.shape[1] != 3:
            raise ValueError("frac_coords must have shape (n_sites, 3)")
        if len(self.species) != frac_coords.shape[0]:
            raise ValueError("species length must match number of fractional coordinates")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "species", tuple(self.species))
        object.__setattr__(self, "frac_coords", frac_coords)

    @property
    def n_sites(self) -> int:
        return len(self.species)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def reciprocal_lattice(self) -> NDArray[np.float64]:
        """Return reciprocal lattice rows with the 2*pi convention."""

        return np.asarray(2.0 * np.pi * np.linalg.inv(self.lattice).T, dtype=np.float64)

    @property
    def cart_coords(self) -> NDArray[np.float64]:
        return self.frac_coords @ self.lattice

    @classmethod
    def from_cartesian(
        cls,
        lattice: ArrayLike,
        species: list[str] | tuple[str, ...],
        cart_coords: ArrayLike,
        *,
        name: str | None = None,
    ) -> Structure:
        lattice_arr = as_matrix3(lattice, name="lattice")
        cart_arr = np.asarray(cart_coords, dtype=float)
        frac = cart_arr @ np.linalg.inv(lattice_arr)
        return cls(lattice_arr, tuple(species), frac, name=name)

    def formula_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for symbol in self.species:
            counts[symbol] = counts.get(symbol, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lattice": self.lattice.tolist(),
            "species": list(self.species),
            "frac_coords": self.frac_coords.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Structure:
        return cls(
            lattice=data["lattice"],
            species=tuple(data["species"]),
            frac_coords=data["frac_coords"],
            name=data.get("name"),
        )
