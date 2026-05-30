"""Generic valley definitions.

Valley labels are convention- and material-dependent. The core library stores
explicit user or symmetry-derived definitions; it does not assume any built-in
valley names or coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import as_array3


@dataclass(frozen=True)
class ValleyDefinition:
    """A named reciprocal-space region around a user-defined center."""

    label: str
    center_frac: NDArray[np.float64]
    radius: float
    reference_bz: str | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("valley label must not be empty")
        if self.radius <= 0:
            raise ValueError("valley radius must be positive")
        object.__setattr__(self, "center_frac", as_array3(self.center_frac, name="center_frac"))
        object.__setattr__(self, "radius", float(self.radius))

    @classmethod
    def from_config(cls, label: str, config: dict[str, Any]) -> ValleyDefinition:
        return cls(
            label=label,
            center_frac=config["center_frac"],
            radius=config["radius"],
            reference_bz=config.get("reference_bz"),
        )

    def contains(self, kpoint_frac: ArrayLike) -> bool:
        """Return whether a fractional k-point lies inside this valley region."""

        delta = as_array3(kpoint_frac, name="kpoint_frac") - self.center_frac
        delta -= np.rint(delta)
        return bool(np.linalg.norm(delta) <= self.radius)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "center_frac": self.center_frac.tolist(),
            "radius": self.radius,
            "reference_bz": self.reference_bz,
        }
