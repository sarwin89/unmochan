"""Reading primitive-cell force constants from JSON.

The format mirrors the model-Hamiltonian one, because the two problems are the
same problem with a different name:

.. code-block:: json

    {
      "masses": [28.085],
      "force_constants": [
        {"cell": [0, 0, 0], "matrix": [[2.0, 0, 0], [0, 0, 0], [0, 0, 0]]},
        {"cell": [1, 0, 0], "matrix": [[-1.0, 0, 0], [0, 0, 0], [0, 0, 0]]}
      ],
      "symmetrize": true,
      "enforce_acoustic_sum_rule": false,
      "perturbations": [{"cell": 1, "mode": 0, "value": 0.5}]
    }

``matrix`` is the ``(3 n_atoms, 3 n_atoms)`` block ``Phi(cell)`` in the ordering
``3 * atom + direction``.  With ``"symmetrize": true`` (the default) the partner
``Phi(-D) = Phi(D)^T`` is filled in automatically, so only one member of each
pair has to be listed.  ``enforce_acoustic_sum_rule`` is *off* by default: a set
of force constants that violates it is a physics problem the user should see,
not something a reader should quietly patch up.

``perturbations`` are additions to the mass-weighted on-site terms of a single
mode of a single primitive cell of the supercell — a bond stiffened by a defect,
or a distortion frozen into one cell.

A *mass* defect is a different thing and gets its own optional block,

.. code-block:: json

    {"site_masses": [{"cell": 1, "atom": 0, "mass": 8.0}]}

because changing a mass rescales a whole row and column of the dynamical matrix
instead of shifting one diagonal entry, and no ``perturbations`` entry can
imitate that (``UnfoldLab.mass_change_not_diagonal_shift``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unfoldlab.core.phonons import ForceConstantModel

__all__ = [
    "PhononProblem",
    "PhononPerturbation",
    "PhononSubstitution",
    "read_force_constant_model",
    "read_force_constant_problem",
]

PhononPerturbation = tuple[int, int, float]
PhononSubstitution = tuple[int, int, float]


@dataclass(frozen=True)
class PhononProblem:
    """A force-constant model together with its supercell defects.

    ``perturbations`` are on-site (force-constant) terms and ``substitutions``
    are ``(cell, atom, mass)`` mass defects; the two act on the dynamical matrix
    in genuinely different ways, which is why they are kept apart.
    """

    model: ForceConstantModel
    perturbations: list[PhononPerturbation] = field(default_factory=list)
    substitutions: list[PhononSubstitution] = field(default_factory=list)


def read_force_constant_model(
    path: str | Path,
) -> tuple[ForceConstantModel, list[PhononPerturbation]]:
    """Read a force-constant model and its on-site supercell perturbations.

    Kept for callers that do not handle mass defects; use
    :func:`read_force_constant_problem` to get the substitutions as well.
    """

    problem = read_force_constant_problem(path)
    return problem.model, problem.perturbations


def read_force_constant_problem(path: str | Path) -> PhononProblem:
    """Read a force-constant model with its perturbations and mass defects."""

    payload: Any = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("a force-constant model must be a JSON object")
    if "masses" not in payload:
        raise ValueError("a force-constant model needs 'masses'")
    masses = np.asarray(payload["masses"], dtype=float).reshape(-1)
    size = 3 * masses.size

    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    for entry in payload.get("force_constants", []):
        if not isinstance(entry, dict):
            raise ValueError("each force-constant entry must be a JSON object")
        cell = tuple(int(component) for component in entry["cell"])
        if len(cell) != 3:
            raise ValueError("'cell' must be an integer 3-vector")
        matrix = np.asarray(entry["matrix"], dtype=float)
        if matrix.shape != (size, size):
            raise ValueError(f"force-constant blocks must be {size}x{size}")
        blocks[cell] = blocks.get(cell, np.zeros((size, size))) + matrix

    model = ForceConstantModel(masses, blocks)
    if payload.get("symmetrize", True):
        model = model.symmetrized()
    if payload.get("enforce_acoustic_sum_rule", False):
        model = model.enforce_acoustic_sum_rule()

    perturbations: list[PhononPerturbation] = []
    for entry in payload.get("perturbations", []):
        if not isinstance(entry, dict):
            raise ValueError("each perturbation must be a JSON object")
        perturbations.append((int(entry["cell"]), int(entry["mode"]), float(entry["value"])))

    substitutions: list[PhononSubstitution] = []
    for entry in payload.get("site_masses", []):
        if not isinstance(entry, dict):
            raise ValueError("each site-mass entry must be a JSON object")
        mass = float(entry["mass"])
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("a site mass must be finite and positive")
        substitutions.append((int(entry["cell"]), int(entry["atom"]), mass))

    return PhononProblem(model, perturbations, substitutions)
