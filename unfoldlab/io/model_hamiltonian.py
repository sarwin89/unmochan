"""Reading tight-binding / Wannier model Hamiltonians from JSON.

The format is deliberately small, because a model Hamiltonian is the one
"backend" that needs no electronic-structure code at all:

.. code-block:: json

    {
      "n_orbitals": 2,
      "hoppings": [
        {"cell": [0, 0, 0], "matrix": [[0.0, -0.5], [-0.5, 0.0]]},
        {"cell": [1, 0, 0], "matrix": [[0.0, 0.0], [-1.0, 0.0]]}
      ],
      "perturbations": [{"cell": 0, "orbital": 1, "value": 0.8}],
      "hermitize": true
    }

Matrix entries are real numbers, or ``[re, im]`` pairs for a complex amplitude.
``cell`` is an integer primitive translation and ``matrix[a][b]`` is
``<0 a| H |cell b>``.  With ``"hermitize": true`` (the default) the conjugate
hoppings ``t(-D) = t(D)^dagger`` are filled in automatically, so only one member
of each pair has to be listed.

``perturbations`` are on-site terms added to a single orbital of a single
primitive cell of the supercell, indexed as in
:func:`unfoldlab.core.tight_binding.supercell_cells`.  They are what turns a
perfect crystal into a defect, distortion, or alloy supercell.

An optional ``"overlaps"`` block, in exactly the same layout as ``hoppings``,
declares a **non-orthogonal** basis: ``overlaps[D][a][b] = <phi_{a,0} |
phi_{b,D}>``.  It is read by :func:`read_overlap_model`, and the unfolding then
has to solve ``H c = E S c`` and weight the states with the overlap-aware
formula of :mod:`unfoldlab.core.lcao`.  Omit it and the basis is orthonormal,
``S = 1``, which is the assumption the plain tight-binding path makes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from unfoldlab.core.tight_binding import TightBindingModel

__all__ = ["Perturbation", "read_overlap_model", "read_tight_binding_model"]

Perturbation = tuple[int, int, complex]


def _complex_entry(value: Any) -> complex:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("a complex entry must be given as [re, im]")
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, str):
        return complex(value)
    return complex(float(value))


def _read_blocks(
    payload: dict[str, Any], key: str, n_orbitals: int
) -> dict[tuple[int, int, int], np.ndarray]:
    """Read a ``hoppings``-shaped list of ``{cell, matrix}`` entries."""

    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    for entry in payload.get(key, ()):
        cell = tuple(int(component) for component in entry["cell"])
        if len(cell) != 3:
            raise ValueError(f"{key} cells must be integer 3-vectors")
        block = np.array(
            [[_complex_entry(value) for value in row] for row in entry["matrix"]],
            dtype=np.complex128,
        )
        if block.shape != (n_orbitals, n_orbitals):
            raise ValueError(
                f"{key} matrix for cell {list(cell)} must be {n_orbitals}x{n_orbitals}"
            )
        blocks[cell] = blocks.get(cell, np.zeros_like(block)) + block
    return blocks


def read_overlap_model(path: str | Path) -> TightBindingModel | None:
    """Read the optional ``overlaps`` block of a model file.

    Returns ``None`` when the file declares no overlaps, which means the basis
    is orthonormal and the ordinary tight-binding unfolding applies.  Otherwise
    the returned model carries ``<phi_{a,0} | phi_{b,D}>`` and is hermitized on
    the same rule as the hoppings, since an overlap matrix is Hermitian.

    A file that lists overlaps but no on-site block is rejected: an LCAO basis
    always has ``<phi_a | phi_a> > 0``, and a missing ``[0, 0, 0]`` entry is far
    more likely a mistake than a deliberately singular basis.
    """

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("a model file must contain a JSON object")
    if "overlaps" not in payload:
        return None
    n_orbitals = int(payload["n_orbitals"])
    blocks = _read_blocks(payload, "overlaps", n_orbitals)
    if not blocks:
        return None
    if (0, 0, 0) not in blocks:
        raise ValueError(
            "an overlaps block must include the on-site cell [0, 0, 0]; "
            "without it the basis functions have zero norm"
        )
    model = TightBindingModel(n_orbitals, blocks)
    if bool(payload.get("hermitize", True)):
        model = model.hermitized()
    return model


def read_tight_binding_model(
    path: str | Path,
) -> tuple[TightBindingModel, list[Perturbation]]:
    """Read a model Hamiltonian and its supercell perturbations."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("a model file must contain a JSON object")

    try:
        n_orbitals = int(payload["n_orbitals"])
    except KeyError as exc:  # pragma: no cover - message is the point
        raise ValueError("a model file must define n_orbitals") from exc

    hoppings = _read_blocks(payload, "hoppings", n_orbitals)

    model = TightBindingModel(n_orbitals, hoppings)
    if bool(payload.get("hermitize", True)):
        model = model.hermitized()

    perturbations: list[Perturbation] = []
    for entry in payload.get("perturbations", ()):
        perturbations.append(
            (
                int(entry["cell"]),
                int(entry["orbital"]),
                _complex_entry(entry["value"]),
            )
        )
    return model, perturbations
