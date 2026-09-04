"""CLI for unfolding tight-binding / Wannier model Hamiltonians.

This backend needs no electronic-structure code: the supercell is built from a
primitive model and an integer transformation matrix, diagonalized, and its
eigenstates unfolded onto the primitive Brillouin zone.  It is the quickest way
to check a workflow, a supercell choice, or a defect model end to end, and it
exercises exactly the sum rules proved in
``RequestProject/Unfolding/TightBinding.lean``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray
from rich.console import Console

from unmochan.core.kpoints import cartesian_path_distances, interpolate_segment
from unmochan.core.lcao import (
    overlap_neglect_from_model,
    unfold_lcao_model,
    unfold_lcao_path,
)
from unmochan.core.spectral import EffectiveBandStructure
from unmochan.core.tight_binding import (
    TightBindingModel,
    diagnose_tight_binding_weights,
    supercell_cells,
    unfold_tight_binding_model,
    unfold_tight_binding_path,
    unfold_tight_binding_path_projected,
)
from unmochan.core.transformations import TransformationMatrix
from unmochan.io.model_hamiltonian import (
    Perturbation,
    read_overlap_model,
    read_tight_binding_model,
)
from unmochan.io.serialization import write_ebs_json
from unmochan.io.wannier90 import read_wannier90_hr, read_wannier90_tb

console = Console()

model_app = typer.Typer(
    name="model",
    help="Unfold tight-binding, Wannier, and model-Hamiltonian supercells.",
    no_args_is_help=True,
)


def _parse_matrix(raw: str) -> np.ndarray:
    values = [float(item) for item in raw.replace(",", " ").split()]
    if len(values) != 9:
        raise typer.BadParameter("matrix needs nine numbers in row-major order")
    return np.array(values, dtype=float).reshape(3, 3)


def _parse_kpoint(raw: str) -> np.ndarray:
    values = [float(item) for item in raw.replace(",", " ").split()]
    if len(values) != 3:
        raise typer.BadParameter("a k-point needs three fractional coordinates")
    return np.array(values, dtype=float)


def _parse_onsite(raw: str) -> Perturbation:
    parts = raw.replace(",", " ").split()
    if len(parts) != 3:
        raise typer.BadParameter("an on-site term is cell,orbital,value")
    try:
        return (int(parts[0]), int(parts[1]), complex(float(parts[2])))
    except ValueError as exc:
        raise typer.BadParameter(f"could not read the on-site term {raw!r}") from exc


def _read_model(path: Path, fmt: str) -> tuple[TightBindingModel, list[Perturbation]]:
    resolved = fmt
    if resolved == "auto":
        if path.suffix.lower() == ".json":
            resolved = "json"
        elif path.name.lower().endswith("_tb.dat"):
            resolved = "tb"
        else:
            resolved = "hr"
    if resolved == "json":
        return read_tight_binding_model(path)
    if resolved == "hr":
        return read_wannier90_hr(path), []
    if resolved == "tb":
        return read_wannier90_tb(path).model, []
    raise typer.BadParameter("format must be auto, json, hr, or tb")


def _read_overlap(path: Path, fmt: str, n_orbitals: int) -> TightBindingModel | None:
    """The optional non-orthogonal basis of a JSON model file.

    Only the JSON format can carry one: a Wannier90 ``hr``/``tb`` file describes
    Wannier functions, which are orthonormal by construction.
    """

    if fmt not in ("auto", "json"):
        return None
    if fmt == "auto" and path.suffix.lower() != ".json":
        return None
    overlap = read_overlap_model(path)
    if overlap is not None and overlap.n_orbitals != n_orbitals:
        raise typer.BadParameter(
            "the overlaps block must have the same number of orbitals as the hoppings"
        )
    return overlap


def _parse_group(raw: str) -> list[int]:
    """Parse ``0,1,4`` into a list of orbital indices."""

    parts = [item for item in raw.replace(",", " ").split() if item]
    if not parts:
        raise typer.BadParameter("an orbital group needs at least one index")
    try:
        return [int(item) for item in parts]
    except ValueError as exc:
        raise typer.BadParameter(f"could not read the orbital group {raw!r}") from exc


@model_app.command("unfold")
def model_unfold_command(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            exists=True,
            readable=True,
            help="Model Hamiltonian: JSON, or a Wannier90 seedname_hr.dat.",
        ),
    ],
    matrix: Annotated[
        str,
        typer.Option("--matrix", help="Transformation matrix, nine numbers row-major."),
    ],
    kpoint: Annotated[
        list[str],
        typer.Option(
            "--kpoint",
            help="Supercell k-point in primitive fractional coordinates, x,y,z.",
        ),
    ],
    model_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Model file format: auto (by extension), json, or hr (Wannier90).",
        ),
    ] = "auto",
    onsite: Annotated[
        list[str] | None,
        typer.Option(
            "--onsite",
            help="Extra supercell on-site term, cell,orbital,value. Repeatable.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--json", help="Write the unfolded band structure as JSON."),
    ] = None,
) -> None:
    """Diagonalize the supercell model and unfold it onto primitive k-points.

    Each supercell k-point expands into the |det T| primitive k-points that fold
    onto it, and every supercell eigenstate is given its weight on each of them.
    The weights of one state add up to one over the fiber, and the weights of
    all states add up to the number of primitive bands at each k-point; both are
    reported as a diagnostic.
    """

    transform = TransformationMatrix(_parse_matrix(matrix))
    tb_model, perturbations = _read_model(model, model_format)
    perturbations = [*perturbations, *(_parse_onsite(item) for item in onsite or ())]
    points = [_parse_kpoint(item) for item in kpoint]
    if not points:
        raise typer.BadParameter("at least one --kpoint is required")

    overlap = _read_overlap(model, model_format, tb_model.n_orbitals)
    if overlap is None:
        kpoints, energies, weights = unfold_tight_binding_model(
            tb_model, transform, points, perturbations=perturbations
        )
    else:
        kpoints, energies, weights = unfold_lcao_model(
            tb_model, overlap, transform, points, perturbations=perturbations
        )
    report = diagnose_tight_binding_weights(
        weights[: transform.multiplicity], n_orbitals=tb_model.n_orbitals
    )

    payload: dict[str, object] = {
        "multiplicity": transform.multiplicity,
        "n_orbitals": tb_model.n_orbitals,
        "basis": "orthonormal" if overlap is None else "non-orthogonal (LCAO)",
        "cells": supercell_cells(transform).tolist(),
        "kpoints": kpoints.tolist(),
        "energies": energies.tolist(),
        "weights": weights.tolist(),
        "sum_rules": report.to_dict(),
    }
    if overlap is not None:
        payload["overlap_neglect"] = overlap_neglect_from_model(
            tb_model, overlap, transform, points[0], perturbations=perturbations
        ).to_dict()
    console.print_json(json.dumps(payload))
    if report.violated(1e-8):
        console.print("[yellow]Warning:[/yellow] unfolding sum rules violated")

    if output is not None:
        ebs = EffectiveBandStructure(
            kpoints=kpoints,
            energies=energies,
            weights=weights,
            metadata={
                "source_code": "tight-binding model",
                "transformation_matrix": transform.matrix.tolist(),
                "n_orbitals": tb_model.n_orbitals,
                "basis": "orthonormal" if overlap is None else "non-orthogonal (LCAO)",
            },
        )
        write_ebs_json(output, ebs)
        console.print(f"Wrote {output}")


def _parse_path(raw: str) -> list[NDArray[np.float64]]:
    """Parse ``0,0,0:0.5,0,0:0.5,0.5,0`` into a list of corner k-points."""

    corners = [chunk for chunk in raw.split(":") if chunk.strip()]
    if len(corners) < 2:
        raise typer.BadParameter("a path needs at least two corners, k1:k2[:k3...]")
    return [_parse_kpoint(corner) for corner in corners]


@model_app.command("bands")
def model_bands_command(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            exists=True,
            readable=True,
            help="Model Hamiltonian: JSON, or a Wannier90 seedname_hr.dat.",
        ),
    ],
    matrix: Annotated[
        str,
        typer.Option("--matrix", help="Transformation matrix, nine numbers row-major."),
    ],
    path: Annotated[
        str,
        typer.Option(
            "--path",
            help="Primitive band path as corners k1:k2[:k3...], each x,y,z.",
        ),
    ],
    points: Annotated[
        int,
        typer.Option("--points", min=2, help="k-points per path segment."),
    ] = 21,
    model_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Model file format: auto (by extension), json, hr or tb (Wannier90).",
        ),
    ] = "auto",
    onsite: Annotated[
        list[str] | None,
        typer.Option(
            "--onsite",
            help="Extra supercell on-site term, cell,orbital,value. Repeatable.",
        ),
    ] = None,
    group: Annotated[
        list[str] | None,
        typer.Option(
            "--group",
            help=(
                "Orbital group for a fat band, e.g. 0,1. Repeatable; the groups "
                "are written alongside the total weights."
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--json", help="Write the unfolded band structure as JSON."),
    ] = None,
) -> None:
    """Unfold a supercell model along a path in the primitive Brillouin zone.

    This is the effective band structure as it is usually plotted: one column of
    weighted eigenvalues per primitive k-point of the path.  A perfect crystal
    reproduces its primitive bands with weight one and leaves the folded copies
    at weight zero; a defect or a distortion smears that weight, and how much it
    smears is the physics being asked for.
    """

    transform = TransformationMatrix(_parse_matrix(matrix))
    tb_model, perturbations = _read_model(model, model_format)
    perturbations = [*perturbations, *(_parse_onsite(item) for item in onsite or ())]

    corners = _parse_path(path)
    kpoints: list[NDArray[np.float64]] = []
    for start, end in zip(corners, corners[1:], strict=False):
        segment = [point.fractional for point in interpolate_segment(start, end, n_points=points)]
        kpoints.extend(segment if not kpoints else segment[1:])
    grid = np.array(kpoints, dtype=float)

    overlap = _read_overlap(model, model_format, tb_model.n_orbitals)
    groups = [_parse_group(item) for item in group or ()]
    projected: NDArray[np.float64] | None = None
    if groups and overlap is not None:
        raise typer.BadParameter(
            "orbital-projected fat bands are not defined for a non-orthogonal "
            "basis: the projection of a state onto a group of overlapping "
            "orbitals is not the sum of the group's squared amplitudes"
        )
    if groups:
        try:
            energies, weights, projected = unfold_tight_binding_path_projected(
                tb_model, transform, grid, groups, perturbations=perturbations
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    elif overlap is not None:
        energies, weights = unfold_lcao_path(
            tb_model, overlap, transform, grid, perturbations=perturbations
        )
    else:
        energies, weights = unfold_tight_binding_path(
            tb_model, transform, grid, perturbations=perturbations
        )
    band_deviation = float(np.max(np.abs(weights.sum(axis=1) - tb_model.n_orbitals)))

    metadata: dict[str, object] = {
        "source_code": "tight-binding model",
        "transformation_matrix": transform.matrix.tolist(),
        "n_orbitals": tb_model.n_orbitals,
        "max_band_deviation": band_deviation,
        "basis": "orthonormal" if overlap is None else "non-orthogonal (LCAO)",
    }
    if projected is not None:
        metadata["orbital_groups"] = groups
        metadata["projected_weights"] = projected.tolist()
    ebs = EffectiveBandStructure(
        kpoints=grid,
        energies=energies,
        weights=weights,
        distances=cartesian_path_distances(grid),
        metadata=metadata,
    )
    summary: dict[str, object] = {
        "basis": "orthonormal" if overlap is None else "non-orthogonal (LCAO)",
        "n_kpoints": int(grid.shape[0]),
        "n_states": int(energies.shape[1]),
        "n_orbitals": tb_model.n_orbitals,
        "multiplicity": transform.multiplicity,
        "max_band_deviation": band_deviation,
        "energy_range": [float(energies.min()), float(energies.max())],
    }
    if projected is not None:
        summary["orbital_groups"] = groups
        # How much of the total weight each group accounts for over the path.
        total = float(weights.sum())
        summary["group_shares"] = [
            float(part.sum() / total) if total > 0.0 else 0.0 for part in projected
        ]
    console.print_json(json.dumps(summary))
    if band_deviation > 1e-8:
        console.print("[yellow]Warning:[/yellow] the band sum rule is violated")
    if output is not None:
        write_ebs_json(output, ebs)
        console.print(f"Wrote {output}")
