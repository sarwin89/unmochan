"""CLI for unfolding phonon supercells from primitive force constants.

Phonon unfolding is the same discrete Fourier duality as electronic unfolding
with ``(atom, Cartesian direction)`` in place of the orbital index, so this
command reuses the tight-binding kernel and the sum rules proved in
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

from unfoldlab.core.kpoints import cartesian_path_distances, interpolate_segment
from unfoldlab.core.phonons import supercell_site_masses, unfold_phonon_path
from unfoldlab.core.spectral import EffectiveBandStructure
from unfoldlab.core.transformations import TransformationMatrix
from unfoldlab.io.force_constants import (
    PhononPerturbation,
    PhononSubstitution,
    read_force_constant_problem,
)
from unfoldlab.io.serialization import write_ebs_json

console = Console()

phonon_app = typer.Typer(
    name="phonon",
    help="Unfold phonon supercells given primitive force constants.",
    no_args_is_help=True,
)


def _parse_matrix(raw: str) -> NDArray[np.float64]:
    values = [float(item) for item in raw.replace(",", " ").split()]
    if len(values) != 9:
        raise typer.BadParameter("matrix needs nine numbers in row-major order")
    return np.array(values, dtype=float).reshape(3, 3)


def _parse_kpoint(raw: str) -> NDArray[np.float64]:
    values = [float(item) for item in raw.replace(",", " ").split()]
    if len(values) != 3:
        raise typer.BadParameter("a k-point needs three fractional coordinates")
    return np.array(values, dtype=float)


def _parse_path(raw: str) -> list[NDArray[np.float64]]:
    corners = [chunk for chunk in raw.split(":") if chunk.strip()]
    if len(corners) < 2:
        raise typer.BadParameter("a path needs at least two corners, k1:k2[:k3...]")
    return [_parse_kpoint(corner) for corner in corners]


def _parse_mode_term(raw: str) -> PhononPerturbation:
    parts = raw.replace(",", " ").split()
    if len(parts) != 3:
        raise typer.BadParameter("a mode term is cell,mode,value")
    try:
        return (int(parts[0]), int(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise typer.BadParameter(f"could not read the mode term {raw!r}") from exc


def _parse_site_mass(raw: str) -> PhononSubstitution:
    parts = raw.replace(",", " ").split()
    if len(parts) != 3:
        raise typer.BadParameter("a site mass is cell,atom,mass")
    try:
        mass = float(parts[2])
    except ValueError as exc:
        raise typer.BadParameter(f"could not read the site mass {raw!r}") from exc
    if not np.isfinite(mass) or mass <= 0.0:
        raise typer.BadParameter("a site mass must be finite and positive")
    try:
        return (int(parts[0]), int(parts[1]), mass)
    except ValueError as exc:
        raise typer.BadParameter(f"could not read the site mass {raw!r}") from exc


@phonon_app.command("bands")
def phonon_bands_command(
    model: Annotated[
        Path,
        typer.Option("--model", exists=True, readable=True, help="Force-constant model JSON."),
    ],
    matrix: Annotated[
        str,
        typer.Option("--matrix", help="Transformation matrix, nine numbers row-major."),
    ],
    path: Annotated[
        str,
        typer.Option("--path", help="Primitive band path as corners k1:k2[:k3...], each x,y,z."),
    ],
    points: Annotated[int, typer.Option("--points", min=2, help="k-points per path segment.")] = 21,
    mode_term: Annotated[
        list[str] | None,
        typer.Option(
            "--mode-term",
            help="Extra mass-weighted on-site term, cell,mode,value. Repeatable.",
        ),
    ] = None,
    site_mass: Annotated[
        list[str] | None,
        typer.Option(
            "--site-mass",
            help="Mass defect (isotope, substitution), cell,atom,mass. Repeatable.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--json", help="Write the unfolded phonon bands as JSON."),
    ] = None,
) -> None:
    """Unfold phonons of a supercell along a path in the primitive zone.

    Frequencies are reported as ``sign(omega^2) sqrt(|omega^2|)``, so an
    unstable mode of a distorted supercell shows up as a negative frequency
    rather than disappearing.  The acoustic sum rule of the input force
    constants is reported too, because acoustic branches that miss zero at Gamma
    are almost always its fault rather than the unfolding's.
    """

    transform = TransformationMatrix.from_values(_parse_matrix(matrix))
    problem = read_force_constant_problem(model)
    force_model = problem.model
    perturbations = [
        *problem.perturbations,
        *(_parse_mode_term(item) for item in mode_term or ()),
    ]
    substitutions = [
        *problem.substitutions,
        *(_parse_site_mass(item) for item in site_mass or ()),
    ]
    try:
        site_masses = (
            supercell_site_masses(force_model, transform, substitutions) if substitutions else None
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    corners = _parse_path(path)
    kpoints: list[NDArray[np.float64]] = []
    for start, end in zip(corners, corners[1:], strict=False):
        segment = [point.fractional for point in interpolate_segment(start, end, n_points=points)]
        kpoints.extend(segment if not kpoints else segment[1:])
    grid = np.array(kpoints, dtype=float)

    frequencies, weights = unfold_phonon_path(
        force_model,
        transform,
        grid,
        perturbations=perturbations,
        site_masses=site_masses,
    )
    band_deviation = float(np.max(np.abs(weights.sum(axis=1) - force_model.n_modes)))
    acoustic = force_model.acoustic_sum_rule_deviation()

    ebs = EffectiveBandStructure(
        kpoints=grid,
        energies=frequencies,
        weights=weights,
        distances=cartesian_path_distances(grid),
        metadata={
            "source_code": "force-constant model",
            "transformation_matrix": transform.matrix.tolist(),
            "n_modes": force_model.n_modes,
            "acoustic_sum_rule_deviation": acoustic,
            "max_band_deviation": band_deviation,
            "n_mass_defects": len(substitutions),
        },
    )
    console.print_json(
        json.dumps(
            {
                "n_kpoints": int(grid.shape[0]),
                "n_states": int(frequencies.shape[1]),
                "n_modes": force_model.n_modes,
                "multiplicity": transform.multiplicity,
                "n_mass_defects": len(substitutions),
                "acoustic_sum_rule_deviation": acoustic,
                "max_band_deviation": band_deviation,
                "frequency_range": [
                    float(frequencies.min()),
                    float(frequencies.max()),
                ],
            }
        )
    )
    if band_deviation > 1e-8:
        console.print("[yellow]Warning:[/yellow] the band sum rule is violated")
    if acoustic > 1e-8:
        console.print(
            "[yellow]Warning:[/yellow] the force constants violate the acoustic "
            "sum rule, so the acoustic branches will not reach zero at Gamma"
        )
    if output is not None:
        write_ebs_json(output, ebs)
        console.print(f"Wrote {output}")
