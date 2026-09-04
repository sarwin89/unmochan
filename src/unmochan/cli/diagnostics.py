"""Diagnostics of an unfolded run: `fiber`, `cut`, `ensemble`, `norms`,
`layers`, `degeneracy`.

None of these produce an unfolded band structure.  They answer questions *about*
one -- which primitive k-points make up a fiber, how much weight lies in an
energy window, how much of a state the wavefunction file actually stores, how
that state is distributed over the layers of a slab, and
how much a per-band weight depends on the gauge inside a degenerate multiplet.

The commands that analyse the *inputs* of a run, or extract physical numbers
from a finished one, are in `unmochan.cli.analysis`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.table import Table

from unmochan.cli.app import app
from unmochan.cli.common import (
    console,
    parse_broadening_kind,
    parse_kpoint,
    parse_matrix,
)
from unmochan.core.augmentation import diagnose_augmentation
from unmochan.core.degeneracy import (
    average_degenerate_weights,
    degeneracy_report,
    subspace_weights,
)
from unmochan.core.ensemble import (
    band_moments,
    disorder_broadening,
    stack_configurations,
)
from unmochan.core.kpoints import fiber_kpoints
from unmochan.core.layers import diagnose_layer_conservation, layer_edges_from_structure
from unmochan.core.transformations import TransformationMatrix, detect_transformation
from unmochan.core.unfolding import diagnose_state_norms
from unmochan.core.windows import constant_energy_cut, energy_window_weight
from unmochan.io.qe import QEPath, write_kmap
from unmochan.io.serialization import read_ebs, write_ebs
from unmochan.io.vasp import read_poscar


@app.command("fiber")
def fiber_command(
    kpoint: Annotated[
        list[str],
        typer.Option("--kpoint", help="Supercell k-point, x,y,z in fractional coordinates."),
    ],
    matrix: Annotated[
        str | None,
        typer.Option("--matrix", help="Transformation matrix, nine numbers row-major."),
    ] = None,
    primitive: Annotated[
        Path | None,
        typer.Option("--primitive", exists=True, readable=True, help="Primitive POSCAR."),
    ] = None,
    supercell: Annotated[
        Path | None,
        typer.Option("--supercell", exists=True, readable=True, help="Supercell POSCAR."),
    ] = None,
    kmap: Annotated[
        Path | None,
        typer.Option("--kmap", help="Write a kmap listing the whole fiber."),
    ] = None,
) -> None:
    """List the primitive k-points that fold onto given supercell k-points.

    Folding is many-to-one: |det T| primitive k-points share each supercell
    k-point, and an unfolded band structure needs all of them.  This inverts the
    fold exactly, via the Smith normal form of the transformation matrix.

    A k-map written with --kmap repeats each supercell k-point |det T| times, so
    the later `weights` and `unfold` runs need --reuse-kpoints.
    """

    if matrix is not None:
        transform = TransformationMatrix(parse_matrix(matrix))
    elif primitive is not None and supercell is not None:
        transform = detect_transformation(read_poscar(primitive), read_poscar(supercell))
    else:
        raise typer.BadParameter("provide either --matrix or both --primitive and --supercell")

    folded = [parse_kpoint(item).fractional for item in kpoint]
    members = [fiber_kpoints(point, transform) for point in folded]

    payload = {
        "multiplicity": transform.multiplicity,
        "fibers": [
            {"supercell": point.tolist(), "primitive": group.tolist()}
            for point, group in zip(folded, members, strict=True)
        ],
    }
    console.print_json(json.dumps(payload))

    if kmap is not None:
        stacked = np.concatenate(members, axis=0) if members else np.zeros((0, 3))
        qe_path = QEPath(
            transform=transform.matrix,
            kpoints=stacked,
            distances=np.arange(len(stacked), dtype=float),
            labels=("",) * len(stacked),
        )
        write_kmap(kmap, qe_path)
        console.print(f"Wrote {kmap}")


@app.command("cut")
def cut_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="Effective band structure, JSON or HDF5 as written by `unfold --json`.",
        ),
    ],
    broadening: Annotated[
        float,
        typer.Option("--broadening", min=0.0, help="Kernel width in the energy unit of the file."),
    ],
    energy: Annotated[
        float | None,
        typer.Option("--energy", help="Constant-energy cut A(k, E), relative to the reference."),
    ] = None,
    window: Annotated[
        str | None,
        typer.Option(
            "--window",
            help=(
                "Instead of a cut, the weight inside an energy window, given as "
                "lo,hi; 'inf' and '-inf' are accepted."
            ),
        ),
    ] = None,
    kind: Annotated[
        str, typer.Option("--kind", help="Broadening kernel: gaussian or lorentzian.")
    ] = "gaussian",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the k-point, path abscissa and value as a table."),
    ] = None,
) -> None:
    """Constant-energy cuts and energy-window weights of an unfolded band structure.

    A cut at the Fermi level is the unfolded Fermi surface; a cut elsewhere is
    the ARPES-like constant-energy map.  A window instead reports how much
    unfolded weight lies between two energies -- the occupied weight below the
    Fermi level, or the weight a broadened plot leaks into a gap.

    Both are evaluated in closed form from the discrete spectrum, so neither
    depends on the resolution of an energy grid.
    """

    if (energy is None) == (window is None):
        raise typer.BadParameter("give exactly one of --energy and --window")
    if broadening <= 0.0:
        raise typer.BadParameter("--broadening must be positive")
    broadening_kind = parse_broadening_kind(kind)

    structure, _ = read_ebs(input_path)
    if energy is not None:
        values = constant_energy_cut(structure, energy, broadening=broadening, kind=broadening_kind)
        title = f"A(k, E = {energy:g})"
        header = "# distance  kx ky kz  intensity"
    else:
        bounds = [part.strip() for part in str(window).split(",")]
        if len(bounds) != 2:
            raise typer.BadParameter("--window takes two comma-separated energies")
        try:
            lower, upper = (float(bound) for bound in bounds)
        except ValueError as error:
            raise typer.BadParameter(f"could not read --window {window}") from error
        values = energy_window_weight(
            structure, lower, upper, broadening=broadening, kind=broadening_kind
        )
        title = f"weight in [{lower:g}, {upper:g}]"
        header = "# distance  kx ky kz  weight"

    table = Table(title=title)
    table.add_column("distance", justify="right")
    table.add_column("k")
    table.add_column("value", justify="right")
    for index in range(structure.n_kpoints):
        table.add_row(
            f"{structure.distances[index]:.6f}",
            " ".join(f"{value: .4f}" for value in structure.kpoints[index]),
            f"{values[index]:.6f}",
        )
    console.print(table)

    if out is not None:
        lines = [header]
        for index in range(structure.n_kpoints):
            k = " ".join(f"{value: .8f}" for value in structure.kpoints[index])
            lines.append(f"{structure.distances[index]: .8f}  {k}  {values[index]: .8e}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"Wrote {out}")


@app.command("ensemble")
def ensemble_command(
    inputs: Annotated[
        list[Path],
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help=(
                "Effective band structure of one configuration, JSON or HDF5 as "
                "written by `unfold --json`; repeat once per configuration."
            ),
        ),
    ],
    weight: Annotated[
        list[float] | None,
        typer.Option(
            "--weight",
            help=(
                "Probability or multiplicity of each configuration, in the order "
                "the inputs were given; equal weights if omitted."
            ),
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Write the averaged effective band structure. A .h5/.hdf5 suffix "
                "writes HDF5, anything else JSON."
            ),
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write the per-k-point broadening table as JSON."),
    ] = None,
) -> None:
    """Average unfolded configurations: alloy, defect-ensemble or SQS disorder.

    A disordered supercell has no single band structure.  Unfold each
    configuration separately, then average here: the states are merged with
    their probabilities, so the averaged spectral function is the convex
    combination of the configurations' and carries the same total weight.

    The width of the averaged band splits, exactly, into the width a typical
    configuration already has and the spread of the configurations' band
    centres.  The second term is the disorder broadening and is what this
    command reports.
    """

    if not inputs:
        raise typer.BadParameter("provide at least one --input")
    if weight is not None and len(weight) != len(inputs):
        raise typer.BadParameter("give one --weight per --input, or none at all")

    structures = [read_ebs(path)[0] for path in inputs]
    merged = stack_configurations(structures, weight)
    split = disorder_broadening(structures, weight)
    moments = band_moments(merged)

    table = Table(title="Configurational average")
    table.add_column("k")
    table.add_column("weight", justify="right")
    table.add_column("centre", justify="right")
    table.add_column("intrinsic width", justify="right")
    table.add_column("disorder width", justify="right")
    for index in range(merged.n_kpoints):
        table.add_row(
            " ".join(f"{value: .4f}" for value in merged.kpoints[index]),
            f"{moments.total[index]:.6f}",
            f"{split.centre[index]: .6f}",
            f"{np.sqrt(max(split.intrinsic[index], 0.0)):.6f}",
            f"{split.disorder_width[index]:.6f}",
        )
    console.print(table)
    console.print(
        f"Configurations: {len(structures)}; variance decomposition residual {split.residual:.3e}"
    )

    if out is not None:
        write_ebs(out, merged)
        console.print(f"Wrote {out}")
    if report is not None:
        payload = {
            "configuration_weights": split.probabilities.tolist(),
            "kpoints": merged.kpoints.tolist(),
            "total_weight": moments.total.tolist(),
            "centre": split.centre.tolist(),
            "intrinsic_variance": split.intrinsic.tolist(),
            "disorder_variance": split.disorder.tolist(),
            "total_variance": split.total.tolist(),
            "residual": split.residual,
        }
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Wrote {report}")


@app.command("norms")
def norms_command(
    wavecar: Annotated[
        Path | None,
        typer.Option("--wavecar", exists=True, readable=True, help="VASP WAVECAR."),
    ] = None,
    qe_save_dir: Annotated[
        Path | None,
        typer.Option(
            "--qe-save-dir",
            "--save-dir",
            exists=True,
            file_okay=False,
            help="Quantum ESPRESSO .save directory.",
        ),
    ] = None,
    nbnd: Annotated[
        int | None,
        typer.Option("--nbnd", min=1, help="Bands to check; required for a QE .save."),
    ] = None,
    spin: Annotated[int, typer.Option("--spin", min=1, max=2)] = 1,
    band_chunk: Annotated[
        int | None,
        typer.Option("--band-chunk", min=1, help="Bands held in memory at once."),
    ] = None,
    wfc_pattern: Annotated[str | None, typer.Option("--wfc-pattern")] = None,
    wfc_format: Annotated[str, typer.Option("--wfc-format", help="auto, hdf5, or dat.")] = "auto",
    gamma_half: Annotated[
        str,
        typer.Option("--gamma-half", help="Half-space convention of a gamma-only WAVECAR."),
    ] = "x",
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write the per-state norms and the summary as JSON."),
    ] = None,
) -> None:
    """Report the norms of the stored plane-wave states of a wavefunction file.

    The unfolding weight is a ratio and does not depend on how the states are
    normalized, so this command changes nothing about a run.  What it tells you
    is how much of each state the file actually contains.  A norm-conserving
    calculation stores states of norm one; a PAW or ultrasoft one stores only
    the pseudo part, whose norm falls short by the augmentation charge, and the
    weights are then accurate only to `delta / (1 - delta)` with `delta` the
    missing fraction.  A norm of zero means an empty state, which is reported as
    a weight of zero and would otherwise be invisible.
    """

    if wavecar is not None and qe_save_dir is not None:
        raise typer.BadParameter("pass exactly one of --wavecar and --qe-save-dir")
    if wavecar is not None:
        from unmochan.io.vasp_wfc import state_norms_from_wavecar

        norms = state_norms_from_wavecar(
            wavecar, spin=spin, gamma_half=gamma_half, band_chunk=band_chunk
        )
        if nbnd is not None:
            if nbnd > norms.shape[1]:
                raise typer.BadParameter(
                    f"--nbnd {nbnd} exceeds the {norms.shape[1]} bands in the WAVECAR"
                )
            norms = norms[:, :nbnd]
        source = str(wavecar)
    elif qe_save_dir is not None:
        if nbnd is None:
            raise typer.BadParameter("--nbnd is required with --qe-save-dir")
        from unmochan.io.qe_wfc import state_norms_from_qe_save

        norms = state_norms_from_qe_save(
            qe_save_dir,
            nbnd,
            spin=spin,
            pattern=wfc_pattern,
            file_format=wfc_format,
            band_chunk=band_chunk,
        )
        source = str(qe_save_dir)
    else:
        raise typer.BadParameter("pass exactly one of --wavecar and --qe-save-dir")

    summary = diagnose_state_norms(norms)
    table = Table(title="Stored plane-wave norms")
    table.add_column("quantity")
    table.add_column("value", justify="right")
    table.add_row("k-points", str(summary.n_kpoints))
    table.add_row("bands", str(summary.n_bands))
    table.add_row("min norm", f"{summary.min_norm:.6f}")
    table.add_row("mean norm", f"{summary.mean_norm:.6f}")
    table.add_row("max norm", f"{summary.max_norm:.6f}")
    table.add_row("empty states", str(summary.n_zero_states))
    table.add_row("missing fraction", f"{summary.max_missing_fraction:.3e}")
    table.add_row("weight error bound", f"{summary.weight_error_bound:.3e}")
    augmentation = diagnose_augmentation(norms)
    table.add_row("PAW weight error bound", f"{augmentation.weight_error_bound:.3e}")
    console.print(table)
    console.print(augmentation.summary())
    if summary.n_zero_states:
        console.print(
            f"[yellow]{summary.n_zero_states} state(s) have no plane-wave content; "
            "their weights are reported as zero.[/yellow]"
        )

    if report is not None:
        payload = {
            "source": source,
            "summary": summary.to_dict(),
            "augmentation": augmentation.to_dict(),
            "norms": norms.tolist(),
        }
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Wrote {report}")


@app.command("layers")
def layers_command(
    wavecar: Annotated[
        Path,
        typer.Option("--wavecar", exists=True, readable=True, help="VASP WAVECAR."),
    ],
    matrix: Annotated[
        str,
        typer.Option("--matrix", help="Transformation matrix, nine numbers row-major."),
    ],
    edges: Annotated[
        str | None,
        typer.Option(
            "--edges",
            help="Layer boundaries along the axis, fractional, comma separated.",
        ),
    ] = None,
    structure: Annotated[
        Path | None,
        typer.Option(
            "--structure",
            exists=True,
            readable=True,
            help="POSCAR from which to place one boundary between each pair of planes.",
        ),
    ] = None,
    axis: Annotated[int, typer.Option("--axis", min=0, max=2)] = 2,
    kpoint_index: Annotated[
        int,
        typer.Option("--kpoint-index", min=1, help="One-based k-point index in the file."),
    ] = 1,
    spin: Annotated[int, typer.Option("--spin", min=1, max=2)] = 1,
    nbnd: Annotated[
        int | None,
        typer.Option("--nbnd", min=1, help="Bands to analyse, from the first."),
    ] = None,
    gamma_half: Annotated[
        str,
        typer.Option("--gamma-half", help="Half-space convention of a gamma-only WAVECAR."),
    ] = "x",
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the layer weights and the report as JSON."),
    ] = None,
) -> None:
    """Split the unfolding weight of a slab among its layers, and say what that means.

    At each primitive k-point the split is exact: the layer shares add up to the
    weight, whatever the boundaries (UnfoldLab.sum_regionCharge_layers).  Summing a
    layer's share over the fiber, however, does not return the charge the layer
    holds, because a layer is diagonal in real space and therefore a full matrix
    in the plane-wave basis.  The gap is the cross term between the k-point
    blocks (UnfoldLab.regionCharge_eq_blockSum_add_crossForm) and is reported here,
    so a k-summed layer weight is read with the right expectations.
    """

    from unmochan.io.vasp_wfc import WavecarReader

    transform = TransformationMatrix(parse_matrix(matrix))
    with WavecarReader(wavecar, gamma_half=gamma_half) as reader:
        if kpoint_index > reader.header.n_kpoints:
            raise typer.BadParameter(
                f"--kpoint-index {kpoint_index} exceeds the "
                f"{reader.header.n_kpoints} k-points in the WAVECAR"
            )
        k_header, coefficients = reader.read_kpoint_wavefunction(
            spin, kpoint_index, band_count=nbnd
        )
        g_vectors = k_header.g_vectors
        folded = np.asarray(k_header.kpoint, dtype=float)

    if edges is not None:
        boundaries = [float(value) for value in edges.replace(",", " ").split()]
    elif structure is not None:
        boundaries = list(layer_edges_from_structure(read_poscar(structure), axis=axis))
    else:
        raise typer.BadParameter("provide either --edges or --structure")

    fiber = fiber_kpoints(folded, transform)
    report = diagnose_layer_conservation(
        g_vectors,
        coefficients,
        fiber,
        folded,
        transform.matrix,
        boundaries,
        axis=axis,
    )

    table = Table(title=f"Layer weights summed over the fiber (k-point {kpoint_index})")
    table.add_column("band", justify="right")
    for index in range(len(boundaries)):
        table.add_column(f"layer {index}", justify="right")
    table.add_column("cross term", justify="right")
    for band, (row, gap) in enumerate(
        zip(report.summed_weights, report.cross_term, strict=True), start=1
    ):
        table.add_row(
            str(band),
            *(f"{value:.4f}" for value in row),
            f"{float(np.max(np.abs(gap))):.3e}",
        )
    console.print(table)
    console.print(report.summary())

    if json_out is not None:
        payload = report.to_dict()
        payload["edges"] = boundaries
        payload["axis"] = axis
        payload["kpoint"] = folded.tolist()
        payload["fiber"] = fiber.tolist()
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Wrote {json_out}")


@app.command("degeneracy")
def degeneracy_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="Effective band structure, JSON or HDF5 as written by `unfold --json`.",
        ),
    ],
    tol: Annotated[
        float,
        typer.Option(
            "--tol",
            min=0.0,
            help="Energy tolerance grouping bands into a multiplet, in the file's unit.",
        ),
    ] = 1e-4,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Write a copy with the weights of each multiplet averaged, so the "
                "weights no longer depend on the diagonalizer's basis. A .h5/.hdf5 "
                "suffix writes HDF5, anything else JSON."
            ),
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write the multiplet table and summary as JSON."),
    ] = None,
) -> None:
    """Report, and optionally remove, the gauge dependence of degenerate bands.

    Inside a degenerate multiplet a diagonalizer returns an arbitrary basis, and
    the unfolded weight of an individual band of the multiplet changes with it.
    Only the weight of the whole multiplet is physical.  This command groups the
    bands by energy, says how much of the reported per-band weight is gauge
    artefact (`max spread`), and with ``--out`` writes a copy in which each
    multiplet shares its weight equally -- which changes no sum rule, because
    the group total is preserved.
    """

    structure, manifest = read_ebs(input_path)
    summary = degeneracy_report(structure.energies, structure.weights, tol=tol)
    groups = subspace_weights(structure.energies, structure.weights, tol=tol)

    table = Table(title=f"Degenerate multiplets (tol = {tol:g})")
    table.add_column("k")
    table.add_column("energy", justify="right")
    table.add_column("multiplicity", justify="right")
    table.add_column("subspace weight", justify="right")
    for index, per_kpoint in enumerate(groups):
        label = " ".join(f"{value: .4f}" for value in structure.kpoints[index])
        for energy, multiplicity, weight_sum in per_kpoint:
            if multiplicity == 1:
                continue
            table.add_row(label, f"{energy: .6f}", str(multiplicity), f"{weight_sum:.6f}")
            label = ""
    if summary.gauge_sensitive:
        console.print(table)
    else:
        console.print(f"No degenerate multiplets within tol = {tol:g}.")
    console.print(
        f"Multiplets: {summary.n_groups} "
        f"({summary.n_degenerate_groups} degenerate, largest {summary.max_multiplicity}); "
        f"gauge spread of a per-band weight up to {summary.max_spread:.3e}"
    )

    if out is not None:
        write_ebs(out, average_degenerate_weights(structure, tol=tol), manifest=manifest)
        console.print(f"Wrote {out}")
    if report is not None:
        payload = {
            "summary": summary.to_dict(),
            "kpoints": structure.kpoints.tolist(),
            "multiplets": [
                [
                    {"energy": energy, "multiplicity": multiplicity, "weight": weight_sum}
                    for energy, multiplicity, weight_sum in per_kpoint
                ]
                for per_kpoint in groups
            ],
        }
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Wrote {report}")
