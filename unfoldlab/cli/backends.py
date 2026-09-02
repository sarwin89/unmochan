"""Code-specific subcommands: `unfoldlab vasp ...` and `unfoldlab qe ...`.

Everything here is a convenience wrapper that either detects a transform from a
pair of structures or drives one backend's reader directly.  The backend-neutral
path goes through `unfoldlab.cli.commands`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from unfoldlab.cli.app import qe_app, vasp_app
from unfoldlab.cli.commands import make_kpoints
from unfoldlab.cli.common import (
    console,
    infer_n_bands,
    parse_kpoint,
    parse_matrix,
    resolve_qe_save_dir,
)
from unfoldlab.io.qe import read_kmap, write_weight_table
from unfoldlab.io.qe_wfc import compute_weights_from_qe_save
from unfoldlab.workflows.problem import UnfoldingProblem
from unfoldlab.workflows.qe import unfold_qe_bands


@vasp_app.command("map")
def vasp_map(
    primitive: Annotated[Path, typer.Option("--primitive", exists=True, readable=True)],
    supercell: Annotated[Path, typer.Option("--supercell", exists=True, readable=True)],
) -> None:
    """Detect the primitive-to-supercell transformation matrix."""

    problem = UnfoldingProblem.from_vasp(primitive=primitive, supercell=supercell)
    console.print_json(json.dumps(problem.find_transformation().to_dict()))


@vasp_app.command("fold-kpoints")
def vasp_fold_kpoints(
    primitive: Annotated[Path, typer.Option("--primitive", exists=True, readable=True)],
    supercell: Annotated[Path, typer.Option("--supercell", exists=True, readable=True)],
    kpoint: Annotated[
        list[str],
        typer.Option("--kpoint", help="label:x,y,z or x,y,z; repeatable."),
    ],
) -> None:
    """Fold primitive k-points into the supercell Brillouin zone."""

    problem = UnfoldingProblem.from_vasp(primitive=primitive, supercell=supercell)
    points = [parse_kpoint(item) for item in kpoint]
    mappings = problem.generate_supercell_kpoints(points)
    console.print_json(json.dumps([mapping.to_dict() for mapping in mappings]))


@vasp_app.command("validate-eigenval")
def vasp_validate_eigenval(
    primitive: Annotated[Path, typer.Option("--primitive", exists=True, readable=True)],
    supercell: Annotated[Path, typer.Option("--supercell", exists=True, readable=True)],
    eigenval: Annotated[Path, typer.Option("--eigenval", exists=True, readable=True)],
    kpoint: Annotated[
        list[str] | None,
        typer.Option("--kpoint", help="Primitive k-point required in the EIGENVAL; repeatable."),
    ] = None,
    atol: Annotated[
        float,
        typer.Option("--atol", help="Fractional k-point tolerance."),
    ] = 1e-8,
) -> None:
    """Report required folded k-points missing from a VASP EIGENVAL file."""

    problem = UnfoldingProblem.from_vasp(primitive=primitive, supercell=supercell)
    points = [parse_kpoint(item) for item in kpoint] if kpoint else None
    missing = problem.validate_eigenval_kpoints(eigenval, points, atol=atol)
    payload = {
        "missing_count": len(missing),
        "missing": [mapping.to_dict() for mapping in missing],
    }
    console.print_json(json.dumps(payload))


@qe_app.command("make-kpoints")
def qe_make_kpoints(
    path_json: Annotated[Path, typer.Argument(exists=True, readable=True)],
    qe_kpoints: Annotated[Path, typer.Option("--qe-kpoints")] = Path("qe_kpoints_supercell.in"),
    kmap: Annotated[Path, typer.Option("--kmap")] = Path("kmap.tsv"),
    ticks: Annotated[Path, typer.Option("--ticks")] = Path("path_ticks.tsv"),
) -> None:
    """Generate QE folded-path K_POINTS, kmap, and tick files."""

    make_kpoints(code="qe", path_json=path_json, kpoints=qe_kpoints, kmap=kmap, ticks=ticks)


@qe_app.command("weights")
def qe_weights(
    kmap: Annotated[Path, typer.Option("--kmap", exists=True, readable=True)],
    matrix: Annotated[
        str,
        typer.Option("--matrix", help='Nine numbers, e.g. "2 0 0 0 2 0 0 0 1".'),
    ],
    save_dir: Annotated[
        Path | None,
        typer.Option("--save-dir", "--qe-save-dir", exists=True, file_okay=False),
    ] = None,
    prefix: Annotated[str | None, typer.Option("--prefix")] = None,
    outdir: Annotated[Path | None, typer.Option("--outdir")] = None,
    bands: Annotated[Path | None, typer.Option("--bands", exists=True, readable=True)] = None,
    nbnd: Annotated[
        int | None,
        typer.Option("--nbnd", help="Number of bands if --bands is omitted."),
    ] = None,
    coefficients: Annotated[
        Path | None,
        typer.Option("--coefficients", exists=True, readable=True),
    ] = None,
    spin: Annotated[int | None, typer.Option("--spin", min=1, max=2)] = None,
    wfc_format: Annotated[
        str,
        typer.Option("--wfc-format", help="auto, hdf5, or dat."),
    ] = "auto",
    wfc_pattern: Annotated[str | None, typer.Option("--wfc-pattern")] = None,
    tol: Annotated[float, typer.Option("--tol")] = 1e-6,
    out: Annotated[Path, typer.Option("--out")] = Path("weights.dat"),
) -> None:
    """Compute primitive-cell unfolding weights from QE wavefunctions or coefficients."""

    transform = parse_matrix(matrix)
    n_bands = infer_n_bands(bands, nbnd)
    mapping = read_kmap(kmap)
    resolved_save_dir = resolve_qe_save_dir(save_dir, outdir, prefix)
    if coefficients is not None:
        from unfoldlab.io.qe import weights_from_coefficient_table

        weights = weights_from_coefficient_table(
            coefficients,
            mapping,
            transform,
            n_bands,
            tol=tol,
        )
    elif resolved_save_dir is not None:
        weights = compute_weights_from_qe_save(
            resolved_save_dir,
            mapping.primitive_kpoints,
            mapping.supercell_folded_kpoints,
            transform,
            n_bands,
            tol=tol,
            spin=spin,
            pattern=wfc_pattern,
            file_format=wfc_format,
        )
    else:
        raise typer.BadParameter("provide either --save-dir or --coefficients")

    write_weight_table(out, weights)
    console.print(f"Wrote {out}")


@qe_app.command("unfold")
def qe_unfold(
    bands: Annotated[Path, typer.Option("--bands", exists=True, readable=True)],
    kmap: Annotated[Path, typer.Option("--kmap", exists=True, readable=True)],
    ticks: Annotated[Path | None, typer.Option("--ticks", exists=True, readable=True)] = None,
    matrix: Annotated[
        str | None,
        typer.Option(
            "--matrix",
            help="Nine numbers, required with --qe-save-dir or --coefficients.",
        ),
    ] = None,
    weights: Annotated[
        Path | None,
        typer.Option("--weights", exists=True, readable=True),
    ] = None,
    qe_save_dir: Annotated[
        Path | None,
        typer.Option("--qe-save-dir", "--save-dir", exists=True, file_okay=False),
    ] = None,
    prefix: Annotated[str | None, typer.Option("--prefix")] = None,
    outdir: Annotated[Path | None, typer.Option("--outdir")] = None,
    coefficients: Annotated[
        Path | None,
        typer.Option("--coefficients", exists=True, readable=True),
    ] = None,
    spin: Annotated[int | None, typer.Option("--spin", min=1, max=2)] = None,
    wfc_format: Annotated[
        str,
        typer.Option("--wfc-format", help="auto, hdf5, or dat."),
    ] = "auto",
    wfc_pattern: Annotated[str | None, typer.Option("--wfc-pattern")] = None,
    write_weights: Annotated[Path | None, typer.Option("--write-weights")] = None,
    fermi: Annotated[float, typer.Option("--fermi")] = 0.0,
    emin: Annotated[float | None, typer.Option("--emin")] = None,
    emax: Annotated[float | None, typer.Option("--emax")] = None,
    marker_scale: Annotated[float, typer.Option("--marker-scale")] = 28.0,
    tol: Annotated[float, typer.Option("--tol")] = 1e-6,
    out: Annotated[Path, typer.Option("--out")] = Path("unfolded_bands.dat"),
    plot: Annotated[Path, typer.Option("--plot")] = Path("unfolded_bands.png"),
) -> None:
    """Write and plot primitive-path unfolded QE supercell bands."""

    transform = parse_matrix(matrix) if matrix is not None else None
    result = unfold_qe_bands(
        bands=bands,
        kmap=kmap,
        ticks=ticks,
        weights=weights,
        qe_save_dir=resolve_qe_save_dir(qe_save_dir, outdir, prefix),
        coefficients=coefficients,
        transform=transform,
        fermi=fermi,
        emin=emin,
        emax=emax,
        marker_scale=marker_scale,
        out=out,
        plot=plot,
        write_weights=write_weights,
        tol=tol,
        spin=spin,
        wfc_pattern=wfc_pattern,
        wfc_format=wfc_format,
    )
    console.print(f"Wrote {result.output_path}")
    if result.weights_path is not None:
        console.print(f"Wrote {result.weights_path}")
    if result.plot_path is not None:
        console.print(f"Wrote {result.plot_path}")
    console.print(f"Weight mode: {result.mode}")
