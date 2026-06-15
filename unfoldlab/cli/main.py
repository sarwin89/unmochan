"""Typer command line interface for UnfoldLab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import click
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from unfoldlab import __version__
from unfoldlab.core.kpoints import KPoint
from unfoldlab.io.qe import (
    read_gnu_blocks,
    read_kmap,
    write_weight_table,
)
from unfoldlab.io.qe_wfc import compute_weights_from_qe_save
from unfoldlab.workflows.backend import (
    compute_backend_weights,
    unfold_backend_bands,
    write_backend_path_files,
)
from unfoldlab.workflows.problem import UnfoldingProblem
from unfoldlab.workflows.qe import unfold_qe_bands

console = Console()

app = typer.Typer(
    name="unfoldlab",
    help="Band unfolding workflows for plane-wave electronic-structure codes.",
    invoke_without_command=True,
    no_args_is_help=False,
)
vasp_app = typer.Typer(help="VASP-oriented workflows.", no_args_is_help=True)
qe_app = typer.Typer(help="Quantum ESPRESSO workflows.", no_args_is_help=True)
app.add_typer(vasp_app, name="vasp")
app.add_typer(qe_app, name="qe")


def version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def root(
    ctx: typer.Context,
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Run UnfoldLab commands."""

    if ctx.invoked_subcommand is None:
        run_guided_cli()


@app.command("guide")
def guide() -> None:
    """Launch the guided interactive workflow menu."""

    run_guided_cli()


@app.command("make-kpoints")
def make_kpoints(
    code: Annotated[str, typer.Option("--code", help="Backend: vasp or qe")],
    path_json: Annotated[Path, typer.Argument(exists=True, readable=True)],
    kpoints: Annotated[Path | None, typer.Option("--kpoints")] = None,
    kmap: Annotated[Path, typer.Option("--kmap")] = Path("kmap.tsv"),
    ticks: Annotated[Path, typer.Option("--ticks")] = Path("path_ticks.tsv"),
) -> None:
    """Generate folded-path k-points for VASP or QE."""

    backend = _parse_code(code)
    output_kpoints = kpoints or Path("KPOINTS" if backend == "vasp" else "qe_kpoints_supercell.in")
    path_data = write_backend_path_files(
        backend,
        path_json,
        kpoints=output_kpoints,
        kmap=kmap,
        ticks=ticks,
    )
    console.print(f"Wrote {len(path_data.kpoints)} {backend.upper()} k-points to {output_kpoints}")
    console.print(f"Wrote k-point map to {kmap}")
    console.print(f"Wrote plot ticks to {ticks}")


@app.command("weights")
def weights_command(
    code: Annotated[str, typer.Option("--code", help="Backend: vasp or qe")],
    kmap: Annotated[Path, typer.Option("--kmap", exists=True, readable=True)],
    matrix: Annotated[
        str,
        typer.Option("--matrix", help='Nine numbers, e.g. "2 0 0 0 2 0 0 0 1".'),
    ],
    out: Annotated[Path, typer.Option("--out")] = Path("weights.dat"),
    bands: Annotated[Path | None, typer.Option("--bands", exists=True, readable=True)] = None,
    nbnd: Annotated[int | None, typer.Option("--nbnd")] = None,
    coefficients: Annotated[
        Path | None,
        typer.Option("--coefficients", exists=True, readable=True),
    ] = None,
    qe_save_dir: Annotated[
        Path | None,
        typer.Option("--qe-save-dir", "--save-dir", exists=True, file_okay=False),
    ] = None,
    prefix: Annotated[str | None, typer.Option("--prefix")] = None,
    outdir: Annotated[Path | None, typer.Option("--outdir")] = None,
    wavecar: Annotated[Path | None, typer.Option("--wavecar", exists=True, readable=True)] = None,
    spin: Annotated[int, typer.Option("--spin", min=1)] = 1,
    wfc_format: Annotated[str, typer.Option("--wfc-format")] = "auto",
    wfc_pattern: Annotated[str | None, typer.Option("--wfc-pattern")] = None,
    tol: Annotated[float, typer.Option("--tol")] = 1e-6,
) -> None:
    """Compute spectral weights for VASP or QE through the common kernel."""

    diagnostics = compute_backend_weights(
        _parse_code(code),
        kmap=kmap,
        transform=_parse_matrix(matrix),
        out=out,
        bands=bands,
        nbnd=nbnd,
        coefficients=coefficients,
        qe_save_dir=_resolve_qe_save_dir(qe_save_dir, outdir, prefix),
        wavecar=wavecar,
        spin=spin,
        wfc_format=wfc_format,
        wfc_pattern=wfc_pattern,
        tol=tol,
    )
    console.print(f"Wrote {out}")
    console.print_json(json.dumps(diagnostics.to_dict()))


@app.command("unfold")
def unfold_command(
    code: Annotated[str, typer.Option("--code", help="Backend: vasp or qe")],
    kmap: Annotated[Path, typer.Option("--kmap", exists=True, readable=True)],
    matrix: Annotated[str | None, typer.Option("--matrix")] = None,
    ticks: Annotated[Path | None, typer.Option("--ticks", exists=True, readable=True)] = None,
    bands: Annotated[Path | None, typer.Option("--bands", exists=True, readable=True)] = None,
    weights: Annotated[Path | None, typer.Option("--weights", exists=True, readable=True)] = None,
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
    wavecar: Annotated[Path | None, typer.Option("--wavecar", exists=True, readable=True)] = None,
    eigenval: Annotated[Path | None, typer.Option("--eigenval", exists=True, readable=True)] = None,
    fermi: Annotated[float, typer.Option("--fermi")] = 0.0,
    out: Annotated[Path, typer.Option("--out")] = Path("unfolded_bands.dat"),
    plot: Annotated[Path, typer.Option("--plot")] = Path("unfolded_bands.png"),
    write_weights: Annotated[Path | None, typer.Option("--write-weights")] = None,
    spin: Annotated[int, typer.Option("--spin", min=1)] = 1,
    tol: Annotated[float, typer.Option("--tol")] = 1e-6,
    wfc_format: Annotated[str, typer.Option("--wfc-format")] = "auto",
    wfc_pattern: Annotated[str | None, typer.Option("--wfc-pattern")] = None,
) -> None:
    """Build an unfolded effective band table and plot for VASP or QE."""

    transform = _parse_matrix(matrix) if matrix is not None else None
    result = unfold_backend_bands(
        _parse_code(code),
        kmap=kmap,
        transform=transform,
        ticks=ticks,
        bands=bands,
        weights=weights,
        qe_save_dir=_resolve_qe_save_dir(qe_save_dir, outdir, prefix),
        coefficients=coefficients,
        wavecar=wavecar,
        eigenval=eigenval,
        fermi=fermi,
        out=out,
        plot=plot,
        write_weights=write_weights,
        spin=spin,
        tol=tol,
        wfc_format=wfc_format,
        wfc_pattern=wfc_pattern,
    )
    console.print(f"Wrote {result.output_path}")
    if result.weights_path is not None:
        console.print(f"Wrote {result.weights_path}")
    if result.plot_path is not None:
        console.print(f"Wrote {result.plot_path}")
    console.print(f"Weight mode: {result.mode}")


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
    points = [_parse_kpoint(item) for item in kpoint]
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
    points = [_parse_kpoint(item) for item in kpoint] if kpoint else None
    missing = problem.validate_eigenval_kpoints(eigenval, points, atol=atol)
    payload = {
        "missing_count": len(missing),
        "missing": [mapping.to_dict() for mapping in missing],
    }
    console.print_json(json.dumps(payload))


@qe_app.command("make-kpoints")
def qe_make_kpoints(
    path_json: Annotated[Path, typer.Argument(exists=True, readable=True)],
    qe_kpoints: Annotated[Path, typer.Option("--qe-kpoints")] = Path(
        "qe_kpoints_supercell.in"
    ),
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

    transform = _parse_matrix(matrix)
    n_bands = _infer_n_bands(bands, nbnd)
    mapping = read_kmap(kmap)
    resolved_save_dir = _resolve_qe_save_dir(save_dir, outdir, prefix)
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
            help='Nine numbers, required with --qe-save-dir or --coefficients.',
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

    transform = _parse_matrix(matrix) if matrix is not None else None
    result = unfold_qe_bands(
        bands=bands,
        kmap=kmap,
        ticks=ticks,
        weights=weights,
        qe_save_dir=_resolve_qe_save_dir(qe_save_dir, outdir, prefix),
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


def run_guided_cli() -> None:
    """Run an interactive menu for users who prefer a guided workflow."""

    console.print(
        Panel(
            "Choose a workflow and UnfoldLab will ask for the required files.\n"
            "The guided menu calls the same backend-neutral APIs as the batch CLI.",
            title="UnfoldLab Guided Mode",
            border_style="cyan",
        )
    )
    backend = Prompt.ask("Select backend", choices=["vasp", "qe"], default="vasp")
    while True:
        _print_guided_menu(backend)
        choice = Prompt.ask("Select an option", choices=[str(i) for i in range(6)], default="0")
        if choice == "0":
            console.print("Leaving guided mode.")
            return
        try:
            if choice == "1":
                _guided_set_backend()
            elif choice == "2":
                _guided_make_kpoints(backend)
            elif choice == "3":
                _guided_weights(backend)
            elif choice == "4":
                _guided_unfold(backend)
            elif choice == "5":
                _guided_validate_backend(backend)
            if choice == "1":
                backend = _GUIDED_BACKEND_STATE["backend"]
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
        if not Confirm.ask("Return to the main menu?", default=True):
            return


_GUIDED_BACKEND_STATE = {"backend": "vasp"}


def _print_guided_menu(backend: str) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("No.", justify="right", width=4)
    table.add_column("Workflow")
    table.add_column("Purpose")
    table.add_row("1", f"Backend: {backend.upper()}", "Switch between VASP and QE")
    table.add_row("2", "make-kpoints", "Create backend k-points, kmap, and tick files")
    table.add_row("3", "weights", "Compute spectral weights through the common kernel")
    table.add_row("4", "unfold", "Build unfolded table and plot")
    table.add_row("5", "validate", "Run backend-specific validation/preflight")
    table.add_row("0", "Exit", "Leave guided mode")
    console.print(table)


def _guided_set_backend() -> None:
    _GUIDED_BACKEND_STATE["backend"] = Prompt.ask(
        "Select backend",
        choices=["vasp", "qe"],
        default=_GUIDED_BACKEND_STATE["backend"],
    )


def _guided_make_kpoints(backend: str) -> None:
    path_json = _prompt_existing_path("Primitive path JSON")
    default_name = "KPOINTS.unfold" if backend == "vasp" else "qe_kpoints_supercell.in"
    kpoints = _prompt_output_path(f"Output {backend.upper()} k-points file", default_name)
    kmap = _prompt_output_path("Output kmap TSV", "kmap.tsv")
    ticks = _prompt_output_path("Output tick TSV", "path_ticks.tsv")
    make_kpoints(code=backend, path_json=path_json, kpoints=kpoints, kmap=kmap, ticks=ticks)


def _guided_weights(backend: str) -> None:
    kmap = _prompt_existing_path("kmap TSV")
    matrix = Prompt.ask('Transform matrix, nine numbers, e.g. "2 0 0 0 2 0 0 0 1"')
    out = _prompt_output_path("Output weights table", "weights.dat")
    if backend == "vasp":
        wavecar = _prompt_existing_path("VASP WAVECAR")
        spin = _prompt_optional_int("Spin channel (blank for 1)") or 1
        weights_command(code=backend, kmap=kmap, matrix=matrix, wavecar=wavecar, spin=spin, out=out)
        return

    source = Prompt.ask("QE weight source", choices=["save", "coefficients"], default="save")
    bands = None
    nbnd = None
    if Confirm.ask("Infer band count from bands.dat.gnu?", default=True):
        bands = _prompt_existing_path("QE bands.dat.gnu")
    else:
        nbnd = int(Prompt.ask("Number of bands"))
    if source == "coefficients":
        coefficients = _prompt_existing_path("Coefficient table")
        weights_command(
            code=backend,
            kmap=kmap,
            matrix=matrix,
            coefficients=coefficients,
            bands=bands,
            nbnd=nbnd,
            out=out,
        )
    else:
        save_dir = _prompt_existing_path("QE prefix.save directory")
        spin = _prompt_optional_int("Spin channel (blank for 1)") or 1
        weights_command(
            code=backend,
            kmap=kmap,
            matrix=matrix,
            qe_save_dir=save_dir,
            bands=bands,
            nbnd=nbnd,
            spin=spin,
            out=out,
        )


def _guided_unfold(backend: str) -> None:
    kmap = _prompt_existing_path("kmap TSV")
    ticks = _prompt_optional_existing_path("Tick TSV, blank to infer from kmap")
    matrix = Prompt.ask(
        "Transform matrix, nine numbers, blank if using external weights",
        default="",
    )
    transform = matrix or None
    fermi = float(Prompt.ask("Fermi energy to subtract", default="0.0"))
    out = _prompt_output_path("Output unfolded table", "unfolded_bands.dat")
    plot = _prompt_output_path("Output plot", "unfolded_bands.png")
    write_weights = None
    if backend == "vasp":
        source = Prompt.ask("VASP source", choices=["wavecar", "eigenval"], default="wavecar")
        if source == "wavecar":
            wavecar = _prompt_existing_path("VASP WAVECAR")
            spin = _prompt_optional_int("Spin channel (blank for 1)") or 1
            if Confirm.ask("Write computed weights?", default=True):
                write_weights = _prompt_output_path("Output weights table", "weights.dat")
            unfold_command(
                code=backend,
                kmap=kmap,
                matrix=transform,
                ticks=ticks,
                wavecar=wavecar,
                fermi=fermi,
                out=out,
                plot=plot,
                write_weights=write_weights,
                spin=spin,
            )
        else:
            eigenval = _prompt_existing_path("VASP EIGENVAL")
            weights = _prompt_optional_existing_path("Weights table, blank for unit weights")
            unfold_command(
                code=backend,
                kmap=kmap,
                ticks=ticks,
                eigenval=eigenval,
                weights=weights,
                fermi=fermi,
                out=out,
                plot=plot,
            )
        return

    bands = _prompt_existing_path("QE bands.dat.gnu")
    source = Prompt.ask(
        "QE spectral weight source",
        choices=["save", "weights", "coefficients", "unit"],
        default="save",
    )
    kwargs: dict[str, Path | str | None] = {}
    if source == "save":
        kwargs["qe_save_dir"] = _prompt_existing_path("QE prefix.save directory")
    elif source == "weights":
        kwargs["weights"] = _prompt_existing_path("Existing weights table")
    elif source == "coefficients":
        kwargs["coefficients"] = _prompt_existing_path("Coefficient table")
    if source in {"save", "coefficients"} and Confirm.ask("Write computed weights?", default=True):
        write_weights = _prompt_output_path("Output weights table", "weights.dat")
    unfold_command(
        code=backend,
        kmap=kmap,
        matrix=transform,
        ticks=ticks,
        bands=bands,
        fermi=fermi,
        out=out,
        plot=plot,
        write_weights=write_weights,
        **kwargs,
    )


def _guided_validate_backend(backend: str) -> None:
    if backend == "vasp":
        if Confirm.ask("Validate primitive/supercell mapping?", default=True):
            _guided_vasp_map()
        if Confirm.ask("Validate VASP EIGENVAL k-points?", default=False):
            _guided_vasp_validate_eigenval()
    else:
        console.print("QE validation is performed during make-kpoints, weights, and unfold.")


def _guided_vasp_map() -> None:
    primitive = _prompt_existing_path("Primitive POSCAR/CONTCAR")
    supercell = _prompt_existing_path("Supercell POSCAR/CONTCAR")
    vasp_map(primitive=primitive, supercell=supercell)


def _guided_vasp_fold_kpoints() -> None:
    primitive = _prompt_existing_path("Primitive POSCAR/CONTCAR")
    supercell = _prompt_existing_path("Supercell POSCAR/CONTCAR")
    points = _prompt_repeated("Primitive k-point, e.g. G:0,0,0 or 0.5,0,0")
    vasp_fold_kpoints(primitive=primitive, supercell=supercell, kpoint=points)


def _guided_vasp_validate_eigenval() -> None:
    primitive = _prompt_existing_path("Primitive POSCAR/CONTCAR")
    supercell = _prompt_existing_path("Supercell POSCAR/CONTCAR")
    eigenval = _prompt_existing_path("VASP EIGENVAL")
    points = None
    if Confirm.ask("Provide primitive k-points to validate?", default=True):
        points = _prompt_repeated("Primitive k-point, e.g. G:0,0,0 or 0.5,0,0")
    atol = float(Prompt.ask("Fractional k-point tolerance", default="1e-8"))
    vasp_validate_eigenval(
        primitive=primitive,
        supercell=supercell,
        eigenval=eigenval,
        kpoint=points,
        atol=atol,
    )


def _guided_qe_make_kpoints() -> None:
    path_json = _prompt_existing_path("Primitive path JSON")
    qe_kpoints = _prompt_output_path("Output QE K_POINTS file", "qe_kpoints_supercell.in")
    kmap = _prompt_output_path("Output kmap TSV", "kmap.tsv")
    ticks = _prompt_output_path("Output tick TSV", "path_ticks.tsv")
    qe_make_kpoints(path_json=path_json, qe_kpoints=qe_kpoints, kmap=kmap, ticks=ticks)


def _guided_qe_weights() -> None:
    kmap = _prompt_existing_path("kmap TSV")
    matrix = Prompt.ask('Transform matrix, nine numbers, e.g. "2 0 0 0 2 0 0 0 1"')
    source = Prompt.ask(
        "Weight source",
        choices=["save", "coefficients"],
        default="save",
    )
    bands = None
    nbnd = None
    if Confirm.ask("Infer band count from bands.dat.gnu?", default=True):
        bands = _prompt_existing_path("QE bands.dat.gnu")
    else:
        nbnd = int(Prompt.ask("Number of bands"))
    out = _prompt_output_path("Output weights table", "weights.dat")
    if source == "coefficients":
        coefficients = _prompt_existing_path("Coefficient table")
        qe_weights(
            kmap=kmap,
            matrix=matrix,
            coefficients=coefficients,
            bands=bands,
            nbnd=nbnd,
            out=out,
        )
    else:
        save_dir = _prompt_existing_path("QE prefix.save directory")
        spin = _prompt_optional_int("Spin channel (blank for none)")
        qe_weights(
            kmap=kmap,
            matrix=matrix,
            save_dir=save_dir,
            bands=bands,
            nbnd=nbnd,
            spin=spin,
            out=out,
        )


def _guided_qe_unfold() -> None:
    bands = _prompt_existing_path("QE bands.dat.gnu")
    kmap = _prompt_existing_path("kmap TSV")
    ticks = _prompt_optional_existing_path("Tick TSV, blank to infer from kmap")
    source = Prompt.ask(
        "Spectral weight source",
        choices=["save", "weights", "coefficients", "unit"],
        default="save",
    )
    matrix = None
    weights = None
    qe_save_dir = None
    coefficients = None
    if source in {"save", "coefficients"}:
        matrix = Prompt.ask('Transform matrix, nine numbers, e.g. "2 0 0 0 2 0 0 0 1"')
    if source == "save":
        qe_save_dir = _prompt_existing_path("QE prefix.save directory")
    elif source == "weights":
        weights = _prompt_existing_path("Existing weights table")
    elif source == "coefficients":
        coefficients = _prompt_existing_path("Coefficient table")
    fermi = float(Prompt.ask("Fermi energy to subtract", default="0.0"))
    out = _prompt_output_path("Output unfolded table", "unfolded_bands.dat")
    plot = _prompt_output_path("Output plot", "unfolded_bands.png")
    write_weights = None
    if source in {"save", "coefficients"} and Confirm.ask("Write computed weights?", default=True):
        write_weights = _prompt_output_path("Output weights table", "weights.dat")
    qe_unfold(
        bands=bands,
        kmap=kmap,
        ticks=ticks,
        matrix=matrix,
        weights=weights,
        qe_save_dir=qe_save_dir,
        coefficients=coefficients,
        write_weights=write_weights,
        fermi=fermi,
        out=out,
        plot=plot,
    )


def _prompt_existing_path(label: str) -> Path:
    while True:
        value = Path(Prompt.ask(label)).expanduser()
        if value.exists():
            return value
        console.print(f"[red]Path does not exist:[/red] {value}")


def _prompt_optional_existing_path(label: str) -> Path | None:
    value = Prompt.ask(label, default="").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _prompt_output_path(label: str, default: str) -> Path:
    return Path(Prompt.ask(label, default=default)).expanduser()


def _prompt_optional_int(label: str) -> int | None:
    value = Prompt.ask(label, default="").strip()
    return int(value) if value else None


def _prompt_repeated(label: str) -> list[str]:
    values: list[str] = []
    while True:
        values.append(Prompt.ask(label))
        if not Confirm.ask("Add another?", default=False):
            return values


def _parse_kpoint(raw: str) -> KPoint:
    label = None
    coords = raw
    if ":" in raw:
        label, coords = raw.split(":", 1)
    values = [float(part) for part in coords.split(",")]
    if len(values) != 3:
        raise typer.BadParameter("k-point must have three comma-separated coordinates")
    return KPoint(values, label=label or None)


def _parse_matrix(raw: str) -> np.ndarray:
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 9:
        raise typer.BadParameter("matrix must contain exactly nine numeric entries")
    return np.array([float(token) for token in tokens], dtype=float).reshape(3, 3)


def _parse_code(raw: str) -> str:
    code = raw.strip().lower()
    if code not in {"vasp", "qe"}:
        raise typer.BadParameter("code must be vasp or qe")
    return code


def _resolve_qe_save_dir(
    explicit_save_dir: Path | None,
    outdir: Path | None,
    prefix: str | None,
) -> Path | None:
    if explicit_save_dir is not None:
        return explicit_save_dir
    if prefix is None:
        return None
    return (outdir or Path(".")) / f"{prefix}.save"


def _infer_n_bands(bands: Path | None, nbnd: int | None) -> int:
    if bands is not None:
        _, energies = read_gnu_blocks(bands)
        return int(energies.shape[1])
    if nbnd is not None:
        if nbnd <= 0:
            raise typer.BadParameter("--nbnd must be positive")
        return int(nbnd)
    raise typer.BadParameter("provide either --bands or --nbnd")


def main(argv: list[str] | None = None) -> int:
    command = typer.main.get_command(app)
    try:
        command(args=argv, prog_name="unfoldlab", standalone_mode=False)
    except click.exceptions.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.exceptions.Abort:
        console.print("Aborted.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
