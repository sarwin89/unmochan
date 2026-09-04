"""The interactive guided menu.

Every branch here calls the same command function the batch CLI exposes, so the
guided mode cannot drift away from the documented behaviour.
"""

from __future__ import annotations

from pathlib import Path

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from unmochan.cli.backends import (
    qe_make_kpoints,
    qe_unfold,
    qe_weights,
    vasp_fold_kpoints,
    vasp_map,
    vasp_validate_eigenval,
)
from unmochan.cli.commands import make_kpoints, unfold_command, weights_command
from unmochan.cli.common import (
    console,
    prompt_existing_path,
    prompt_optional_existing_path,
    prompt_optional_int,
    prompt_output_path,
    prompt_repeated,
)


def run_guided_cli() -> None:
    """Run an interactive menu for users who prefer a guided workflow."""

    console.print(
        Panel(
            "Run the same workflow you would do by hand after a supercell DFT calculation:\n"
            "prepare folded k-points, compute spectral weights, then build and plot the "
            "unfolded bands.\n\n"
            "Every answer here maps directly to a scriptable command, so the run can be "
            "repeated later from the shell.",
            title="Unmochan Guided Mode",
            border_style="cyan",
        )
    )
    backend = Prompt.ask("DFT code", choices=["vasp", "qe"], default="vasp")
    while True:
        _print_guided_menu(backend)
        choice = Prompt.ask("Workflow number", choices=[str(i) for i in range(6)], default="0")
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
    table.add_column("Step")
    table.add_column("What it does")
    table.add_row("1", f"DFT code: {backend.upper()}", "Switch between VASP and Quantum ESPRESSO")
    table.add_row("2", "make-kpoints", "Write the folded supercell k-path for the DFT run")
    table.add_row("3", "weights", "Read WAVECAR/QE wavefunctions and compute spectral weights")
    table.add_row("4", "unfold", "Combine bands and weights into an unfolded table and plot")
    table.add_row("5", "validate", "Check structures, k-point order, and backend-specific inputs")
    table.add_row("0", "Exit", "Leave guided mode")
    console.print(table)


def _guided_set_backend() -> None:
    _GUIDED_BACKEND_STATE["backend"] = Prompt.ask(
        "DFT code",
        choices=["vasp", "qe"],
        default=_GUIDED_BACKEND_STATE["backend"],
    )


def _guided_make_kpoints(backend: str) -> None:
    path_json = prompt_existing_path("Primitive-cell k-path JSON")
    default_name = "KPOINTS.unfold" if backend == "vasp" else "qe_kpoints_supercell.in"
    kpoints = prompt_output_path(f"Output {backend.upper()} k-points file", default_name)
    kmap = prompt_output_path("Output kmap TSV", "kmap.tsv")
    ticks = prompt_output_path("Output tick TSV", "path_ticks.tsv")
    make_kpoints(code=backend, path_json=path_json, kpoints=kpoints, kmap=kmap, ticks=ticks)


def _guided_weights(backend: str) -> None:
    kmap = prompt_existing_path("kmap TSV from make-kpoints")
    matrix = Prompt.ask('Supercell transform T, nine integers, e.g. "2 0 0 0 2 0 0 0 1"')
    out = prompt_output_path("Output weights table", "weights.dat")
    if backend == "vasp":
        wavecar = prompt_existing_path("VASP WAVECAR from the supercell run")
        spin = prompt_optional_int("Spin channel, 1 or 2 (blank for 1)") or 1
        weights_command(code=backend, kmap=kmap, matrix=matrix, wavecar=wavecar, spin=spin, out=out)
        return

    source = Prompt.ask("QE wavefunction source", choices=["save", "coefficients"], default="save")
    bands = None
    nbnd = None
    if Confirm.ask("Infer band count from bands.dat.gnu?", default=True):
        bands = prompt_existing_path("QE bands.dat.gnu from bands.x")
    else:
        nbnd = int(Prompt.ask("Number of bands"))
    if source == "coefficients":
        coefficients = prompt_existing_path("Exported coefficient table")
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
        save_dir = prompt_existing_path("QE prefix.save directory")
        spin = prompt_optional_int("Spin channel, 1 or 2 (blank for 1)") or 1
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
    kmap = prompt_existing_path("kmap TSV from make-kpoints")
    ticks = prompt_optional_existing_path("Tick TSV, blank to infer from kmap")
    matrix = Prompt.ask(
        "Supercell transform T, nine integers, blank if using external weights",
        default="",
    )
    transform = matrix or None
    fermi = float(Prompt.ask("Energy zero / Fermi level to subtract (eV)", default="0.0"))
    out = prompt_output_path("Output unfolded table", "unfolded_bands.dat")
    plot = prompt_output_path("Output plot", "unfolded_bands.png")
    write_weights = None
    if backend == "vasp":
        source = Prompt.ask("VASP band source", choices=["wavecar", "eigenval"], default="wavecar")
        if source == "wavecar":
            wavecar = prompt_existing_path("VASP WAVECAR from the supercell run")
            spin = prompt_optional_int("Spin channel, 1 or 2 (blank for 1)") or 1
            if Confirm.ask("Write computed weights?", default=True):
                write_weights = prompt_output_path("Output weights table", "weights.dat")
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
            eigenval = prompt_existing_path("VASP EIGENVAL from the supercell run")
            weights = prompt_optional_existing_path("Weights table, blank for unit weights")
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

    bands = prompt_existing_path("QE bands.dat.gnu from bands.x")
    source = Prompt.ask(
        "QE spectral-weight source",
        choices=["save", "weights", "coefficients", "unit"],
        default="save",
    )
    qe_save_dir: Path | None = None
    qe_weights: Path | None = None
    coefficients: Path | None = None
    if source == "save":
        qe_save_dir = prompt_existing_path("QE prefix.save directory")
    elif source == "weights":
        qe_weights = prompt_existing_path("Existing weights table")
    elif source == "coefficients":
        coefficients = prompt_existing_path("Exported coefficient table")
    if source in {"save", "coefficients"} and Confirm.ask("Write computed weights?", default=True):
        write_weights = prompt_output_path("Output weights table", "weights.dat")
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
        qe_save_dir=qe_save_dir,
        weights=qe_weights,
        coefficients=coefficients,
    )


def _guided_validate_backend(backend: str) -> None:
    if backend == "vasp":
        if Confirm.ask("Validate primitive/supercell mapping?", default=True):
            _guided_vasp_map()
        if Confirm.ask("Validate VASP EIGENVAL k-points?", default=False):
            _guided_vasp_validate_eigenval()
    else:
        console.print(
            "QE preflight is built into make-kpoints, weights, and unfold: XML metadata, "
            "k-point order, wavefunction storage, and band/weight shapes are checked when "
            "those files are read."
        )


def _guided_vasp_map() -> None:
    primitive = prompt_existing_path("Primitive POSCAR/CONTCAR")
    supercell = prompt_existing_path("Supercell POSCAR/CONTCAR")
    vasp_map(primitive=primitive, supercell=supercell)


def _guided_vasp_fold_kpoints() -> None:
    primitive = prompt_existing_path("Primitive POSCAR/CONTCAR")
    supercell = prompt_existing_path("Supercell POSCAR/CONTCAR")
    points = prompt_repeated("Primitive k-point, e.g. G:0,0,0 or 0.5,0,0")
    vasp_fold_kpoints(primitive=primitive, supercell=supercell, kpoint=points)


def _guided_vasp_validate_eigenval() -> None:
    primitive = prompt_existing_path("Primitive POSCAR/CONTCAR")
    supercell = prompt_existing_path("Supercell POSCAR/CONTCAR")
    eigenval = prompt_existing_path("VASP EIGENVAL")
    points = None
    if Confirm.ask("Provide primitive k-points to validate?", default=True):
        points = prompt_repeated("Primitive k-point, e.g. G:0,0,0 or 0.5,0,0")
    atol = float(Prompt.ask("Fractional k-point tolerance", default="1e-8"))
    vasp_validate_eigenval(
        primitive=primitive,
        supercell=supercell,
        eigenval=eigenval,
        kpoint=points,
        atol=atol,
    )


def _guided_qe_make_kpoints() -> None:
    path_json = prompt_existing_path("Primitive path JSON")
    qe_kpoints = prompt_output_path("Output QE K_POINTS file", "qe_kpoints_supercell.in")
    kmap = prompt_output_path("Output kmap TSV", "kmap.tsv")
    ticks = prompt_output_path("Output tick TSV", "path_ticks.tsv")
    qe_make_kpoints(path_json=path_json, qe_kpoints=qe_kpoints, kmap=kmap, ticks=ticks)


def _guided_qe_weights() -> None:
    kmap = prompt_existing_path("kmap TSV")
    matrix = Prompt.ask('Transform matrix, nine numbers, e.g. "2 0 0 0 2 0 0 0 1"')
    source = Prompt.ask(
        "Weight source",
        choices=["save", "coefficients"],
        default="save",
    )
    bands = None
    nbnd = None
    if Confirm.ask("Infer band count from bands.dat.gnu?", default=True):
        bands = prompt_existing_path("QE bands.dat.gnu")
    else:
        nbnd = int(Prompt.ask("Number of bands"))
    out = prompt_output_path("Output weights table", "weights.dat")
    if source == "coefficients":
        coefficients = prompt_existing_path("Coefficient table")
        qe_weights(
            kmap=kmap,
            matrix=matrix,
            coefficients=coefficients,
            bands=bands,
            nbnd=nbnd,
            out=out,
        )
    else:
        save_dir = prompt_existing_path("QE prefix.save directory")
        spin = prompt_optional_int("Spin channel (blank for none)")
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
    bands = prompt_existing_path("QE bands.dat.gnu")
    kmap = prompt_existing_path("kmap TSV")
    ticks = prompt_optional_existing_path("Tick TSV, blank to infer from kmap")
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
        qe_save_dir = prompt_existing_path("QE prefix.save directory")
    elif source == "weights":
        weights = prompt_existing_path("Existing weights table")
    elif source == "coefficients":
        coefficients = prompt_existing_path("Coefficient table")
    fermi = float(Prompt.ask("Fermi energy to subtract", default="0.0"))
    out = prompt_output_path("Output unfolded table", "unfolded_bands.dat")
    plot = prompt_output_path("Output plot", "unfolded_bands.png")
    write_weights = None
    if source in {"save", "coefficients"} and Confirm.ask("Write computed weights?", default=True):
        write_weights = prompt_output_path("Output weights table", "weights.dat")
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
