"""The Typer application object and its root callback.

Kept separate from the command bodies so that `commands`, `backends` and
`guided` can register against it without importing each other.
"""

from __future__ import annotations

from typing import Annotated

import typer

from unfoldlab import __version__
from unfoldlab.cli.common import console
from unfoldlab.cli.model import model_app
from unfoldlab.cli.phonon import phonon_app

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
app.add_typer(model_app, name="model")
app.add_typer(phonon_app, name="phonon")


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
        # Imported here: `guided` drives the commands, which register on `app`.
        from unfoldlab.cli.guided import run_guided_cli

        run_guided_cli()


@app.command("guide")
def guide() -> None:
    """Launch the guided interactive workflow menu."""

    from unfoldlab.cli.guided import run_guided_cli

    run_guided_cli()
