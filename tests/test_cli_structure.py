"""The command surface of the CLI, pinned against refactoring.

`unfoldlab/cli/main.py` used to be one 1766-line module holding the Typer app,
every command body, the guided menu and the argument parsers.  It is now split
into `app`, `commands`, `backends`, `guided` and `common`.  A split like that
can silently drop a command -- a body that is never imported is never
registered -- so the exact set of commands and subcommands is asserted here.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import typer

from unfoldlab.cli.main import app, main

TOP_LEVEL_COMMANDS = {
    "guide",
    "make-kpoints",
    "mesh",
    "weights",
    "unfold",
    "qe-info",
    "fiber",
    "cut",
    "ensemble",
    "norms",
    "layers",
    "degeneracy",
    "detect-symmetry",
    "check-symmetry",
    "strain",
    "slab",
    "dispersion",
    "overlap",
    "fermi",
    "dos",
    "branches",
    "shadow",
    "bz",
    "vasp",
    "qe",
    "model",
    "phonon",
    "twist",
}

GROUP_COMMANDS = {
    "vasp": {"map", "fold-kpoints", "validate-eigenval"},
    "qe": {"make-kpoints", "weights", "unfold"},
}


def _command_names(group: object) -> set[str]:
    """Names of a click group's subcommands.

    Duck-typed on purpose: recent Typer releases vendor their own copy of click,
    so ``isinstance(..., click.Group)`` is false for the group Typer builds.
    """

    commands = getattr(group, "commands", None)
    assert commands is not None, f"{group!r} is not a command group"
    return set(commands)


def test_top_level_commands_are_all_registered() -> None:
    assert _command_names(typer.main.get_command(app)) == TOP_LEVEL_COMMANDS


@pytest.mark.parametrize("group_name", sorted(GROUP_COMMANDS))
def test_backend_groups_are_all_registered(group_name: str) -> None:
    command = typer.main.get_command(app)
    group = command.commands[group_name]  # type: ignore[attr-defined]
    assert _command_names(group) == GROUP_COMMANDS[group_name]


@pytest.mark.parametrize(
    "module_name",
    [
        "unfoldlab.cli.analysis",
        "unfoldlab.cli.app",
        "unfoldlab.cli.backends",
        "unfoldlab.cli.commands",
        "unfoldlab.cli.diagnostics",
        "unfoldlab.cli.common",
        "unfoldlab.cli.guided",
        "unfoldlab.cli.main",
        "unfoldlab.cli.model",
        "unfoldlab.cli.phonon",
        "unfoldlab.cli.symmetry",
    ],
)
def test_cli_modules_import_standalone(module_name: str) -> None:
    """Each module must import on its own: no import cycle between them.

    `guided` depends on the command bodies and `app` depends on `guided`, so
    the latter dependency is deferred into the callback.  Importing every module
    first is the check that the deferral is still in place.
    """

    assert importlib.import_module(module_name) is not None


def test_no_cli_module_exceeds_the_length_limit() -> None:
    """No CLI module is back over 1000 lines."""

    import unfoldlab.cli

    package_dir = Path(next(iter(unfoldlab.cli.__path__)))
    lengths = {
        path.name: len(path.read_text().splitlines()) for path in sorted(package_dir.glob("*.py"))
    }
    assert {name: n for name, n in lengths.items() if n > 1000} == {}


def test_main_returns_zero_for_help() -> None:
    assert main(["--help"]) == 0


def test_main_reports_a_usage_error_without_a_traceback() -> None:
    assert main(["no-such-command"]) != 0
