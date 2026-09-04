"""Entry point and command registration for the `unmochan` executable."""

from __future__ import annotations

import typer

from unmochan.cli import analysis as _analysis  # noqa: F401  (registers commands)
from unmochan.cli import backends as _backends  # noqa: F401  (registers commands)
from unmochan.cli import commands as _commands  # noqa: F401  (registers commands)
from unmochan.cli import diagnostics as _diagnostics  # noqa: F401  (registers commands)
from unmochan.cli import symmetry as _symmetry  # noqa: F401  (registers commands)
from unmochan.cli.app import app, qe_app, vasp_app
from unmochan.cli.common import click_exception_classes, console

__all__ = ["app", "main", "qe_app", "vasp_app"]


def main(argv: list[str] | None = None, *, prog_name: str = "unmochan") -> int:
    command = typer.main.get_command(app)
    click_exceptions = click_exception_classes("ClickException")
    exit_exceptions = click_exception_classes("Exit")
    abort_exceptions = click_exception_classes("Abort")
    try:
        command(args=argv, prog_name=prog_name, standalone_mode=False)
    except click_exceptions as exc:
        exc.show()
        return int(exc.exit_code)
    except exit_exceptions as exc:
        return int(exc.exit_code)
    except abort_exceptions:
        console.print("Aborted.")
        return 1
    except (ValueError, OSError) as exc:
        console.print(f"Error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
