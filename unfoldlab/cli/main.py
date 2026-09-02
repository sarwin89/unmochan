"""Entry points for the `unmochan` and legacy `unfoldlab` executables.

The command bodies live in `unfoldlab.cli.commands`, `unfoldlab.cli.backends`,
`unfoldlab.cli.diagnostics`, `unfoldlab.cli.analysis` and
`unfoldlab.cli.symmetry`; importing them here is what registers them on the
Typer application.  `app` and `main` are re-exported from this module because
that is where they have always been.
"""

from __future__ import annotations

import typer

from unfoldlab.cli import analysis as _analysis  # noqa: F401  (registers commands)
from unfoldlab.cli import backends as _backends  # noqa: F401  (registers commands)
from unfoldlab.cli import commands as _commands  # noqa: F401  (registers commands)
from unfoldlab.cli import diagnostics as _diagnostics  # noqa: F401  (registers commands)
from unfoldlab.cli import symmetry as _symmetry  # noqa: F401  (registers commands)
from unfoldlab.cli.app import app, qe_app, vasp_app
from unfoldlab.cli.common import click_exception_classes, console

__all__ = ["app", "main", "qe_app", "unfoldlab_main", "vasp_app"]


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
        # Invalid inputs and unreadable files are user errors, not crashes.
        console.print(f"Error: {exc}")
        return 2
    return 0


def unfoldlab_main(argv: list[str] | None = None) -> int:
    """Compatibility entry point for the historical `unfoldlab` command."""

    return main(argv, prog_name="unfoldlab")


if __name__ == "__main__":
    raise SystemExit(main())
