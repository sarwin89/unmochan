import re

from typer.testing import CliRunner

import unmochan
from unmochan.cli.main import app, main

runner = CliRunner()

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def test_unmochan_import_exposes_the_project_version() -> None:
    assert unmochan.__version__ == "0.1.0"


def test_primary_cli_name_is_unmochan() -> None:
    result = runner.invoke(app, ["--help"])
    output = _plain(result.output)

    assert result.exit_code == 0, result.output
    assert "Usage: unmochan" in output
    assert "UNfolding MOmentum-space Crystal Hamiltonian ANalysis" in output


def test_console_entrypoint_program_name_is_unmochan() -> None:
    assert main(["--help"]) == 0
