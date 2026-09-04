from typer.testing import CliRunner

import unmochan
from unmochan.cli.main import app, main

runner = CliRunner()


def test_unmochan_import_exposes_the_project_version() -> None:
    assert unmochan.__version__ == "0.1.0"


def test_primary_cli_name_is_unmochan() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Usage: unmochan" in result.output
    assert "UNfolding MOmentum-space Crystal Hamiltonian ANalysis" in result.output


def test_console_entrypoint_program_name_is_unmochan() -> None:
    assert main(["--help"]) == 0
