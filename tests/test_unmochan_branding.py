from typer.testing import CliRunner

import unfoldlab
from unfoldlab.cli.main import app, main, unfoldlab_main

runner = CliRunner()


def test_unmochan_import_is_a_public_compatibility_alias() -> None:
    import unmochan

    assert unmochan.__version__ == unfoldlab.__version__
    assert unmochan.EffectiveBandStructure is unfoldlab.EffectiveBandStructure


def test_primary_cli_name_is_unmochan() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Usage: unmochan" in result.output
    assert "UNfolding MOmentum-space Crystal Hamiltonian ANalysis" in result.output


def test_console_entrypoint_program_names_are_explicit() -> None:
    assert main(["--help"], prog_name="unmochan") == 0
    assert unfoldlab_main(["--help"]) == 0
