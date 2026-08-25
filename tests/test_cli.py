from typer.testing import CliRunner

from patchpoint.cli import app

runner = CliRunner()


def test_demo_command_runs_offline() -> None:
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "bm25" in result.stdout
