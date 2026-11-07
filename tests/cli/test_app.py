from __future__ import annotations

import inspect

import pytest
import typer.main
from rich.text import Text
from typer.testing import CliRunner

from jarvislabs.cli import commands, connect, subscription  # noqa: F401  (registers the commands)
from jarvislabs.cli.app import app

runner = CliRunner()


@pytest.mark.parametrize(
    "command",
    [
        ["connect"],
        ["disconnect"],
        ["logout"],
        ["status"],
        ["subscribe"],
        ["usage"],
        ["usage", "use-balance"],
        ["usage", "manage"],
    ],
    ids=" ".join,
)
def test_new_commands_help_wraps_cleanly(command):
    group = typer.main.get_command(app)
    for name in command:
        group = group.commands[name]
    paragraphs = inspect.cleandoc(group.help or "").split("\n\n")

    # Wide enough that the help itself wraps no line.
    result = runner.invoke(app, [*command, "--help"], env={"COLUMNS": "1000"})

    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in result.output.splitlines()]
    # Each paragraph on one line: the terminal wraps it, never a line break inside it. An example, a
    # comment and its command, keeps its own lines.
    for paragraph in (Text.from_markup(p).plain for p in paragraphs):
        if paragraph.lstrip().startswith("#"):
            assert all(line.strip() in lines for line in paragraph.splitlines())
        else:
            assert " ".join(paragraph.split()) in lines


def test_help_says_what_each_command_is_for():
    for command, purpose in (
        ("status", "See your account and coding tools. Plan usage: jl usage."),
        ("connect", "Write the Jarvislabs provider into your coding tools."),
        ("disconnect", "Remove jl's provider and key from your coding tools."),
    ):
        result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "1000"})
        assert result.exit_code == 0, result.output
        assert purpose in " ".join(result.output.split())

    result = runner.invoke(app, ["connect", "--help"], env={"COLUMNS": "1000"})
    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in result.output.splitlines()]
    assert "# Connect one tool, or see how to install it" in lines
    assert "jl connect qwen" in lines
