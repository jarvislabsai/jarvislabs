from __future__ import annotations

import inspect
import re

import pytest
import typer.main
import typer.rich_utils as rich_utils
from rich.text import Text
from typer.testing import CliRunner

from jarvislabs.cli import commands, connect, instance, subscription  # noqa: F401  (registers the commands)
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


def test_logout_help_names_its_flags_without_backticks():
    result = runner.invoke(app, ["logout", "--help"], env={"COLUMNS": "1000"})

    assert result.exit_code == 0, result.output
    assert "`" not in result.output
    assert "--keep-tools leaves them connected; that keeps the key saved" in " ".join(result.output.split())


def test_create_help_has_an_example_with_a_comment_for_each_kind_of_machine():
    result = runner.invoke(app, ["create", "--help"], env={"COLUMNS": "1000"})

    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in result.output.splitlines()]
    for comment, command in (
        ("# A container with PyTorch and Jupyter", "jl create --gpu A100"),
        ("# A bare-metal VM with SSH only", "jl create --gpu H100 --vm"),
        ("# A CPU-only VM, with 8 vCPUs and 32 GB of RAM", "jl create --cpu --vcpus 8 --ram 32 --vm"),
    ):
        assert comment in lines
        assert command in lines
    # The docstring holds the examples, so the help has no plain epilog repeating them.
    assert result.output.count("jl create --gpu A100") == 1


def test_root_help_replaces_the_tools_row_with_one_coding_agent_line():
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "80"})

    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in result.output.splitlines()]
    agent_line = "Using a coding agent? jl setup adds the jl skill; jl connect adds our models."
    assert agent_line in lines
    assert len(agent_line) <= 80
    assert "Tools   jl connect" not in result.output


def test_subcommand_help_uses_no_warning_or_success_colours():
    result = runner.invoke(
        app,
        ["connect", "--help"],
        env={"FORCE_COLOR": "1", "TERM": "xterm-256color", "COLUMNS": "100"},
    )

    assert result.exit_code == 0, result.output
    # Yellow (33) means warning and green (32) means success in jl, so help never uses either.
    sgr = {
        int(code)
        for match in re.finditer(r"\x1b\[([0-9;]*)m", result.output)
        for code in match.group(1).split(";")
        if code
    }
    assert 33 not in sgr
    assert 32 not in sgr


def test_option_defaults_print_in_body_text_not_dim(monkeypatch):
    # CliRunner's output is not a terminal, so the console has to be told to colour it.
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", True)

    result = runner.invoke(app, ["create", "--help"], color=True, env={"COLUMNS": "100"})

    assert result.exit_code == 0, result.output
    assert "[default: pytorch]" in result.output
    assert "\x1b[2m[default:" not in result.output


def test_help_shows_a_commands_help_and_refuses_an_unknown_name():
    result = runner.invoke(app, ["help", "connect"], env={"COLUMNS": "1000"})
    assert result.exit_code == 0, result.output
    assert "Write the Jarvislabs provider into your coding tools." in result.output

    # A name after a command that has no subcommands is unknown too, not a crash.
    for args in (["help", "nosuch"], ["help", "status", "extra"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1, result.output
        assert "Unknown command" in result.output


def test_metavars_are_bold_under_every_typer_name_for_them():
    import typer.rich_utils as rich_utils

    # Typer 0.28 renamed STYLE_METAVAR to STYLE_TYPES; uv tool install picks the newest Typer, so a
    # name jl doesn't set would turn <id> and <command> back to Typer's bold yellow.
    assert rich_utils.STYLE_METAVAR == "bold"
    assert rich_utils.STYLE_TYPES == "bold"
