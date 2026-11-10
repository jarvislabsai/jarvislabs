from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from jarvislabs import config
from jarvislabs.cli import render, setup
from jarvislabs.cli.app import app

runner = CliRunner()

WORDMARK_TOP_ROW = "    ▄▄▄▄▄▖                       ▄▄▖        ▐██          ██▌"


@pytest.fixture()
def terminal(server, monkeypatch):
    """A first-time setup in an interactive terminal 80 columns wide; the API key typed is "jarvis-key"."""
    monkeypatch.setattr(render, "can_prompt", lambda: True)
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(setup.questionary, "password", lambda message, **kwargs: MagicMock(ask=lambda: "jarvis-key"))
    monkeypatch.setattr("rich.console.getpass", lambda prompt, stream=None: "jarvis-key")
    return monkeypatch


def test_first_setup_opens_with_the_wordmark(terminal):
    result = runner.invoke(app, ["setup"], input="n\n")

    assert result.exit_code == 0, result.output
    assert WORDMARK_TOP_ROW in result.output
    assert "CLI Setup" not in result.output


@pytest.mark.parametrize(
    ("saved_key", "columns", "tty", "args"),
    [
        pytest.param(True, "79", True, ["setup"], id="re-run-narrow"),
        pytest.param(False, "79", True, ["setup"], id="narrow"),
        pytest.param(False, "80", False, ["setup"], id="piped"),
        pytest.param(False, "80", True, ["setup", "--yes"], id="yes"),
    ],
)
def test_setup_shows_the_one_line_header_otherwise(terminal, saved_key, columns, tty, args):
    if saved_key:
        config.save_config({"auth": {"token": "jarvis-key"}})
    terminal.setenv("COLUMNS", columns)
    terminal.setattr(render, "can_prompt", lambda: tty)

    result = runner.invoke(app, args, input="n\nn\n")

    assert result.exit_code == 0, result.output
    assert "\nJarvislabs CLI setup\n" in result.output
    assert WORDMARK_TOP_ROW not in result.output


def test_setup_asks_for_the_jarvis_key(terminal):
    terminal.setattr(render, "can_prompt", lambda: False)

    result = runner.invoke(app, ["setup"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "Get your API key at: https://jarvislabs.ai/settings/api-keys" in result.output
    assert "? API key: " in result.output
    assert "JarvisLabs" not in result.output
    assert "API token" not in result.output


def test_a_repeat_setup_in_a_wide_terminal_still_shows_the_wordmark(terminal):
    config.save_config({"auth": {"token": "jarvis-key"}})
    terminal.setenv("COLUMNS", "80")
    terminal.setattr(render, "can_prompt", lambda: True)

    result = runner.invoke(app, ["setup"], input="n\nn\n")

    assert result.exit_code == 0, result.output
    assert WORDMARK_TOP_ROW in result.output
