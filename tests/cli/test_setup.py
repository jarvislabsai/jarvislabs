"""`jl setup`'s login, run end to end with the servers faked at Transport."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console
from typer.testing import CliRunner

from jarvislabs import client as client_module, config
from jarvislabs.cli import render, setup
from jarvislabs.cli.app import app
from jarvislabs.exceptions import AuthError
from tests.cli.test_connect import ctrl_c_at

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture()
def typed(server: MagicMock, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A first-time setup in a terminal: fill the returned list with the Jarvis keys typed, in order.

    Only "jarvis-key" is a valid key; the server refuses any other with "Invalid token".
    """
    monkeypatch.delenv("JL_API_KEY")
    monkeypatch.setattr(render, "can_prompt", lambda: True)
    refused = MagicMock(request=MagicMock(side_effect=AuthError("Invalid token")))
    monkeypatch.setattr(client_module, "Transport", lambda token: server if token == "jarvis-key" else refused)
    keys: list[str] = []

    def password(message: str, **kwargs):
        return MagicMock(ask=lambda: keys.pop(0))

    monkeypatch.setattr(setup.questionary, "password", password)
    return keys


def test_the_key_is_asked_with_a_masked_questionary_prompt(typed: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    typed += ["jarvis-key"]
    asked = []
    answered = []

    def password(message: str, **kwargs):
        question = MagicMock(ask=lambda: "jarvis-key")
        asked.append((message, kwargs, question))
        return question

    real_answer = render.answer

    def answer(question):
        answered.append(question)
        return real_answer(question)

    monkeypatch.setattr(setup.questionary, "password", password)
    monkeypatch.setattr(setup.render, "answer", answer)

    result = runner.invoke(app, ["setup"], input="n\n")

    assert result.exit_code == 0, result.output
    # Through questionary's masked prompt and render.answer, which turns Ctrl-C into exit 130.
    assert [message for message, _, _ in asked] == ["Jarvis key:"]
    assert asked[0][1]["style"] is render.PROMPT_STYLE
    assert answered == [asked[0][2]]


def test_ctrl_c_at_the_key_prompt_aborts_with_130(typed: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup.questionary, "password", lambda message, **kwargs: MagicMock(ask=lambda: None))

    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 130, result.output


def test_the_key_prompt_shows_a_star_per_character(server: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JL_API_KEY")
    monkeypatch.setattr(render, "can_prompt", lambda: True)
    refused = MagicMock(request=MagicMock(side_effect=AuthError("Invalid token")))
    monkeypatch.setattr(client_module, "Transport", lambda token: server if token == "jarvis-key" else refused)
    screen = io.StringIO()
    real_password = setup.questionary.password

    def password(message: str, **kwargs):
        size = Size(rows=40, columns=80)
        with create_pipe_input() as keys:
            keys.send_text("jarvis-key\r")
            output = Vt100_Output(screen, lambda: size, term="dumb")
            typed = real_password(message, input=keys, output=output, **kwargs).ask()
        return MagicMock(ask=lambda: typed)

    monkeypatch.setattr(setup.questionary, "password", password)

    result = runner.invoke(app, ["setup"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "Jarvis key:" in screen.getvalue()
    assert "**********" in screen.getvalue()
    assert "jarvis-key" not in screen.getvalue()


def test_setup_asks_again_after_a_wrong_key(typed: list[str]) -> None:
    typed += ["wrong-key", "jarvis-key"]

    # No to installing the skill files.
    result = runner.invoke(app, ["setup"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "Authentication failed: Invalid token" in result.output
    assert "Logged in as Ada" in result.output
    assert config.load_config()["auth"] == {"token": "jarvis-key"}


def test_setup_stops_after_three_wrong_keys(typed: list[str]) -> None:
    typed += ["wrong-1", "wrong-2", "wrong-3", "jarvis-key"]

    result = runner.invoke(app, ["setup"], input="n\n")

    assert result.exit_code == 1
    assert result.output.count("Authentication failed: Invalid token") == 3
    assert typed == ["jarvis-key"]
    assert "auth" not in config.load_config()


def test_setup_from_the_installer_skips_the_next_steps(typed: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    typed += ["jarvis-key"]
    monkeypatch.setenv("JL_FROM_INSTALLER", "1")

    result = runner.invoke(app, ["setup"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "get started" not in result.output
    assert "jl create --gpu A100" not in result.output
    assert "Next:" not in result.output


def test_setup_ends_with_a_neutral_next_step(typed: list[str]) -> None:
    typed += ["jarvis-key"]

    result = runner.invoke(app, ["setup"], input="n\n")

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[-1] == "Next: see all commands: jl --help"
    assert lines[-2] == ""
    assert "Get started" in result.output
    assert "quick reference" not in result.output


def test_enter_at_install_agent_skills_installs_them(
    typed: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    typed += ["jarvis-key"]
    monkeypatch.setattr(setup, "UNIVERSAL_PATH", tmp_path / "skills" / "SKILL.md")
    # No agent-specific path ticked.
    monkeypatch.setattr(setup.questionary, "checkbox", lambda message, choices, **kwargs: MagicMock(ask=lambda: []))

    result = runner.invoke(app, ["setup"], input="\n")

    assert result.exit_code == 0, result.output
    assert "? Install agent skills? (Y/n)" in result.output
    assert (tmp_path / "skills" / "SKILL.md").exists()


def test_the_installed_skill_path_is_a_dim_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(setup, "UNIVERSAL_PATH", tmp_path / "skills" / "SKILL.md")
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, color_system="standard", theme=render.theme)
    monkeypatch.setattr(render, "console", console)

    setup._skill_install_flow("universal")

    line = next(line for line in buffer.getvalue().splitlines() if "Agent skill installed" in line)
    assert "\x1b[2m~/skills/SKILL.md\x1b[0m" in line


@pytest.mark.parametrize("question", ["skills", "agents"])
def test_ctrl_c_in_setup_aborts_with_130(typed: list[str], monkeypatch: pytest.MonkeyPatch, question: str) -> None:
    typed += ["jarvis-key"]
    if question == "skills":
        ctrl_c_at(monkeypatch, "Install agent skills?")
    else:
        # One extra agent has no picker now, so give it a second row to Ctrl-C on.
        monkeypatch.setitem(setup.ADDITIONAL_AGENTS, "other-agent", ("Other Agent", None))
    # questionary's questions answer None to Ctrl-C.
    monkeypatch.setattr(setup.questionary, "checkbox", lambda message, choices, **kwargs: MagicMock(ask=lambda: None))

    # Yes to installing the skills, for the agents picker that follows.
    result = runner.invoke(app, ["setup"], input="y\n")

    assert result.exit_code == 130, result.output


@pytest.fixture()
def script(server: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A script or coding agent running jl setup: no terminal, nothing on stdin. Returns the skills folder."""
    monkeypatch.delenv("JL_API_KEY")
    monkeypatch.setattr(render, "can_prompt", lambda: False)
    skills = tmp_path / "skills"
    monkeypatch.setattr(setup, "UNIVERSAL_PATH", skills / "SKILL.md")
    return skills


def test_setup_without_a_terminal_logs_in_and_skips_the_skills(script: Path) -> None:
    result = runner.invoke(app, ["setup", "--token", "jarvis-key"])

    assert result.exit_code == 0, result.output
    assert "Logged in as Ada" in result.output
    assert "Skipped agent skills" not in result.output
    assert config.load_config()["auth"] == {"token": "jarvis-key"}
    assert not script.exists()


def test_setup_without_a_terminal_keeps_the_saved_login(script: Path) -> None:
    config.save_config({"auth": {"token": "jarvis-key"}})

    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 0, result.output
    assert "Logged in as Ada" in result.output
    assert config.load_config()["auth"] == {"token": "jarvis-key"}
