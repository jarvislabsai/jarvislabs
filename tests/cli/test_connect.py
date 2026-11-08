from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import webbrowser
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

import httpx
import pytest
import questionary
import typer
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console
from typer.testing import CliRunner

from jarvislabs import client as client_module, config, harnesses
from jarvislabs.cli import commands, connect, render, setup, update  # importing them registers the commands
from jarvislabs.cli.app import app
from jarvislabs.config import Connection, Replaced
from jarvislabs.exceptions import APIError, JarvislabsError, NotFoundError, error_from_response

runner = CliRunner()

USAGE = {
    "plan": {"id": 2, "name": "Pro"},
    "period": {"start": "2026-10-20T09:00:00Z", "renews_at": "2026-11-20T09:00:00Z", "cancel_at_period_end": False},
    "use_balance": False,
    "renewal_pending": False,
    "windows": {
        "five_hour": {"percent_used": 0.0, "resets_at": "2026-11-05T16:05:00Z"},
        "weekly": {"percent_used": 0.0, "resets_at": "2026-11-09T00:00:00Z"},
        "monthly": {"percent_used": 0.0, "resets_at": "2026-11-20T09:00:00Z"},
    },
    "slots": {"in_use": 0, "limit": 4},
}

DEEPSEEK = {
    "id": "deepseek-v4.1-flash-1017",
    "name": "DeepSeek V4.1 Flash",
    "context_length": 1048576,
    "max_output_tokens": 65536,
    "input_modalities": ["text", "image", "video"],
    "output_modalities": ["text"],
    "capabilities": {"tool_calling": True, "structured_output": None},
}
GLM = {
    "id": "glm-5.3",
    "name": "GLM-5.3",
    "context_length": 230000,
    "input_modalities": ["text"],
    "output_modalities": ["text"],
    "capabilities": {"tool_calling": True, "structured_output": None},
}
GEMMA = {
    "id": "gemma-3-1b-it",
    "name": "Gemma 3 1B IT",
    "context_length": 32768,
    "max_output_tokens": None,
    "input_modalities": ["text"],
    "output_modalities": ["text"],
    "capabilities": {"tool_calling": None, "structured_output": None},
}

CREATED_KEY = {
    "id": "key_1",
    "name": "jl-devbox-f7a7",
    "kind": "subscription",
    "token": "jlsk-new-secret",
    "created_at": "2026-11-05T10:00:00Z",
}


REPLY = {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}


@pytest.fixture()
def subscribed(server, monkeypatch):
    """A subscriber on the machine "devbox" with no key yet; the catalog holds DeepSeek, GLM and Gemma.

    The OS gives the machine ID "machine-1", and the account's keys key_0 and key_1 are active.
    """
    monkeypatch.setattr(socket, "gethostname", lambda: "devbox")
    monkeypatch.setattr(connect, "_os_machine_id", lambda: "machine-1")
    server.routes[("GET", "users/api_keys")] = [
        {"id": "key_0", "name": "jl-devbox-f7a7", "kind": "subscription"},
        {"id": "key_1", "name": "jl-devbox-f7a7", "kind": "subscription"},
    ]
    server.routes[("GET", "subscription/usage")] = USAGE
    server.routes[("POST", "users/api_keys")] = CREATED_KEY
    server.routes[("GET", "models")] = {"object": "list", "data": [DEEPSEEK, GLM, GEMMA]}
    server.routes[("POST", "chat/completions")] = REPLY
    return server


@pytest.fixture()
def opencode_config(tmp_path):
    """OpenCode's global config file, under XDG_CONFIG_HOME (tmp_path, set by the server fixture)."""
    path = tmp_path / "opencode" / "opencode.json"
    path.parent.mkdir(parents=True)
    return path


def read_json(path):
    return json.loads(path.read_text())


def test_connect_writes_the_provider_models_and_default_into_opencode(subscribed, opencode_config):
    opencode_config.write_text('{\n  "theme": "tokyonight",\n  "model": "anthropic/claude-sonnet-4"\n}\n')

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    written = read_json(opencode_config)
    provider = written["providers"]["jarvislabs"]
    assert written["theme"] == "tokyonight"
    assert written["model"] == "jarvislabs/deepseek-v4.1-flash-1017"
    assert provider["name"] == "Jarvislabs"
    assert provider["package"] == "@opencode/ai/providers/openai-compatible"
    assert provider["settings"] == {"baseURL": "https://models.jarvislabs.net/v1", "apiKey": "jlsk-new-secret"}
    # OpenCode takes video, and a model the catalog doesn't mark as tool-calling gets no tools.
    assert {model_id: (m["name"], m["capabilities"]) for model_id, m in provider["models"].items()} == {
        "deepseek-v4.1-flash-1017": (
            "DeepSeek V4.1 Flash",
            {"tools": True, "input": ["text", "image", "video"], "output": ["text"], "reasoning": {"supported": False}},
        ),
        "glm-5.3": (
            "GLM-5.3",
            {"tools": True, "input": ["text"], "output": ["text"], "reasoning": {"supported": False}},
        ),
        "gemma-3-1b-it": (
            "Gemma 3 1B IT",
            {"tools": False, "input": ["text"], "output": ["text"], "reasoning": {"supported": False}},
        ),
    }
    # The V1 entry stays beside it: OpenCode installs that predate the providers one, and Kilo, read it.
    assert written["provider"]["jarvislabs"]["npm"] == "@ai-sdk/openai-compatible"
    assert written["provider"]["jarvislabs"]["options"] == {
        "baseURL": "https://models.jarvislabs.net/v1",
        "apiKey": "jlsk-new-secret",
    }


def test_connect_creates_a_subscription_key_when_subscribed(subscribed, opencode_config):
    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert ("POST", "users/api_keys", {"name": "jl-devbox-f7a7", "kind": "subscription"}) in subscribed.calls
    assert config.load_connection() == Connection(
        key_id="key_1",
        name="jl-devbox-f7a7",
        kind="subscription",
        token="jlsk-new-secret",
        harnesses={"opencode": ANY},
    )
    assert config.load_connection().harnesses["opencode"].created == (opencode_config,)


def test_connecting_a_second_harness_reuses_the_machine_key(subscribed, opencode_config):
    pi_models = opencode_config.parent.parent / ".pi" / "agent" / "models.json"
    config.save_connection(
        Connection(
            key_id="key_0",
            name="jl-devbox-f7a7",
            kind="subscription",
            token="jlsk-machine-secret",
            harnesses={"pi": Replaced({pi_models: {("defaultModel",): "glm-4.6"}})},
        )
    )

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert [call for call in subscribed.calls if call[:2] == ("POST", "users/api_keys")] == []
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-machine-secret"
    assert set(config.load_connection().harnesses) == {"pi", "opencode"}


def test_each_model_gets_its_context_and_max_output(subscribed, opencode_config):
    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    models = read_json(opencode_config)["provider"]["jarvislabs"]["models"]
    # GLM has no max_output_tokens and Gemma's is null: both fall back to a tenth of the context.
    assert {model_id: m["limit"] for model_id, m in models.items()} == {
        "deepseek-v4.1-flash-1017": {"context": 1048576, "output": 65536},
        "glm-5.3": {"context": 230000, "output": 23000},
        "gemma-3-1b-it": {"context": 32768, "output": 3276},
    }


@pytest.mark.parametrize(
    ("args", "default"),
    [([], "jarvislabs/deepseek-v4.1-flash-1017"), (["--model", "glm-5.3"], "jarvislabs/glm-5.3")],
    ids=["recommended", "flag"],
)
def test_the_default_model_is_the_recommended_one_or_the_flag(subscribed, opencode_config, args, default):
    result = runner.invoke(app, ["connect", "opencode", "--yes", *args])

    assert result.exit_code == 0, result.output
    assert read_json(opencode_config)["model"] == default


@pytest.mark.skipif(os.name == "nt", reason="Windows has no owner-only file mode")
def test_written_configs_are_owner_only(subscribed, opencode_config):
    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert opencode_config.stat().st_mode & 0o777 == 0o600
    assert config.config_path().stat().st_mode & 0o777 == 0o600


def test_reconnecting_refreshes_models_and_the_default(subscribed, opencode_config):
    runner.invoke(app, ["connect", "opencode", "--yes", "--model", "glm-5.3"])
    subscribed.routes[("GET", "models")] = {"object": "list", "data": [DEEPSEEK, GEMMA]}

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    written = read_json(opencode_config)
    assert set(written["provider"]["jarvislabs"]["models"]) == {"deepseek-v4.1-flash-1017", "gemma-3-1b-it"}
    assert written["model"] == "jarvislabs/deepseek-v4.1-flash-1017"


def test_the_key_name_is_the_hostname_without_its_domain_and_a_machine_id_hash(
    subscribed, opencode_config, monkeypatch
):
    monkeypatch.setattr(socket, "gethostname", lambda: "Adas-MacBook-Air.local")

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    # sha256("machine-1") starts with f7a7.
    name = "jl-Adas-MacBook-Air-f7a7"
    assert ("POST", "users/api_keys", {"name": name, "kind": "subscription"}) in subscribed.calls


def test_without_an_os_machine_id_a_random_one_is_saved_and_reused(subscribed, opencode_config, monkeypatch):
    monkeypatch.setattr(connect, "_os_machine_id", lambda: None)
    runner.invoke(app, ["connect", "opencode", "--yes"])
    config.save_connection(None)

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    names = [
        body["name"] for call, body in ((c[:2], c[2]) for c in subscribed.calls) if call == ("POST", "users/api_keys")
    ]
    assert len(names) == 2
    assert names[0] == names[1]
    assert re.fullmatch(r"jl-devbox-[0-9a-f]{4}", names[0])


def test_an_unknown_model_flag_writes_nothing(subscribed, opencode_config):
    result = runner.invoke(app, ["connect", "opencode", "--yes", "--model", "gpt-9"])

    assert result.exit_code == 1
    assert "Unknown model gpt-9" in result.output
    assert not opencode_config.exists()


def test_dry_run_prints_the_preview_and_writes_nothing(subscribed, opencode_config):
    original = '{\n  "model": "anthropic/claude-sonnet-4"\n}\n'
    opencode_config.write_text(original)

    # A "y" waits on stdin: a dry run must stop before it would be read.
    result = runner.invoke(app, ["connect", "opencode", "--dry-run"], input="y\n")

    assert result.exit_code == 0, result.output
    # One row per tool: mark, name, file; what it replaces of the user's on the line under it.
    assert result.stdout.splitlines() == [
        "jl will add the Jarvislabs provider  3 models, default deepseek-v4.1-flash-1017",
        "~ change what is already there",
        "",
        "~ OpenCode  ~/opencode/opencode.json",
        "            replaces anthropic/claude-sonnet-4",
    ]
    assert opencode_config.read_text() == original
    assert not opencode_config.with_name("opencode.json.jl-backup").exists()
    assert not config.config_path().exists()
    assert [call for call in subscribed.calls if call[0] == "POST"] == []


@pytest.mark.parametrize(
    ("args", "kept", "shown"),
    [
        (["connect", "--print", "zed", "--dry-run"], None, "--print writes no file, so it takes no --dry-run"),
        # A key kept after the last harness was disconnected, which --rotate-key alone revokes.
        (["connect", "--rotate-key", "--dry-run"], {}, "Would revoke this machine's key jl-devbox-f7a7."),
        (
            ["connect", "--rotate-key", "--dry-run"],
            {"opencode": Replaced()},
            "Would replace this machine's key jl-devbox-f7a7 and update 1 tool.",
        ),
    ],
    ids=["print", "rotate-key-alone", "rotate-key-with-a-tool"],
)
def test_a_dry_run_creates_or_revokes_no_key(subscribed, args, kept, shown):
    if kept is not None:
        config.save_connection(Connection("key_0", "jl-devbox-f7a7", "subscription", "jlsk-old", kept))
    before = config.load_connection()

    result = runner.invoke(app, args)

    assert shown in " ".join(result.output.split())
    assert [call for call in subscribed.calls if call[0] in ("POST", "DELETE")] == []
    assert config.load_connection() == before


def test_the_first_connect_saves_a_backup_once(subscribed, opencode_config):
    original = '{\n  // my theme\n  "theme": "tokyonight"\n}\n'
    opencode_config.write_text(original)
    backup = opencode_config.with_name("opencode.json.jl-backup")

    first = runner.invoke(app, ["connect", "opencode", "--yes"])
    second = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert backup.read_text() == original
    # Under the tool's row, the home folder as ~.
    assert "✓ OpenCode  added\n            your old config is at ~/opencode/opencode.json.jl-backup\n" in first.output
    assert "jl-backup" not in second.output
    assert config.load_connection().harnesses["opencode"].backups == (backup,)


def test_declining_the_preview_writes_nothing(subscribed, opencode_config):
    original = '{\n  "model": "anthropic/claude-sonnet-4"\n}\n'
    opencode_config.write_text(original)

    result = runner.invoke(app, ["connect", "opencode"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "Write these changes?" in result.output
    assert opencode_config.read_text() == original
    assert not opencode_config.with_name("opencode.json.jl-backup").exists()


def test_enter_at_the_preview_writes_the_changes(subscribed, opencode_config):
    result = runner.invoke(app, ["connect", "opencode"], input="\n")

    assert result.exit_code == 0, result.output
    assert "? Write these changes? (Y/n)" in result.output
    assert read_json(opencode_config)["model"] == "jarvislabs/deepseek-v4.1-flash-1017"


@pytest.mark.parametrize(
    ("args", "question"),
    [(["connect", "opencode"], "Write these changes?"), (["disconnect", "opencode"], "Disconnect OpenCode?")],
    ids=["connect", "disconnect"],
)
def test_no_terminal_stops_with_a_message(subscribed, opencode_config, args, question):
    if args[0] == "disconnect":
        runner.invoke(app, ["connect", "opencode", "--yes"])
    before = opencode_config.read_text() if opencode_config.exists() else None

    # A script: nothing on stdin answers the question.
    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert f'"{question}" needs an answer: run this in a terminal, or pass --yes.' in " ".join(result.output.split())
    assert (opencode_config.read_text() if opencode_config.exists() else None) == before


def test_a_config_jl_created_never_gets_a_backup(subscribed, opencode_config):
    first = runner.invoke(app, ["connect", "opencode", "--yes"])
    second = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert not opencode_config.with_name("opencode.json.jl-backup").exists()
    assert "jl-backup" not in second.output
    assert config.load_connection().harnesses["opencode"].created == (opencode_config,)


def test_reconnecting_previews_an_update_of_the_provider(subscribed, opencode_config):
    runner.invoke(app, ["connect", "opencode", "--yes", "--model", "glm-5.3"])

    result = runner.invoke(app, ["connect", "opencode", "--dry-run"])

    assert result.exit_code == 0, result.output
    # jl's own provider entry isn't listed; the default model it set before is.
    assert "~ OpenCode  ~/opencode/opencode.json\n            replaces jarvislabs/glm-5.3\n" in result.stdout


def test_declining_the_preview_creates_no_key(subscribed, opencode_config):
    result = runner.invoke(app, ["connect", "opencode"], input="n\n")

    assert result.exit_code == 0, result.output
    assert [call for call in subscribed.calls if call[0] == "POST"] == []
    assert not config.config_path().exists()


@pytest.fixture()
def terminal(server, tmp_path, monkeypatch):
    """An interactive terminal whose PATH holds only OpenCode.

    Set ``terminal.picked`` (the harness IDs ticked in the picker). The picker's arguments, and the
    server calls made before it, land in ``terminal.picker``; the questions asked in
    ``terminal.asked``. Every yes/no question is answered yes.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("opencode", "opencode.cmd"):  # .cmd so that Windows finds it too
        (bin_dir / name).write_text("")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(render, "can_prompt", lambda: True)
    fake = SimpleNamespace(picked=[], picker=None, asked=[])

    def checkbox(message, choices, **kwargs):
        fake.picker = SimpleNamespace(choices=choices, calls_before=list(server.calls), **kwargs)
        return MagicMock(ask=lambda: fake.picked)

    def confirm(message, **kwargs):
        fake.asked.append(message)
        return MagicMock(ask=lambda: True)

    def select(message, choices, **kwargs):
        fake.asked.append(message)
        return MagicMock(ask=lambda: None)

    monkeypatch.setattr(connect.questionary, "checkbox", checkbox)
    monkeypatch.setattr(connect.questionary, "confirm", confirm)
    monkeypatch.setattr(connect.questionary, "select", select)
    return fake


def test_connect_list_names_every_tool(server):
    result = runner.invoke(app, ["connect", "--list"])

    assert result.exit_code == 0, result.output
    for harness in harnesses.HARNESSES:
        assert harness.id in result.output
        assert harness.name in result.output
    # No tool is print-only today, so the legend doesn't explain a label no row shows.
    assert "print only" not in result.output
    # The whole list is data, so a pipe gets the legend too.
    assert "connect      jl writes the config" in result.stdout
    assert server.calls == []


def test_connect_list_says_print_only_only_where_print_works(server):
    result = runner.invoke(app, ["connect", "--list"])

    rows = {line.split()[0]: line for line in result.stdout.splitlines() if line.startswith("  ") and ":" not in line}
    # VS Code Copilot has no writer, so --print would refuse it: coming soon, not print only.
    assert rows["zed"].endswith("connect")
    assert rows["vscode-copilot"].endswith("coming soon")
    assert rows["opencode"].endswith("connect")


@pytest.mark.parametrize("flags", ([], ["--yes"]), ids=["terminal", "yes"])
def test_connect_never_starts_a_tool(subscribed, opencode_config, terminal, monkeypatch, flags):
    started = []

    def trip(argv, *args, **kwargs):
        started.append(argv)
        raise OSError("jl must start no program here")

    monkeypatch.setattr(subprocess, "call", trip)
    monkeypatch.setattr(subprocess, "Popen", trip)

    result = runner.invoke(app, ["connect", "opencode", *flags], input="y\n")

    assert result.exit_code == 0, result.output
    assert started == []


def ends_with_next_step(result, commands):
    """Assert the run ends with the next actions, after the blank line that separates them from the block above."""
    lines = result.output.splitlines()
    assert lines[-2] == f"Next: {commands}"
    assert lines[-1] == "      start a tool in your project folder"
    assert lines[-3] == ""
    assert "\n\n\nNext:" not in result.output  # one blank line, not two


def test_connect_ends_with_how_to_start_each_tool(subscribed, opencode_config, terminal, tmp_path):
    put_cline_on_path(tmp_path)

    result = runner.invoke(app, ["connect", "opencode", "cline", "--yes"])

    assert result.exit_code == 0, result.output
    ends_with_next_step(result, "opencode  or  cline -i")
    # One result, not two endings: the test is a secondary line, not a second ✓.
    assert "API test passed in " in result.output
    assert "Test request:" not in result.output
    assert not re.search(r"✓ API test passed", result.output)
    assert "Add another tool: jl connect" in said(result)


def test_under_the_installer_another_tool_needs_a_new_terminal(subscribed, opencode_config, monkeypatch):
    # curl | sh runs in a child shell: jl isn't on this window's PATH until a new terminal.
    monkeypatch.setenv("JL_FROM_INSTALLER", "1")

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Add another tool later: open a new terminal, then run jl connect" in said(result)
    ends_with_next_step(result, "opencode")


def test_the_picker_ticks_installed_harnesses_and_lists_coming_soon_last(subscribed, opencode_config, terminal):
    terminal.picked = ["opencode"]

    result = runner.invoke(app, ["connect", "--all", "--yes"])

    assert result.exit_code == 0, result.output
    assert terminal.picker.use_search_filter
    # Asked only once the subscription is known to be active.
    assert ("GET", "subscription/usage", None) in terminal.picker.calls_before
    rows = [(choice.value, choice.checked, choice.disabled) for choice in terminal.picker.choices]
    assert rows[0] == ("opencode", True, None)
    groups = [
        "installed" if checked else "coming soon" if disabled.startswith("coming soon") else "not installed"
        for _, checked, disabled in rows
    ]
    assert groups == sorted(groups, key=["installed", "not installed", "coming soon"].index)
    coming_soon = {value: disabled for value, _, disabled in rows if disabled}
    assert set(coming_soon) >= {"claude-code", "codex", "vscode-copilot", "zed"}
    # The ticked harness is the one connected.
    assert read_json(opencode_config)["model"] == "jarvislabs/deepseek-v4.1-flash-1017"


def test_an_empty_picker_says_how_to_come_back(subscribed, opencode_config, terminal):
    terminal.picked = []

    result = runner.invoke(app, ["connect", "--all"])

    assert result.exit_code == 0, result.output
    assert result.output.endswith("Next: jl connect\n      set up a coding tool when you want\n")
    assert [call for call in subscribed.calls if call[0] == "POST"] == []
    assert not opencode_config.exists()


# Each tool's own install page: jl connect never installs a tool.
OPENCODE_INSTALL = "https://opencode.ai/docs/#install"
PI_INSTALL = "https://pi.dev/docs/latest/quickstart"
KILO_INSTALL = "https://kilo.ai/cli"
CLINE_INSTALL = "https://docs.cline.bot/getting-started/installing-cline#cli"
QWEN_INSTALL = "https://qwenlm.github.io/qwen-code-docs/en/users/quickstart/"
# The real one, before the server fixture replaces it.
REAL_OUTPUT = harnesses._output


def said(result):
    return " ".join(result.output.split())


def test_no_tool_installed_points_to_install_pages(subscribed, terminal, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-installed"))

    result = runner.invoke(app, ["connect"])

    assert result.exit_code == 0, result.output
    assert (
        "! No supported coding tool found Install one, then run jl connect: "
        f"OpenCode {OPENCODE_INSTALL} pi {PI_INSTALL} Kilo {KILO_INSTALL} "
        "Every tool jl knows: jl connect --list "
        "Just installed one? Open a new terminal, then run jl connect."
    ) in said(result)
    assert terminal.picker is None
    # Told before the subscription check: there's nothing to pay for yet.
    assert subscribed.calls == []
    assert not config.config_path().exists()


def test_no_tool_message_says_to_open_a_new_terminal(subscribed, tmp_path, monkeypatch):
    """A tool installed after this terminal opened is named as needing a new one."""
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-installed"))

    result = runner.invoke(app, ["connect"])

    assert result.exit_code == 0, result.output
    assert "Just installed one? Open a new terminal, then run jl connect." in said(result)


def test_pi_links_its_quickstart():
    """pi's install page is its quickstart."""
    pi = next(h for h in harnesses.HARNESSES if h.id == "pi")

    assert pi.install_url == "https://pi.dev/docs/latest/quickstart"


def test_no_tool_install_rows_show_names_bold_and_urls_accent(subscribed, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-installed"))
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, color_system="standard", theme=render.theme)
    monkeypatch.setattr(render, "console", console)

    with pytest.raises(typer.Exit):
        connect._stop_without_a_tool()

    out = buffer.getvalue()
    assert re.search(r"\x1b\[1mOpenCode  \x1b\[0m", out)
    # Accent and underlined; the terminal also gets OSC 8, so it is clickable.
    assert re.search(r"\x1b\[4;36mhttps://opencode\.ai/docs/#install\x1b\[0m", out)
    assert "\x1b]8;" in out


@pytest.mark.parametrize("args", [[], ["--yes"]], ids=["no-terminal", "no-terminal-yes"])
def test_with_no_terminal_and_no_tool_connect_still_points_to_install_pages(subscribed, tmp_path, monkeypatch, args):
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-installed"))
    monkeypatch.setattr(render, "can_prompt", lambda: False)

    result = runner.invoke(app, ["connect", *args])

    assert result.exit_code == 0, result.output
    assert f"OpenCode {OPENCODE_INSTALL}" in said(result)
    assert "Name the tools to connect" not in said(result)


@pytest.mark.parametrize("rotate", [False, True], ids=["print", "rotate-key"])
def test_print_and_rotate_key_work_with_no_tool_installed(subscribed, opencode_config, tmp_path, monkeypatch, rotate):
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-installed"))
    if rotate:
        connection = Connection(
            key_id="key_0",
            name="jl-devbox-f7a7",
            kind="subscription",
            token="jlsk-old",
            harnesses={"opencode": Replaced()},
        )
        config.save_connection(connection)
        subscribed.routes[("DELETE", "users/api_keys/key_0")] = {"success": True}

    result = runner.invoke(app, ["connect", "--rotate-key"] if rotate else ["connect", "--print", "opencode"])

    assert result.exit_code == 0, result.output
    assert "No coding tool found" not in said(result)
    if rotate:
        assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-new-secret"
    else:
        assert "jlsk-new-secret" in result.stdout


def program(folder, name):
    """Put an executable ``name`` in ``folder``, with a .cmd twin so that Windows finds it too."""
    folder.mkdir(parents=True, exist_ok=True)
    for file in (folder / name, folder / f"{name}.cmd"):
        file.write_text("")
        file.chmod(0o755)


# Where each tool's own installer puts it, on PATH only in terminals opened after the install.
@pytest.mark.parametrize(
    ("folder", "env", "name", "harness"),
    [
        # OpenCode's install script: INSTALL_DIR=$HOME/.opencode/bin.
        (".opencode/bin", {}, "opencode", "OpenCode"),
        # Kilo's: INSTALL_DIR=$HOME/.kilo/bin.
        (".kilo/bin", {}, "kilo", "Kilo"),
        # pi's, when no folder of its list is on PATH: ~/.pi/agent/bin.
        (".pi/agent/bin", {}, "pi", "pi"),
        # npm's global packages under nvm's Node.js: $NVM_DIR, else $XDG_CONFIG_HOME/nvm, else ~/.nvm.
        ("my-nvm/versions/node/v24.11.0/bin", {"NVM_DIR": "my-nvm"}, "cline", "Cline"),
        ("nvm/versions/node/v24.11.0/bin", {}, "cline", "Cline"),
        (".nvm/versions/node/v24.11.0/bin", {"XDG_CONFIG_HOME": None}, "cline", "Cline"),
    ],
    ids=["opencode", "kilo", "pi", "nvm-dir", "nvm-xdg-config-home", "nvm-home"],
)
def test_a_tool_in_its_installers_folder_counts_as_installed(
    subscribed, tmp_path, monkeypatch, folder, env, name, harness
):
    # A terminal opened before the install: the folder isn't on its PATH yet.
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-installed"))
    for var, value in env.items():
        if value is None:
            monkeypatch.delenv(var)
        else:
            monkeypatch.setenv(var, str(tmp_path / value))
    program(tmp_path / folder, name)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    rows = [re.split(r"\s{2,}", line.strip("│ ")) for line in result.output.splitlines()]
    # An installed tool that isn't connected names the command that connects it.
    assert [row for row in rows if row[0] == harness] == [[harness, "installed", f"jl connect {name}"]]


@pytest.mark.skipif(sys.platform == "win32", reason="nvm runs on macOS and Linux")
def test_reading_a_tools_output_in_nvms_folder_uses_the_node_next_to_it(tmp_path, monkeypatch):
    nvm_bin = tmp_path / ".nvm" / "versions" / "node" / "v26.13.0" / "bin"
    nvm_bin.mkdir(parents=True)
    (nvm_bin / "node").write_text("#!/bin/sh\necho v26.13.0\n")
    (nvm_bin / "crush").write_text("#!/usr/bin/env node\n")
    for name in ("node", "crush"):
        (nvm_bin / name).chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NVM_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    assert harnesses._output(["crush", "--version"]) == "v26.13.0\n"


@pytest.mark.skipif(sys.platform == "win32", reason="nvm runs on macOS and Linux")
def test_a_relative_nvm_dir_finds_an_absolute_path(tmp_path, monkeypatch):
    nvm_bin = tmp_path / ".nvm" / "versions" / "node" / "v26.13.0" / "bin"
    nvm_bin.mkdir(parents=True)
    (nvm_bin / "node").write_text(f"#!/bin/sh\necho ran > {tmp_path / 'node-ran'}\n")
    (nvm_bin / "cline").write_text("#!/usr/bin/env node\n")
    for name in ("node", "cline"):
        (nvm_bin / name).chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NVM_DIR", ".nvm")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    assert harnesses._find("cline") == str(nvm_bin / "cline")


def test_a_vscode_extension_ends_with_how_to_open_vs_code(subscribed, terminal, tmp_path):
    # Kilo's extension is in VS Code, but VS Code's code command isn't on PATH (common on macOS).
    (tmp_path / ".vscode" / "extensions" / "kilocode.kilo-code-1.0.0").mkdir(parents=True)

    result = runner.invoke(app, ["connect", "kilo-vscode", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(tmp_path / "kilo" / "kilo.jsonc")
    assert terminal.asked == []
    assert "Next: open VS Code" in said(result)
    assert "start a tool in your project folder" in said(result)


def test_uninstalled_tools_are_disabled_in_the_picker(subscribed, opencode_config, terminal):
    terminal.picked = ["opencode"]

    result = runner.invoke(app, ["connect", "--all", "--yes"])

    assert result.exit_code == 0, result.output
    rows = {choice.value: choice.disabled for choice in terminal.picker.choices}
    assert rows["opencode"] is None
    # questionary shows the reason in brackets, and a disabled row can't be ticked: the command that names
    # where to get the tool.
    assert rows["cline"] == "jl connect cline"
    assert rows["kilo-vscode"] == "jl connect kilo-vscode"
    supported = [h.id for h in harnesses.HARNESSES if h.support == "connect" and h.id != "opencode"]
    assert [i for i in supported if rows[i] != f"jl connect {i}"] == []


def test_a_named_uninstalled_tool_is_skipped_with_its_page(subscribed, opencode_config, terminal, tmp_path):
    result = runner.invoke(app, ["connect", "cline"])

    # Nothing connected.
    assert result.exit_code == 1, result.output
    assert f"! Cline skipped: not installed Install it from {CLINE_INSTALL}, then run jl connect cline." in said(result)
    assert not (tmp_path / ".cline").exists()
    assert [call for call in subscribed.calls if call[0] == "POST"] == []


@pytest.mark.parametrize(
    ("args", "picked", "on_path"),
    [
        # An older jl offered OpenCode here, to install.
        (["connect"], [], False),
        (["connect", "pi", "--yes"], [], True),
        # A tick that the picker's disabled row refuses, in case it gets through.
        (["connect"], ["pi"], True),
        (["connect", "opencode", "pi", "kilo", "cline"], [], True),
    ],
    ids=["no-tool", "named-missing", "picker", "connected"],
)
def test_connect_never_runs_an_installer(
    subscribed, opencode_config, terminal, tmp_path, monkeypatch, args, picked, on_path
):
    if not on_path:
        monkeypatch.setenv("PATH", str(tmp_path / "nothing-installed"))
    terminal.picked = picked
    # The real one, so that any program jl reads reaches subprocess.
    monkeypatch.setattr(harnesses, "_output", REAL_OUTPUT)
    started, downloaded = [], []

    def popen(argv, *a, **kwargs):
        started.append(argv)
        raise OSError("jl must start no program here")

    def send(self, request, *a, **kwargs):
        downloaded.append(str(request.url))
        raise httpx.ConnectError("jl must download nothing here")

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(httpx.Client, "send", send)

    result = runner.invoke(app, args, input="y\n")

    assert result.exit_code in (0, 1), result.output
    assert started == []
    assert downloaded == []


def test_every_cli_tool_has_an_install_page():
    cli = [h for h in harnesses.HARNESSES if h.kind == "cli" and h.support == "connect"]

    assert [h.id for h in cli if not (h.install_url or "").startswith("https://")] == []


@pytest.mark.parametrize(
    ("harness", "step", "config_file"),
    [
        ("kilo-vscode", "code --install-extension kilocode.kilo-code --pre-release", "kilo/kilo.jsonc"),
        ("cline-vscode", "code --install-extension saoudrizwan.claude-dev", ".cline/data/settings/providers.json"),
    ],
)
def test_a_missing_extension_gets_its_install_step_and_no_install(
    subscribed, terminal, tmp_path, harness, step, config_file
):
    result = runner.invoke(app, ["connect", harness, "--yes"])

    # Nothing connected.
    assert result.exit_code == 1, result.output
    assert step in said(result)
    assert f"then run jl connect {harness}" in said(result)
    assert not (tmp_path / config_file).exists()
    # Nothing to connect, so no key is created either.
    assert ("POST", "users/api_keys") not in {call[:2] for call in subscribed.calls}

    # Once the extension is in VS Code's extensions folder, jl connects it.
    extension = step.split()[2]
    (tmp_path / ".vscode" / "extensions" / f"{extension}-1.0.0").mkdir(parents=True)

    result = runner.invoke(app, ["connect", harness, "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(tmp_path / config_file)


def test_a_missing_harness_doesnt_stop_the_other_harnesses(subscribed, opencode_config, terminal):
    # Yes to the preview.
    result = runner.invoke(app, ["connect", "opencode", "qwen"], input="y\n")

    assert result.exit_code == 0, result.output
    assert read_json(opencode_config)["model"] == "jarvislabs/deepseek-v4.1-flash-1017"
    assert set(config.load_connection().harnesses) == {"opencode"}
    assert "API test passed in" in said(result)


def test_a_skipped_tool_links_its_install_page(subscribed, opencode_config, terminal):
    result = runner.invoke(app, ["connect", "opencode", "qwen", "--yes"])

    assert result.exit_code == 0, result.output
    # The reason ends the headline; what to do is the hint under it.
    assert "! Qwen Code skipped: not installed\n  Install it from " in result.output
    assert f"Install it from {QWEN_INSTALL}, then run jl connect qwen." in said(result)


def test_skipped_harnesses_are_listed_with_their_reason(subscribed, opencode_config, terminal):
    # Yes to the preview.
    result = runner.invoke(app, ["connect", "opencode", "cline", "kilo-vscode", "qwen"], input="y\n")

    assert result.exit_code == 0, result.output
    # One block before the preview, in the order named.
    summary = (
        f"! Cline skipped: not installed Install it from {CLINE_INSTALL}, then run jl connect cline. "
        "! Kilo (VS Code) skipped: the extension isn't installed Install it with code --install-extension "
        "kilocode.kilo-code --pre-release, then run jl connect kilo-vscode. "
        f"! Qwen Code skipped: not installed Install it from {QWEN_INSTALL}, then run jl connect qwen. "
        "jl will add the Jarvislabs provider"
    )
    assert summary in said(result)


def test_a_dry_run_lists_the_skipped_harnesses(subscribed, opencode_config, terminal):
    result = runner.invoke(app, ["connect", "opencode", "qwen", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert f"! Qwen Code skipped: not installed Install it from {QWEN_INSTALL}, then run jl connect qwen." in said(
        result
    )


def test_without_a_terminal_a_skipped_harness_and_nothing_connected_exits_1(
    subscribed, opencode_config, terminal, monkeypatch
):
    monkeypatch.setattr(render, "can_prompt", lambda: False)

    # Nothing on stdin, so the preview question reads no answer.
    result = runner.invoke(app, ["connect", "opencode", "cline"])

    assert result.exit_code == 1, result.output
    assert f"! Cline skipped: not installed Install it from {CLINE_INSTALL}, then run jl connect cline." in said(result)
    assert not opencode_config.exists()


def fake_windows_registry(monkeypatch, paths):
    """Fake Windows' os.name and winreg, whose Path value is paths[(root, key name)]."""
    winreg = SimpleNamespace(
        HKEY_CURRENT_USER="HKCU",
        HKEY_LOCAL_MACHINE="HKLM",
        OpenKey=lambda root, path: contextlib.nullcontext((root, path)),
        QueryValueEx=lambda key, name: (paths[key], 1),
    )
    monkeypatch.setitem(sys.modules, "winreg", winreg)
    monkeypatch.setattr(harnesses.os, "name", "nt")
    # Only harnesses sees Windows: shutil.which on this OS would break under a global sys.platform.
    monkeypatch.setattr(harnesses, "sys", SimpleNamespace(platform="win32"))


MACHINE_ENVIRONMENT = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"


def test_windows_finds_a_tool_on_the_registry_path(tmp_path, monkeypatch):
    """A tool only on the registry's Path is found, with %VARS% expanded."""
    # A terminal opened before Node.js's installer ran keeps its old PATH; the registry has the new one.
    nodejs = tmp_path / "nodejs"
    program(nodejs, "kilo")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "old-terminal"))
    fake_windows_registry(
        monkeypatch,
        {
            ("HKCU", "Environment"): r"%APPDATA%/npm",
            ("HKLM", MACHINE_ENVIRONMENT): str(nodejs),
        },
    )

    assert harnesses._find("kilo") == str(nodejs / "kilo")


def test_windows_reads_a_tools_output_with_the_registry_path(tmp_path, monkeypatch):
    """Reading a tool's output gets the registry's Path, so it finds node beside its launcher."""
    nodejs = tmp_path / "nodejs"
    npm = tmp_path / "npm"
    nodejs.mkdir()
    program(npm, "kilo")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "old-terminal"))
    fake_windows_registry(
        monkeypatch,
        {
            ("HKCU", "Environment"): r"%APPDATA%/npm",
            ("HKLM", MACHINE_ENVIRONMENT): str(nodejs),
        },
    )
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=b"0.99.0\n")

    monkeypatch.setattr(harnesses.subprocess, "run", run)

    assert harnesses._output(["kilo", "--version"]) == "0.99.0\n"

    ((argv, kwargs),) = calls
    assert argv == [str(npm / "kilo"), "--version"]
    # node is in the machine folder, the .cmd in the user one: the tool read needs both.
    assert str(nodejs) in kwargs["env"]["PATH"]
    assert str(npm) in kwargs["env"]["PATH"]


def test_a_programs_output_is_read_as_utf8_whatever_the_code_page(monkeypatch):
    # Go programs print UTF-8; Windows' ANSI code page (here cp1252) can't decode the "Á" of a profile folder.
    printed = "crush version 0.99.0 (C:\\Users\\Álvaro)\n".encode()

    def run(argv, **kwargs):
        stdout = printed.decode("cp1252") if kwargs.get("text") else printed
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(subprocess, "run", run)

    assert harnesses._output(["crush", "--version"]) == "crush version 0.99.0 (C:\\Users\\Álvaro)\n"


def test_ctrl_c_in_the_picker_aborts_with_130(subscribed, monkeypatch):
    # questionary answers None to Ctrl-C.
    monkeypatch.setattr(render, "can_prompt", lambda: True)
    monkeypatch.setattr(questionary, "checkbox", lambda *args, **kwargs: SimpleNamespace(ask=lambda: None))

    result = runner.invoke(app, ["connect"])

    assert result.exit_code == 130, result.output
    assert [call for call in subscribed.calls if call[0] == "POST"] == []


def test_a_single_installed_tool_skips_the_picker(subscribed, terminal):
    result = runner.invoke(app, ["connect", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert terminal.picker is None
    assert "OpenCode is the only installed tool" in result.output


def test_the_picker_connects_the_ticks_without_asking_again(subscribed, monkeypatch):
    monkeypatch.setattr(render, "can_prompt", lambda: True)
    monkeypatch.setattr(
        questionary,
        "checkbox",
        lambda message, choices, **kwargs: SimpleNamespace(ask=lambda: ["opencode", "pi"]),
    )
    # The picker's Enter is the answer; there is no second "Connect these 2?".
    monkeypatch.setattr(questionary, "confirm", lambda *args, **kwargs: pytest.fail("asked again"))

    result = runner.invoke(app, ["connect", "--dry-run"])

    assert result.exit_code == 0, result.output
    previewed = [line.split()[1] for line in result.stdout.splitlines() if line.startswith("+ ") and "~/" in line]
    assert previewed == ["OpenCode", "pi"]


def test_yes_writes_without_another_question(subscribed, monkeypatch):
    monkeypatch.setattr(render, "can_prompt", lambda: True)
    monkeypatch.setattr(questionary, "checkbox", lambda message, choices, **kwargs: SimpleNamespace(ask=lambda: ["pi"]))
    monkeypatch.setattr(questionary, "confirm", lambda message, **kwargs: pytest.fail(f"asked {message!r}"))

    result = runner.invoke(app, ["connect", "--dry-run", "--yes"])

    assert result.exit_code == 0, result.output
    assert "+ pi  ~/.pi/agent/models.json, settings.json" in result.output


@pytest.fixture
def screen(monkeypatch):
    """The real picker, drawn into the returned buffer; Enter accepts the ticked rows."""
    monkeypatch.setattr(render, "can_prompt", lambda: True)
    screen = io.StringIO()
    real_checkbox = questionary.checkbox

    def checkbox(*args, **kwargs):
        size = Size(rows=60, columns=200)
        with create_pipe_input() as keys:
            keys.send_text("\r")
            output = Vt100_Output(screen, lambda: size, term="dumb")
            answer = real_checkbox(*args, input=keys, output=output, **kwargs).ask()
        return SimpleNamespace(ask=lambda: answer)

    monkeypatch.setattr(questionary, "checkbox", checkbox)
    return screen


def _drawn(screen: io.StringIO) -> str:
    """Return what was drawn on the screen, without its escape codes."""
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", screen.getvalue())


def test_a_coming_soon_row_shows_its_docs_link_in_one_pair_of_brackets(subscribed, screen):
    result = runner.invoke(app, ["connect", "--all", "--dry-run"])

    assert result.exit_code == 0, result.output
    drawn = _drawn(screen)
    assert "Coming soon: jl connect <id> links its docs" in drawn
    assert "- VS Code Copilot (jl connect vscode-copilot)" in drawn
    # A preview tool is coming soon while its endpoint isn't public.
    assert "- Claude Code (jl connect claude-code)" in drawn
    assert "((" not in drawn


def test_the_picker_says_space_ticks_and_enter_continues(subscribed, screen):
    result = runner.invoke(app, ["connect", "--dry-run"])

    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in _drawn(screen).splitlines()]
    assert "Space toggles · Enter continues · type to filter" in lines


def test_the_picker_uses_ascii_separators_when_marks_are_ascii(subscribed, screen, monkeypatch):
    monkeypatch.setattr(render, "MARKS", render.ASCII_MARKS)

    result = runner.invoke(app, ["connect", "--dry-run"])

    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in _drawn(screen).splitlines()]
    assert "Space toggles | Enter continues | type to filter" in lines


def test_connect_opens_with_a_heading_for_the_phase(subscribed, opencode_config, monkeypatch):
    monkeypatch.setenv("JL_FROM_INSTALLER", "1")
    result = runner.invoke(app, ["connect", "opencode", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert result.stderr.startswith("\nCONNECT TOOLS")
    assert "Sets up the Jarvislabs provider in your coding tools." in result.stderr


def test_standalone_connect_opens_with_the_same_heading(subscribed, opencode_config):
    result = runner.invoke(app, ["connect", "opencode", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert result.stderr.startswith("\nCONNECT TOOLS")
    assert "Sets up the Jarvislabs provider in your coding tools." in result.stderr


def test_connect_shows_the_test_reply_or_the_server_error(subscribed, opencode_config, monkeypatch):
    tokens = []
    monkeypatch.setattr(client_module, "Transport", lambda token: tokens.append(token) or subscribed)

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert re.search(r"API test passed in \d+\.\d s", result.output)
    # The old second ✓ is gone: the test is a secondary line, not a competing success.
    assert "Test request:" not in result.output
    assert not re.search(r"✓ API test passed", result.output)
    # One tiny request to the default model, sent with this machine's key, not the Jarvis key.
    [(_, _, body)] = [call for call in subscribed.calls if call[1] == "chat/completions"]
    assert body["model"] == "deepseek-v4.1-flash-1017"
    assert body["max_tokens"] <= 32
    assert tokens[-1] == "jlsk-new-secret"

    subscribed.routes[("POST", "chat/completions")] = error_from_response(
        429, {"error": {"message": "Plan limit reached. Resets at 16:05 UTC."}}
    )

    opencode_config.unlink()

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    # The configs are kept: they're right, and the error says why the harness can't answer yet.
    assert result.exit_code == 1
    assert (
        "✗ Test request to deepseek-v4.1-flash-1017 failed\n"
        "  Plan limit reached. Resets at 16:05 UTC.\n"
        "  Your tools are set up; they work once requests go through.\n"
    ) in result.output
    assert read_json(opencode_config)["model"] == "jarvislabs/deepseek-v4.1-flash-1017"


def ctrl_c_at(monkeypatch, question):
    """Press Ctrl-C at ``question``, a yes/no question jl asks itself."""
    real_input = render.console.input

    def console_input(prompt, **kwargs):
        if question in str(prompt):
            raise KeyboardInterrupt
        return real_input(prompt, **kwargs)

    monkeypatch.setattr(render.console, "input", console_input)


@pytest.mark.parametrize("question", ["picker", "preview"])
def test_ctrl_c_aborts_with_130(subscribed, opencode_config, terminal, monkeypatch, question):
    # questionary's questions answer None to Ctrl-C.
    terminal.picked = None
    if question == "preview":
        ctrl_c_at(monkeypatch, "Write these changes?")

    # The picker opens only when no harness is named.
    result = runner.invoke(app, ["connect", "--all"] if question == "picker" else ["connect", "opencode"], input="y\n")

    assert result.exit_code == 130, result.output
    assert not opencode_config.exists()


def test_a_single_tool_ends_with_its_start_command(subscribed, opencode_config, terminal):
    # Yes to the preview.
    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    assert result.exit_code == 0, result.output
    ends_with_next_step(result, "opencode")
    assert "API test passed in " in result.output
    assert "Test request:" not in result.output
    assert not re.search(r"✓ API test passed", result.output)


@pytest.mark.parametrize("in_terminal", [True, False], ids=["terminal", "no-terminal"])
def test_several_tools_not_started_get_one_line(
    subscribed, opencode_config, terminal, tmp_path, monkeypatch, in_terminal
):
    put_cline_on_path(tmp_path)
    monkeypatch.setattr(render, "can_prompt", lambda: in_terminal)

    # In a terminal, yes to the preview; without one, --yes answers it.
    result = runner.invoke(app, ["connect", "opencode", "cline", *([] if in_terminal else ["--yes"])], input="y\n")

    assert result.exit_code == 0, result.output
    ends_with_next_step(result, "opencode  or  cline -i")


def test_a_two_tool_connect_fits_one_screen(subscribed, opencode_config, terminal, tmp_path):
    put_cline_on_path(tmp_path)

    result = runner.invoke(app, ["connect", "opencode", "cline", "--yes"])

    assert result.exit_code == 0, result.output
    # DESIGN.md rule 42: the default success view is one screen. This counts every line a 2-tool connect prints
    # (16 lines here), so one extra line of ceremony fails the budget rather than drifting silently.
    assert len(result.output.splitlines()) <= 20, result.output


def test_connect_yes_help_names_what_it_does(server):
    result = runner.invoke(app, ["connect", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    shown = " ".join(result.output.replace("│", " ").split())
    assert (
        "Answer yes to every question: write the changes, and move this machine from the wallet to your plan."
    ) in shown
    assert "jl connect opencode --model glm-5.3" in shown


def put_cline_on_path(tmp_path):
    for name in ("cline", "cline.cmd"):  # .cmd so that Windows finds it too
        (tmp_path / "bin" / name).write_text("")
        (tmp_path / "bin" / name).chmod(0o755)


def connect_before(terminal, *names):
    """Connect ``names`` in an earlier run."""
    result = runner.invoke(app, ["connect", *names, "--yes"])
    assert result.exit_code == 0, result.output
    terminal.asked.clear()


def test_a_repeat_connect_with_nothing_to_change_says_already_connected(subscribed, opencode_config, terminal):
    """A second connect that would change nothing says so instead of opening the picker."""
    connect_before(terminal, "opencode")

    result = runner.invoke(app, ["connect"])

    assert result.exit_code == 0, result.output
    assert "OpenCode is already connected" in said(result)
    assert "jl status" in said(result)
    assert "jl disconnect <tool>" in said(result)
    assert "jl disconnect opencode" not in said(result)
    assert terminal.picker is None


def test_a_wallet_key_on_a_subscribed_account_is_not_up_to_date(subscribed, opencode_config, terminal):
    """A machine on the wallet whose account now has a plan goes on to the switch, not "up to date"."""
    connect_before(terminal, "opencode")
    config.save_connection(dataclasses.replace(config.load_connection(), kind="payg"))

    result = runner.invoke(app, ["connect"])

    assert "up to date" not in said(result)


def test_a_revoked_key_is_not_up_to_date(subscribed, opencode_config, terminal):
    """A key revoked on the console means the tools are broken, so connect goes on to make a new one."""
    connect_before(terminal, "opencode")
    subscribed.routes[("GET", "users/api_keys")] = []

    result = runner.invoke(app, ["connect"])

    assert "up to date" not in said(result)


def test_a_new_installed_tool_still_opens_the_picker(subscribed, opencode_config, terminal, tmp_path):
    """A tool installed since the last connect isn't connected yet, so the picker opens."""
    connect_before(terminal, "opencode")
    program(tmp_path / "bin", "kilo")
    terminal.picked = ["kilo"]

    result = runner.invoke(app, ["connect", "--yes"])

    assert result.exit_code == 0, result.output
    assert terminal.picker is not None
    assert (tmp_path / "kilo" / "kilo.jsonc").exists()


def test_a_changed_config_still_runs_the_flow(subscribed, opencode_config, terminal):
    """A config the refresh would change means the normal flow, which writes it back."""
    connect_before(terminal, "opencode")
    opencode_config.write_text('{\n  "theme": "tokyonight"\n}\n')
    terminal.picked = ["opencode"]

    result = runner.invoke(app, ["connect", "--yes"])

    assert result.exit_code == 0, result.output
    assert terminal.picker is None
    assert read_json(opencode_config)["model"] == "jarvislabs/deepseek-v4.1-flash-1017"


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("opencod", "Unknown tool opencod. See every tool ID: jl connect --list"),
        ("claude-code", "Claude Code is coming soon: see https://docs.jarvislabs.ai/models-api/harnesses/claude-code"),
        ("vscode-copilot", "VS Code Copilot is coming soon: see https://"),
    ],
    ids=["typo", "preview", "coming-soon"],
)
def test_a_harness_jl_cant_connect_stops_before_any_server_call(server, name, message):
    result = runner.invoke(app, ["connect", name, "--yes"])

    assert result.exit_code == 1
    assert message in " ".join(result.output.split())
    assert server.calls == []


def test_amp_is_an_unknown_harness(server):
    result = runner.invoke(app, ["connect", "amp", "--yes"])

    assert result.exit_code == 1
    assert "Unknown tool amp." in said(result)
    assert server.calls == []


@pytest.mark.parametrize(
    ("connected", "on_path", "expected"),
    [
        (True, True, ["✓ OpenCode", "connected", "default glm-5.3"]),
        (False, False, ["OpenCode", "not installed", "jl connect opencode"]),
    ],
    ids=["connected", "not-connected"],
)
def test_status_lists_harnesses_and_their_connection(
    subscribed, opencode_config, monkeypatch, connected, on_path, expected
):
    if connected:
        runner.invoke(app, ["connect", "opencode", "--yes", "--model", "glm-5.3"])
    monkeypatch.setattr(shutil, "which", lambda name, path=None: f"/usr/bin/{name}" if on_path else None)

    result = runner.invoke(app, ["status"] if connected else ["status", "--all"])

    assert result.exit_code == 0, result.output
    (line,) = [line for line in result.output.splitlines() if "OpenCode" in line]
    assert re.split(r"\s{2,}", line.strip("│ ")) == expected


def test_status_lists_connected_tools_first_then_installed_then_the_rest(subscribed, opencode_config):
    runner.invoke(app, ["connect", "pi", "--yes"])
    # Every CLI tool is on PATH (the server fixture); no VS Code extension is installed.

    result = runner.invoke(app, ["status", "--all"])

    assert result.exit_code == 0, result.output
    panel = result.output[result.output.index("Coding tools") :]
    rows = [re.split(r"\s{2,}", line.strip("│ ")) for line in panel.splitlines() if line.strip("│ ") and line[0] == "│"]
    states = [row[1] for row in rows]
    assert states == ["connected", *["installed"] * 10, *["not installed"] * 3]


def test_a_partial_connect_table_is_a_clean_error(subscribed):
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[connect]\nid = "key_1"\ntoken = "t"\n')

    result = runner.invoke(app, ["status"])

    # main() turns this into one line; the point is that it is jl's error, not a KeyError traceback.
    assert isinstance(result.exception, JarvislabsError)
    assert "Can't read this machine's saved connection" in str(result.exception)
    assert "[connect]" in str(result.exception)


def test_a_wrong_shaped_connect_table_is_a_clean_error(subscribed):
    path = config.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # A string where the harness records belong, as a hand edit might leave it.
    path.write_text('[connect]\nid = "key_1"\nname = "n"\nkind = "subscription"\ntoken = "t"\nharnesses = "opencode"\n')

    result = runner.invoke(app, ["status"])

    assert isinstance(result.exception, JarvislabsError)
    assert "Can't read this machine's saved connection" in str(result.exception)


def test_status_says_a_connected_tool_was_uninstalled(subscribed, opencode_config, monkeypatch):
    """A tool connected earlier and uninstalled since isn't shown as working."""
    runner.invoke(app, ["connect", "pi", "--yes"])
    monkeypatch.setattr(commands, "installed", lambda harness: harness.id != "pi")

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if " pi " in line)
    assert "not installed" in row
    assert "jl disconnect pi" in row


def test_status_not_installed_rows_name_their_command(subscribed, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path=None: None)

    result = runner.invoke(app, ["status", "--all"])

    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if "OpenCode" in line)
    assert re.split(r"\s{2,}", row.strip("│ ")) == ["OpenCode", "not installed", "jl connect opencode"]


def test_status_hides_tools_that_are_not_installed(subscribed, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, path=None: None)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "OpenCode" not in result.output
    # Nothing is shown, so the hint says what --all lists rather than "more".
    assert re.search(r"\d+ tools jl can connect: jl status --all", result.output)

    full = runner.invoke(app, ["status", "--all"])
    assert "OpenCode" in full.output


def test_status_names_a_missing_skill_as_a_hint_not_a_warning(subscribed, monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "UNIVERSAL_PATH", tmp_path / "none" / "SKILL.md")
    monkeypatch.setattr(setup, "ADDITIONAL_AGENTS", {})

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "No jl agent skill is installed. Run jl setup to add one." in result.output
    assert "! No jl agent skill" not in result.output


def test_status_ends_with_what_to_run_next(subscribed, opencode_config):
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert result.output.endswith("Next: jl connect\n      connect a tool\n")


@pytest.fixture()
def up_to_date(monkeypatch):
    """jl is already the latest version; the skill files live in the real home, so they are left alone."""
    monkeypatch.setattr(update, "_is_editable", lambda: False)
    monkeypatch.setattr(update, "_fetch_latest", lambda: update.current_version())
    monkeypatch.setattr(setup, "refresh_skills", lambda: [])


@pytest.mark.parametrize("args", [["update"], ["update", "--skills-only"]], ids=["up-to-date", "after-upgrade"])
def test_update_refreshes_connected_harnesses(subscribed, opencode_config, up_to_date, args):
    runner.invoke(app, ["connect", "opencode", "--yes", "--model", "glm-5.3"])
    qwen = {**GLM, "id": "qwen3.6-coder", "name": "Qwen3.6 Coder"}
    subscribed.routes[("GET", "models")] = {"object": "list", "data": [DEEPSEEK, GLM, qwen]}

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    written = read_json(opencode_config)
    assert set(written["provider"]["jarvislabs"]["models"]) == {"deepseek-v4.1-flash-1017", "glm-5.3", "qwen3.6-coder"}
    # The default the user chose at connect stays.
    assert written["model"] == "jarvislabs/glm-5.3"
    assert "Refreshed OpenCode" in result.output
    # jl created the file, so a refresh backs nothing up.
    assert not opencode_config.with_name("opencode.json.jl-backup").exists()


@pytest.mark.parametrize(
    ("users_default", "catalog", "expected", "warning"),
    [
        ("anthropic/claude-sonnet-4", [DEEPSEEK, GEMMA], "anthropic/claude-sonnet-4", None),
        (
            "jarvislabs/glm-5.3",
            [DEEPSEEK, GEMMA],
            "jarvislabs/deepseek-v4.1-flash-1017",
            "glm-5.3 is no longer offered, so OpenCode's default model is now deepseek-v4.1-flash-1017",
        ),
    ],
    ids=["another-provider", "jarvislabs-model-dropped"],
)
def test_update_keeps_the_users_default_model(
    subscribed, opencode_config, up_to_date, users_default, catalog, expected, warning
):
    runner.invoke(app, ["connect", "opencode", "--yes"])
    written = read_json(opencode_config)
    opencode_config.write_text(json.dumps({**written, "model": users_default}))
    subscribed.routes[("GET", "models")] = {"object": "list", "data": catalog}

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0, result.output
    written = read_json(opencode_config)
    assert set(written["provider"]["jarvislabs"]["models"]) == {"deepseek-v4.1-flash-1017", "gemma-3-1b-it"}
    assert written["model"] == expected
    if warning:
        assert warning in " ".join(result.output.split())


def test_print_shows_the_config_with_the_key_filled_in(subscribed, tmp_path, monkeypatch):
    result = runner.invoke(app, ["connect", "--print", "zed"])

    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "settings.json" in out.splitlines()[0]
    settings = json.loads(out[out.index("{") : out.rindex("}") + 1])
    provider = settings["language_models"]["openai_compatible"]["Jarvislabs"]
    assert provider["api_url"] == "https://models.jarvislabs.net/v1"
    # Zed requires all four capability fields once the object is there, or it drops the provider;
    # parallel tool calls follow tool calling, which the Models API answers with several at once.
    assert [(m["name"], m["max_tokens"], m["capabilities"]) for m in provider["available_models"]] == [
        (
            "deepseek-v4.1-flash-1017",
            1048576,
            {"tools": True, "images": True, "parallel_tool_calls": True, "prompt_cache_key": False},
        ),
        ("glm-5.3", 230000, {"tools": True, "images": False, "parallel_tool_calls": True, "prompt_cache_key": False}),
        (
            "gemma-3-1b-it",
            32768,
            {"tools": False, "images": False, "parallel_tool_calls": False, "prompt_cache_key": False},
        ),
    ]
    assert settings["agent"]["default_model"] == {"provider": "Jarvislabs", "model": "deepseek-v4.1-flash-1017"}
    assert "jlsk-new-secret" in out
    # Printing writes no harness file, and records the harness so jl status shows it.
    assert not list(tmp_path.rglob("settings.json"))
    assert config.load_connection().harnesses == {"zed": Replaced()}


def test_connect_writes_goose_aider_and_crush_in_their_formats(subscribed, tmp_path):
    goose = goose_dir(tmp_path)
    goose.mkdir(parents=True)
    (goose / "config.yaml").write_text(
        "# Model\nactive_provider: anthropic\nproviders:\n  anthropic:\n    enabled: true\n"
        "    model: claude-sonnet-4-5\n    configured: true\nGOOSE_MODE: smart_approve\n"
    )
    (goose / "secrets.yaml").write_text("ANTHROPIC_API_KEY: sk-ant-1\n")
    (tmp_path / ".aider.conf.yml").write_text("# Aider\nmodel: gpt-4o\ndark-mode: true\n")
    crushrc = tmp_path / "crush" / "crushrc"
    crushrc.parent.mkdir()
    crushrc.write_text("option progress false\n")

    result = runner.invoke(app, ["connect", "goose", "aider", "crush", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(goose / "custom_providers" / "jarvislabs.json") == {
        "name": "jarvislabs",
        "engine": "openai",
        "display_name": "Jarvislabs",
        "base_url": "https://models.jarvislabs.net/v1/chat/completions",
        "models": [
            {
                "name": "deepseek-v4.1-flash-1017",
                "context_limit": 1048576,
                "supports_vision": True,
                "request_params": {"max_tokens": 65536},
            },
            {
                "name": "glm-5.3",
                "context_limit": 230000,
                "supports_vision": False,
                "request_params": {"max_tokens": 23000},
            },
            {
                "name": "gemma-3-1b-it",
                "context_limit": 32768,
                "supports_vision": False,
                "request_params": {"max_tokens": 3276},
            },
        ],
        "dynamic_models": False,
        "supports_streaming": True,
        # The key rides in the provider's own headers, so Goose needn't find it in its keyring.
        "requires_auth": False,
        "headers": {"Authorization": "Bearer jlsk-new-secret"},
    }
    assert (goose / "config.yaml").read_text() == (
        "# Model\nactive_provider: jarvislabs\nproviders:\n  anthropic:\n    enabled: true\n"
        "    model: claude-sonnet-4-5\n    configured: true\n  jarvislabs:\n    enabled: true\n"
        "    model: deepseek-v4.1-flash-1017\n    configured: true\nGOOSE_MODE: smart_approve\n"
    )
    assert (goose / "secrets.yaml").read_text() == "ANTHROPIC_API_KEY: sk-ant-1\n"
    assert (tmp_path / ".aider.conf.yml").read_text() == (
        "# Aider\nmodel: openai/deepseek-v4.1-flash-1017\ndark-mode: true\n"
        "openai-api-base: https://models.jarvislabs.net/v1\nopenai-api-key: jlsk-new-secret\n"
    )
    assert read_json(tmp_path / ".aider.model.metadata.json") == {
        "openai/deepseek-v4.1-flash-1017": {
            "max_input_tokens": 1048576,
            "max_output_tokens": 65536,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_vision": True,
        },
        "openai/glm-5.3": {
            "max_input_tokens": 230000,
            "max_output_tokens": 23000,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_vision": False,
        },
        "openai/gemma-3-1b-it": {
            "max_input_tokens": 32768,
            "max_output_tokens": 3276,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_vision": False,
        },
    }
    assert crushrc.read_text() == (
        "option progress false\n"
        "# >>> jl >>>\n"
        "provider add jarvislabs --type completions --name Jarvislabs"
        " --base-url https://models.jarvislabs.net/v1 --api-key jlsk-new-secret\n"
        "model add jarvislabs/deepseek-v4.1-flash-1017 --name 'DeepSeek V4.1 Flash'"
        " --context-window 1048576 --default-max-tokens 65536 --supports-images true\n"
        "model add jarvislabs/glm-5.3 --name GLM-5.3 --context-window 230000 --default-max-tokens 23000"
        " --supports-images false\n"
        "model add jarvislabs/gemma-3-1b-it --name 'Gemma 3 1B IT' --context-window 32768"
        " --default-max-tokens 3276 --supports-images false\n"
        "model large jarvislabs/deepseek-v4.1-flash-1017\n"
        "model small jarvislabs/deepseek-v4.1-flash-1017\n"
        "# <<< jl <<<\n"
    )


def test_an_unreadable_yaml_config_skips_the_harness_and_writes_nothing(subscribed, tmp_path):
    aider_config = tmp_path / ".aider.conf.yml"
    aider_config.write_text("model: [gpt-4o\n")

    result = runner.invoke(app, ["connect", "aider"])

    # Nothing connected.
    assert result.exit_code == 1, result.output
    assert "! Aider skipped Can't read" in " ".join(result.output.split())
    assert aider_config.read_text() == "model: [gpt-4o\n"
    assert not (tmp_path / ".aider.model.metadata.json").exists()


def test_goose_sends_each_models_default_effort(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = {
        "object": "list",
        "data": [
            {
                **DEEPSEEK,
                "thinking": {"supported": True, "supported_efforts": ["low", "high", "max"], "default_effort": "high"},
            },
            {**GEMMA, "thinking": {"supported": True, "supported_efforts": []}},
        ],
    }
    goose = goose_dir(tmp_path)
    goose.mkdir(parents=True)

    result = runner.invoke(app, ["connect", "goose", "--yes"])

    assert result.exit_code == 0, result.output
    # Goose sends request_params verbatim with every request for the model, whichever of them
    # the user picks; max_tokens is always there, and the effort joins it where the model has one.
    models = {m["name"]: m for m in read_json(goose / "custom_providers" / "jarvislabs.json")["models"]}
    assert models["deepseek-v4.1-flash-1017"]["request_params"] == {
        "max_tokens": 65536,
        "reasoning_effort": "high",
    }
    assert models["gemma-3-1b-it"]["request_params"] == {"max_tokens": 3276}


def test_aider_pins_each_models_default_effort_beside_the_users_own(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = {
        "object": "list",
        "data": [
            {
                **DEEPSEEK,
                "thinking": {"supported": True, "supported_efforts": ["low", "high", "max"], "default_effort": "high"},
            },
            {**GLM, "thinking": {"supported": True, "supported_efforts": ["medium"]}},
            {**GEMMA, "thinking": {"supported": True, "supported_efforts": []}},
        ],
    }
    settings = tmp_path / ".aider.model.settings.yml"
    settings.write_text(
        "# Mine\n- name: openai/gpt-4o\n  extra_params:\n    extra_body:\n      reasoning_effort: high\n"
    )

    result = runner.invoke(app, ["connect", "aider", "--yes"])

    assert result.exit_code == 0, result.output
    written = settings.read_text()
    # The user's own entry stays; jl's sit in a marked block, one per model, each with its
    # output limit and the effort where the model has one.
    assert "- name: openai/gpt-4o" in written
    assert (
        "- name: openai/deepseek-v4.1-flash-1017\n  extra_params:\n    max_tokens: 65536\n"
        "    extra_body:\n      reasoning_effort: high" in written
    )
    assert (
        "- name: openai/glm-5.3\n  extra_params:\n    max_tokens: 23000\n"
        "    extra_body:\n      reasoning_effort: medium" in written
    )
    assert "- name: openai/gemma-3-1b-it\n  extra_params:\n    max_tokens: 3276" in written


def test_crush_carries_the_default_models_effort_on_its_slots(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = {
        "object": "list",
        "data": [
            {
                **DEEPSEEK,
                "thinking": {"supported": True, "supported_efforts": ["low", "high", "max"], "default_effort": "high"},
            },
            {**GEMMA, "thinking": {"supported": True, "supported_efforts": []}},
        ],
    }
    crushrc = tmp_path / "crush" / "crushrc"
    crushrc.parent.mkdir()

    result = runner.invoke(app, ["connect", "crush", "--yes"])

    assert result.exit_code == 0, result.output
    written = crushrc.read_text()
    # Only the slot lines carry the effort: model add takes no provider options, and a
    # provider-level one would send a single effort for every model.
    assert (
        'model large jarvislabs/deepseek-v4.1-flash-1017 --provider-options \'{"reasoning_effort": "high"}\'\n'
        in written
    )
    assert (
        'model small jarvislabs/deepseek-v4.1-flash-1017 --provider-options \'{"reasoning_effort": "high"}\'\n'
        in written
    )
    assert "--provider-options" not in written.split("model large")[0]

    # A default model without efforts gets the bare slots.
    result = runner.invoke(app, ["connect", "crush", "--yes", "--model", "gemma-3-1b-it"])
    assert result.exit_code == 0, result.output
    assert "model large jarvislabs/gemma-3-1b-it\n" in crushrc.read_text()


def goose_dir(tmp_path):
    return tmp_path / "Block" / "goose" / "config" if os.name == "nt" else tmp_path / "goose"


def test_update_refreshes_goose_aider_and_crush_and_keeps_their_default(subscribed, tmp_path, up_to_date):
    runner.invoke(app, ["connect", "goose", "aider", "crush", "--yes", "--model", "glm-5.3"])
    qwen = {**GLM, "id": "qwen3.6-coder", "name": "Qwen3.6 Coder"}
    subscribed.routes[("GET", "models")] = {"object": "list", "data": [DEEPSEEK, GLM, qwen]}

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0, result.output
    goose = goose_dir(tmp_path)
    goose_models = read_json(goose / "custom_providers" / "jarvislabs.json")["models"]
    assert [m["name"] for m in goose_models] == ["deepseek-v4.1-flash-1017", "glm-5.3", "qwen3.6-coder"]
    assert "    model: glm-5.3\n" in (goose / "config.yaml").read_text()
    assert set(read_json(tmp_path / ".aider.model.metadata.json")) >= {"openai/qwen3.6-coder"}
    assert "model: openai/glm-5.3\n" in (tmp_path / ".aider.conf.yml").read_text()
    crushrc = (tmp_path / "crush" / "crushrc").read_text()
    assert "model add jarvislabs/qwen3.6-coder " in crushrc
    assert "model large jarvislabs/glm-5.3\nmodel small jarvislabs/glm-5.3\n" in crushrc


def test_status_reads_back_the_default_of_goose_aider_and_crush(subscribed, tmp_path, monkeypatch):
    runner.invoke(app, ["connect", "goose", "aider", "crush", "--yes", "--model", "glm-5.3"])
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    rows = [re.split(r"\s{2,}", line.strip("│ ")) for line in result.output.splitlines()]
    assert [row for row in rows if row[0] in ("✓ Goose", "✓ Aider", "✓ Crush")] == [
        [f"✓ {name}", "connected", "default glm-5.3"] for name in ("Goose", "Aider", "Crush")
    ]


def test_update_leaves_goose_alone_when_another_provider_is_active(subscribed, tmp_path, up_to_date):
    runner.invoke(app, ["connect", "goose", "--yes", "--model", "glm-5.3"])
    config_yaml = goose_dir(tmp_path) / "config.yaml"
    config_yaml.write_text(config_yaml.read_text().replace("active_provider: jarvislabs", "active_provider: anthropic"))
    subscribed.routes[("GET", "models")] = {"object": "list", "data": [DEEPSEEK, GEMMA]}

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0, result.output
    assert "active_provider: anthropic\n" in config_yaml.read_text()
    assert "no longer offered" not in result.output


def test_print_shows_crushs_block_with_the_key(subscribed, tmp_path):
    result = runner.invoke(app, ["connect", "--print", "crush"])

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert "crushrc" in lines[0]
    assert lines[1:3] == [
        "# >>> jl >>>",
        "provider add jarvislabs --type completions --name Jarvislabs"
        " --base-url https://models.jarvislabs.net/v1 --api-key jlsk-new-secret",
    ]
    assert not (tmp_path / "crush" / "crushrc").exists()


def crush_says(monkeypatch, version):
    """Make ``crush --version`` print ``version``, as Crush does when its output isn't a terminal."""
    monkeypatch.setattr(harnesses, "_output", lambda argv: version if argv == ["crush", "--version"] else "")


def test_an_old_crush_is_skipped_with_its_version(subscribed, tmp_path, monkeypatch):
    # winget's Crush: it ignores the provider type jl writes, which is new in Crush 0.99.0.
    crush_says(monkeypatch, "crush version 0.98.1\n")

    result = runner.invoke(app, ["connect", "crush", "aider", "--yes"])

    assert result.exit_code == 0, result.output
    assert (
        "! Crush skipped Crush 0.98.1 is too old for jl's setup: update it to 0.99.0 or later, then run jl connect crush."
    ) in " ".join(result.output.split())
    assert not (tmp_path / "crush" / "crushrc").exists()
    assert set(config.load_connection().harnesses) == {"aider"}


def test_crush_0_99_0_is_connected(subscribed, tmp_path, monkeypatch):
    # Built from Crush's AUR recipe, whose version starts with a v.
    crush_says(monkeypatch, "crush version v0.99.0\n")

    result = runner.invoke(app, ["connect", "crush", "--yes"])

    assert result.exit_code == 0, result.output
    assert "provider add jarvislabs --type completions" in (tmp_path / "crush" / "crushrc").read_text()


def test_crush_is_checked_once_per_connect(subscribed, tmp_path, monkeypatch):
    # Crush updated to... an older one while jl runs: the answer jl checked before the preview stands.
    answers = iter(["crush version 0.99.0\n", "crush version 0.98.1\n"])
    monkeypatch.setattr(harnesses, "_output", lambda argv: next(answers) if argv == ["crush", "--version"] else "")

    result = runner.invoke(app, ["connect", "crush", "--yes"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "crush" / "crushrc").exists()


CRUSH = next(h for h in harnesses.HARNESSES if h.id == "crush")


@pytest.mark.parametrize(
    "printed",
    [
        # Compared number by number, not as text: "0.100.0" < "0.99.0".
        "crush version 0.100.0\n",
        # A development build, or a Crush that doesn't start: jl can't tell, so it doesn't refuse.
        "crush version devel\n",
        "",
        # A pre-release of 0.99.0 counts as 0.99.0.
        "crush version v0.99.0-rc.1\n",
    ],
    ids=["0.100.0", "devel", "no-output", "pre-release"],
)
def test_check_version_lets_crush_through(monkeypatch, printed):
    crush_says(monkeypatch, printed)

    harnesses.check_version(CRUSH)


def test_check_version_refuses_a_pre_release_of_an_older_crush(monkeypatch):
    crush_says(monkeypatch, "crush version 0.98.3-nightly\n")

    with pytest.raises(JarvislabsError) as refused:
        harnesses.check_version(CRUSH)

    assert str(refused.value).startswith("Crush 0.98.3 is too old for jl's setup")


def test_the_preview_names_a_replaced_key_without_showing_it(subscribed, tmp_path):
    (tmp_path / ".aider.conf.yml").write_text("model: gpt-4o\nopenai-api-key: sk-openai-1\n")

    result = runner.invoke(app, ["connect", "aider", "--dry-run"])

    assert result.exit_code == 0, result.output
    preview = " ".join(result.output.split())
    assert "replaces openai-api-key, gpt-4o" in preview
    assert "sk-openai-1" not in preview


def key_changes(server):
    """Return the requests that created or revoked a key."""
    return [call for call in server.calls if call[1].startswith("users/api_keys") and call[0] != "GET"]


class InTurn(list):
    """Responses a fake route gives to successive requests, one each."""


class Routes(dict):
    """The fake server's routes, where an InTurn value answers each request with its next response."""

    def __getitem__(self, route):
        response = super().__getitem__(route)
        return response.pop(0) if isinstance(response, InTurn) else response


@pytest.fixture()
def picker(monkeypatch):
    """Answer questionary.select prompts with ``picker.answers``, in order, as if in a terminal.

    ``picker.offered`` records each prompt's choice values (titles for plain-text choices). Yes/no
    questions are declined.
    """
    monkeypatch.setattr(connect.render, "can_prompt", lambda: True)
    monkeypatch.setattr(questionary, "confirm", lambda message, **kwargs: SimpleNamespace(ask=lambda: False))
    picker = SimpleNamespace(answers=[], offered=[])

    def select(message, choices, **kwargs):
        picker.offered.append([getattr(c, "value", c) for c in choices if not isinstance(c, questionary.Separator)])
        answer = picker.answers.pop(0)
        return SimpleNamespace(ask=lambda: answer, application=SimpleNamespace())

    monkeypatch.setattr(questionary, "select", select)
    return picker


@pytest.fixture()
def unsubscribed(subscribed, monkeypatch):
    """The "devbox" machine of an account with no subscription; the browser is faked."""
    subscribed.routes[("GET", "subscription/usage")] = NotFoundError("This account has no active subscription.")
    subscribed.opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *args, **kwargs: subscribed.opened.append(url) or True)
    return subscribed


def test_connect_without_a_subscription_offers_subscribe_or_wallet(unsubscribed, opencode_config, picker):
    picker.answers = ["Subscribe now"]

    result = runner.invoke(app, ["connect", "opencode"])

    assert result.exit_code == 0, result.output
    # The fact, then what to do.
    # The state once, before the question; the wallet warning comes only after a wallet choice.
    assert "No subscription on this account. Just subscribed? It can take a minute to show.\n" in result.output
    assert "! No subscription found" not in result.output
    assert picker.offered == [["Continue on wallet", "Subscribe now", "Not now"]]
    assert unsubscribed.opened == ["https://jarvislabs.ai/products/coding-plan"]
    assert "Next: jl connect" in result.output
    assert "after paying, connect your tools" in result.output
    assert "jl will add the Jarvislabs provider" not in result.output
    assert [call for call in unsubscribed.calls if call[0] == "POST"] == []
    assert not opencode_config.exists()


def test_not_now_at_the_pay_question_leaves_without_changes(unsubscribed, opencode_config, picker):
    picker.answers = ["Not now"]

    result = runner.invoke(app, ["connect", "opencode"])

    assert result.exit_code == 0, result.output
    assert result.output.endswith("Next: jl connect\n      set up a coding tool when you want\n")
    assert [call for call in unsubscribed.calls if call[0] in ("POST", "DELETE")] == []
    # No wallet choice saved either, so the next jl connect asks again.
    assert not config.config_path().exists()
    assert not opencode_config.exists()


WALLET_KEY = {**CREATED_KEY, "kind": "payg", "token": "jlsk-wallet-secret"}


def test_continue_on_wallet_creates_a_wallet_key(unsubscribed, opencode_config, picker):
    picker.answers = ["Continue on wallet"]
    unsubscribed.routes[("POST", "users/api_keys")] = WALLET_KEY

    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    assert result.exit_code == 0, result.output
    assert ("POST", "users/api_keys", {"name": "jl-devbox-f7a7", "kind": "payg"}) in unsubscribed.calls
    assert config.load_connection().kind == "payg"
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-wallet-secret"
    # Said at the choice, not only on the next run.
    assert "On wallet (no subscription) Requests are charged to your wallet." in " ".join(result.output.split())


def test_the_wallet_choice_outlasts_the_last_disconnect(unsubscribed, opencode_config, picker):
    picker.answers = ["Continue on wallet"]
    unsubscribed.routes[("POST", "users/api_keys")] = WALLET_KEY
    unsubscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}
    runner.invoke(app, ["connect", "opencode"], input="y\n")
    runner.invoke(app, ["disconnect", "opencode", "--yes"])

    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    assert result.exit_code == 0, result.output
    # Asked only on the first connect.
    assert picker.offered == [["Continue on wallet", "Subscribe now", "Not now"]]
    assert "On wallet (no subscription)" in result.output
    assert config.load_connection().kind == "payg"


def test_a_dry_run_saves_no_wallet_choice(unsubscribed, opencode_config, picker):
    picker.answers = ["Continue on wallet"]

    result = runner.invoke(app, ["connect", "opencode", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "+ OpenCode  ~/opencode/opencode.json" in result.output
    assert not config.config_path().exists()


def test_the_wallet_choice_survives_an_early_stop(unsubscribed, opencode_config, picker):
    picker.answers = ["Continue on wallet"]
    unsubscribed.routes[("POST", "users/api_keys")] = WALLET_KEY
    # Stopped at the preview, before the wallet key is created.
    runner.invoke(app, ["connect", "opencode"], input="n\n")

    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    assert result.exit_code == 0, result.output
    # Asked only on the first connect.
    assert picker.offered == [["Continue on wallet", "Subscribe now", "Not now"]]
    assert config.load_connection().kind == "payg"


def test_a_key_of_the_wrong_kind_is_revoked_and_connect_stops(subscribed, opencode_config):
    # A created key whose response has no kind is a wallet key.
    subscribed.routes[("POST", "users/api_keys")] = {k: v for k, v in CREATED_KEY.items() if k != "kind"}
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 1
    assert "Subscriptions aren't available yet" in result.output
    assert ("DELETE", "users/api_keys/key_1", None) in subscribed.calls
    assert config.load_connection() is None
    assert not opencode_config.exists()


def api_key(key_id, name, kind="payg"):
    return {"id": key_id, "name": name, "kind": kind, "created_at": "2026-10-01T10:00:00Z"}


def test_at_the_key_cap_connect_offers_to_revoke_an_old_jl_key(subscribed, opencode_config, picker):
    subscribed.routes = Routes(subscribed.routes)
    subscribed.routes[("POST", "users/api_keys")] = InTurn(
        [APIError(429, "You can have at most 6 active API keys. Expire one before creating another."), CREATED_KEY]
    )
    subscribed.routes[("GET", "users/api_keys")] = [
        api_key("key_laptop", "jl-laptop", "subscription"),
        api_key("key_prod", "prod"),
        api_key("key_ci", "ci"),
        api_key("key_desktop", "jl-old-desktop", "subscription"),
        api_key("key_staging", "staging"),
        api_key("key_notebook", "notebook"),
    ]
    subscribed.routes[("DELETE", "users/api_keys/key_desktop")] = {"success": True}
    picker.answers = ["key_desktop"]

    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    assert result.exit_code == 0, result.output
    # Only jl's own keys are offered, with a way out that revokes nothing.
    assert picker.offered == [["key_laptop", "key_desktop", ""]]
    assert [call for call in subscribed.calls if call[0] == "DELETE"] == [
        ("DELETE", "users/api_keys/key_desktop", None)
    ]
    assert config.load_connection().token == "jlsk-new-secret"


@pytest.fixture()
def at_the_key_cap(subscribed, monkeypatch):
    """The account is at its key cap: two other machines' jl keys, one named by an older jl, and two of the user's own.

    Set ``at_the_key_cap.pick`` to the ID picked in the key list ("" keeps every key). The list's
    rows land in ``at_the_key_cap.rows``.
    """
    subscribed.routes[("POST", "users/api_keys")] = APIError(
        429, "You can have at most 6 active API keys. Expire one before creating another."
    )
    subscribed.routes[("GET", "users/api_keys")] = [
        api_key("key_win", "jl-DESKTOP-1A2B3C-c2a1"),
        api_key("key_laptop", "jl-laptop", "subscription"),
        api_key("key_prod", "prod"),
        api_key("key_ci", "ci"),
    ]
    monkeypatch.setattr(connect.render, "can_prompt", lambda: True)
    fake = SimpleNamespace(pick="", rows=[])

    def select(message, choices, **kwargs):
        fake.rows = [" ".join(choice.title.split()) for choice in choices]
        return SimpleNamespace(ask=lambda: fake.pick, application=SimpleNamespace())

    monkeypatch.setattr(questionary, "select", select)
    return fake


def test_the_key_cap_names_each_keys_machine(subscribed, opencode_config, at_the_key_cap):
    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    # The heading row, then one row per jl key, then the way out.
    assert at_the_key_cap.rows == [
        "NAME MACHINE PAYS FROM CREATED",
        "jl-DESKTOP-1A2B3C-c2a1 DESKTOP-1A2B3C wallet 2026-10-01",
        "jl-laptop laptop plan 2026-10-01",
        "Keep every key",
    ]
    assert (
        "Revoking another machine's key disconnects the coding tools on that machine until it runs jl connect again."
    ) in " ".join(result.output.split())


def test_keeping_every_key_echoes_one_line(subscribed, opencode_config, at_the_key_cap):
    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    assert result.exit_code == 1
    assert "? Revoke one of jl's keys to make room for this machine's key: Keep every key\n" in result.output
    assert "PAYS FROM" not in result.output


def test_at_the_key_cap_this_machines_lost_key_is_revoked_without_asking(subscribed, opencode_config, monkeypatch):
    # No key is saved, so a key with this machine's name is one whose secret was lost.
    subscribed.routes = Routes(subscribed.routes)
    subscribed.routes[("POST", "users/api_keys")] = InTurn(
        [APIError(429, "You can have at most 6 active API keys. Expire one before creating another."), CREATED_KEY]
    )
    subscribed.routes[("GET", "users/api_keys")] = [
        api_key("key_win", "jl-DESKTOP-1A2B3C-c2a1"),
        api_key("key_here", "jl-devbox-f7a7", "subscription"),
    ]
    subscribed.routes[("DELETE", "users/api_keys/key_here")] = {"success": True}
    monkeypatch.setattr(connect.render, "can_prompt", lambda: True)
    monkeypatch.setattr(questionary, "select", lambda message, **kwargs: pytest.fail(f"asked {message!r}"))
    # Any questionary question beyond the pay one is declined.
    monkeypatch.setattr(questionary, "confirm", lambda message, **kwargs: SimpleNamespace(ask=lambda: False))

    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    assert result.exit_code == 0, result.output
    assert [call for call in subscribed.calls if call[0] == "DELETE"] == [("DELETE", "users/api_keys/key_here", None)]
    assert "Revoked the key jl-devbox-f7a7, whose secret this machine no longer has." in " ".join(result.output.split())
    assert config.load_connection().token == "jlsk-new-secret"


def test_a_lost_local_key_is_replaced_on_the_server(subscribed, opencode_config):
    subscribed.routes = Routes(subscribed.routes)
    subscribed.routes[("POST", "users/api_keys")] = InTurn(
        [APIError(409, "A key with this name already exists. Please choose a different name."), CREATED_KEY]
    )
    subscribed.routes[("GET", "users/api_keys")] = [
        api_key("key_laptop", "jl-laptop", "subscription"),
        api_key("key_0", "jl-devbox-f7a7", "subscription"),
    ]
    subscribed.routes[("DELETE", "users/api_keys/key_0")] = {"success": True}

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert [call for call in subscribed.calls if call[0] == "DELETE"] == [("DELETE", "users/api_keys/key_0", None)]
    assert config.load_connection().token == "jlsk-new-secret"
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-new-secret"


@pytest.fixture()
def wallet_machine(subscribed, monkeypatch):
    """The "devbox" machine already connected to OpenCode with a wallet key; the account has since subscribed.

    It runs in a terminal, where any yes/no question is declined.
    """
    monkeypatch.setattr(connect.render, "can_prompt", lambda: True)
    monkeypatch.setattr(questionary, "confirm", lambda message, **kwargs: SimpleNamespace(ask=lambda: False))
    config.save_connection(
        Connection(
            key_id="key_0",
            name="jl-devbox-f7a7",
            kind="payg",
            token="jlsk-wallet-secret",
            harnesses={"opencode": Replaced()},
        )
    )
    subscribed.routes[("DELETE", "users/api_keys/key_0")] = {"success": True}
    return subscribed


def test_a_subscribed_wallet_user_is_offered_the_switch(wallet_machine, opencode_config):
    result = runner.invoke(app, ["connect", "opencode"], input="y\ny\n")

    assert result.exit_code == 0, result.output
    assert "Switch this machine's tools from your wallet to your plan?" in " ".join(result.output.split())
    assert ("DELETE", "users/api_keys/key_0", None) in wallet_machine.calls
    assert ("POST", "users/api_keys", {"name": "jl-devbox-f7a7", "kind": "subscription"}) in wallet_machine.calls
    assert config.load_connection() == Connection(
        key_id="key_1",
        name="jl-devbox-f7a7",
        kind="subscription",
        token="jlsk-new-secret",
        harnesses={"opencode": ANY},
    )
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-new-secret"


# Without a terminal, a "y" waiting on stdin for the switch must not be read.
@pytest.mark.parametrize(
    ("terminal", "stdin", "hint"), [(True, "y\nn\n", False), (False, "y\ny\n", True)], ids=["declined", "no-terminal"]
)
def test_the_wallet_key_stays_unless_the_user_agrees_to_switch(
    wallet_machine, opencode_config, monkeypatch, terminal, stdin, hint
):
    monkeypatch.setattr(connect.render, "can_prompt", lambda: terminal)

    result = runner.invoke(app, ["connect", "opencode"], input=stdin)

    assert result.exit_code == 0, result.output
    assert key_changes(wallet_machine) == []
    # With nobody to ask, the user is told how to switch.
    assert (
        "Run jl connect in a terminal to switch it to your plan, or run jl connect again with --yes."
        in " ".join(result.output.split())
    ) is hint
    assert config.load_connection().kind == "payg"
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-wallet-secret"


# Without a terminal too: a script that passes --yes wants the plan.
def test_yes_switches_a_wallet_machine_to_the_plan(wallet_machine, opencode_config, monkeypatch):
    monkeypatch.setattr(connect.render, "can_prompt", lambda: False)

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert key_changes(wallet_machine) == [
        ("DELETE", "users/api_keys/key_0", None),
        ("POST", "users/api_keys", {"name": "jl-devbox-f7a7", "kind": "subscription"}),
    ]
    assert config.load_connection().kind == "subscription"
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-new-secret"


def test_rotate_key_replaces_the_key_in_every_harness(subscribed, opencode_config):
    config.save_connection(
        Connection(
            key_id="key_0",
            name="jl-devbox-f7a7",
            kind="subscription",
            token="jlsk-leaked",
            harnesses={"opencode": Replaced()},
        )
    )
    subscribed.routes[("DELETE", "users/api_keys/key_0")] = {"success": True}

    result = runner.invoke(app, ["connect", "--rotate-key"])

    assert result.exit_code == 0, result.output
    assert ("DELETE", "users/api_keys/key_0", None) in subscribed.calls
    assert config.load_connection() == Connection(
        key_id="key_1",
        name="jl-devbox-f7a7",
        kind="subscription",
        token="jlsk-new-secret",
        harnesses={"opencode": ANY},
    )
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-new-secret"
    assert "Replaced this machine's key; updated 1 tool." in result.output


def test_continue_on_wallet_replaces_a_saved_subscription_key(unsubscribed, opencode_config, picker):
    # The plan has ended: the Models API refuses this subscription key, so the wallet choice needs a wallet key.
    config.save_connection(
        Connection(key_id="key_0", name="jl-devbox-f7a7", kind="subscription", token="jlsk-plan-secret", harnesses={})
    )
    unsubscribed.routes[("DELETE", "users/api_keys/key_0")] = {"success": True}
    unsubscribed.routes[("POST", "users/api_keys")] = WALLET_KEY
    picker.answers = ["Continue on wallet"]

    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    assert result.exit_code == 0, result.output
    assert ("DELETE", "users/api_keys/key_0", None) in unsubscribed.calls
    assert config.load_connection().kind == "payg"
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-wallet-secret"


IOREG_OUTPUT = """+-o Mac-EXAMPLE  <class IOPlatformExpertDevice, id 0x100000201, registered, matched, active, busy 0>
    {
      "IOPlatformSerialNumber" = "C02XXXXXX"
      "IOPlatformUUID" = "00000000-0000-4000-8000-000000000001"
    }
"""


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_the_machine_id_is_read_from_each_os(platform, tmp_path, monkeypatch):
    machine_id_file = tmp_path / "machine-id"
    machine_id_file.write_text("11111111222233334444555555555555\n")
    monkeypatch.setattr(connect, "MACHINE_ID_FILE", machine_id_file)
    monkeypatch.setattr(
        connect.subprocess, "run", lambda argv, **kwargs: SimpleNamespace(stdout=IOREG_OUTPUT, returncode=0)
    )
    registry = {(r"SOFTWARE\Microsoft\Cryptography", "MachineGuid"): "11111111-2222-3333-4444-555555555555"}
    winreg = SimpleNamespace(
        HKEY_LOCAL_MACHINE="HKLM",
        OpenKey=lambda root, path: contextlib.nullcontext(path),
        QueryValueEx=lambda key, name: (registry[(key, name)], 1),
    )
    monkeypatch.setitem(sys.modules, "winreg", winreg)
    monkeypatch.setattr(sys, "platform", platform)

    machine_id = connect._os_machine_id()

    assert (
        machine_id
        == {
            "linux": "11111111222233334444555555555555",
            "darwin": "00000000-0000-4000-8000-000000000001",
            "win32": "11111111-2222-3333-4444-555555555555",
        }[platform]
    )


def test_an_unreadable_machine_id_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(connect, "MACHINE_ID_FILE", tmp_path / "missing")
    monkeypatch.setattr(sys, "platform", "linux")

    assert connect._os_machine_id() is None


def test_a_saved_key_revoked_on_the_server_is_replaced(subscribed, opencode_config):
    config.save_connection(
        Connection(
            key_id="key_0",
            name="jl-devbox-f7a7",
            kind="subscription",
            token="jlsk-revoked",
            harnesses={"opencode": Replaced()},
        )
    )
    subscribed.routes[("GET", "users/api_keys")] = [api_key("key_laptop", "jl-laptop-d934", "subscription")]

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert ("POST", "users/api_keys", {"name": "jl-devbox-f7a7", "kind": "subscription"}) in subscribed.calls
    assert config.load_connection().token == "jlsk-new-secret"
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-new-secret"


def test_a_wallet_user_without_a_plan_is_not_asked_again(unsubscribed, opencode_config, picker):
    config.save_connection(
        Connection(key_id="key_0", name="jl-devbox-f7a7", kind="payg", token="jlsk-wallet-secret", harnesses={})
    )

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert picker.offered == []
    assert "On wallet (no subscription) Requests are charged to your wallet. Run jl subscribe for a plan." in " ".join(
        result.output.split()
    )
    assert key_changes(unsubscribed) == []
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-wallet-secret"


def test_without_a_subscription_and_no_one_to_ask_connect_stops(unsubscribed, opencode_config, monkeypatch):
    monkeypatch.setattr(connect.render, "can_prompt", lambda: False)

    # Without a terminal, connect still reads a "y" from stdin for the preview.
    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    assert result.exit_code == 1
    assert "To pay from your wallet instead, run jl connect --wallet." in " ".join(result.output.split())
    assert key_changes(unsubscribed) == []
    assert not opencode_config.exists()


def test_the_wallet_flag_creates_a_wallet_key_without_asking(unsubscribed, opencode_config):
    unsubscribed.routes[("POST", "users/api_keys")] = WALLET_KEY

    result = runner.invoke(app, ["connect", "opencode", "--wallet", "--yes"])

    assert result.exit_code == 0, result.output
    assert ("POST", "users/api_keys", {"name": "jl-devbox-f7a7", "kind": "payg"}) in unsubscribed.calls
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-wallet-secret"


# The catalog's USD prices per million tokens: DeepSeek's cache reads are cheaper than its input,
# GLM has no cached price, and Gemma's row has no price at all.
PRICED_CATALOG = {
    "object": "list",
    "data": [
        {
            **DEEPSEEK,
            "pricing": {
                "currency": "USD",
                "input_per_million_tokens": 0.14,
                "cached_input_per_million_tokens": 0.014,
                "output_per_million_tokens": 0.28,
            },
        },
        {
            **GLM,
            "pricing": {
                "currency": "USD",
                "input_per_million_tokens": 0.55,
                "cached_input_per_million_tokens": None,
                "output_per_million_tokens": 2.2,
            },
        },
        GEMMA,
    ],
}


def test_a_wallet_connect_writes_the_usd_prices_opencode_shows(unsubscribed, opencode_config):
    unsubscribed.routes[("GET", "models")] = PRICED_CATALOG
    unsubscribed.routes[("POST", "users/api_keys")] = WALLET_KEY

    result = runner.invoke(app, ["connect", "opencode", "--wallet", "--yes"])

    assert result.exit_code == 0, result.output
    written = read_json(opencode_config)
    # OpenCode shows a session's spend from the V2 cost, in USD per million tokens.
    assert {
        model_id: model.get("cost") for model_id, model in written["providers"]["jarvislabs"]["models"].items()
    } == {
        "deepseek-v4.1-flash-1017": {"input": 0.14, "output": 0.28, "cache": {"read": 0.014}},
        # No cached price, so no cache key.
        "glm-5.3": {"input": 0.55, "output": 2.2},
        "gemma-3-1b-it": None,
    }
    # The V1 entry, read by OpenCode 1.x, puts the cached rate beside the other two.
    assert {model_id: model.get("cost") for model_id, model in written["provider"]["jarvislabs"]["models"].items()} == {
        "deepseek-v4.1-flash-1017": {"input": 0.14, "output": 0.28, "cache_read": 0.014},
        "glm-5.3": {"input": 0.55, "output": 2.2},
        "gemma-3-1b-it": None,
    }


def test_a_subscription_connect_writes_no_prices(subscribed, opencode_config):
    subscribed.routes[("GET", "models")] = PRICED_CATALOG

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    written = read_json(opencode_config)
    # The plan pays nothing per token, so a cost readout there would be wrong.
    models = {**written["providers"]["jarvislabs"]["models"], **written["provider"]["jarvislabs"]["models"]}
    assert [model_id for model_id, model in models.items() if "cost" in model] == []


def test_a_model_priced_in_rupees_uses_its_usd_rates(unsubscribed, opencode_config):
    unsubscribed.routes[("GET", "models")] = {
        "object": "list",
        "data": [
            {
                **DEEPSEEK,
                "pricing": {
                    "currency": "INR",
                    "input_per_million_tokens": 12.0,
                    "cached_input_per_million_tokens": 1.2,
                    "output_per_million_tokens": 24.0,
                    "usd": {
                        "currency": "USD",
                        "input_per_million_tokens": 0.14,
                        "cached_input_per_million_tokens": 0.014,
                        "output_per_million_tokens": 0.28,
                    },
                },
            }
        ],
    }
    unsubscribed.routes[("POST", "users/api_keys")] = WALLET_KEY

    result = runner.invoke(app, ["connect", "opencode", "--wallet", "--yes"])

    assert result.exit_code == 0, result.output
    models = read_json(opencode_config)["providers"]["jarvislabs"]["models"]
    assert models["deepseek-v4.1-flash-1017"]["cost"] == {"input": 0.14, "output": 0.28, "cache": {"read": 0.014}}


def test_a_model_priced_in_rupees_without_usd_rates_gets_no_prices(unsubscribed, opencode_config):
    # A row priced in INR without a usd block: jl has no USD rate to write.
    unsubscribed.routes[("GET", "models")] = {
        "object": "list",
        "data": [
            {
                **DEEPSEEK,
                "pricing": {
                    "currency": "INR",
                    "input_per_million_tokens": 12.0,
                    "cached_input_per_million_tokens": 1.2,
                    "output_per_million_tokens": 24.0,
                },
            }
        ],
    }
    unsubscribed.routes[("POST", "users/api_keys")] = WALLET_KEY

    result = runner.invoke(app, ["connect", "opencode", "--wallet", "--yes"])

    assert result.exit_code == 0, result.output
    written = read_json(opencode_config)
    assert "cost" not in written["providers"]["jarvislabs"]["models"]["deepseek-v4.1-flash-1017"]
    assert "cost" not in written["provider"]["jarvislabs"]["models"]["deepseek-v4.1-flash-1017"]


def test_switching_to_the_plan_removes_the_prices(unsubscribed, opencode_config):
    unsubscribed.routes[("GET", "models")] = PRICED_CATALOG
    unsubscribed.routes[("POST", "users/api_keys")] = WALLET_KEY
    runner.invoke(app, ["connect", "opencode", "--wallet", "--yes"])
    assert "cost" in read_json(opencode_config)["providers"]["jarvislabs"]["models"]["deepseek-v4.1-flash-1017"]
    # The account takes a plan: the switch replaces the wallet key, and the wallet's prices go.
    unsubscribed.routes[("GET", "subscription/usage")] = USAGE
    unsubscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}
    unsubscribed.routes[("POST", "users/api_keys")] = CREATED_KEY

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert "cost" not in opencode_config.read_text()


def test_a_failed_write_revokes_the_key_it_just_made(subscribed, opencode_config, monkeypatch):
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}
    failure = PermissionError(13, "Permission denied", str(opencode_config))
    monkeypatch.setattr(connect, "apply", MagicMock(side_effect=failure))

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 1, result.output
    assert "Could not write" in " ".join(result.output.split())
    # The key this run made is revoked, so the account doesn't hold one no tool can use.
    assert any(call[:2] == ("DELETE", "users/api_keys/key_1") for call in subscribed.calls)
    assert config.load_connection() is None


def test_a_failed_rotation_says_the_tools_hold_a_dead_key(subscribed, opencode_config):
    # A tool is connected, so the rotation replaces the key it holds.
    runner.invoke(app, ["connect", "opencode", "--yes"])
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}
    subscribed.routes[("POST", "users/api_keys")] = APIError(500, "server exploded")

    result = runner.invoke(app, ["connect", "--rotate-key"])

    # main() prints this one line; the point is the message, not a traceback.
    assert isinstance(result.exception, JarvislabsError)
    said = str(result.exception)
    assert "server exploded" in said
    assert "already revoked jl-devbox-f7a7" in said
    assert "dead key" in said


def test_at_the_key_cap_with_no_one_to_ask_connect_lists_the_keys_and_stops(subscribed, opencode_config, monkeypatch):
    monkeypatch.setattr(connect.render, "can_prompt", lambda: False)
    subscribed.routes[("POST", "users/api_keys")] = APIError(
        429, "You can have at most 6 active API keys. Expire one before creating another."
    )
    subscribed.routes[("GET", "users/api_keys")] = [
        api_key("key_laptop", "jl-laptop-d934", "subscription"),
        api_key("key_prod", "prod"),
    ]

    result = runner.invoke(app, ["connect", "opencode"], input="y\n")

    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "jl-laptop-d934 (plan, created 2026-10-01)" in output
    assert "prod (wallet, created 2026-10-01)" in output
    assert "https://jarvislabs.ai/settings/api-keys" in output
    assert key_changes(subscribed) == [("POST", "users/api_keys", {"name": "jl-devbox-f7a7", "kind": "subscription"})]


# Neither is a plain yes: one spends wallet money, the other breaks another machine's tools.
@pytest.mark.parametrize(
    ("routes", "message", "changes"),
    [
        (
            {("GET", "subscription/usage"): NotFoundError("This account has no active subscription.")},
            "To pay from your wallet instead, run jl connect --wallet.",
            [],
        ),
        (
            {
                ("POST", "users/api_keys"): APIError(
                    429, "You can have at most 6 active API keys. Expire one before creating another."
                ),
                ("GET", "users/api_keys"): [api_key("key_laptop", "jl-laptop-d934", "subscription")],
            },
            "jl-laptop-d934 (plan, created 2026-10-01)",
            [("POST", "users/api_keys", {"name": "jl-devbox-f7a7", "kind": "subscription"})],
        ),
    ],
    ids=["no-subscription", "key-cap"],
)
def test_yes_still_stops_at_no_subscription_and_the_key_cap(
    subscribed, opencode_config, picker, routes, message, changes
):
    subscribed.routes.update(routes)

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 1
    assert message in " ".join(result.output.split())
    # Stopped without asking, though there's a terminal to ask in.
    assert picker.offered == []
    assert key_changes(subscribed) == changes
    assert not opencode_config.exists()


def test_print_revokes_a_key_of_the_wrong_kind(subscribed):
    subscribed.routes[("POST", "users/api_keys")] = {k: v for k, v in CREATED_KEY.items() if k != "kind"}
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}

    result = runner.invoke(app, ["connect", "--print", "zed"])

    assert result.exit_code == 1
    assert "jlsk-new-secret" not in result.stdout
    assert ("DELETE", "users/api_keys/key_1", None) in subscribed.calls
    assert config.load_connection() is None


def test_print_with_a_new_key_rewrites_the_connected_harnesses(wallet_machine, opencode_config):
    result = runner.invoke(app, ["connect", "--print", "zed"], input="y\n")

    assert result.exit_code == 0, result.output
    assert read_json(opencode_config)["provider"]["jarvislabs"]["options"]["apiKey"] == "jlsk-new-secret"


@pytest.mark.parametrize("args", [[], ["--json"]], ids=["text", "json"])
def test_status_offers_the_switch_to_a_subscribed_wallet_user(wallet_machine, args):
    result = runner.invoke(app, ["status", *args])

    assert result.exit_code == 0, result.output
    if args:
        assert json.loads(result.stdout)["switch_available"] is True
    else:
        said = " ".join(result.output.split())
        assert "! These tools pay from your wallet" in said
        assert "You have a subscription now. Run jl connect to switch them to your plan." in said


def test_status_offers_no_switch_without_a_plan(wallet_machine):
    wallet_machine.routes[("GET", "subscription/usage")] = NotFoundError("This account has no active subscription.")

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["switch_available"] is False


def test_a_help_example_for_print_names_a_printable_tool():
    """Each --print example in the help is one --print accepts."""
    result = runner.invoke(app, ["connect", "--help"], env={"COLUMNS": "200"})

    assert "jl connect --print zed" in result.output
    assert "--print claude-code" not in result.output


def test_a_mixed_preview_says_set_up(subscribed, opencode_config, terminal, tmp_path):
    """One tool to add and one to update: the headline doesn't call it all an add."""
    connect_before(terminal, "opencode")
    program(tmp_path / "bin", "kilo")

    result = runner.invoke(app, ["connect", "opencode", "kilo", "--dry-run"])

    assert "jl will set up the Jarvislabs provider" in said(result)


def zed_installed(tmp_path):
    """Put Zed's command on PATH, as Zed's "Install CLI" does."""
    for name in ("zed", "zed.cmd"):
        (tmp_path / "installed-harnesses" / name).write_text("")
        (tmp_path / "installed-harnesses" / name).chmod(0o755)


def test_zed_connects_and_its_key_goes_to_the_clipboard(subscribed, tmp_path, monkeypatch):
    zed_installed(tmp_path)
    copied = []
    monkeypatch.setattr(connect, "_copy_to_clipboard", lambda text: copied.append(text) or True)

    result = runner.invoke(app, ["connect", "zed", "--yes"])

    assert result.exit_code == 0, result.output
    provider = read_json(tmp_path / "zed" / "settings.json")["language_models"]["openai_compatible"]["Jarvislabs"]
    assert provider["api_url"] == "https://models.jarvislabs.net/v1"
    # Zed keeps the key in the OS keychain: never in its settings file.
    assert "jlsk-new-secret" not in (tmp_path / "zed" / "settings.json").read_text()
    assert copied == ["jlsk-new-secret"]
    text = said(result)
    # Said before the gate; after it, a "!" row, not a ✓ that reads as done, with Zed's own steps
    # and guide, and the key off the screen.
    assert "then you add the key in Zed yourself: jl shows how" in text
    assert "! Zed added, one step left: add your key in Zed" in text
    assert 'run "agent: open settings".' in text
    assert "2. Under LLM Providers, find Jarvislabs and paste your key; it's on your clipboard." in text
    assert "Zed's guide: https://zed.dev/docs/ai/use-api-access#api-keys" in text
    assert "✓ Zed" not in result.output
    assert "jlsk-new-secret" not in result.output


def test_zed_shows_the_key_when_there_is_no_clipboard(subscribed, tmp_path, monkeypatch):
    zed_installed(tmp_path)
    monkeypatch.setattr(connect, "_copy_to_clipboard", lambda text: False)

    result = runner.invoke(app, ["connect", "zed", "--yes"])

    assert result.exit_code == 0, result.output
    assert "2. Under LLM Providers, find Jarvislabs and paste your key: jlsk-new-secret" in said(result)


def test_reconnecting_zed_still_shows_how_to_add_the_key(subscribed, tmp_path, monkeypatch):
    zed_installed(tmp_path)
    monkeypatch.setattr(connect, "_copy_to_clipboard", lambda text: True)
    runner.invoke(app, ["connect", "zed", "--yes"])

    result = runner.invoke(app, ["connect", "zed", "--yes", "--model", "glm-5.3"])

    assert result.exit_code == 0, result.output
    # jl can't see Zed's keychain, so it never assumes the key was pasted: the steps come back,
    # worded for a key that may already be there.
    assert "! Zed updated: add your key in Zed if it isn't there yet" in said(result)
    assert "find Jarvislabs and paste your key" in said(result)
    assert "✓ Zed" not in result.output


def test_a_new_key_tells_zed_users_to_paste_it(subscribed, opencode_config, tmp_path, monkeypatch):
    zed_installed(tmp_path)
    monkeypatch.setattr(connect, "_copy_to_clipboard", lambda text: True)
    runner.invoke(app, ["connect", "opencode", "zed", "--yes"])
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}

    result = runner.invoke(app, ["connect", "--rotate-key", "--yes"])

    assert result.exit_code == 0, result.output
    # The refresh rewrote OpenCode's file; Zed's keychain is the user's to update.
    assert "Zed still holds the old key" in said(result)
    assert "find Jarvislabs and paste your key; it's on your clipboard." in said(result)


def test_zed_counts_as_installed_from_its_app_alone(subscribed, tmp_path, monkeypatch):
    app_bundle = tmp_path / "Zed.app"
    app_bundle.mkdir()
    zed = next(h for h in harnesses.HARNESSES if h.id == "zed")
    zed = dataclasses.replace(zed, apps=(str(app_bundle),))
    monkeypatch.setattr(shutil, "which", lambda name, path=None: None)

    assert harnesses.installed(zed)
    # No zed command, so the next step is to open the app.
    assert connect._start_command(zed) == "open Zed"


def test_no_clipboard_is_used_over_ssh(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")
    # Over SSH the clipboard would be the server's, not the user's.
    assert connect._copy_to_clipboard("key") is False


def test_disconnecting_zed_says_the_key_stays_in_zed(subscribed, tmp_path, monkeypatch):
    zed_installed(tmp_path)
    monkeypatch.setattr(connect, "_copy_to_clipboard", lambda text: True)
    runner.invoke(app, ["connect", "zed", "--yes"])
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}

    result = runner.invoke(app, ["disconnect", "zed", "--yes"])

    assert result.exit_code == 0, result.output
    # jl created the file, so it goes with jl's entries.
    assert not (tmp_path / "zed" / "settings.json").exists()
    assert "the key stays in Zed's LLM Providers settings until you remove it there" in said(result)


def test_a_tool_whose_file_was_removed_reads_added_again(subscribed, opencode_config):
    runner.invoke(app, ["connect", "opencode", "--yes"])
    # The user (or a restore) removes the file; jl's record of the connection stays.
    opencode_config.unlink()

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    # The preview says + add, so the result row says added, not updated.
    assert "+ OpenCode" in result.output
    assert "✓ OpenCode  added" in result.output


def test_status_counts_one_hidden_tool_in_the_singular(subscribed, monkeypatch):
    # Every tool installed but Aider: exactly one hidden from the panel.
    hidden = next(h for h in harnesses.HARNESSES if h.id == "aider")
    monkeypatch.setattr(commands, "installed", lambda h: h is not hidden)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "1 more tool: jl status --all" in said(result)
