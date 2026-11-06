"""`jl disconnect` and `jl logout`, run end to end with the servers faked at Transport."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from jarvislabs import config
from jarvislabs.cli import connect, render
from jarvislabs.cli.app import app
from jarvislabs.exceptions import APIError
from tests.cli import test_connect
from tests.cli.test_connect import CREATED_KEY, DEEPSEEK, GLM, ctrl_c_at, goose_dir, read_json

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import Result

runner = CliRunner()

# Re-exported, so pytest finds the fixtures here too.
subscribed = test_connect.subscribed
opencode_config = test_connect.opencode_config
up_to_date = test_connect.up_to_date

REVOKE = ("DELETE", "users/api_keys/key_1")


@pytest.fixture(autouse=True)
def revocable(subscribed: MagicMock) -> None:
    """The machine key that connect creates, key_1, can be revoked."""
    subscribed.routes[REVOKE] = {"success": True}


def connected() -> set[str] | None:
    """Return the harnesses this machine's saved key is recorded for, or None when no key is saved."""
    connection = config.load_connection()
    return set(connection.harnesses) if connection else None


def revokes(server: MagicMock) -> list[tuple[str, str, object]]:
    return [call for call in server.calls if call[0] == "DELETE"]


@pytest.fixture()
def terminal(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A terminal whose disconnect picker answers ``terminal.answer``; it records each question asked."""
    monkeypatch.setattr(render, "can_prompt", lambda: True)
    fake = SimpleNamespace(answer=[], questions=[])

    def checkbox(message, choices, **kwargs):
        fake.questions.append(SimpleNamespace(message=message, choices=choices, **kwargs))
        return MagicMock(ask=lambda: fake.answer)

    monkeypatch.setattr(connect.questionary, "checkbox", checkbox)
    return fake


def test_disconnect_without_a_tool_asks_which_connected_tools(subscribed: MagicMock, terminal: SimpleNamespace) -> None:
    runner.invoke(app, ["connect", "opencode", "aider", "--yes"])
    terminal.answer = ["opencode"]

    result = runner.invoke(app, ["disconnect"], input="y\n")

    assert result.exit_code == 0, result.output
    (question,) = terminal.questions
    assert question.message == "Disconnect which tool?"
    assert [choice.title for choice in question.choices] == ["OpenCode", "Aider"]
    assert f"Space toggles{render.MARKS['sep']}Enter continues" in question.instruction
    assert "? Disconnect OpenCode?" in result.output
    assert connected() == {"aider"}


def test_disconnect_without_a_tool_disconnects_every_tool_ticked(
    subscribed: MagicMock, terminal: SimpleNamespace
) -> None:
    runner.invoke(app, ["connect", "opencode", "aider", "--yes"])
    terminal.answer = ["opencode", "aider"]

    result = runner.invoke(app, ["disconnect"], input="y\ny\n")

    assert result.exit_code == 0, result.output
    assert connected() is None
    # The last tool's key goes too.
    assert revokes(subscribed) == [(*REVOKE, None)]


def test_disconnect_names_the_only_connected_tool_in_the_confirmation(
    subscribed: MagicMock, opencode_config: Path, terminal: SimpleNamespace
) -> None:
    runner.invoke(app, ["connect", "opencode", "--yes"])

    result = runner.invoke(app, ["disconnect"], input="y\n")

    assert result.exit_code == 0, result.output
    assert terminal.questions == []
    assert "? Disconnect OpenCode?" in result.output
    assert not opencode_config.exists()


def test_disconnect_with_no_connected_tools_says_so(subscribed: MagicMock, terminal: SimpleNamespace) -> None:
    result = runner.invoke(app, ["disconnect"])

    assert result.exit_code == 0, result.output
    assert "No tools are connected." in result.output
    assert terminal.questions == []


def test_disconnect_ticking_nothing_changes_nothing(subscribed: MagicMock, terminal: SimpleNamespace) -> None:
    runner.invoke(app, ["connect", "opencode", "aider", "--yes"])

    result = runner.invoke(app, ["disconnect"])

    assert result.exit_code == 0, result.output
    assert "No tools disconnected." in result.output
    assert connected() == {"opencode", "aider"}


def test_disconnect_without_a_tool_keeps_the_error_without_a_terminal(
    subscribed: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(render, "can_prompt", lambda: False)

    result = runner.invoke(app, ["disconnect"])

    assert result.exit_code == 2
    assert "Missing argument 'HARNESS'." in result.output


def test_ctrl_c_at_the_disconnect_picker_aborts_with_130(subscribed: MagicMock, terminal: SimpleNamespace) -> None:
    runner.invoke(app, ["connect", "opencode", "aider", "--yes"])
    # questionary's questions answer None to Ctrl-C.
    terminal.answer = None

    result = runner.invoke(app, ["disconnect"])

    assert result.exit_code == 130, result.output
    assert connected() == {"opencode", "aider"}


def test_disconnect_restores_replaced_values_and_keeps_later_edits(
    subscribed: MagicMock, opencode_config: Path
) -> None:
    original = (
        '{\n  // My settings\n  "theme": "tokyonight",\n  "model": "anthropic/claude-sonnet-4",\n  "provider": {}\n}\n'
    )
    opencode_config.write_text(original)
    runner.invoke(app, ["connect", "opencode", "--yes"])
    # Edited after connecting: the theme changes.
    opencode_config.write_text(opencode_config.read_text().replace("tokyonight", "gruvbox"))

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert opencode_config.read_text() == original.replace("tokyonight", "gruvbox")


def test_disconnecting_kilo_keeps_the_file_its_extension_uses(subscribed: MagicMock, tmp_path: Path) -> None:
    (tmp_path / ".vscode" / "extensions" / "kilocode.kilo-code-1.0.0").mkdir(parents=True)
    runner.invoke(app, ["connect", "kilo", "kilo-vscode", "--yes"])
    path = tmp_path / "kilo" / "kilo.jsonc"
    assert path.exists()

    result = runner.invoke(app, ["disconnect", "kilo", "--yes"])

    assert result.exit_code == 0, result.output
    # The extension still uses the file: jl empties its own entries instead of deleting it.
    assert path.exists()
    assert "Kilo (VS Code) shares" in " ".join(result.output.split())
    assert "jl connect kilo-vscode" in " ".join(result.output.split())


def test_disconnect_gives_back_kimi_codes_config(subscribed: MagicMock, tmp_path: Path) -> None:
    path = tmp_path / ".kimi-code" / "config.toml"
    path.parent.mkdir(parents=True)
    original = '# Mine\ndefault_model = "kimi-code/k3"\ntelemetry = false\n'
    path.write_text(original)
    runner.invoke(app, ["connect", "kimi", "--yes"])

    result = runner.invoke(app, ["disconnect", "kimi", "--yes"])

    assert result.exit_code == 0, result.output
    # The user's provider, default and comments are back; jl's models are gone.
    assert path.read_text() == original


def test_disconnect_previews_a_removal_row_and_names_the_tool(subscribed: MagicMock, opencode_config: Path) -> None:
    opencode_config.write_text('{\n  "model": "anthropic/claude-sonnet-4"\n}\n')
    runner.invoke(app, ["connect", "opencode", "--yes"])

    result = runner.invoke(app, ["disconnect", "opencode"], input="y\n")

    assert result.exit_code == 0, result.output
    # The row of the connect preview, with - for removal; the question names the tool.
    assert (
        "- OpenCode  ~/opencode/opencode.json\n"
        "            remove the Jarvislabs entries, put back your model\n"
        "? Disconnect OpenCode? (y/N) "
    ) in result.output
    # The last tool: its key is revoked, and the backup from connect stays.
    assert result.output.endswith(
        "✓ OpenCode disconnected\n"
        "  revoked this machine's key jl-devbox-f7a7\n"
        "  your config from before jl is still at ~/opencode/opencode.json.jl-backup\n"
    )


def test_disconnect_keeps_a_default_the_user_chose_after_connecting(
    subscribed: MagicMock, opencode_config: Path
) -> None:
    opencode_config.write_text('{\n  "model": "anthropic/claude-sonnet-4"\n}\n')
    runner.invoke(app, ["connect", "opencode", "--yes"])
    written = read_json(opencode_config)
    opencode_config.write_text(opencode_config.read_text().replace(written["model"], "openai/gpt-5"))

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(opencode_config) == {"model": "openai/gpt-5"}


def test_jls_provider_goes_even_after_the_user_edited_it(subscribed: MagicMock, opencode_config: Path) -> None:
    opencode_config.write_text('{\n  "theme": "tokyonight"\n}\n')
    runner.invoke(app, ["connect", "opencode", "--yes"])
    opencode_config.write_text(opencode_config.read_text().replace('"GLM-5.3"', '"My GLM"'))

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(opencode_config) == {"theme": "tokyonight"}


def test_disconnect_restores_the_default_from_before_the_latest_connect(
    subscribed: MagicMock, opencode_config: Path
) -> None:
    opencode_config.write_text('{\n  "model": "anthropic/claude-sonnet-4"\n}\n')
    runner.invoke(app, ["connect", "opencode", "--yes"])
    written = read_json(opencode_config)
    opencode_config.write_text(opencode_config.read_text().replace(written["model"], "openai/gpt-5"))
    runner.invoke(app, ["connect", "opencode", "--yes"])

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(opencode_config) == {"model": "openai/gpt-5"}


def test_disconnect_gives_back_goose_aider_and_crush_configs(
    subscribed: MagicMock, tmp_path: Path, up_to_date: None
) -> None:
    goose = goose_dir(tmp_path)
    goose.mkdir(parents=True)
    originals = {
        goose / "config.yaml": "# Model\nactive_provider: anthropic\nGOOSE_MODE: smart_approve\n",
        tmp_path / ".aider.conf.yml": "# Aider\nmodel: gpt-4o\ndark-mode: true\n",
        tmp_path / ".aider.model.metadata.json": '{\n  "openai/gpt-4o": {"max_input_tokens": 128000}\n}\n',
        tmp_path / "crush" / "crushrc": "option progress false\n",
    }
    (tmp_path / "crush").mkdir()
    for path, text in originals.items():
        path.write_text(text)
    runner.invoke(app, ["connect", "goose", "aider", "crush", "--yes"])
    # jl update adds a model's entries after the connect; disconnect takes those out too.
    qwen = {**GLM, "id": "qwen3.6-coder", "name": "Qwen3.6 Coder"}
    subscribed.routes[("GET", "models")] = {"object": "list", "data": [DEEPSEEK, GLM, qwen]}
    runner.invoke(app, ["update"])

    for harness in ("goose", "aider", "crush"):
        result = runner.invoke(app, ["disconnect", harness, "--yes"])
        assert result.exit_code == 0, result.output

    assert {path: path.read_text() for path in originals} == originals
    # The provider file jl created held only jl's entries, so it's gone.
    assert not (goose / "custom_providers" / "jarvislabs.json").exists()


def test_a_file_jl_created_goes_unless_the_user_added_to_it(subscribed: MagicMock, opencode_config: Path) -> None:
    runner.invoke(app, ["connect", "opencode", "--yes"])

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert not opencode_config.exists()

    runner.invoke(app, ["connect", "opencode", "--yes"])
    opencode_config.write_text(opencode_config.read_text().replace("{", '{\n  "theme": "gruvbox",', 1))

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(opencode_config) == {"theme": "gruvbox"}


def test_disconnecting_the_last_harness_revokes_the_key(
    subscribed: MagicMock, opencode_config: Path, tmp_path: Path
) -> None:
    runner.invoke(app, ["connect", "opencode", "aider", "--yes"])

    result = runner.invoke(app, ["disconnect", "aider", "--yes"])

    assert result.exit_code == 0, result.output
    assert revokes(subscribed) == []
    assert connected() == {"opencode"}

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert revokes(subscribed) == [(*REVOKE, None)]
    assert config.load_connection() is None
    assert not opencode_config.exists()


def test_a_failed_revoke_comes_after_the_last_harness_is_removed(subscribed: MagicMock, opencode_config: Path) -> None:
    runner.invoke(app, ["connect", "opencode", "--yes"])
    subscribed.routes[REVOKE] = APIError(503, "Server unavailable")

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 1
    assert "jl-devbox-f7a7" in result.output
    assert not opencode_config.exists()
    # The key stays saved, so nothing is lost: the next connect reuses it.
    assert connected() == set()


def test_a_printed_harness_only_loses_its_record_and_still_holds_the_key(
    subscribed: MagicMock, opencode_config: Path, tmp_path: Path
) -> None:
    runner.invoke(app, ["connect", "--print", "zed"])
    runner.invoke(app, ["connect", "opencode", "--yes"])

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    # Zed may hold the key by hand: it stays.
    assert result.exit_code == 0, result.output
    assert revokes(subscribed) == []
    assert connected() == {"zed"}

    result = runner.invoke(app, ["disconnect", "zed", "--yes"])

    assert result.exit_code == 0, result.output
    assert revokes(subscribed) == [(*REVOKE, None)]
    assert config.load_connection() is None
    assert not list(tmp_path.rglob("settings.json"))


def test_declining_the_disconnect_changes_nothing(subscribed: MagicMock, opencode_config: Path) -> None:
    runner.invoke(app, ["connect", "opencode", "--yes"])
    text = opencode_config.read_text()

    result = runner.invoke(app, ["disconnect", "opencode"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "- OpenCode  ~/opencode/opencode.json\n            delete the file jl created\n" in result.output
    assert opencode_config.read_text() == text
    assert connected() == {"opencode"}


def test_ctrl_c_at_the_disconnect_question_aborts_with_130(
    subscribed: MagicMock, opencode_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["connect", "opencode", "--yes"])
    written = opencode_config.read_text()
    ctrl_c_at(monkeypatch, "Disconnect OpenCode?")

    result = runner.invoke(app, ["disconnect", "opencode"])

    assert result.exit_code == 130, result.output
    assert opencode_config.read_text() == written
    assert connected() == {"opencode"}


def test_disconnecting_a_harness_that_isnt_connected_stops(subscribed: MagicMock) -> None:
    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert "OpenCode isn't connected" in result.output
    assert revokes(subscribed) == []


def test_logout_keep_tools_keeps_the_machine_key_for_disconnect(subscribed: MagicMock, opencode_config: Path) -> None:
    runner.invoke(app, ["connect", "opencode", "--yes"])
    config.save_config({**config.load_config(), "auth": {"token": "jarvis-key"}})

    result = runner.invoke(app, ["logout", "--keep-tools"])

    assert result.exit_code == 0, result.output
    assert "auth" not in config.load_config()
    assert connected() == {"opencode"}
    # The server key stays, so the harnesses keep working.
    assert revokes(subscribed) == []
    assert "Your coding tools stay connected; run jl disconnect <tool> to remove them" in " ".join(
        result.output.split()
    )


def test_logout_removes_tools_and_revokes_the_key(subscribed: MagicMock, opencode_config: Path) -> None:
    runner.invoke(app, ["connect", "opencode", "--yes"])
    config.save_config({**config.load_config(), "auth": {"token": "jarvis-key"}})

    result = runner.invoke(app, ["logout", "--yes"])

    assert result.exit_code == 0, result.output
    assert "auth" not in config.load_config()
    assert connected() is None
    assert config.load_connection() is None
    assert revokes(subscribed) == [(*REVOKE, None)]
    assert not opencode_config.exists()


def disconnect_logged_out(monkeypatch: pytest.MonkeyPatch) -> Result:
    """Connect OpenCode, log out (with no Jarvis key in the environment), then disconnect OpenCode."""
    runner.invoke(app, ["connect", "opencode", "--yes"])
    monkeypatch.delenv("JL_API_KEY")
    runner.invoke(app, ["logout", "--keep-tools"])
    return runner.invoke(app, ["disconnect", "opencode", "--yes"])


def test_disconnect_after_logout_cleans_files_and_explains_the_key(
    subscribed: MagicMock, opencode_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = disconnect_logged_out(monkeypatch)

    assert result.exit_code == 0, result.output
    assert not opencode_config.exists()
    assert revokes(subscribed) == []
    # The key stays saved with no harness, so a disconnect after logging in can still revoke it.
    assert connected() == set()
    assert (
        "Run jl setup, then jl disconnect opencode again to revoke this machine's key, "
        "or revoke jl-devbox-f7a7 at https://jarvislabs.ai/settings/api-keys." in " ".join(result.output.split())
    )


def test_disconnect_after_logging_in_again_revokes_the_kept_key(
    subscribed: MagicMock, opencode_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disconnect_logged_out(monkeypatch)
    monkeypatch.setenv("JL_API_KEY", "jarvis-key")

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert revokes(subscribed) == [(*REVOKE, None)]
    assert config.load_connection() is None
    assert "Revoked this machine's key jl-devbox-f7a7." in result.output


def test_rotate_key_revokes_a_kept_key_no_tool_uses(
    subscribed: MagicMock, opencode_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disconnect_logged_out(monkeypatch)
    monkeypatch.setenv("JL_API_KEY", "jarvis-key")
    calls = len(subscribed.calls)

    result = runner.invoke(app, ["connect", "--rotate-key"])

    assert result.exit_code == 0, result.output
    assert revokes(subscribed) == [(*REVOKE, None)]
    # No tool would hold a new key, so none is created.
    assert [call for call in subscribed.calls[calls:] if call[0] == "POST"] == []
    assert config.load_connection() is None
    assert "Revoked this machine's key jl-devbox-f7a7." in result.output


CLINE_SLOT = ("providers", "openai-compatible")


def cline_file(tmp_path: Path) -> Path:
    return tmp_path / ".cline" / "data" / "settings" / "providers.json"


def test_cline_keeps_its_version_and_providers_added_since(subscribed: MagicMock, tmp_path: Path) -> None:
    runner.invoke(app, ["connect", "cline", "--yes"])
    path = cline_file(tmp_path)
    data = read_json(path)
    # Cline saved a provider of its own into the file jl created.
    data["providers"]["anthropic"] = {"settings": {"provider": "anthropic", "apiKey": "sk-ant"}}
    path.write_text(json.dumps(data, indent=2))

    result = runner.invoke(app, ["disconnect", "cline", "--yes"])

    assert result.exit_code == 0, result.output
    # Without its version, Cline reads the file as empty and its next save drops every provider.
    assert read_json(path) == {"version": 1, "providers": {"anthropic": data["providers"]["anthropic"]}}


def test_a_cline_file_holding_only_jls_entries_goes(subscribed: MagicMock, tmp_path: Path) -> None:
    runner.invoke(app, ["connect", "cline", "--yes"])

    result = runner.invoke(app, ["disconnect", "cline", "--yes"])

    assert result.exit_code == 0, result.output
    assert not cline_file(tmp_path).exists()


def test_disconnect_after_a_rotate_gives_back_cline(subscribed: MagicMock, tmp_path: Path) -> None:
    cline = cline_file(tmp_path)
    originals = {
        cline: json.dumps(
            {
                "version": 1,
                "lastUsedProvider": "openrouter",
                "providers": {
                    "openai-compatible": {"settings": {"baseUrl": "https://openrouter.ai/api/v1", "apiKey": "sk-or"}}
                },
            },
            indent=2,
        ),
    }
    for path, text in originals.items():
        path.parent.mkdir(parents=True)
        path.write_text(text)
    runner.invoke(app, ["connect", "cline", "--yes"])
    rotated = {**CREATED_KEY, "id": "key_2", "token": "jlsk-rotated"}
    subscribed.routes[("POST", "users/api_keys")] = rotated
    subscribed.routes[("DELETE", "users/api_keys/key_2")] = {"success": True}
    # The rotation refreshes only the key, at paths inside the values the connect wrote.
    result = runner.invoke(app, ["connect", "--rotate-key"])
    assert result.exit_code == 0, result.output
    assert "jlsk-rotated" in cline.read_text()

    result = runner.invoke(app, ["disconnect", "cline", "--yes"])
    assert result.exit_code == 0, result.output

    assert {path: path.read_text() for path in originals} == originals


def test_disconnect_prints_where_the_backups_are(subscribed: MagicMock, opencode_config: Path) -> None:
    opencode_config.write_text('{\n  "theme": "tokyonight"\n}\n')
    runner.invoke(app, ["connect", "opencode", "--yes"])
    backup = opencode_config.with_name("opencode.json.jl-backup")

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert backup.exists()
    assert "your config from before jl is still at ~/opencode/opencode.json.jl-backup" in result.output


GPT = {"id": "gpt-4o", "envKey": "OPENAI_API_KEY", "baseUrl": "https://api.openai.com/v1"}
OLLAMA = {"name": "ollama", "type": "openai", "baseUrl": "http://localhost:11434/v1"}
LLAMA = {"id": "llama3", "provider": "ollama"}


def shared_lists(tmp_path: Path) -> tuple[Path, Path]:
    """Write Qwen Code's and Copilot CLI's lists holding one entry of the user's, and connect both."""
    qwen = tmp_path / ".qwen" / "settings.json"
    copilot = tmp_path / ".copilot" / "providers.json"
    for path, data in (
        (qwen, {"modelProviders": {"openai": [GPT]}}),
        (copilot, {"providers": [OLLAMA], "models": [LLAMA]}),
    ):
        path.parent.mkdir()
        path.write_text(json.dumps(data, indent=2))
    result = runner.invoke(app, ["connect", "qwen", "copilot", "--yes"])
    assert result.exit_code == 0, result.output
    return qwen, copilot


def disconnect_qwen_and_copilot() -> str:
    """Disconnect Qwen Code, then Copilot CLI, and return what they printed."""
    output = ""
    for harness in ("qwen", "copilot"):
        result = runner.invoke(app, ["disconnect", harness, "--yes"])
        assert result.exit_code == 0, result.output
        output += result.output
    return output


def test_disconnect_keeps_items_the_user_added_to_shared_lists(subscribed: MagicMock, tmp_path: Path) -> None:
    qwen, copilot = shared_lists(tmp_path)
    claude = {"id": "claude-sonnet-4", "envKey": "OPENROUTER_API_KEY", "baseUrl": "https://openrouter.ai/api/v1"}
    lmstudio = {"name": "lmstudio", "type": "openai", "baseUrl": "http://localhost:1234/v1"}
    qwen2 = {"id": "qwen2", "provider": "lmstudio"}
    # Added after connecting, after jl's entries.
    data = read_json(qwen)
    data["modelProviders"]["openai"].append(claude)
    qwen.write_text(json.dumps(data, indent=2))
    data = read_json(copilot)
    data["providers"].append(lmstudio)
    data["models"].append(qwen2)
    copilot.write_text(json.dumps(data, indent=2))

    output = disconnect_qwen_and_copilot()

    assert read_json(qwen) == {"modelProviders": {"openai": [GPT, claude]}}
    assert read_json(copilot) == {"providers": [OLLAMA, lmstudio], "models": [LLAMA, qwen2]}
    # The lists aren't put back as they were before jl: only jl's entries are taken out.
    assert "put back" not in output


def test_disconnect_keeps_a_jl_list_item_the_user_edited(subscribed: MagicMock, tmp_path: Path) -> None:
    qwen, copilot = shared_lists(tmp_path)
    data = read_json(qwen)
    glm = next(entry for entry in data["modelProviders"]["openai"] if entry["id"] == GLM["id"])
    glm["name"] = "My GLM"
    qwen.write_text(json.dumps(data, indent=2))
    data = read_json(copilot)
    deepseek = next(model for model in data["models"] if model["id"] == DEEPSEEK["id"])
    deepseek["maxOutputTokens"] = 4096
    copilot.write_text(json.dumps(data, indent=2))

    disconnect_qwen_and_copilot()

    assert read_json(qwen) == {"modelProviders": {"openai": [GPT, glm]}}
    assert read_json(copilot) == {"providers": [OLLAMA], "models": [LLAMA, deepseek]}


def test_disconnect_after_a_rotate_takes_out_the_new_entries(subscribed: MagicMock, tmp_path: Path) -> None:
    qwen, copilot = shared_lists(tmp_path)
    subscribed.routes[("POST", "users/api_keys")] = {**CREATED_KEY, "id": "key_2", "token": "jlsk-rotated"}
    subscribed.routes[("DELETE", "users/api_keys/key_2")] = {"success": True}
    # The rotation rewrites Copilot CLI's provider entry, which holds the key.
    result = runner.invoke(app, ["connect", "--rotate-key"])
    assert result.exit_code == 0, result.output
    assert "jlsk-rotated" in copilot.read_text()

    disconnect_qwen_and_copilot()

    assert read_json(qwen) == {"modelProviders": {"openai": [GPT]}}
    assert read_json(copilot) == {"providers": [OLLAMA], "models": [LLAMA]}


def test_qwen_and_copilot_files_jl_created_go(subscribed: MagicMock, tmp_path: Path) -> None:
    runner.invoke(app, ["connect", "qwen", "copilot", "--yes"])

    disconnect_qwen_and_copilot()

    assert not (tmp_path / ".qwen" / "settings.json").exists()
    assert not (tmp_path / ".copilot" / "providers.json").exists()


def test_a_record_saved_without_list_entries_still_disconnects(subscribed: MagicMock, opencode_config: Path) -> None:
    opencode_config.write_text('{\n  "theme": "tokyonight"\n}\n')
    runner.invoke(app, ["connect", "opencode", "--yes"])
    # A record saved before jl tracked entries in shared lists has no lists or items.
    saved = config.load_config()
    for written in saved["connect"]["harnesses"]["opencode"]["written"]:
        del written["lists"], written["items"]
    config.save_config(saved)

    result = runner.invoke(app, ["disconnect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(opencode_config) == {"theme": "tokyonight"}


AIDER_CONF = "# Aider\nmodel: gpt-4o\ndark-mode: true\n"


def connect_aider_with_an_old_record(tmp_path: Path) -> Path:
    """Connect Aider over the user's own config, then make its record one an older jl saved.

    That record doesn't say what jl wrote, and lists jl's own key among the values it replaced.
    """
    conf = tmp_path / ".aider.conf.yml"
    conf.write_text(AIDER_CONF)
    runner.invoke(app, ["connect", "aider", "--yes"])
    saved = config.load_config()
    record = saved["connect"]["harnesses"]["aider"]
    del record["written"]
    record.pop("untracked", None)
    record["replaced"].append({"file": str(conf), "path": ["openai-api-key"], "value": json.dumps("jlsk-old-key")})
    config.save_config(saved)
    return conf


def test_an_old_record_is_disconnected_without_guessing(subscribed: MagicMock, tmp_path: Path) -> None:
    conf = connect_aider_with_an_old_record(tmp_path)

    result = runner.invoke(app, ["disconnect", "aider", "--yes"])

    assert result.exit_code == 0, result.output
    # The base URL names Jarvislabs, so the key and model jl wrote beside it go with it; nothing is put back.
    assert conf.read_text() == "# Aider\ndark-mode: true\n"
    said = " ".join(result.output.split())
    assert "Aider was connected by an older jl, which didn't record the settings it replaced" in said
    assert "your config from before jl is still at ~/.aider.conf.yml.jl-backup" in said


def test_reconnecting_over_an_old_record_puts_back_none_of_its_values(subscribed: MagicMock, tmp_path: Path) -> None:
    conf = connect_aider_with_an_old_record(tmp_path)
    runner.invoke(app, ["connect", "aider", "--yes"])

    result = runner.invoke(app, ["disconnect", "aider", "--yes"])

    assert result.exit_code == 0, result.output
    assert conf.read_text() == "# Aider\ndark-mode: true\n"
    assert "Aider was connected by an older jl" in " ".join(result.output.split())
