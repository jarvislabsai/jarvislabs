"""`jl connect` for the JSON-family harnesses: Kilo, pi, Cline, Qwen Code, Copilot CLI and Junie CLI."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from typer.testing import CliRunner

from jarvislabs import config
from jarvislabs.cli import connect
from jarvislabs.cli.app import app
from jarvislabs.harnesses import HARNESSES, installed
from tests.cli import test_connect
from tests.cli.test_connect import CREATED_KEY, DEEPSEEK, GEMMA, GLM, read_json

runner = CliRunner()
# A subscriber with no key yet, and the DeepSeek, GLM and Gemma catalog.
subscribed = test_connect.subscribed
# jl is already the latest version, for jl update.
up_to_date = test_connect.up_to_date

URL = "https://models.jarvislabs.net/v1"
KEY = "jlsk-new-secret"
JSON_FAMILY = ("kilo", "pi", "cline", "qwen", "copilot", "junie")
NAMES = ("Kilo", "pi", "Cline", "Qwen Code", "Copilot CLI", "Junie CLI")
# The catalog with GLM marked as a thinking model.
CATALOG = {"object": "list", "data": [DEEPSEEK, {**GLM, "thinking": {"supported": True}}, GEMMA]}


def test_connect_writes_each_json_family_harness_in_its_own_format(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = CATALOG

    result = runner.invoke(app, ["connect", *JSON_FAMILY, "--yes"])

    assert result.exit_code == 0, result.output
    kilo = read_json(tmp_path / "kilo" / "kilo.jsonc")
    assert kilo["model"] == "jarvislabs/deepseek-v4.1-flash-1017"
    assert kilo["provider"]["jarvislabs"]["options"] == {"baseURL": URL, "apiKey": KEY}
    assert {model_id: m["limit"] for model_id, m in kilo["provider"]["jarvislabs"]["models"].items()} == {
        "deepseek-v4.1-flash-1017": {"context": 1048576, "output": 65536},
        "glm-5.3": {"context": 230000, "output": 32768},
        "gemma-3-1b-it": {"context": 32768, "output": 32768},
    }

    pi = read_json(tmp_path / ".pi" / "agent" / "models.json")["providers"]["jarvislabs"]
    assert pi == {
        "name": "Jarvislabs",
        "baseUrl": URL,
        "api": "openai-completions",
        "apiKey": KEY,
        # Without these pi takes us for OpenAI: it sends store, the developer role and max_completion_tokens.
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
        "models": [
            {
                "id": "deepseek-v4.1-flash-1017",
                "name": "DeepSeek V4.1 Flash",
                "reasoning": False,
                "input": ["text", "image"],
                "contextWindow": 1048576,
                "maxTokens": 65536,
            },
            {
                "id": "glm-5.3",
                "name": "GLM-5.3",
                "reasoning": True,
                "input": ["text"],
                "contextWindow": 230000,
                "maxTokens": 32768,
            },
            {
                "id": "gemma-3-1b-it",
                "name": "Gemma 3 1B IT",
                "reasoning": False,
                "input": ["text"],
                "contextWindow": 32768,
                "maxTokens": 32768,
            },
        ],
    }
    assert read_json(tmp_path / ".pi" / "agent" / "settings.json") == {
        "defaultProvider": "jarvislabs",
        "defaultModel": "deepseek-v4.1-flash-1017",
    }

    cline = read_json(tmp_path / ".cline" / "data" / "settings" / "providers.json")
    # Cline requires an ISO time with milliseconds and a Z, and drops every provider if the file breaks its schema.
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", cline["providers"]["openai-compatible"].pop("updatedAt")
    )
    assert cline == {
        "version": 1,
        "lastUsedProvider": "openai-compatible",
        "providers": {
            "openai-compatible": {
                "tokenSource": "manual",
                # Cline's OpenAI-compatible provider holds one model: the default.
                "settings": {
                    "provider": "openai-compatible",
                    "apiKey": KEY,
                    "baseUrl": URL,
                    "model": "deepseek-v4.1-flash-1017",
                    "contextWindow": 1048576,
                    "maxTokens": 65536,
                    "capabilities": ["streaming", "tools", "vision"],
                },
            }
        },
    }

    qwen = read_json(tmp_path / ".qwen" / "settings.json")
    # Qwen Code reads the key from the env var an entry names, here set in the file's env block.
    assert qwen["env"] == {"JARVISLABS_API_KEY": KEY}
    assert qwen["security"] == {"auth": {"selectedType": "openai"}}
    assert qwen["model"] == {"name": "deepseek-v4.1-flash-1017"}
    assert qwen["modelProviders"] == {
        "openai": [
            {
                "id": "deepseek-v4.1-flash-1017",
                "name": "DeepSeek V4.1 Flash",
                "envKey": "JARVISLABS_API_KEY",
                "baseUrl": URL,
                "generationConfig": {
                    "contextWindowSize": 1048576,
                    "samplingParams": {"max_tokens": 65536},
                    "modalities": {"image": True},
                },
            },
            {
                "id": "glm-5.3",
                "name": "GLM-5.3",
                "envKey": "JARVISLABS_API_KEY",
                "baseUrl": URL,
                "generationConfig": {
                    "contextWindowSize": 230000,
                    "samplingParams": {"max_tokens": 32768},
                    "modalities": {"image": False},
                },
            },
            {
                "id": "gemma-3-1b-it",
                "name": "Gemma 3 1B IT",
                "envKey": "JARVISLABS_API_KEY",
                "baseUrl": URL,
                "generationConfig": {
                    "contextWindowSize": 32768,
                    "samplingParams": {"max_tokens": 32768},
                    "modalities": {"image": False},
                },
            },
        ]
    }

    # Copilot CLI takes "$VAR" in this file literally, so the key is written as is.
    assert read_json(tmp_path / ".copilot" / "providers.json") == {
        "providers": [
            {"name": "jarvislabs", "type": "openai", "wireApi": "completions", "baseUrl": URL, "apiKey": KEY}
        ],
        "models": [
            {
                "id": "deepseek-v4.1-flash-1017",
                "provider": "jarvislabs",
                "name": "DeepSeek V4.1 Flash",
                "maxPromptTokens": 1048576,
                "maxOutputTokens": 65536,
            },
            {
                "id": "glm-5.3",
                "provider": "jarvislabs",
                "name": "GLM-5.3",
                "maxPromptTokens": 230000,
                "maxOutputTokens": 32768,
            },
            {
                "id": "gemma-3-1b-it",
                "provider": "jarvislabs",
                "name": "Gemma 3 1B IT",
                "maxPromptTokens": 32768,
                "maxOutputTokens": 32768,
            },
        ],
    }
    assert read_json(tmp_path / ".copilot" / "settings.json") == {"model": "jarvislabs/deepseek-v4.1-flash-1017"}

    # Junie CLI keeps one model per file, named for the profile, and takes the full endpoint as its base URL.
    junie = tmp_path / ".junie"
    assert sorted(path.name for path in (junie / "models").iterdir()) == [
        "jarvislabs-deepseek-v4.1-flash-1017.json",
        "jarvislabs-gemma-3-1b-it.json",
        "jarvislabs-glm-5.3.json",
    ]
    assert read_json(junie / "models" / "jarvislabs-glm-5.3.json") == {
        "id": "glm-5.3",
        "displayName": "GLM-5.3",
        "baseUrl": URL + "/chat/completions",
        "apiType": "OpenAICompletion",
        "apiKey": KEY,
        "maxContextLength": 230000,
        "extraBody": {"max_tokens": 32768},
    }
    assert read_json(junie / "config.json") == {"model": "custom:jarvislabs-deepseek-v4.1-flash-1017"}


def test_connect_writes_kimi_code_config_in_toml(subscribed, tmp_path, monkeypatch):
    subscribed.routes[("GET", "models")] = CATALOG

    result = runner.invoke(app, ["connect", "kimi", "--yes"])

    assert result.exit_code == 0, result.output
    written = tomllib.loads((tmp_path / ".kimi-code" / "config.toml").read_text())
    assert written == {
        # Kimi Code requires the default to be one of the aliases below.
        "default_model": "jarvislabs/deepseek-v4.1-flash-1017",
        "providers": {"jarvislabs": {"type": "openai", "base_url": URL, "api_key": KEY}},
        "models": {
            "jarvislabs/deepseek-v4.1-flash-1017": {
                "provider": "jarvislabs",
                "model": "deepseek-v4.1-flash-1017",
                "max_context_size": 1048576,
                "display_name": "DeepSeek V4.1 Flash",
                "capabilities": ["tool_use", "image_in", "video_in"],
            },
            "jarvislabs/glm-5.3": {
                "provider": "jarvislabs",
                "model": "glm-5.3",
                "max_context_size": 230000,
                "display_name": "GLM-5.3",
                # The catalog says it reasons and has no toggle, so it always thinks.
                "capabilities": ["tool_use", "always_thinking"],
            },
            "jarvislabs/gemma-3-1b-it": {
                "provider": "jarvislabs",
                "model": "gemma-3-1b-it",
                "max_context_size": 32768,
                "display_name": "Gemma 3 1B IT",
            },
        },
    }
    # jl reads the same file back, so status can name the default.
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    status = runner.invoke(app, ["status"])
    assert status.exit_code == 0, status.output
    assert re.search(r"✓ Kimi Code\s+connected\s+default deepseek-v4\.1-flash-1017", status.output)


def test_kimi_keeps_the_users_config_and_follows_kimi_code_home(subscribed, tmp_path, monkeypatch):
    elsewhere = tmp_path / "kimi-home"
    elsewhere.mkdir()
    original = '# Mine\ndefault_model = "kimi-code/k3"\ntelemetry = false\n'
    (elsewhere / "config.toml").write_text(original)
    monkeypatch.setenv("KIMI_CODE_HOME", str(elsewhere))

    result = runner.invoke(app, ["connect", "kimi", "--yes"])

    assert result.exit_code == 0, result.output
    written = (elsewhere / "config.toml").read_text()
    assert "# Mine" in written
    assert "telemetry = false" in written
    assert 'default_model = "jarvislabs/deepseek-v4.1-flash-1017"' in written
    assert not (tmp_path / ".kimi-code").exists()


def test_pi_thinking_levels_map_to_supported_efforts(subscribed, tmp_path):
    efforts = {"supported": True, "supported_efforts": ["low", "high", "max"]}
    subscribed.routes[("GET", "models")] = {"object": "list", "data": [{**DEEPSEEK, "thinking": efforts}]}

    result = runner.invoke(app, ["connect", "pi", "--yes"])

    assert result.exit_code == 0, result.output
    (model,) = read_json(tmp_path / ".pi" / "agent" / "models.json")["providers"]["jarvislabs"]["models"]
    # null takes a level out of pi's picker, so pi's default, medium, moves up to high instead of
    # sending an effort the model refuses.
    assert model["thinkingLevelMap"] == {
        "minimal": None,
        "low": "low",
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }


def test_cline_preview_warns_when_its_slot_is_taken(subscribed, tmp_path):
    providers = tmp_path / ".cline" / "data" / "settings" / "providers.json"
    providers.parent.mkdir(parents=True)
    slot = {
        "settings": {"provider": "openai-compatible", "baseUrl": "https://openrouter.ai/api/v1", "model": "qwen3"},
        "updatedAt": "2026-10-01T00:00:00.000Z",
    }
    original = json.dumps(
        {"version": 1, "lastUsedProvider": "openai-compatible", "providers": {"openai-compatible": slot}}
    )
    providers.write_text(original)

    result = runner.invoke(app, ["connect", "cline", "--dry-run"])

    assert result.exit_code == 0, result.output
    # lastUsedProvider already names the slot, so it isn't listed as replaced.
    assert (
        "~ Cline ~/.cline/data/settings/providers.json replaces the OpenAI-compatible provider at "
        "https://openrouter.ai/api/v1 (Cline holds only one) default model only"
    ) in " ".join(result.stdout.split())
    assert providers.read_text() == original


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_each_harness_config_path_per_os(subscribed, tmp_path, monkeypatch, platform):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setattr(sys, "platform", platform)
    # shutil.which can't run on a faked Windows: every harness is installed.
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    result = runner.invoke(app, ["connect", *JSON_FAMILY, "--yes"])

    assert result.exit_code == 0, result.output
    # Every one of them builds its path from the home folder on all three OSes, none from APPDATA; Kilo
    # uses ~/.config there too when XDG_CONFIG_HOME isn't set.
    files = [path for path in home.rglob("*") if path.is_file() and path != config.config_path()]
    assert {path.relative_to(home).as_posix() for path in files} == {
        ".config/kilo/kilo.jsonc",
        ".pi/agent/models.json",
        ".pi/agent/settings.json",
        ".cline/data/settings/providers.json",
        ".qwen/settings.json",
        ".copilot/providers.json",
        ".copilot/settings.json",
        ".junie/config.json",
        ".junie/settings.json",
        ".junie/models/jarvislabs-deepseek-v4.1-flash-1017.json",
        ".junie/models/jarvislabs-glm-5.3.json",
        ".junie/models/jarvislabs-gemma-3-1b-it.json",
    }
    assert not (tmp_path / "appdata").exists()


@pytest.mark.parametrize("harness", ["kilo", "pi", "cline", "qwen", "copilot", "junie"])
def test_reconnecting_previews_an_update_that_replaces_nothing(subscribed, harness):
    runner.invoke(app, ["connect", harness, "--yes"])

    result = runner.invoke(app, ["connect", harness, "--dry-run"])

    assert result.exit_code == 0, result.output
    # Everything there is jl's own from the first connect: an update, with no line under its row.
    lines = result.stdout.splitlines()
    legend_index = lines.index("~ update")
    heading = " ".join(line.strip() for line in lines[:legend_index])
    assert heading.startswith("jl will update the Jarvislabs provider ")
    assert lines[legend_index + 2].startswith("~ ")
    assert "replaces" not in " ".join(lines[legend_index + 3 :])


@pytest.fixture()
def shared_lists(tmp_path):
    """Qwen Code and Copilot CLI configs that already hold one OpenAI-compatible provider of the user's."""
    gpt = {"id": "gpt-4o", "envKey": "OPENAI_API_KEY", "baseUrl": "https://api.openai.com/v1"}
    (tmp_path / ".qwen").mkdir()
    (tmp_path / ".qwen" / "settings.json").write_text(json.dumps({"modelProviders": {"openai": [gpt]}}))
    ollama = {"name": "ollama", "type": "openai", "baseUrl": "http://localhost:11434/v1"}
    (tmp_path / ".copilot").mkdir()
    (tmp_path / ".copilot" / "providers.json").write_text(
        json.dumps({"providers": [ollama], "models": [{"id": "llama3", "provider": "ollama"}]})
    )
    return gpt, ollama


def test_connect_keeps_the_users_own_entries_in_shared_provider_lists(subscribed, tmp_path, shared_lists):
    gpt, ollama = shared_lists

    result = runner.invoke(app, ["connect", "qwen", "copilot", "--yes"])

    assert result.exit_code == 0, result.output
    qwen = read_json(tmp_path / ".qwen" / "settings.json")["modelProviders"]["openai"]
    assert [entry["id"] for entry in qwen] == ["gpt-4o", "deepseek-v4.1-flash-1017", "glm-5.3", "gemma-3-1b-it"]
    assert qwen[0] == gpt
    copilot = read_json(tmp_path / ".copilot" / "providers.json")
    assert [provider["name"] for provider in copilot["providers"]] == ["ollama", "jarvislabs"]
    assert copilot["providers"][0] == ollama
    assert [(m["provider"], m["id"]) for m in copilot["models"]][:2] == [
        ("ollama", "llama3"),
        ("jarvislabs", DEEPSEEK["id"]),
    ]


def test_the_preview_doesnt_list_shared_provider_lists_as_replaced(subscribed, shared_lists):
    result = runner.invoke(app, ["connect", "qwen", "copilot", "--dry-run"])

    assert result.exit_code == 0, result.output
    # Both add to a list of the user's: added, not changed, and nothing replaced.
    rows = result.stdout.splitlines()[3:]
    assert [row.split("  ")[0] for row in rows] == ["+ Qwen Code", "+ Copilot CLI"]


# What Junie wrote on its first launch: its settings file keeps every value as a string.
JUNIE_SETTINGS = '{\n    "sessionCount": "1",\n    "effortPerModel": "{}"\n}\n'


def test_junie_gets_the_default_model(subscribed, tmp_path):
    settings = tmp_path / ".junie" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(JUNIE_SETTINGS)

    result = runner.invoke(app, ["connect", "junie", "--yes"])

    assert result.exit_code == 0, result.output
    # The model Junie starts with, which also skips its "Select model" screen.
    assert read_json(settings) == {
        "sessionCount": "1",
        "effortPerModel": "{}",
        "modelForLaunch": "custom:jarvislabs-deepseek-v4.1-flash-1017",
    }


def test_disconnecting_junie_leaves_only_its_own_settings(subscribed, tmp_path):
    settings = tmp_path / ".junie" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(JUNIE_SETTINGS)
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}
    runner.invoke(app, ["connect", "junie", "--yes"])

    result = runner.invoke(app, ["disconnect", "junie", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(settings) == {"sessionCount": "1", "effortPerModel": "{}"}


def test_connecting_cline_tells_existing_extension_users_what_to_change(subscribed):
    result = runner.invoke(app, ["connect", "cline", "--yes", "--model", "glm-5.3"])

    assert result.exit_code == 0, result.output
    # The extension reads providers.json only while no provider is picked in it.
    assert (
        "Already use Cline in VS Code? In Cline's settings, set API Provider to OpenAI Compatible "
        "and Model ID to glm-5.3."
    ) in " ".join(result.output.split())


def test_status_reads_back_the_default_of_each_json_family_harness(subscribed, monkeypatch):
    runner.invoke(app, ["connect", *JSON_FAMILY, "--yes", "--model", "glm-5.3"])
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    rows = {row[0]: row[1:] for row in (re.split(r"\s{2,}", line.strip("│ ")) for line in result.output.splitlines())}
    assert {name: rows[f"✓ {name}"] for name in NAMES} == {name: ["connected", "default glm-5.3"] for name in NAMES}


def test_rotating_the_key_reaches_every_json_family_harness(subscribed, tmp_path):
    # Connected with the key that leaked, so the files hold the values jl wrote.
    config.save_connection(config.Connection("key_0", "jl-devbox-f7a7", "subscription", "jlsk-leaked"))
    runner.invoke(app, ["connect", *JSON_FAMILY, "--yes"])
    subscribed.routes[("DELETE", "users/api_keys/key_0")] = {"success": True}

    result = runner.invoke(app, ["connect", "--rotate-key"])

    assert result.exit_code == 0, result.output
    # Cline holds only the default, which a rotation leaves as it is; its key changes too.
    still_leaked = sorted(
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
        if path.is_file() and "jlsk-leaked" in path.read_text()
    )
    assert still_leaked == []


@pytest.mark.parametrize(
    ("harness", "name", "file", "ours", "theirs"),
    [
        ("pi", "pi", ".pi/agent/settings.json", '"defaultProvider": "jarvislabs"', '"defaultProvider": "anthropic"'),
        ("qwen", "Qwen Code", ".qwen/settings.json", '"selectedType": "openai"', '"selectedType": "qwen-oauth"'),
    ],
)
def test_status_shows_no_default_when_another_provider_is_picked(
    subscribed, tmp_path, monkeypatch, harness, name, file, ours, theirs
):
    runner.invoke(app, ["connect", harness, "--yes", "--model", "glm-5.3"])
    settings = tmp_path / file
    # The user picked another provider; the model name jl wrote beside it is still there.
    settings.write_text(settings.read_text().replace(ours, theirs))
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    rows = {row[0]: row[1:] for row in (re.split(r"\s{2,}", line.strip("│ ")) for line in result.output.splitlines())}
    assert rows[f"✓ {name}"] == ["connected"]


def test_update_leaves_clines_slot_alone_when_it_holds_another_provider(subscribed, tmp_path, up_to_date):
    runner.invoke(app, ["connect", "cline", "--yes"])
    providers = tmp_path / ".cline" / "data" / "settings" / "providers.json"
    theirs = {"provider": "openai-compatible", "baseUrl": "https://openrouter.ai/api/v1", "apiKey": "sk-or"}
    slot = {"settings": theirs, "updatedAt": "2026-11-05T00:00:00.000Z"}
    providers.write_text(json.dumps({"version": 1, "providers": {"openai-compatible": slot}}))

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0, result.output
    assert read_json(providers)["providers"]["openai-compatible"]["settings"] == theirs


def install_cline_extension(tmp_path):
    """Put Cline's extension in VS Code's extensions folder, where jl finds it."""
    (tmp_path / ".vscode" / "extensions" / "saoudrizwan.claude-dev-1.0.0").mkdir(parents=True)


def cline_data(tmp_path):
    return tmp_path / ".cline" / "data"


def test_an_extension_vscode_marked_obsolete_is_not_installed(subscribed, tmp_path):
    """VS Code keeps an uninstalled extension's folder until it restarts, and lists it in .obsolete."""
    install_cline_extension(tmp_path)
    extensions = tmp_path / ".vscode" / "extensions"
    cline_vscode = next(h for h in HARNESSES if h.id == "cline-vscode")
    assert installed(cline_vscode)

    (extensions / ".obsolete").write_text(json.dumps({"saoudrizwan.claude-dev-1.0.0": True}))

    assert not installed(cline_vscode)


def test_cline_extension_gets_its_own_provider_state(subscribed, tmp_path):
    install_cline_extension(tmp_path)

    result = runner.invoke(app, ["connect", "cline-vscode", "--yes", "--model", "glm-5.3"])

    assert result.exit_code == 0, result.output
    # providers.json alone is ignored by the extension once its own settings name a provider.
    slot = read_json(cline_data(tmp_path) / "settings" / "providers.json")["providers"]["openai-compatible"]
    assert slot["settings"]["model"] == "glm-5.3"
    assert slot["settings"]["apiKey"] == KEY
    assert read_json(cline_data(tmp_path) / "globalState.json") == {
        "welcomeViewCompleted": True,
        "planModeApiProvider": "openai",
        "actModeApiProvider": "openai",
        "openAiBaseUrl": URL,
        "planModeOpenAiModelId": "glm-5.3",
        "actModeOpenAiModelId": "glm-5.3",
    }
    assert read_json(cline_data(tmp_path) / "secrets.json") == {"openAiApiKey": KEY}


@pytest.mark.skipif(os.name == "nt", reason="Windows has no owner-only file mode")
def test_cline_extension_secret_is_owner_only(subscribed, tmp_path):
    install_cline_extension(tmp_path)

    result = runner.invoke(app, ["connect", "cline-vscode", "--yes"])

    assert result.exit_code == 0, result.output
    assert (cline_data(tmp_path) / "secrets.json").stat().st_mode & 0o777 == 0o600


def test_disconnect_restores_cline_extensions_previous_provider(subscribed, tmp_path):
    install_cline_extension(tmp_path)
    data = cline_data(tmp_path)
    data.mkdir(parents=True)
    previous = {
        "welcomeViewCompleted": True,
        "planModeApiProvider": "cline",
        "actModeApiProvider": "cline",
        "openAiBaseUrl": "https://api.example.com/v1",
        "planModeOpenAiModelId": "gpt-4o",
        "actModeOpenAiModelId": "gpt-4o",
    }
    (data / "globalState.json").write_text(json.dumps(previous))
    (data / "secrets.json").write_text(json.dumps({"openAiApiKey": "sk-old"}))
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}

    result = runner.invoke(app, ["connect", "cline-vscode", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(data / "globalState.json")["actModeApiProvider"] == "openai"

    result = runner.invoke(app, ["disconnect", "cline-vscode", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(data / "globalState.json") == previous
    assert read_json(data / "secrets.json") == {"openAiApiKey": "sk-old"}


def test_cline_extension_is_skipped_while_vscode_runs(subscribed, tmp_path, monkeypatch):
    install_cline_extension(tmp_path)
    monkeypatch.setattr(connect, "vscode_running", lambda: True)

    result = runner.invoke(app, ["connect", "cline-vscode", "--yes"])

    assert result.exit_code == 1, result.output
    output = " ".join(result.output.split())
    assert "! Cline (VS Code) skipped: VS Code is running" in output
    assert "Quit VS Code first: Cline rewrites its settings while it runs. Then run jl connect cline-vscode." in output
    assert not cline_data(tmp_path).exists()

    monkeypatch.setattr(connect, "vscode_running", lambda: False)
    result = runner.invoke(app, ["connect", "cline-vscode", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(cline_data(tmp_path) / "globalState.json")["actModeApiProvider"] == "openai"


def test_cline_extension_keeps_its_other_state_keys(subscribed, tmp_path):
    install_cline_extension(tmp_path)
    data = cline_data(tmp_path)
    data.mkdir(parents=True)
    (data / "globalState.json").write_text(
        json.dumps({"welcomeViewCompleted": True, "mode": "plan", "autoApprovalEnabled": True})
    )
    (data / "secrets.json").write_text(json.dumps({"clineApiKey": "cline-token"}))

    result = runner.invoke(app, ["connect", "cline-vscode", "--yes"])

    assert result.exit_code == 0, result.output
    state = read_json(data / "globalState.json")
    assert {key: state[key] for key in ("mode", "autoApprovalEnabled")} == {"mode": "plan", "autoApprovalEnabled": True}
    assert state["planModeApiProvider"] == "openai"
    assert read_json(data / "secrets.json") == {"clineApiKey": "cline-token", "openAiApiKey": KEY}


def test_a_rotation_updates_the_cline_extensions_secret(subscribed, tmp_path):
    install_cline_extension(tmp_path)
    runner.invoke(app, ["connect", "cline-vscode", "--yes"])
    data = cline_data(tmp_path)
    subscribed.routes[("POST", "users/api_keys")] = {**CREATED_KEY, "id": "key_2", "token": "jlsk-rotated"}
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}

    result = runner.invoke(app, ["connect", "--rotate-key"])

    assert result.exit_code == 0, result.output
    assert read_json(data / "secrets.json")["openAiApiKey"] == "jlsk-rotated"
    # The model stays jl's while the extension still points at the Models API.
    assert read_json(data / "globalState.json")["actModeOpenAiModelId"] == DEEPSEEK["id"]


@pytest.mark.parametrize(
    ("variable", "folder"),
    [("CLINE_DATA_DIR", ""), ("CLINE_DIR", "data")],
    ids=["data-dir", "cline-dir"],
)
def test_cline_extension_honours_clines_data_dir(subscribed, tmp_path, monkeypatch, variable, folder):
    install_cline_extension(tmp_path)
    custom = tmp_path / "custom"
    monkeypatch.setenv(variable, str(custom))

    result = runner.invoke(app, ["connect", "cline-vscode", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(custom / folder / "globalState.json")["actModeApiProvider"] == "openai"
    assert not (tmp_path / ".cline").exists()
