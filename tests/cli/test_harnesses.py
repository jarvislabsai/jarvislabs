"""`jl connect` for the JSON-family harnesses: Kilo, pi, Cline, Qwen Code, Copilot CLI and Junie CLI."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys

import json5
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from typer.testing import CliRunner

from jarvislabs import config, harnesses
from jarvislabs.cli import connect
from jarvislabs.cli.app import app
from jarvislabs.harnesses import HARNESSES, installed
from tests.cli import test_connect
from tests.cli.test_connect import CREATED_KEY, DEEPSEEK, GEMMA, GLM, PRICED_CATALOG, WALLET_KEY, read_json

runner = CliRunner()
# A subscriber with no key yet, and the DeepSeek, GLM and Gemma catalog.
subscribed = test_connect.subscribed
opencode_config = test_connect.opencode_config
# jl is already the latest version, for jl update.
up_to_date = test_connect.up_to_date
# An account with no subscription, whose connect takes the wallet.
unsubscribed = test_connect.unsubscribed

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
        "glm-5.3": {"context": 230000, "output": 23000},
        "gemma-3-1b-it": {"context": 32768, "output": 3276},
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
                "maxTokens": 23000,
            },
            {
                "id": "gemma-3-1b-it",
                "name": "Gemma 3 1B IT",
                "reasoning": False,
                "input": ["text"],
                "contextWindow": 32768,
                "maxTokens": 3276,
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
    # providerProtocol tells Qwen Code that jl's custom provider speaks OpenAI.
    assert qwen["providerProtocol"] == {"jarvislabs": "openai"}
    assert qwen["modelProviders"] == {
        "jarvislabs": [
            {
                "id": "deepseek-v4.1-flash-1017",
                "name": "DeepSeek V4.1 Flash",
                "envKey": "JARVISLABS_API_KEY",
                "baseUrl": URL,
                "generationConfig": {
                    "contextWindowSize": 1048576,
                    "samplingParams": {"max_tokens": 65536},
                    "modalities": {"image": True, "video": True},
                },
            },
            {
                "id": "glm-5.3",
                "name": "GLM-5.3",
                "envKey": "JARVISLABS_API_KEY",
                "baseUrl": URL,
                "generationConfig": {
                    "contextWindowSize": 230000,
                    "samplingParams": {"max_tokens": 23000},
                    "modalities": {"image": False, "video": False},
                },
            },
            {
                "id": "gemma-3-1b-it",
                "name": "Gemma 3 1B IT",
                "envKey": "JARVISLABS_API_KEY",
                "baseUrl": URL,
                "generationConfig": {
                    "contextWindowSize": 32768,
                    "samplingParams": {"max_tokens": 3276},
                    "modalities": {"image": False, "video": False},
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
                "capabilities": {"supports": {"vision": True}},
            },
            {
                "id": "glm-5.3",
                "provider": "jarvislabs",
                "name": "GLM-5.3",
                "maxPromptTokens": 230000,
                "maxOutputTokens": 23000,
            },
            {
                "id": "gemma-3-1b-it",
                "provider": "jarvislabs",
                "name": "Gemma 3 1B IT",
                "maxPromptTokens": 32768,
                "maxOutputTokens": 3276,
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
        "extraBody": {"max_tokens": 23000},
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
                "max_output_size": 65536,
                "display_name": "DeepSeek V4.1 Flash",
                "capabilities": ["tool_use", "image_in", "video_in"],
            },
            "jarvislabs/glm-5.3": {
                "provider": "jarvislabs",
                "model": "glm-5.3",
                "max_context_size": 230000,
                "max_output_size": 23000,
                "display_name": "GLM-5.3",
                # The catalog says it reasons and has no toggle, so it always thinks.
                "capabilities": ["tool_use", "always_thinking"],
            },
            "jarvislabs/gemma-3-1b-it": {
                "provider": "jarvislabs",
                "model": "gemma-3-1b-it",
                "max_context_size": 32768,
                "max_output_size": 3276,
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


def test_connect_writes_claude_codes_env_block(subscribed, tmp_path, monkeypatch):
    monkeypatch.setenv("JL_PREVIEW_HARNESSES", "1")

    result = runner.invoke(app, ["connect", "claude-code", "--yes"])

    assert result.exit_code == 0, result.output
    settings = read_json(tmp_path / ".claude" / "settings.json")
    assert settings == {
        "env": {
            # Claude Code's SDK appends /v1/messages to this host.
            "ANTHROPIC_BASE_URL": "https://models.jarvislabs.net",
            "ANTHROPIC_AUTH_TOKEN": KEY,
            # Every Claude tier points at the one model jl wrote, so /model can't pick a model we lack.
            "ANTHROPIC_MODEL": "deepseek-v4.1-flash-1017",
            "ANTHROPIC_SMALL_FAST_MODEL": "deepseek-v4.1-flash-1017",
            "CLAUDE_CODE_SUBAGENT_MODEL": "deepseek-v4.1-flash-1017",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4.1-flash-1017",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4.1-flash-1017",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4.1-flash-1017",
            "ANTHROPIC_DEFAULT_FABLE_MODEL": "deepseek-v4.1-flash-1017",
            # One extra /model row with our name, not only Claude's aliases.
            "ANTHROPIC_CUSTOM_MODEL_OPTION": "deepseek-v4.1-flash-1017",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "DeepSeek V4.1 Flash",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "Served by Jarvislabs",
        }
    }


def test_connect_writes_codexs_responses_provider(subscribed, tmp_path, monkeypatch):
    monkeypatch.setenv("JL_PREVIEW_HARNESSES", "1")

    result = runner.invoke(app, ["connect", "codex", "--yes"])

    assert result.exit_code == 0, result.output
    written = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text())
    assert written == {
        "model": "deepseek-v4.1-flash-1017",
        "model_provider": "jarvislabs",
        "model_context_window": 1048576,
        "model_providers": {
            "jarvislabs": {
                "name": "Jarvislabs",
                # Codex appends /responses to this URL; wire_api only takes "responses".
                "base_url": "https://models.jarvislabs.net/v1",
                "wire_api": "responses",
                # Inline, so the user doesn't have to export a variable.
                "experimental_bearer_token": KEY,
            }
        },
    }


def test_status_names_claude_codes_default_when_preview_is_on(subscribed, monkeypatch):
    monkeypatch.setenv("JL_PREVIEW_HARNESSES", "1")
    runner.invoke(app, ["connect", "claude-code", "--yes"])
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert re.search(r"✓ Claude Code\s+connected\s+default deepseek-v4\.1-flash-1017", result.output)


def test_a_preview_harness_needs_the_flag(subscribed):
    result = runner.invoke(app, ["connect", "claude-code"])

    assert result.exit_code == 1, result.output
    said = " ".join(result.output.split())
    # Coming soon to a user; the testing flag isn't theirs to know about.
    assert "Claude Code is coming soon" in said
    assert "JL_PREVIEW_HARNESSES" not in said


def test_the_flag_marks_preview_harnesses_connectable_in_the_list(subscribed, monkeypatch):
    off = runner.invoke(app, ["connect", "--list"])
    row = next(line for line in off.stdout.splitlines() if line.strip().startswith("claude-code"))
    # Its config would point at an endpoint that isn't public yet, so it isn't printable either.
    assert row.endswith("coming soon")
    printed = runner.invoke(app, ["connect", "--print", "claude-code"])
    assert printed.exit_code == 1
    assert "Claude Code is coming soon: see https://docs.jarvislabs.ai/models-api/harnesses/claude-code" in " ".join(
        printed.output.split()
    )

    monkeypatch.setenv("JL_PREVIEW_HARNESSES", "1")
    on = runner.invoke(app, ["connect", "--list"])
    row = next(line for line in on.stdout.splitlines() if line.strip().startswith("claude-code"))
    assert row.endswith("connect")


def test_codex_catalog_carries_each_models_own_efforts(subscribed, tmp_path, monkeypatch):
    monkeypatch.setenv("JL_PREVIEW_HARNESSES", "1")
    monkeypatch.setattr(harnesses, "_codex_instructions", lambda: "You are Codex.")
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG

    result = runner.invoke(app, ["connect", "codex", "--yes"])

    assert result.exit_code == 0, result.output
    written = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text())
    assert written["model_catalog_json"] == str(tmp_path / ".codex" / "models.json")
    models = {m["slug"]: m for m in read_json(tmp_path / ".codex" / "models.json")["models"]}
    # The catalog's entries replace the prompt Codex assembles itself, so they carry its own.
    assert models["deepseek-v4.1-flash-1017"]["base_instructions"] == "You are Codex."
    assert models["deepseek-v4.1-flash-1017"]["supported_reasoning_levels"] == [
        {"effort": "low", "description": "Lighter reasoning"},
        {"effort": "high", "description": "Deep reasoning"},
        {"effort": "max", "description": "Maximum reasoning"},
    ]
    assert models["deepseek-v4.1-flash-1017"]["default_reasoning_level"] == "high"
    assert models["glm-5.3"]["default_reasoning_level"] == "medium"
    # A model that offers no efforts lists none, and defaults to none.
    assert models["gemma-3-1b-it"]["supported_reasoning_levels"] == []
    assert "default_reasoning_level" not in models["gemma-3-1b-it"]


def test_codex_catalog_merges_into_the_users_own(subscribed, tmp_path, monkeypatch):
    monkeypatch.setenv("JL_PREVIEW_HARNESSES", "1")
    monkeypatch.setattr(harnesses, "_codex_instructions", lambda: "You are Codex.")
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG
    catalog = tmp_path / ".codex" / "models.json"
    catalog.parent.mkdir()
    catalog.write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "deepseek-chat", "display_name": "DeepSeek Chat", "description": "The user's own"},
                    {"slug": "a-model-we-dropped", "display_name": "Old", "description": "Served by Jarvislabs"},
                ]
            }
        )
    )

    result = runner.invoke(app, ["connect", "codex", "--yes"])

    assert result.exit_code == 0, result.output
    models = read_json(catalog)["models"]
    slugs = [model["slug"] for model in models]
    # The user's own entry stays, jl's stale one goes, and ours are added.
    assert "deepseek-chat" in slugs
    assert "a-model-we-dropped" not in slugs
    assert "deepseek-v4.1-flash-1017" in slugs


def test_codex_writes_no_catalog_when_it_cant_read_the_stock_prompt(subscribed, tmp_path, monkeypatch):
    monkeypatch.setenv("JL_PREVIEW_HARNESSES", "1")
    monkeypatch.setattr(harnesses, "_codex_instructions", lambda: None)
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG

    result = runner.invoke(app, ["connect", "codex", "--yes"])

    assert result.exit_code == 0, result.output
    # No catalog leaves Codex's own default prompt and no efforts, exactly as before.
    written = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text())
    assert "model_catalog_json" not in written
    assert not (tmp_path / ".codex" / "models.json").exists()


def test_codex_follows_codex_home_and_keeps_the_users_config(subscribed, tmp_path, monkeypatch):
    monkeypatch.setenv("JL_PREVIEW_HARNESSES", "1")
    elsewhere = tmp_path / "codex-home"
    elsewhere.mkdir()
    original = '# Mine\napproval_policy = "never"\n'
    (elsewhere / "config.toml").write_text(original)
    monkeypatch.setenv("CODEX_HOME", str(elsewhere))

    result = runner.invoke(app, ["connect", "codex", "--yes"])

    assert result.exit_code == 0, result.output
    written = (elsewhere / "config.toml").read_text()
    assert "# Mine" in written
    assert 'approval_policy = "never"' in written
    assert not (tmp_path / ".codex").exists()


def test_a_symlinked_config_is_written_through(subscribed, tmp_path):
    # A dotfiles user's link stays the live config; replacing it with a regular file would
    # silently stop their dotfiles from managing the file.
    real = tmp_path / "dotfiles" / "opencode.json"
    real.parent.mkdir()
    real.write_text('{\n  "theme": "dark"\n}\n')
    link = tmp_path / "opencode" / "opencode.json"
    link.parent.mkdir()
    link.symlink_to(real)

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert link.is_symlink()
    assert "jarvislabs" in real.read_text()


def test_kimi_refuses_a_config_with_dotted_keys(subscribed, tmp_path):
    path = tmp_path / ".kimi-code" / "config.toml"
    path.parent.mkdir(parents=True)
    original = 'providers.jarvislabs.type = "openai"\n'
    path.write_text(original)

    result = runner.invoke(app, ["connect", "kimi", "--yes"])

    assert result.exit_code == 1, result.output
    assert "dotted keys" in " ".join(result.output.split())
    assert path.read_text() == original


def test_a_model_the_catalog_dropped_leaves_kimi_config(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = CATALOG
    runner.invoke(app, ["connect", "kimi", "--yes"])
    subscribed.routes[("GET", "models")] = {"object": "list", "data": [DEEPSEEK]}

    result = runner.invoke(app, ["connect", "kimi", "--yes"])

    assert result.exit_code == 0, result.output
    models = tomllib.loads((tmp_path / ".kimi-code" / "config.toml").read_text())["models"]
    # jl's aliases only; a model the catalog no longer serves is gone rather than left to fail later.
    assert list(models) == ["jarvislabs/deepseek-v4.1-flash-1017"]


def test_a_config_jl_cannot_read_is_reported_not_a_traceback(subscribed, opencode_config):
    # A string where jl needs an object: a real file, just not one jl can write into.
    opencode_config.write_text('{\n  "provider": "openai"\n}\n')

    result = runner.invoke(app, ["connect", "opencode", "--yes"])

    assert result.exit_code == 1, result.output
    said = " ".join(result.output.split())
    assert "Can't read" in said
    assert "different shape" in said
    assert "then run jl connect again" in said
    # Nothing was written over it.
    assert opencode_config.read_text() == '{\n  "provider": "openai"\n}\n'


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


def test_pi_starts_each_model_at_its_default_effort(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG

    result = runner.invoke(app, ["connect", "pi", "--yes"])

    assert result.exit_code == 0, result.output
    settings = read_json(tmp_path / ".pi" / "agent" / "settings.json")
    # pi starts every model at its global medium level; modelThinkingLevels names the level whose
    # thinkingLevelMap value is the model's own default effort, and keeps the default keys.
    assert settings == {
        "defaultProvider": "jarvislabs",
        "defaultModel": "deepseek-v4.1-flash-1017",
        "modelThinkingLevels": {
            "jarvislabs/deepseek-v4.1-flash-1017": "high",
            "jarvislabs/glm-5.3": "medium",
        },
    }


def test_opencode_offers_the_models_real_reasoning_efforts(subscribed, tmp_path):
    # OpenCode offers low, medium and high when a model carries no efforts, and the Models API
    # refuses an effort the model doesn't list; the catalog's efforts replace that ladder.
    subscribed.routes[("GET", "models")] = {
        "object": "list",
        "data": [
            {**DEEPSEEK, "thinking": {"supported": True, "supported_efforts": ["low", "high", "max"]}},
            {**GLM, "thinking": {"supported": True}},
        ],
    }

    result = runner.invoke(app, ["connect", "opencode", "kilo", "--yes"])

    assert result.exit_code == 0, result.output
    models = read_json(tmp_path / "opencode" / "opencode.json")["providers"]["jarvislabs"]["models"]
    assert {model_id: m["capabilities"]["reasoning"] for model_id, m in models.items()} == {
        "deepseek-v4.1-flash-1017": {"supported": True, "effort": ["low", "high", "max"]},
        # A model that reasons with a toggle only offers no efforts, not the default ladder.
        "glm-5.3": {"supported": True},
    }
    # Kilo reads only the V1 entry, which cannot express efforts; its own writer pins each
    # model's fixed effort there, OpenCode's copy stays bare.
    kilo = read_json(tmp_path / "kilo" / "kilo.jsonc")["provider"]["jarvislabs"]["models"]
    assert "capabilities" not in kilo["deepseek-v4.1-flash-1017"]
    opencode_v1 = read_json(tmp_path / "opencode" / "opencode.json")["provider"]["jarvislabs"]["models"]
    assert "options" not in opencode_v1["deepseek-v4.1-flash-1017"]


# The catalog for the effort writers: a model with the catalog's default effort, one that offers a
# single effort (so the default falls to it), and one that reasons with a toggle only.
EFFORTS_CATALOG = {
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

# A model that can turn thinking off, and one whose toggle has no effort ladder to send with it.
TOGGLE_CATALOG = {
    "object": "list",
    "data": [
        {
            **DEEPSEEK,
            "thinking": {
                "supported": True,
                "supports_toggle": True,
                "supported_efforts": ["low", "medium", "high"],
                "default_effort": "medium",
            },
        },
        {**GLM, "thinking": {"supported": True, "supports_toggle": True}},
    ],
}


def test_qwen_declares_each_models_own_efforts(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG

    result = runner.invoke(app, ["connect", "qwen", "--yes"])

    assert result.exit_code == 0, result.output
    entries = read_json(tmp_path / ".qwen" / "settings.json")["modelProviders"]["jarvislabs"]
    # The capabilities name the efforts the /effort picker offers and the one to send without a
    # choice; generationConfig's reasoning is dropped once samplingParams is set.
    assert entries[0]["capabilities"] == {
        "reasoning": {"profile": "openai-effort", "efforts": ["low", "high", "max"], "defaultEffort": "high"}
    }
    # A model with a single effort takes it as its default; the API leaves no other choice.
    assert entries[1]["capabilities"] == {
        "reasoning": {"profile": "openai-effort", "efforts": ["medium"], "defaultEffort": "medium"}
    }
    # A model without efforts gets no capabilities at all: an empty ladder makes Qwen Code
    # discard the declaration.
    assert "capabilities" not in entries[2]


def test_qwen_never_defaults_to_an_effort_the_model_refuses(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = {
        "object": "list",
        "data": [
            {
                **DEEPSEEK,
                "thinking": {"supported": True, "supported_efforts": ["low", "high"], "default_effort": "xhigh"},
            }
        ],
    }

    result = runner.invoke(app, ["connect", "qwen", "--yes"])

    assert result.exit_code == 0, result.output
    (entry,) = read_json(tmp_path / ".qwen" / "settings.json")["modelProviders"]["jarvislabs"]
    assert entry["capabilities"] == {"reasoning": {"profile": "openai-effort", "efforts": ["low", "high"]}}


def test_kimi_offers_each_models_own_efforts(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG

    result = runner.invoke(app, ["connect", "kimi", "--yes"])

    assert result.exit_code == 0, result.output
    models = tomllib.loads((tmp_path / ".kimi-code" / "config.toml").read_text())["models"]
    assert models["jarvislabs/deepseek-v4.1-flash-1017"]["support_efforts"] == ["low", "high", "max"]
    assert models["jarvislabs/deepseek-v4.1-flash-1017"]["default_effort"] == "high"
    assert models["jarvislabs/glm-5.3"]["support_efforts"] == ["medium"]
    assert models["jarvislabs/glm-5.3"]["default_effort"] == "medium"
    # default_effort is ignored without support_efforts beside it, so a toggle-only model
    # gets neither and sends no effort at all.
    assert "support_efforts" not in models["jarvislabs/gemma-3-1b-it"]
    assert "default_effort" not in models["jarvislabs/gemma-3-1b-it"]


def test_kimi_writes_off_effort_only_for_a_model_that_can_toggle_thinking(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = TOGGLE_CATALOG

    result = runner.invoke(app, ["connect", "kimi", "--yes"])

    assert result.exit_code == 0, result.output
    models = tomllib.loads((tmp_path / ".kimi-code" / "config.toml").read_text())["models"]
    # Off in Kimi's picker sends off_effort's value; without it Kimi raises off-needs-offeffort.
    assert models["jarvislabs/deepseek-v4.1-flash-1017"]["off_effort"] == "none"
    # A toggle without an effort ladder has nothing to turn off from, so it gets no off_effort.
    assert "off_effort" not in models["jarvislabs/glm-5.3"]


def test_copilot_offers_each_models_own_efforts(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG

    result = runner.invoke(app, ["connect", "copilot", "--yes"])

    assert result.exit_code == 0, result.output
    models = {m["id"]: m for m in read_json(tmp_path / ".copilot" / "providers.json")["models"]}
    # The ladder the picker's Reasoning column cycles through, and image support beside it: without
    # either, the column is hidden or the CLI refuses an image for a vision model.
    assert models["deepseek-v4.1-flash-1017"]["capabilities"] == {
        "supports": {"reasoning_effort": ["low", "high", "max"], "vision": True}
    }
    assert models["glm-5.3"]["capabilities"] == {"supports": {"reasoning_effort": ["medium"]}}
    assert "capabilities" not in models["gemma-3-1b-it"]


def test_kilo_pins_each_models_default_effort(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG

    result = runner.invoke(app, ["connect", "kilo", "--yes"])

    assert result.exit_code == 0, result.output
    # Kilo's V1 entry cannot express a ladder, so each model's effort is pinned as its fixed
    # default; a toggle-only model gets no options, and sends no effort.
    models = read_json(tmp_path / "kilo" / "kilo.jsonc")["provider"]["jarvislabs"]["models"]
    assert models["deepseek-v4.1-flash-1017"]["options"] == {"reasoningEffort": "high"}
    assert models["glm-5.3"]["options"] == {"reasoningEffort": "medium"}
    assert "options" not in models["gemma-3-1b-it"]


def test_aider_sets_each_models_output_limit_alongside_its_effort(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG

    result = runner.invoke(app, ["connect", "aider", "--yes"])

    assert result.exit_code == 0, result.output
    written = (tmp_path / ".aider.model.settings.yml").read_text()
    # max_tokens is the output cap Aider puts on the wire; the model's effort rides in extra_body
    # in the same entry, and a model without one still gets its limit.
    assert (
        "- name: openai/deepseek-v4.1-flash-1017\n  extra_params:\n    max_tokens: 65536\n"
        "    extra_body:\n      reasoning_effort: high" in written
    )
    assert "- name: openai/gemma-3-1b-it\n  extra_params:\n    max_tokens: 3276" in written


def test_aider_lets_vision_models_take_images(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = CATALOG

    result = runner.invoke(app, ["connect", "aider", "--yes"])

    assert result.exit_code == 0, result.output
    metadata = read_json(tmp_path / ".aider.model.metadata.json")
    # Aider refuses an image for a model whose metadata doesn't say it takes one.
    assert metadata["openai/deepseek-v4.1-flash-1017"]["supports_vision"] is True
    assert metadata["openai/glm-5.3"]["supports_vision"] is False


def test_goose_sets_each_models_output_limit_alongside_its_effort(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG
    goose = tmp_path / "Block" / "goose" / "config" if os.name == "nt" else tmp_path / "goose"

    result = runner.invoke(app, ["connect", "goose", "--yes"])

    assert result.exit_code == 0, result.output
    models = {m["name"]: m for m in read_json(goose / "custom_providers" / "jarvislabs.json")["models"]}
    # Goose sends request_params verbatim, so max_tokens is the output cap on every request and
    # the effort joins it where the model has one.
    assert models["deepseek-v4.1-flash-1017"]["request_params"] == {
        "max_tokens": 65536,
        "reasoning_effort": "high",
    }
    assert models["gemma-3-1b-it"]["request_params"] == {"max_tokens": 3276}


def test_zed_sets_each_models_output_limit(subscribed):
    result = runner.invoke(app, ["connect", "--print", "zed"])

    assert result.exit_code == 0, result.output
    out = result.stdout
    settings = json.loads(out[out.index("{") : out.rindex("}") + 1])
    models = settings["language_models"]["openai_compatible"]["Jarvislabs"]["available_models"]
    # max_tokens is the context window; without max_output_tokens Zed sends no output ceiling
    # and the Models API cuts replies at 1024 tokens.
    assert [model["max_output_tokens"] for model in models] == [65536, 23000, 3276]


def vscode_models_file(tmp_path):
    """Put VS Code's code command on PATH, and return where Copilot reads Custom Endpoint models."""
    for name in ("code", "code.cmd"):
        (tmp_path / "installed-harnesses" / name).write_text("")
        (tmp_path / "installed-harnesses" / name).chmod(0o755)
    # HOME, APPDATA and XDG_CONFIG_HOME are all tmp_path (the server fixture).
    if sys.platform == "darwin":
        return tmp_path / "Library" / "Application Support" / "Code" / "User" / "chatLanguageModels.json"
    return tmp_path / "Code" / "User" / "chatLanguageModels.json"


# A group of the user's own, as VS Code writes it once they add Ollama.
OLLAMA_GROUP = {"name": "Ollama", "vendor": "ollama", "url": "http://localhost:11434"}


def test_vscode_copilot_gets_a_custom_endpoint_group_with_the_key_in_its_header(subscribed, tmp_path):
    models_file = vscode_models_file(tmp_path)

    result = runner.invoke(app, ["connect", "vscode-copilot", "--yes"])

    assert result.exit_code == 0, result.output
    (group,) = read_json(models_file)
    assert (group["name"], group["vendor"]) == ("Jarvislabs", "customendpoint")
    # VS Code reads a group's apiKey only from its secret store, so the key rides in the header.
    assert "apiKey" not in group
    # A file that wasn't there is added, not "changed".
    assert "+ VS Code Copilot" in result.output
    deepseek, glm, gemma = group["models"]
    assert deepseek == {
        "id": "deepseek-v4.1-flash-1017",
        "name": "DeepSeek V4.1 Flash",
        "url": "https://models.jarvislabs.net/v1/chat/completions",
        "toolCalling": True,
        "vision": True,
        # 1048576 window - 65536 output = 983040 for the prompt.
        "maxInputTokens": 983040,
        "maxOutputTokens": 65536,
        # Without it the output limit isn't sent and the Models API cuts replies at 1024 tokens.
        "thinking": True,
        "requestHeaders": {"Authorization": "Bearer jlsk-new-secret"},
    }
    # GLM's catalog has no output limit: a tenth of its 230000 window.
    assert (glm["maxInputTokens"], glm["maxOutputTokens"], glm["vision"]) == (207000, 23000, False)
    assert gemma["toolCalling"] is False


def test_vscode_copilot_keeps_the_users_other_model_groups(subscribed, tmp_path):
    models_file = vscode_models_file(tmp_path)
    models_file.parent.mkdir(parents=True)
    models_file.write_text(json.dumps([OLLAMA_GROUP], indent="\t"))

    result = runner.invoke(app, ["connect", "vscode-copilot", "--yes"])

    assert result.exit_code == 0, result.output
    assert [(g["name"], g["vendor"]) for g in read_json(models_file)] == [
        ("Ollama", "ollama"),
        ("Jarvislabs", "customendpoint"),
    ]


def test_vscode_copilot_offers_each_models_own_efforts(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = EFFORTS_CATALOG
    models_file = vscode_models_file(tmp_path)

    result = runner.invoke(app, ["connect", "vscode-copilot", "--yes"])

    assert result.exit_code == 0, result.output
    deepseek, glm, gemma = read_json(models_file)[0]["models"]
    # VS Code sends reasoning_effort only from this list; the Models API refuses any other.
    assert deepseek["supportsReasoningEffort"] == ["low", "high", "max"]
    assert glm["supportsReasoningEffort"] == ["medium"]
    # No efforts: no effort picker, and no reasoning_effort sent.
    assert "supportsReasoningEffort" not in gemma


def test_vscode_copilot_starts_new_agents_window_sessions_on_the_chosen_model(subscribed, tmp_path):
    settings = vscode_models_file(tmp_path).with_name("settings.json")
    settings.parent.mkdir(parents=True)
    settings.write_text('{\n  // mine\n  "editor.fontSize": 14\n}\n')

    result = runner.invoke(app, ["connect", "vscode-copilot", "--yes", "--model", "glm-5.3"])

    assert result.exit_code == 0, result.output
    text = settings.read_text()
    # The Agents window names a Custom Endpoint model <vendor>/<group>/<id>.
    assert json5.loads(text) == {"editor.fontSize": 14, "chat.defaultModel": "customendpoint/Jarvislabs/glm-5.3"}
    assert "// mine" in text


def test_status_reads_vscode_copilots_default_model_back(subscribed, tmp_path):
    vscode_models_file(tmp_path)
    runner.invoke(app, ["connect", "vscode-copilot", "--yes", "--model", "glm-5.3"])

    assert harnesses.default_model(next(h for h in HARNESSES if h.id == "vscode-copilot")) == "glm-5.3"


def test_disconnecting_vscode_copilot_leaves_the_users_other_groups(subscribed, tmp_path):
    models_file = vscode_models_file(tmp_path)
    models_file.parent.mkdir(parents=True)
    models_file.write_text(json.dumps([OLLAMA_GROUP], indent="\t"))
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}
    runner.invoke(app, ["connect", "vscode-copilot", "--yes"])

    result = runner.invoke(app, ["disconnect", "vscode-copilot", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(models_file) == [OLLAMA_GROUP]


def test_disconnecting_vscode_copilot_leaves_an_empty_list_in_a_file_it_found(subscribed, tmp_path):
    models_file = vscode_models_file(tmp_path)
    models_file.parent.mkdir(parents=True)
    # VS Code creates the file as an empty list the first time its models editor opens.
    models_file.write_text("[]")
    subscribed.routes[("DELETE", "users/api_keys/key_1")] = {"success": True}
    runner.invoke(app, ["connect", "vscode-copilot", "--yes"])

    result = runner.invoke(app, ["disconnect", "vscode-copilot", "--yes"])

    assert result.exit_code == 0, result.output
    assert read_json(models_file) == []


def test_opencode_and_kilo_take_video_when_the_catalog_lists_it(subscribed, tmp_path):
    subscribed.routes[("GET", "models")] = CATALOG

    result = runner.invoke(app, ["connect", "opencode", "kilo", "--yes"])

    assert result.exit_code == 0, result.output
    opencode = read_json(tmp_path / "opencode" / "opencode.json")
    v2 = opencode["providers"]["jarvislabs"]["models"]["deepseek-v4.1-flash-1017"]
    assert v2["capabilities"]["input"] == ["text", "image", "video"]
    # The V1 entry, which Kilo reads too, names video as well; Kilo's own copy is the same entry.
    v1 = opencode["provider"]["jarvislabs"]["models"]["deepseek-v4.1-flash-1017"]
    assert v1["modalities"]["input"] == ["text", "image", "video"]
    kilo = read_json(tmp_path / "kilo" / "kilo.jsonc")
    assert kilo["provider"]["jarvislabs"]["models"]["deepseek-v4.1-flash-1017"]["modalities"]["input"] == [
        "text",
        "image",
        "video",
    ]


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
    # Qwen Code: the user's own entries stay in the openai list, jl's go under its own provider.
    qwen = read_json(tmp_path / ".qwen" / "settings.json")["modelProviders"]
    assert qwen["openai"] == [gpt]
    assert [entry["id"] for entry in qwen["jarvislabs"]] == [
        "deepseek-v4.1-flash-1017",
        "glm-5.3",
        "gemma-3-1b-it",
    ]
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


# An entry an earlier jl wrote into Qwen Code's shared modelProviders.openai list, before jl had
# its own provider there.
OLD_JL_GLM = {
    "id": "glm-5.3",
    "name": "GLM-5.3",
    "envKey": "JARVISLABS_API_KEY",
    "baseUrl": URL,
    "generationConfig": {"contextWindowSize": 230000, "samplingParams": {"max_tokens": 32768}},
}


def test_connect_moves_entries_an_earlier_jl_left_in_the_openai_list(subscribed, tmp_path):
    settings = tmp_path / ".qwen" / "settings.json"
    settings.parent.mkdir()
    gpt = {"id": "gpt-4o", "envKey": "OPENAI_API_KEY", "baseUrl": "https://api.openai.com/v1"}
    settings.write_text(json.dumps({"modelProviders": {"openai": [gpt, OLD_JL_GLM]}}))

    result = runner.invoke(app, ["connect", "qwen", "--yes"])

    assert result.exit_code == 0, result.output
    qwen = read_json(settings)
    # The user's own entry stays in the openai list; jl's move under its own provider, which
    # providerProtocol maps to the OpenAI protocol.
    assert qwen["modelProviders"]["openai"] == [gpt]
    assert qwen["providerProtocol"] == {"jarvislabs": "openai"}
    assert [entry["id"] for entry in qwen["modelProviders"]["jarvislabs"]] == [
        "deepseek-v4.1-flash-1017",
        "glm-5.3",
        "gemma-3-1b-it",
    ]


def test_connect_removes_the_openai_list_when_it_held_only_jls_entries(subscribed, tmp_path):
    settings = tmp_path / ".qwen" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"modelProviders": {"openai": [OLD_JL_GLM]}}))

    result = runner.invoke(app, ["connect", "qwen", "--yes"])

    assert result.exit_code == 0, result.output
    qwen = read_json(settings)
    # Nothing of the user's is left in the openai list, so the key goes rather than an empty array.
    assert "openai" not in qwen["modelProviders"]
    assert [entry["id"] for entry in qwen["modelProviders"]["jarvislabs"]] == [
        "deepseek-v4.1-flash-1017",
        "glm-5.3",
        "gemma-3-1b-it",
    ]


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
    info = {
        "contextWindow": slot["settings"]["contextWindow"],
        "maxTokens": slot["settings"]["maxTokens"],
        "supportsImages": "vision" in slot["settings"]["capabilities"],
        "supportsPromptCache": False,
    }
    assert read_json(cline_data(tmp_path) / "globalState.json") == {
        "welcomeViewCompleted": True,
        "planModeApiProvider": "openai",
        "actModeApiProvider": "openai",
        "openAiBaseUrl": URL,
        "planModeOpenAiModelId": "glm-5.3",
        "actModeOpenAiModelId": "glm-5.3",
        # The model's own window, the same limits as providers.json; without it Cline assumes 128k.
        "planModeOpenAiModelInfo": info,
        "actModeOpenAiModelInfo": info,
    }
    assert info["contextWindow"] == 230000
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


def connect_on_wallet(server, *tools):
    """Connect tools on a wallet key, with the USD-priced catalog."""
    server.routes[("GET", "models")] = PRICED_CATALOG
    server.routes[("POST", "users/api_keys")] = WALLET_KEY
    return runner.invoke(app, ["connect", *tools, "--wallet", "--yes"])


def test_a_wallet_connect_writes_kilos_usd_prices(unsubscribed, tmp_path):
    result = connect_on_wallet(unsubscribed, "kilo")

    assert result.exit_code == 0, result.output
    models = read_json(tmp_path / "kilo" / "kilo.jsonc")["provider"]["jarvislabs"]["models"]
    assert {model_id: model.get("cost") for model_id, model in models.items()} == {
        "deepseek-v4.1-flash-1017": {"input": 0.14, "output": 0.28, "cache_read": 0.014},
        # No cached price, so no cache key.
        "glm-5.3": {"input": 0.55, "output": 2.2},
        "gemma-3-1b-it": None,
    }


def test_a_wallet_connect_writes_pis_usd_prices(unsubscribed, tmp_path):
    result = connect_on_wallet(unsubscribed, "pi")

    assert result.exit_code == 0, result.output
    pi = read_json(tmp_path / ".pi" / "agent" / "models.json")["providers"]["jarvislabs"]["models"]
    models = {model["id"]: model for model in pi}
    # pi needs all four rates, so the cache-write price the catalog lacks is written as zero.
    assert models["deepseek-v4.1-flash-1017"]["cost"] == {
        "input": 0.14,
        "output": 0.28,
        "cacheRead": 0.014,
        "cacheWrite": 0,
    }
    # A model without a cached price gets no cost at all: pi refuses a cost missing a rate, and a
    # zero would report cached reads as free.
    assert "cost" not in models["glm-5.3"]


def test_a_wallet_connect_writes_gooses_per_token_prices(unsubscribed, tmp_path):
    result = connect_on_wallet(unsubscribed, "goose")

    assert result.exit_code == 0, result.output
    goose = tmp_path / "Block" / "goose" / "config" if os.name == "nt" else tmp_path / "goose"
    models = {model["name"]: model for model in read_json(goose / "custom_providers" / "jarvislabs.json")["models"]}
    # Goose prices per token, and prints the currency it is given beside the session's cost.
    assert {
        key: models["deepseek-v4.1-flash-1017"][key]
        for key in ("input_token_cost", "output_token_cost", "cache_read_token_cost", "currency")
    } == {
        "input_token_cost": 1.4e-07,
        "output_token_cost": 2.8e-07,
        "cache_read_token_cost": 1.4e-08,
        "currency": "USD",
    }
    assert "cache_read_token_cost" not in models["glm-5.3"]
    assert "input_token_cost" not in models["gemma-3-1b-it"]


def test_a_wallet_connect_writes_aiders_per_token_prices(unsubscribed, tmp_path):
    result = connect_on_wallet(unsubscribed, "aider")

    assert result.exit_code == 0, result.output
    metadata = read_json(tmp_path / ".aider.model.metadata.json")
    # Aider reads litellm's per-token prices for its cost report.
    assert {
        key: metadata["openai/deepseek-v4.1-flash-1017"][key]
        for key in ("input_cost_per_token", "output_cost_per_token", "input_cost_per_token_cache_hit")
    } == {
        "input_cost_per_token": 1.4e-07,
        "output_cost_per_token": 2.8e-07,
        "input_cost_per_token_cache_hit": 1.4e-08,
    }
    assert "input_cost_per_token_cache_hit" not in metadata["openai/glm-5.3"]
    assert "input_cost_per_token" not in metadata["openai/gemma-3-1b-it"]


@pytest.mark.parametrize("harness", ["kilo", "pi", "goose", "aider"])
def test_a_subscription_connect_writes_no_prices(subscribed, tmp_path, harness):
    subscribed.routes[("GET", "models")] = PRICED_CATALOG

    result = runner.invoke(app, ["connect", harness, "--yes"])

    assert result.exit_code == 0, result.output
    # The plan pays nothing per token, so a cost readout there would be wrong.
    written = "\n".join(path.read_text() for path in tmp_path.rglob("*") if path.is_file())
    assert "cost" not in written
