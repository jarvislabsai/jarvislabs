"""`jl connect`'s preview and success lines, and reading back a harness's default model."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from jarvislabs.cli import connect  # noqa: F401  (registers the command)
from jarvislabs.cli.app import app
from tests.cli import test_connect
from tests.cli.test_connect import DEEPSEEK, GEMMA, read_json

runner = CliRunner()
# A subscriber with no key yet, and the DeepSeek, GLM and Gemma catalog.
subscribed = test_connect.subscribed
# jl is already the latest version, for jl update.
up_to_date = test_connect.up_to_date


def _status_default(harness_id: str) -> str | None:
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    (row,) = [h for h in json.loads(result.stdout)["harnesses"] if h["id"] == harness_id]
    return row["default_model"]


@pytest.mark.parametrize(
    ("harness", "name"), [("opencode", "OpenCode"), ("goose", "Goose"), ("aider", "Aider"), ("crush", "Crush")]
)
def test_reconnecting_unchanged_previews_no_replacements(subscribed, harness, name):
    runner.invoke(app, ["connect", harness, "--yes"])

    result = runner.invoke(app, ["connect", harness, "--dry-run"])

    assert result.exit_code == 0, result.output
    # Everything in the files is what the first connect wrote: an update that replaces nothing.
    lines = result.stdout.splitlines()
    legend_index = lines.index("~ update")
    heading = " ".join(" ".join(lines[:legend_index]).split())
    assert heading == "jl will update the Jarvislabs provider 3 models, default deepseek-v4.1-flash-1017"
    assert lines[legend_index + 2].startswith(f"~ {name}  ")
    assert lines[legend_index + 3 :] == []


def test_crush_preview_names_overridden_model_lines(subscribed, tmp_path):
    crushrc = tmp_path / "crush" / "crushrc"
    crushrc.parent.mkdir()
    crushrc.write_text(
        "provider add anthropic --type anthropic --api-key sk-ant-1\n"
        "model large anthropic/claude-sonnet-4\n"
        "  model small anthropic/claude-haiku-4  # cheap\n"
        "model add anthropic/claude-opus-4\n"
    )

    result = runner.invoke(app, ["connect", "crush", "--dry-run"])

    assert result.exit_code == 0, result.output
    # jl's block comes after the user's lines, so its model large and model small win over theirs.
    assert (
        "~ Crush ~/crush/crushrc overrides your model large anthropic/claude-sonnet-4, model small "
        "anthropic/claude-haiku-4"
    ) in " ".join(result.stdout.split())
    assert "sk-ant-1" not in result.output


def test_aider_default_is_ours_only_with_our_base_url(subscribed, tmp_path, up_to_date):
    runner.invoke(app, ["connect", "aider", "--yes"])
    # The user points Aider back at OpenAI, keeping jl's metadata.
    conf = tmp_path / ".aider.conf.yml"
    conf.write_text("model: openai/gpt-4o\nopenai-api-base: https://api.openai.com/v1\nopenai-api-key: sk-openai-1\n")

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0, result.output
    # gpt-4o isn't in the catalog, but it isn't a Jarvislabs default, so jl update leaves it.
    assert "model: openai/gpt-4o\n" in conf.read_text()
    assert "no longer offered" not in result.output
    assert _status_default("aider") is None


def test_a_one_model_harness_reports_its_default_model(subscribed):
    result = runner.invoke(app, ["connect", "cline", "--yes"])

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "jl will add the Jarvislabs provider default deepseek-v4.1-flash-1017" in output
    assert "~/.cline/data/settings/providers.json default model only" in output
    assert "3 models" not in output


def test_cline_default_is_read_back_and_refreshed(subscribed, tmp_path, up_to_date):
    runner.invoke(app, ["connect", "cline", "--yes", "--model", "glm-5.3"])
    subscribed.routes[("GET", "models")] = {"object": "list", "data": [DEEPSEEK, GEMMA]}

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0, result.output
    settings = read_json(tmp_path / ".cline" / "data" / "settings" / "providers.json")["providers"]["openai-compatible"]
    assert settings["settings"]["model"] == "deepseek-v4.1-flash-1017"
    output = " ".join(result.output.split())
    assert "Refreshed Cline. " in output
    assert "glm-5.3 is no longer offered, so Cline's default model is now deepseek-v4.1-flash-1017" in output
    assert _status_default("cline") == "deepseek-v4.1-flash-1017"


def test_cline_default_is_not_ours_with_another_base_url(subscribed, tmp_path):
    runner.invoke(app, ["connect", "cline", "--yes"])
    path = tmp_path / ".cline" / "data" / "settings" / "providers.json"
    providers = read_json(path)
    # The user points Cline's one OpenAI-compatible slot at OpenRouter, in Cline itself.
    providers["providers"]["openai-compatible"]["settings"]["baseUrl"] = "https://openrouter.ai/api/v1"
    path.write_text(json.dumps(providers))

    assert _status_default("cline") is None
