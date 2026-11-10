"""End-to-end check of the preview harnesses against a mock Models API.

Skipped unless ``JL_PREVIEW_E2E=1`` and the real clients are on PATH, because it installs
nothing and runs two node CLIs:

    npm i -g @anthropic-ai/claude-code @openai/codex
    JL_PREVIEW_E2E=1 uv run pytest tests/cli/test_preview_clients.py -q

Each test lets jl write the client's config against a local mock (via JL_ANTHROPIC_URL /
JL_RESPONSES_URL), then runs the real client for one turn. The client's request proves the
config, the key and the endpoint all reached it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from jarvislabs.cli.app import app
from tests.cli import test_connect
from tests.cli.mock_endpoints import MockModelsAPI

runner = CliRunner()
subscribed = test_connect.subscribed

pytestmark = pytest.mark.skipif(
    not os.environ.get("JL_PREVIEW_E2E"),
    reason="set JL_PREVIEW_E2E=1 with claude and codex installed to run the real clients",
)

MODEL = "deepseek-v4.1-flash-1017"
KEY = "jlsk-new-secret"


def _run(argv: list[str], home: str) -> subprocess.CompletedProcess[str]:
    """Run a client in home, off any real config, with a generous timeout for a first start."""
    env = {
        **os.environ,
        "HOME": home,
        "XDG_CONFIG_HOME": home,
        "CODEX_HOME": f"{home}/.codex",
        # Keep the child off any key or endpoint the developer's own shell has.
        "ANTHROPIC_API_KEY": "",
        "OPENAI_API_KEY": "",
    }
    return subprocess.run(
        argv,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        # Codex appends stdin to the prompt when it isn't a TTY; a closed stdin keeps the turn exact.
        stdin=subprocess.DEVNULL,
    )


def test_claude_code_completes_a_turn_through_the_models_api(subscribed, tmp_path, monkeypatch):
    if not shutil.which("claude"):
        pytest.skip("claude is not on PATH")
    monkeypatch.setenv("JL_PREVIEW_HARNESSES", "1")
    with MockModelsAPI() as api:
        monkeypatch.setenv("JL_ANTHROPIC_URL", f"http://127.0.0.1:{api.port}")
        connected = runner.invoke(app, ["connect", "claude-code", "--yes"])
        assert connected.exit_code == 0, connected.output

        # Claude Code's own first-run state, so it doesn't stop at a theme or login prompt.
        (tmp_path / ".claude.json").write_text(
            json.dumps({"hasCompletedOnboarding": True, "theme": "dark", "numStartups": 2})
        )
        result = _run(
            ["claude", "-p", "say hi", "--output-format", "json", "--dangerously-skip-permissions"],
            str(tmp_path),
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Hi from the mock" in result.stdout
        (request,) = [r for r in api.requests if r["path"].startswith("/v1/messages")]
        assert request["body"]["model"] == MODEL
        # Claude Code sends the token as a bearer.
        assert request["headers"].get("authorization") == f"Bearer {KEY}"
        # It reaches the gateway as a coding agent, with its tools, not as a chat box.
        assert request["body"]["tools"]
        assert request["body"]["stream"] is True
        print(
            f"claude says {json.loads(result.stdout)['result']!r} | POST {request['path']} "
            f"model={request['body']['model']} auth={request['headers']['authorization'].split()[0]} "
            f"tools={len(request['body']['tools'])}"
        )


def test_codex_completes_a_turn_through_the_models_api(subscribed, tmp_path, monkeypatch):
    if not shutil.which("codex"):
        pytest.skip("codex is not on PATH")
    monkeypatch.setenv("JL_PREVIEW_HARNESSES", "1")
    with MockModelsAPI() as api:
        monkeypatch.setenv("JL_RESPONSES_URL", f"http://127.0.0.1:{api.port}/v1")
        connected = runner.invoke(app, ["connect", "codex", "--yes"])
        assert connected.exit_code == 0, connected.output

        result = _run(["codex", "exec", "--skip-git-repo-check", "say hi"], str(tmp_path))

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Hi from the mock" in result.stdout
        (request,) = [r for r in api.requests if r["path"].startswith("/v1/responses")]
        assert request["body"]["model"] == MODEL
        assert request["headers"].get("authorization") == f"Bearer {KEY}"
        # Codex sends its tool definitions too, so the endpoint must pass them through.
        assert request["body"]["tools"]
        assert request["body"]["stream"] is True
        reply = next(line for line in result.stdout.splitlines() if "Hi from the mock" in line)
        print(
            f"codex says {reply!r} | POST {request['path']} model={request['body']['model']} "
            f"auth={request['headers']['authorization'].split()[0]} tools={len(request['body']['tools'])}"
        )
