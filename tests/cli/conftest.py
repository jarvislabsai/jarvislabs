"""Fixtures shared by the command tests: a fake home and fake servers."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from jarvislabs import client as client_module, harnesses
from jarvislabs.cli import connect
from jarvislabs.transport import Transport

ACCOUNT = {
    ("GET", "users/user_info"): {"user_id": "u1", "name": "Ada"},
    ("GET", "users/balance"): {"balance": 12.5, "grants": 0},
    ("GET", "misc/resource_metrics"): {
        "running_instances": 0,
        "paused_instances": 0,
        "running_vms": 0,
        "paused_vms": 0,
        "deployments": 0,
        "filesystems": 0,
    },
    ("GET", "misc/"): {"success": False},
}


@pytest.fixture(autouse=True)
def desktop(monkeypatch):
    """Run every command test in a desktop session, where jl may open a browser; a test can change it."""
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")


@pytest.fixture(autouse=True)
def no_vscode(monkeypatch):
    """VS Code isn't running, whatever this machine has open: a test that needs it patches this itself."""
    monkeypatch.setattr(connect, "vscode_running", lambda: False, raising=False)


@pytest.fixture()
def server(tmp_path, monkeypatch):
    """Fake the servers at Transport: set ``server.routes[(method, path)]`` to a response or an exception.

    Every request is recorded in ``server.calls`` as ``(method, path, json)``. Every CLI harness
    is on PATH, so that connecting one never skips it; a program whose output jl reads prints
    nothing.
    """
    for var in ("HOME", "USERPROFILE", "APPDATA", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(var, str(tmp_path))
    installed = tmp_path / "installed-harnesses"
    installed.mkdir()
    for name in ("opencode", "kilo", "pi", "cline", "qwen", "copilot", "junie", "goose", "aider", "crush"):
        for program in (name, name + ".cmd"):  # .cmd so that Windows finds it too
            (installed / program).write_text("")
            (installed / program).chmod(0o755)
    monkeypatch.setenv("PATH", str(installed) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("JL_API_KEY", "jarvis-key")
    monkeypatch.setenv("JL_NO_UPDATE_CHECK", "1")
    transport = MagicMock(spec=Transport)
    transport.routes = dict(ACCOUNT)
    transport.calls = []
    # Programs jl reads from (crush --version) print nothing, as when they can't run.
    monkeypatch.setattr(harnesses, "_output", lambda argv: "")

    def request(method, path, *, json=None, params=None, files=None, base_url=None):
        transport.calls.append((method, path, json))
        response = transport.routes[(method, path)]
        if isinstance(response, Exception):
            raise response
        return response

    transport.request.side_effect = request
    monkeypatch.setattr(client_module, "Transport", lambda token: transport)
    return transport
