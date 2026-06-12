"""Render tests for deployment table / detail / snippet (via StringIO console)."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from jarvislabs.cli import render
from jarvislabs.models import (
    Deployment,
    DeploymentListResult,
    DeploymentStatus,
    DeploymentSummary,
    RegionError,
)


@pytest.fixture()
def capture(monkeypatch):
    buf = StringIO()
    monkeypatch.setattr(render, "stdout_console", Console(file=buf, force_terminal=False, color_system=None, width=200))
    return buf


def _summary(**overrides):
    payload = {
        "deployment_id": "dep1",
        "name": "qwen",
        "status": "running",
        "region": "india-noida-01",
        "framework": "vllm",
        "min_workers": 0,
        "max_workers": 2,
        "gpus_to_use": {"gpus": ["L4"]},
        "concurrent_requests": 4,
        "gpus_per_worker": 1,
    }
    payload.update(overrides)
    return DeploymentSummary(**payload)


def test_table_renders_rows(capture):
    result = DeploymentListResult(deployments=[_summary()])
    render.deployments_table(result)
    out = capture.getvalue()
    assert "dep1" in out
    assert "IN2" in out
    assert "running" in out
    assert "L4" in out
    assert "0-2" in out
    assert "Model" not in out  # no model column (the list endpoint omits it)


def test_table_warns_on_region_errors(capture, monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(render, "warning", lambda m: warnings.append(m))
    result = DeploymentListResult(
        deployments=[_summary()],
        region_errors=[RegionError(region="india-chennai-01", error="timeout")],
    )
    render.deployments_table(result)
    assert any("IN1" in w and "timeout" in w for w in warnings)


def test_table_empty(capture, monkeypatch):
    infos: list[str] = []
    monkeypatch.setattr(render, "info", lambda m: infos.append(m))
    render.deployments_table(DeploymentListResult())
    assert infos == ["No deployments."]


def _deployment(status="running", **overrides):
    payload = {
        "deployment_id": "dep1",
        "name": "qwen",
        "status": status,
        "region": "india-noida-01",
        "framework": "vllm",
        "gpus_to_use": {"gpus": ["L4"]},
        "gpus_per_worker": 1,
        "min_workers": 0,
        "max_workers": 2,
        "concurrent_requests": 4,
        "idle_timeout": 600,
        "wait_time": 60,
        "storage": 50,
        "args": {"model": "Qwen/Qwen3-0.6B", "max-model-len": "8192"},
        "env": {"HF_TOKEN": "hf_****"},
        "workers": {
            "total": 1,
            "healthy": 1,
            "provisioning": 0,
            "list": [{"status": "healthy", "last_used": "now"}],
        },
        "queue_depth": 0,
    }
    payload.update(overrides)
    return Deployment(**payload)


def test_detail_shows_env_and_workers(capture):
    render.deployment_detail(_deployment(), base_url="https://serverlessn.jarvislabs.net/openai/dep1/v1")
    out = capture.getvalue()
    assert "Qwen/Qwen3-0.6B" in out
    assert "max-model-len=8192" in out
    assert "hf_****" in out
    assert "worker 1" in out
    assert "serverlessn.jarvislabs.net/openai/dep1/v1" in out


def test_detail_shows_worker_id_when_present(capture):
    workers = {
        "total": 2,
        "healthy": 1,
        "provisioning": 1,
        "list": [
            {"worker_id": 8841, "status": "healthy", "last_used": "now"},
            {"worker_id": None, "status": "provisioning", "last_used": None},
        ],
    }
    render.deployment_detail(_deployment(workers=workers))
    out = capture.getvalue()
    assert "worker 8841" in out  # real id when the API sends one
    assert "worker 2" in out  # positional fallback while it's null


def test_detail_shows_error_message(capture):
    render.deployment_detail(_deployment(status="failed", error_message="download timeout"))
    assert "download timeout" in capture.getvalue()


def test_detail_workers_dash_when_missing(capture):
    render.deployment_detail(_deployment(min_workers=None, max_workers=None))
    out = capture.getvalue()
    assert "None" not in out
    assert "— (min-max)" in out


def test_status_line(capture):
    render.deployment_status_line(DeploymentStatus(deployment_id="dep1", region="india-noida-01", status="running"))
    out = capture.getvalue()
    assert "IN2" in out
    assert "dep1" in out
    assert "running" in out


def test_running_handoff_snippet_markup_safe(capture):
    # The base URL arrives in /v1 form and is used verbatim — print and snippet identical.
    base_url = "https://serverlessn.jarvislabs.net/openai/dep1/v1"
    render.deployment_running_handoff(base_url, "Qwen/Qwen3-0.6B")
    out = capture.getvalue()
    assert f'base_url="{base_url}"' in out
    assert 'model="Qwen/Qwen3-0.6B"' in out
    # Brackets must survive (markup disabled).
    assert '{"role": "user", "content": "Hello"}' in out
    assert "[" in out and "]" in out
    assert "first request" in out


def test_table_markup_safe_values_survive(capture):
    s = _summary(deployment_id="dep[bold]x", name="my [b]name", gpus_to_use={"gpus": ["org/Model[v1]"]})
    render.deployments_table(DeploymentListResult(deployments=[s]))
    out = capture.getvalue()
    assert "dep[bold]x" in out
    assert "my [b]name" in out
    assert "org/Model[v1]" in out


def test_detail_markup_safe_values_survive(capture):
    d = _deployment(
        deployment_id="dep[bold]x",
        name="my [b]name",
        args={"model": "org/Model[v1]", "a": "b[c]d"},
        error_message="bad [red]err",
    )
    render.deployment_detail(d)
    out = capture.getvalue()
    assert "dep[bold]x" in out
    assert "my [b]name" in out
    assert "org/Model[v1]" in out
    assert "b[c]d" in out
    assert "bad [red]err" in out


def test_status_colors():
    assert render._deployment_status_style("running") == "green"
    assert render._deployment_status_style("starting") == "blue"
    assert render._deployment_status_style("failed") == "red"
    assert render._deployment_status_style("cleaning") == "red"
    assert render._deployment_status_style("deleting") == "yellow"
    assert render._deployment_status_style("deleted") == "dim"


def test_info_suppressed_in_json_mode(monkeypatch):
    from jarvislabs.cli import state

    buf = StringIO()
    monkeypatch.setattr(render, "console", Console(file=buf, force_terminal=False, color_system=None))
    monkeypatch.setattr(state, "json_output", True)
    render.info("should not appear")
    assert buf.getvalue() == ""
    monkeypatch.setattr(state, "json_output", False)
    render.info("visible")
    assert "visible" in buf.getvalue()


def test_confirm_json_mode_dies(monkeypatch, capsys):
    from jarvislabs.cli import state

    monkeypatch.setattr(state, "json_output", True)
    with pytest.raises(SystemExit) as excinfo:
        render.confirm("Delete?", skip=False)
    assert excinfo.value.code == 1
    assert "--json requires --yes" in capsys.readouterr().out


def test_confirm_json_mode_skip_still_wins(monkeypatch):
    from jarvislabs.cli import state

    monkeypatch.setattr(state, "json_output", True)
    assert render.confirm("Delete?", skip=True) is True
