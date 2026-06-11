"""Unit tests for the `jl deploy` CLI commands (direct function calls)."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from jarvislabs.cli import deploy, state
from jarvislabs.cli.app import app
from jarvislabs.exceptions import RegionResolutionError
from jarvislabs.models import (
    Deployment,
    DeploymentListResult,
    DeploymentStatus,
    DeploymentSummary,
)


@pytest.fixture(autouse=True)
def _reset_state():
    state.json_output = False
    state.yes = False
    yield
    state.json_output = False
    state.yes = False


@pytest.fixture(autouse=True)
def _stub_spinner(monkeypatch):
    monkeypatch.setattr(deploy.render, "spinner", lambda *a, **k: nullcontext())


def _install_client(monkeypatch, deployments):
    client = SimpleNamespace(deployments=deployments)
    monkeypatch.setattr(deploy, "get_client", lambda: client)
    return client


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
        "args": {"model": "Qwen/Qwen3-0.6B"},
        "env": {"HF_TOKEN": "hf_****"},
        "queue_depth": 0,
    }
    payload.update(overrides)
    return Deployment(**payload)


# ── create ─────────────────────────────────────────────────────────────────────


def _create_call(deploy_mock, **overrides):
    kwargs = {
        "name": "qwen",
        "region": "IN2",
        "framework": "vllm",
        "gpu": "L4",
        "gpus_per_worker": 1,
        "min_workers": 0,
        "max_workers": 2,
        "idle_timeout": 600,
        "wait_time": 60,
        "storage": 50,
        "model": "Qwen/Qwen3-0.6B",
        "detach": False,
        "yes": True,
        "json_output": False,
    }
    kwargs.update(overrides)
    deploy.deploy_create(**kwargs)


def test_create_prints_id_then_handoff(monkeypatch):
    captured = {}
    deployments = SimpleNamespace(
        create=lambda **kw: captured.update(kw) or "dep1",
        wait_until_running=lambda dep_id, region=None: _deployment(),
        openai_base_url=lambda dep_id, region=None: "https://serverlessn.jarvislabs.net/openai/dep1",
    )
    _install_client(monkeypatch, deployments)

    info_calls: list[str] = []
    handoffs: list[tuple] = []
    monkeypatch.setattr(deploy.render, "info", lambda m: info_calls.append(m))
    monkeypatch.setattr(deploy.render, "deployment_running_handoff", lambda url, model: handoffs.append((url, model)))

    _create_call(None, arg=["max-model-len=8192"], env=["HF_TOKEN=hf_x"])

    assert captured["wait"] is False
    # The CLI passes args raw + model separately; the SDK folds model into args.
    assert captured["args"] == {"max-model-len": "8192"}
    assert captured["model"] == "Qwen/Qwen3-0.6B"
    assert captured["env"] == {"HF_TOKEN": "hf_x"}
    assert any("dep1" in m for m in info_calls)  # id printed early
    assert handoffs == [("https://serverlessn.jarvislabs.net/openai/dep1", "Qwen/Qwen3-0.6B")]


def test_create_detach_does_not_poll(monkeypatch):
    polled = []
    deployments = SimpleNamespace(
        create=lambda **kw: "dep1",
        wait_until_running=lambda *a, **k: polled.append(1),
        openai_base_url=lambda *a, **k: "x",
    )
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "info", lambda m: None)
    _create_call(None, detach=True)
    assert polled == []


def test_create_json_suppresses_early_id_and_snippet(monkeypatch):
    printed = {}
    deployments = SimpleNamespace(
        create=lambda **kw: "dep1",
        wait_until_running=lambda *a, **k: _deployment(),
        openai_base_url=lambda *a, **k: "https://serverlessn.jarvislabs.net/openai/dep1",
    )
    _install_client(monkeypatch, deployments)
    info_calls: list[str] = []
    handoffs: list = []
    monkeypatch.setattr(deploy.render, "info", lambda m: info_calls.append(m))
    monkeypatch.setattr(deploy.render, "deployment_running_handoff", lambda *a: handoffs.append(a))
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.update(data))

    _create_call(None, json_output=True)

    assert info_calls == []  # early id suppressed
    assert handoffs == []  # snippet suppressed
    assert printed["openai_base_url"] == "https://serverlessn.jarvislabs.net/openai/dep1"
    assert printed["deployment_id"] == "dep1"
    assert printed["env"]["HF_TOKEN"] == "hf_****"  # env included in --json output


def test_create_detach_json_emits_id_and_region(monkeypatch):
    printed = {}
    deployments = SimpleNamespace(create=lambda **kw: "dep1", wait_until_running=lambda *a, **k: None)
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.update(data))
    _create_call(None, detach=True, json_output=True)
    assert printed == {"deployment_id": "dep1", "region": "IN2"}


def test_create_ctrl_c_clean_detach(monkeypatch):
    def boom(*a, **k):
        raise KeyboardInterrupt

    deployments = SimpleNamespace(
        create=lambda **kw: "dep1", wait_until_running=boom, openai_base_url=lambda *a, **k: "x"
    )
    _install_client(monkeypatch, deployments)
    info_calls: list[str] = []
    monkeypatch.setattr(deploy.render, "info", lambda m: info_calls.append(m))

    with pytest.raises(typer.Exit) as exc:
        _create_call(None)
    assert exc.value.exit_code == 0
    assert any("keeps running" in m for m in info_calls)


def test_create_invalid_arg_syntax_dies(monkeypatch):
    deployments = SimpleNamespace(create=lambda **kw: "dep1")
    _install_client(monkeypatch, deployments)

    def fake_die(msg, code=1):
        raise SystemExit(msg)

    monkeypatch.setattr(deploy.render, "die", fake_die)
    with pytest.raises(SystemExit) as exc:
        _create_call(None, arg=["no_equals_here"])
    assert "expected key=value" in str(exc.value)


def test_create_arg_split_on_first_equals(monkeypatch):
    captured = {}
    deployments = SimpleNamespace(
        create=lambda **kw: captured.update(kw) or "dep1",
        wait_until_running=lambda *a, **k: _deployment(),
        openai_base_url=lambda *a, **k: "x",
    )
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "info", lambda m: None)
    monkeypatch.setattr(deploy.render, "deployment_running_handoff", lambda *a: None)
    _create_call(None, arg=["key=a=b", "key=c"])  # last wins
    assert captured["args"]["key"] == "c"


def test_create_handoff_uses_served_model_name_override(monkeypatch):
    handoffs: list[tuple] = []
    deployments = SimpleNamespace(
        create=lambda **kw: "dep1",
        wait_until_running=lambda *a, **k: _deployment(),
        openai_base_url=lambda *a, **k: "https://serverlessn.jarvislabs.net/openai/dep1",
    )
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "info", lambda m: None)
    monkeypatch.setattr(deploy.render, "deployment_running_handoff", lambda url, model: handoffs.append((url, model)))

    _create_call(None, arg=["served-model-name=custom"])
    # snippet model= uses the served-model-name override, not --model
    assert handoffs == [("https://serverlessn.jarvislabs.net/openai/dep1", "custom")]


def test_create_missing_required_flag_exits_nonzero():
    """A missing required flag exits non-zero."""
    result = CliRunner().invoke(app, ["deploy", "create", "--name", "x"])  # omits --model and others
    assert result.exit_code != 0


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        (["key=a=b"], {"key": "a=b"}),  # split on first = only
        (["key="], {"key": ""}),  # empty value passes through
        (["enable=true", "disable=false"], {"enable": "true", "disable": "false"}),  # booleans pass through uncoerced
        (["key=a", "key=c"], {"key": "c"}),  # duplicate key, last wins
    ],
)
def test_parse_kv_acceptance_cases(pairs, expected):
    assert deploy._parse_kv(pairs, flag="--arg") == expected


def test_create_confirm_declined_no_call(monkeypatch):
    called = []
    deployments = SimpleNamespace(create=lambda **kw: called.append(kw) or "dep1")
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "confirm", lambda *a, **k: False)
    with pytest.raises(typer.Exit):
        _create_call(None, yes=False)
    assert called == []


def test_create_json_without_yes_dies(monkeypatch):
    called = []
    deployments = SimpleNamespace(create=lambda **kw: called.append(kw) or "dep1")
    _install_client(monkeypatch, deployments)

    def fake_die(msg, code=1):
        raise SystemExit(msg)

    monkeypatch.setattr(deploy.render, "die", fake_die)
    with pytest.raises(SystemExit) as exc:
        _create_call(None, yes=False, json_output=True)
    assert "requires --yes" in str(exc.value)
    assert called == []


def test_env_invalid_syntax_does_not_echo_value(monkeypatch):
    def fake_die(msg, code=1):
        raise SystemExit(msg)

    monkeypatch.setattr(deploy.render, "die", fake_die)
    with pytest.raises(SystemExit) as exc:
        deploy._parse_kv(["HF_TOKEN:hf_secret"], flag="--env", redact=True)
    assert "hf_secret" not in str(exc.value)
    assert "expected key=value" in str(exc.value)


# ── list ────────────────────────────────────────────────────────────────────────


def test_list_renders_table(monkeypatch):
    result = DeploymentListResult(
        deployments=[DeploymentSummary(deployment_id="dep1", status="running", region="india-noida-01")],
    )
    deployments = SimpleNamespace(list=lambda: result)
    _install_client(monkeypatch, deployments)
    rendered = []
    monkeypatch.setattr(deploy.render, "deployments_table", lambda r: rendered.append(r))
    deploy.deploy_list(json_output=False)
    assert rendered == [result]


def test_list_json(monkeypatch):
    result = DeploymentListResult(
        deployments=[DeploymentSummary(deployment_id="dep1", status="running", region="india-noida-01")],
    )
    deployments = SimpleNamespace(list=lambda: result)
    _install_client(monkeypatch, deployments)
    printed = {}
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.update({"r": data}))
    deploy.deploy_list(json_output=True)
    assert printed["r"] is result


# ── get / status ────────────────────────────────────────────────────────────────


def test_get_json_includes_base_url_when_running(monkeypatch):
    deployments = SimpleNamespace(
        get=lambda dep_id, region=None: _deployment(),
        openai_base_url=lambda dep_id, region=None: "https://serverlessn.jarvislabs.net/openai/dep1",
    )
    _install_client(monkeypatch, deployments)
    printed = {}
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.update(data))
    deploy.deploy_get("dep1", region=None, json_output=True)
    assert printed["openai_base_url"] == "https://serverlessn.jarvislabs.net/openai/dep1"
    assert printed["env"]["HF_TOKEN"] == "hf_****"  # env included


def test_get_collision_surfaces_via_global_handler(monkeypatch):
    def boom(dep_id, region=None):
        raise RegionResolutionError("found in IN1 and IN2")

    deployments = SimpleNamespace(get=boom)
    _install_client(monkeypatch, deployments)
    with pytest.raises(RegionResolutionError):
        deploy.deploy_get("dep1", region=None, json_output=False)


def test_status_json_shape(monkeypatch):
    status = DeploymentStatus(deployment_id="dep1", region="india-noida-01", status="running")
    deployments = SimpleNamespace(status=lambda dep_id, region=None: status)
    _install_client(monkeypatch, deployments)
    printed = {}
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.update({"r": data}))
    deploy.deploy_status("dep1", region=None, json_output=True)
    assert printed["r"].model_dump() == {"deployment_id": "dep1", "region": "IN2", "status": "running"}


# ── update ──────────────────────────────────────────────────────────────────────


def test_update_passes_only_given_fields(monkeypatch):
    captured = {}
    deployments = SimpleNamespace(
        update=lambda dep_id, **kw: captured.update(kw) or _deployment(name="new"),
    )
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "deployment_detail", lambda *a, **k: None)
    deploy.deploy_update("dep1", name="new", idle_timeout=None, wait_time=None, region=None, json_output=False)
    assert captured["name"] == "new"
    assert captured["idle_timeout"] is None


# ── delete ──────────────────────────────────────────────────────────────────────


def test_delete_confirm_declined_no_call(monkeypatch):
    deleted = []
    deployments = SimpleNamespace(delete=lambda *a, **k: deleted.append(1))
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "confirm", lambda *a, **k: False)
    with pytest.raises(typer.Exit):
        deploy.deploy_delete("dep1", yes=False, region=None, json_output=False)
    assert deleted == []


def test_delete_yes_skips_prompt(monkeypatch):
    deleted = []
    deployments = SimpleNamespace(delete=lambda dep_id, region=None: deleted.append(dep_id))
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "success", lambda m: None)
    deploy.deploy_delete("dep1", yes=True, region=None, json_output=False)
    assert deleted == ["dep1"]


def test_delete_json_without_yes_dies(monkeypatch):
    deployments = SimpleNamespace(delete=lambda *a, **k: None)
    _install_client(monkeypatch, deployments)

    def fake_die(msg, code=1):
        raise SystemExit(msg)

    monkeypatch.setattr(deploy.render, "die", fake_die)
    with pytest.raises(SystemExit) as exc:
        deploy.deploy_delete("dep1", yes=False, region=None, json_output=True)
    assert "requires --yes" in str(exc.value)


def test_delete_json_with_yes(monkeypatch):
    deployments = SimpleNamespace(delete=lambda dep_id, region=None: True)
    _install_client(monkeypatch, deployments)
    printed = {}
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.update(data))
    deploy.deploy_delete("dep1", yes=True, region=None, json_output=True)
    assert printed == {"success": True, "deployment_id": "dep1"}
