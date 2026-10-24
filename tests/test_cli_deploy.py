"""Unit tests for the `jl deploy` CLI commands (direct function calls)."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import typer
from typer.testing import CliRunner

from jarvislabs.cli import deploy, options as cli_options, state
from jarvislabs.cli.app import app
from jarvislabs.exceptions import NotFoundError, RegionResolutionError
from jarvislabs.models import (
    Deployment,
    DeploymentListResult,
    DeploymentStatus,
    DeploymentSummary,
    ManagedModel,
    ManagedModelListResult,
    ServerMetaGPU,
)
from jarvislabs.token_factory import Models
from jarvislabs.transport import Transport


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


# GPU prices used to price managed models: H200 in IN2 at 400/hr.
FAKE_ACCOUNT = SimpleNamespace(
    gpu_availability=lambda: [ServerMetaGPU(gpu_type="H200", region="india-noida-01", price_per_hour=400.0)],
    currency=lambda: "INR",
)


def _install_client(monkeypatch, deployments):
    client = SimpleNamespace(deployments=deployments, account=FAKE_ACCOUNT)
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
        openai_base_url=lambda dep_id, region=None: "https://serverlessn.jarvislabs.net/openai/dep1/v1",
    )
    _install_client(monkeypatch, deployments)

    info_calls: list[str] = []
    handoffs: list[tuple] = []
    monkeypatch.setattr(deploy.render, "info", lambda m, **k: info_calls.append(m))
    monkeypatch.setattr(deploy.render, "deployment_running_handoff", lambda url, model: handoffs.append((url, model)))

    _create_call(None, arg=["max-model-len=8192"], env=["HF_TOKEN=hf_x"])

    # The CLI passes args raw + model separately; the SDK folds model into args.
    assert captured["args"] == {"max-model-len": "8192"}
    assert captured["model"] == "Qwen/Qwen3-0.6B"
    assert captured["env"] == {"HF_TOKEN": "hf_x"}
    assert any("dep1" in m for m in info_calls)  # id printed early
    assert handoffs == [("https://serverlessn.jarvislabs.net/openai/dep1/v1", "Qwen/Qwen3-0.6B")]


def test_create_detach_does_not_poll(monkeypatch):
    polled = []
    deployments = SimpleNamespace(
        create=lambda **kw: "dep1",
        wait_until_running=lambda *a, **k: polled.append(1),
        openai_base_url=lambda *a, **k: "x",
    )
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "info", lambda m, **k: None)
    _create_call(None, detach=True)
    assert polled == []


def test_create_json_suppresses_early_id_and_snippet(monkeypatch):
    printed = {}
    deployments = SimpleNamespace(
        create=lambda **kw: "dep1",
        wait_until_running=lambda *a, **k: _deployment(),
        openai_base_url=lambda *a, **k: "https://serverlessn.jarvislabs.net/openai/dep1/v1",
    )
    _install_client(monkeypatch, deployments)
    info_calls: list[str] = []
    handoffs: list = []
    monkeypatch.setattr(deploy.render, "info", lambda m, **k: info_calls.append(m))
    monkeypatch.setattr(deploy.render, "deployment_running_handoff", lambda *a: handoffs.append(a))
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.update(data))

    _create_call(None, json_output=True)

    # The early-id info line self-suppresses in json mode (render.info checks state);
    # the snippet must not render — json mode emits exactly one json payload.
    assert handoffs == []
    assert printed["openai_base_url"] == "https://serverlessn.jarvislabs.net/openai/dep1/v1"
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
    monkeypatch.setattr(deploy.render, "info", lambda m, **k: info_calls.append(m))

    with pytest.raises(typer.Exit) as exc:
        _create_call(None)
    assert exc.value.exit_code == 0
    assert any("still starting" in m for m in info_calls)


def test_create_ctrl_c_json_emits_id(monkeypatch):
    """json mode + Ctrl-C must still hand back the id — it must never be silently lost."""

    def boom(*a, **k):
        raise KeyboardInterrupt

    deployments = SimpleNamespace(
        create=lambda **kw: "dep1", wait_until_running=boom, openai_base_url=lambda *a, **k: "x"
    )
    _install_client(monkeypatch, deployments)
    printed = {}
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.update(data))

    with pytest.raises(typer.Exit) as exc:
        _create_call(None, json_output=True)
    assert exc.value.exit_code == 0
    assert printed == {"deployment_id": "dep1", "region": "IN2"}


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
    monkeypatch.setattr(deploy.render, "info", lambda m, **k: None)
    monkeypatch.setattr(deploy.render, "deployment_running_handoff", lambda *a: None)
    _create_call(None, arg=["key=a=b", "key=c"])  # last wins
    assert captured["args"]["key"] == "c"


def test_create_handoff_uses_served_model_name_override(monkeypatch):
    handoffs: list[tuple] = []
    deployments = SimpleNamespace(
        create=lambda **kw: "dep1",
        # The handoff reads served_model off the record the server returned.
        wait_until_running=lambda *a, **k: _deployment(
            args={"model": "Qwen/Qwen3-0.6B", "served-model-name": "custom"}
        ),
        openai_base_url=lambda *a, **k: "https://serverlessn.jarvislabs.net/openai/dep1/v1",
    )
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "info", lambda m, **k: None)
    monkeypatch.setattr(deploy.render, "deployment_running_handoff", lambda url, model: handoffs.append((url, model)))

    _create_call(None, arg=["served-model-name=custom"])
    # snippet model= uses the served-model-name override, not --model
    assert handoffs == [("https://serverlessn.jarvislabs.net/openai/dep1/v1", "custom")]


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
    assert cli_options.parse_kv(pairs, flag="--arg") == expected


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
        cli_options.parse_kv(["HF_TOKEN:hf_secret"], flag="--env", redact=True)
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
    monkeypatch.setattr(deploy.render, "deployments_table", lambda r, wide=False: rendered.append((r, wide)))
    deploy.deploy_list(json_output=False)
    assert rendered == [(result, False)]


def test_list_wide_passes_through(monkeypatch):
    result = DeploymentListResult(
        deployments=[DeploymentSummary(deployment_id="dep1", status="running", region="india-noida-01")],
    )
    _install_client(monkeypatch, SimpleNamespace(list=lambda: result))
    rendered = []
    monkeypatch.setattr(deploy.render, "deployments_table", lambda r, wide=False: rendered.append((r, wide)))
    deploy.deploy_list(wide=True, json_output=False)
    assert rendered == [(result, True)]


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
        openai_base_url=lambda dep_id, region=None: "https://serverlessn.jarvislabs.net/openai/dep1/v1",
    )
    _install_client(monkeypatch, deployments)
    printed = {}
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.update(data))
    deploy.deploy_get("dep1", region=None, json_output=True)
    assert printed["openai_base_url"] == "https://serverlessn.jarvislabs.net/openai/dep1/v1"
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


# ── logs ────────────────────────────────────────────────────────────────────────


def _install_logs(monkeypatch, events, captured=None, record=None):
    def logs(dep_id, **kw):
        if captured is not None:
            captured.update(kw, deployment_id=dep_id)
        yield from events

    deployments = SimpleNamespace(get=lambda dep_id, region=None: record or _deployment(), logs=logs)
    _install_client(monkeypatch, deployments)


def test_logs_lines_to_stdout_notices_to_info(monkeypatch, capsys):
    infos: list[str] = []
    monkeypatch.setattr(deploy.render, "info", lambda m, **k: infos.append(m))
    _install_logs(
        monkeypatch,
        [
            ("notice", "deployment status is running"),
            ("log", "[worker-1] hello"),
            ("log", "[worker-1] world"),
        ],
    )
    deploy.deploy_logs("dep1", region="IN2")
    assert capsys.readouterr().out == "[worker-1] hello\n[worker-1] world\n"
    assert infos == ["deployment status is running"]


def test_logs_passes_flags_and_resolved_region(monkeypatch):
    captured = {}
    _install_logs(monkeypatch, [], captured)
    deploy.deploy_logs("dep1", region="IN2", tail=0, follow=False, worker=12)
    # logs() gets the region from the fetched record, not the raw user input.
    assert captured == {
        "deployment_id": "dep1",
        "region": "india-noida-01",
        "tail": 0,
        "follow": False,
        "worker": 12,
    }


def test_logs_json_rejected_before_any_call(monkeypatch):
    called = []
    _install_client(
        monkeypatch,
        SimpleNamespace(get=lambda *a, **k: called.append("get"), logs=lambda *a, **k: called.append("logs")),
    )

    def fake_die(msg, code=1):
        raise SystemExit(msg)

    monkeypatch.setattr(deploy.render, "die", fake_die)
    with pytest.raises(SystemExit) as exc:
        deploy.deploy_logs("dep1", json_output=True)
    assert "log streaming" in str(exc.value)
    assert called == []


def test_logs_ctrl_c_exits_cleanly(monkeypatch):
    def logs(dep_id, **kw):
        yield ("log", "line")
        raise KeyboardInterrupt

    _install_client(monkeypatch, SimpleNamespace(get=lambda *a, **k: _deployment(), logs=logs))
    with pytest.raises(typer.Exit) as exc:
        deploy.deploy_logs("dep1", region="IN2")
    assert exc.value.exit_code == 0


def test_logs_no_workers_failed_points_to_get(monkeypatch):
    infos: list[str] = []
    monkeypatch.setattr(deploy.render, "info", lambda m, **k: infos.append(m))
    _install_logs(monkeypatch, [("notice", "no active workers")], record=_deployment(status="failed"))
    deploy.deploy_logs("dep1", region="IN2")
    assert any("jl deploy get dep1" in m for m in infos)


def test_logs_no_workers_while_starting_says_retry(monkeypatch):
    infos: list[str] = []
    monkeypatch.setattr(deploy.render, "info", lambda m, **k: infos.append(m))
    _install_logs(monkeypatch, [("notice", "no active workers")], record=_deployment(status="starting"))
    deploy.deploy_logs("dep1", region="IN2")
    assert any("once a worker is running" in m for m in infos)


def test_logs_unknown_id_fails_before_streaming(monkeypatch):
    streamed = []

    def get(*a, **k):
        raise NotFoundError("Deployment ghost not found in any region.")

    _install_client(monkeypatch, SimpleNamespace(get=get, logs=lambda *a, **k: streamed.append(1)))
    with pytest.raises(NotFoundError):
        deploy.deploy_logs("ghost")
    assert streamed == []


def test_logs_prefetches_record_even_with_region_hint(monkeypatch):
    calls = []
    deployments = SimpleNamespace(
        get=lambda dep_id, region=None: calls.append(("get", region)) or _deployment(),
        logs=lambda dep_id, **kw: iter([]),
    )
    _install_client(monkeypatch, deployments)
    deploy.deploy_logs("dep1", region="IN2")
    assert calls == [("get", "IN2")]


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


# ── managed models ──────────────────────────────────────────────────────────────


def test_managed_models_list_renders_table(monkeypatch):
    result = ManagedModelListResult(models=[ManagedModel(model_slug="glm-5.2")])
    _install_client(monkeypatch, SimpleNamespace(list_managed_models=lambda: result))
    rendered = []
    monkeypatch.setattr(deploy.render, "managed_models_table", lambda r: rendered.append(r))
    deploy.managed_models_list(json_output=False)
    assert rendered == [result]


def test_managed_models_list_json(monkeypatch):
    result = ManagedModelListResult(models=[ManagedModel(model_slug="glm-5.2")])
    _install_client(monkeypatch, SimpleNamespace(list_managed_models=lambda: result))
    printed = []
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.append(data))
    deploy.managed_models_list(json_output=True)
    assert printed == [result]


def _catalog(*rows):
    """A catalog with one ManagedModel per (slug, region) row."""
    return ManagedModelListResult(
        models=[
            ManagedModel(
                model_slug=slug,
                display_name=slug.upper(),
                region=region,
                gpu_type="H200",
                gpus_per_worker=8,
                concurrent_requests=32,
            )
            for slug, region in rows
        ]
    )


def _install_managed_client(monkeypatch, catalog, **extra):
    """Fake client whose create_managed records its arguments into the returned dict."""
    created = {}
    deployments = SimpleNamespace(
        list_managed_models=lambda: catalog,
        create_managed=lambda slug, **kw: created.update(slug=slug, **kw) or "dep1",
        **extra,
    )
    _install_client(monkeypatch, deployments)
    monkeypatch.setattr(deploy.render, "info", lambda m, **k: None)
    return created


def _managed_create_call(**overrides):
    kwargs = {
        "model_slug": "glm-5.2",
        "name": "glm",
        "region": "IN2",
        "min_workers": 0,
        "max_workers": 2,
        "idle_timeout": 600,
        "detach": False,
        "yes": True,
        "json_output": False,
    }
    kwargs.update(overrides)
    deploy.managed_models_create(**kwargs)


def _die_raises(monkeypatch):
    def fake_die(msg, code=1):
        raise SystemExit(msg)

    monkeypatch.setattr(deploy.render, "die", fake_die)


def test_managed_create_passes_settings_and_hands_off_api_model(monkeypatch):
    managed = {"model_slug": "glm-5.2", "api_model_name": "zai-org/GLM-5.2"}
    created = _install_managed_client(
        monkeypatch,
        _catalog(("glm-5.2", "india-noida-01")),
        wait_until_running=lambda dep_id, region=None: _deployment(args=None, env=None, managed_model=managed),
        openai_base_url=lambda dep_id, region=None: "https://serverlessn.jarvislabs.net/openai/dep1/v1",
    )
    handoffs: list[tuple] = []
    monkeypatch.setattr(deploy.render, "deployment_running_handoff", lambda url, model: handoffs.append((url, model)))

    _managed_create_call(concurrent=16)

    assert created == {
        "slug": "glm-5.2",
        "name": "glm",
        "region": "india-noida-01",
        "min_workers": 0,
        "max_workers": 2,
        "idle_timeout": 600,
        "concurrent": 16,
    }
    # The snippet uses the managed model's API name, not a (missing) model arg.
    assert handoffs == [("https://serverlessn.jarvislabs.net/openai/dep1/v1", "zai-org/GLM-5.2")]


def test_managed_create_confirm_shows_gpu_and_declined_creates_nothing(monkeypatch):
    created = _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    prompts: list[str] = []
    monkeypatch.setattr(deploy.render, "confirm", lambda msg, **k: prompts.append(msg) and False)
    with pytest.raises(typer.Exit):
        _managed_create_call(yes=False)
    assert "gpu=8x H200/worker" in prompts[0]
    assert "concurrent=32" in prompts[0]  # the model's tested value when --concurrent isn't set
    assert created == {}


def test_managed_create_detach_json_emits_id_and_region(monkeypatch):
    polled = []
    _install_managed_client(
        monkeypatch, _catalog(("glm-5.2", "india-noida-01")), wait_until_running=lambda *a, **k: polled.append(1)
    )
    printed = {}
    monkeypatch.setattr(deploy.render, "print_json", lambda data: printed.update(data))
    _managed_create_call(detach=True, json_output=True)
    assert printed == {"deployment_id": "dep1", "region": "IN2"}
    assert polled == []


def test_managed_create_uses_defaults_and_slug_as_name(monkeypatch):
    created = _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    deploy.managed_models_create("glm-5.2", detach=True, yes=True)
    assert created == {
        "slug": "glm-5.2",
        "name": "glm-5.2",
        "region": "india-noida-01",
        "min_workers": 0,
        "max_workers": 1,
        "idle_timeout": 300,
        "concurrent": None,
    }


def test_managed_create_unknown_slug_dies(monkeypatch):
    _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    _die_raises(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        _managed_create_call(model_slug="nope", region=None)
    assert "No matching managed model" in str(exc.value)


def test_managed_create_bare_command_without_terminal_exits_nonzero(monkeypatch):
    # CliRunner has no real terminal, so the picker must not open.
    created = _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    result = CliRunner().invoke(app, ["deploy", "managed-models", "create"])
    assert result.exit_code != 0
    assert created == {}


# ── managed models: picker ──────────────────────────────────────────────────────


def _stub_select(monkeypatch, pick):
    """Replace questionary.select; ``pick(choices)`` returns the chosen value (None = Ctrl+C)."""
    shown = []

    def fake_select(message, choices, **kwargs):
        # The first entry is the heading row; the rest are the models.
        assert isinstance(choices[0], deploy.render.questionary.Separator)
        models = choices[1:]
        shown.append([c.value for c in models])
        return SimpleNamespace(ask=lambda: pick(models), application=SimpleNamespace())

    monkeypatch.setattr(deploy.render, "can_prompt", lambda: True)
    monkeypatch.setattr(deploy.render.questionary, "select", fake_select)
    return shown


def _stub_text(monkeypatch, answers=None):
    """Replace questionary.text; answers by prompt message, else accept the prefilled default.

    Returns the list of prompt messages asked, in order.
    """
    asked = []

    def fake_text(message, default, validate, **kwargs):
        asked.append(message)
        answer = (answers or {}).get(message, default)
        if answer is not None:
            assert validate(answer) is True, f"{message} rejected {answer!r}"
        return SimpleNamespace(ask=lambda: answer)

    monkeypatch.setattr(deploy.render.questionary, "text", fake_text)
    return asked


SETTING_PROMPTS = [
    "Deployment name:",
    "Min workers:",
    "Max workers:",
    "Idle timeout (seconds):",
    "Concurrent requests per worker:",
]


def test_managed_create_without_slug_picks_model_then_asks_settings(monkeypatch):
    created = _install_managed_client(
        monkeypatch, _catalog(("glm-5.2", "india-noida-01"), ("qwen3", "india-chennai-01"))
    )
    shown = _stub_select(monkeypatch, lambda choices: choices[1].value)
    asked = _stub_text(monkeypatch)

    deploy.managed_models_create(detach=True, yes=True)

    assert len(shown[0]) == 2
    assert asked == SETTING_PROMPTS
    # Region comes from the picked row; accepted defaults: slug as name, 0-1 workers, the model's concurrency.
    assert created == {
        "slug": "qwen3",
        "name": "qwen3",
        "region": "india-chennai-01",
        "min_workers": 0,
        "max_workers": 1,
        "idle_timeout": 300,
        "concurrent": 32,
    }


def test_managed_create_settings_answers_are_used(monkeypatch):
    created = _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    _stub_select(monkeypatch, lambda choices: choices[0].value)
    _stub_text(
        monkeypatch,
        {
            "Deployment name:": "mine",
            "Min workers:": "2",
            "Max workers:": "5",
            "Idle timeout (seconds):": "60",
            "Concurrent requests per worker:": "8",
        },
    )

    deploy.managed_models_create(detach=True, yes=True)

    assert {k: created[k] for k in ("name", "min_workers", "max_workers", "idle_timeout", "concurrent")} == {
        "name": "mine",
        "min_workers": 2,
        "max_workers": 5,
        "idle_timeout": 60,
        "concurrent": 8,
    }


def test_managed_create_settings_skip_flags_given_on_command_line(monkeypatch):
    created = _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    _stub_select(monkeypatch, lambda choices: choices[0].value)
    asked = _stub_text(monkeypatch)
    monkeypatch.setattr(deploy.cli_options, "option_was_explicit", lambda param: param in {"name", "idle_timeout"})

    deploy.managed_models_create(name="given", idle_timeout=900, detach=True, yes=True)

    assert asked == ["Min workers:", "Max workers:", "Concurrent requests per worker:"]
    assert created["name"] == "given"
    assert created["idle_timeout"] == 900


def test_managed_create_max_workers_prefill_follows_min(monkeypatch):
    created = _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    _stub_select(monkeypatch, lambda choices: choices[0].value)
    _stub_text(monkeypatch, {"Min workers:": "3"})

    deploy.managed_models_create(detach=True, yes=True)

    # The default max (1) would be below the new min, so the prefill rises to 3.
    assert created["min_workers"] == 3
    assert created["max_workers"] == 3


def test_managed_create_ctrl_c_in_settings_creates_nothing(monkeypatch):
    created = _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    _stub_select(monkeypatch, lambda choices: choices[0].value)
    _stub_text(monkeypatch, {"Max workers:": None})

    with pytest.raises(typer.Exit):
        deploy.managed_models_create(detach=True, yes=True)
    assert created == {}


def test_managed_create_picker_filters_by_region(monkeypatch):
    _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01"), ("qwen3", "india-chennai-01")))
    shown = _stub_select(monkeypatch, lambda choices: choices[0].value)
    _stub_text(monkeypatch)

    _managed_create_call(model_slug=None, region="IN2", detach=True)

    assert [m.model_slug for m in shown[0]] == ["glm-5.2"]


def test_managed_create_slug_in_one_region_needs_no_prompt(monkeypatch):
    created = _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    shown = _stub_select(monkeypatch, lambda choices: pytest.fail("should not prompt"))
    asked = _stub_text(monkeypatch)

    _managed_create_call(region=None, detach=True)

    assert shown == [] and asked == []
    assert created["region"] == "india-noida-01"


def test_managed_create_slug_in_many_regions_prompts_for_region(monkeypatch):
    created = _install_managed_client(
        monkeypatch, _catalog(("glm-5.2", "india-chennai-01"), ("glm-5.2", "india-noida-01"), ("qwen3", "x"))
    )
    shown = _stub_select(monkeypatch, lambda choices: choices[1].value)
    _stub_text(monkeypatch)

    _managed_create_call(region=None, detach=True)

    assert [(m.model_slug, m.region) for m in shown[0]] == [
        ("glm-5.2", "india-chennai-01"),
        ("glm-5.2", "india-noida-01"),
    ]
    assert created["region"] == "india-noida-01"


def test_managed_create_picker_ctrl_c_creates_nothing(monkeypatch):
    created = _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    _stub_select(monkeypatch, lambda choices: None)

    with pytest.raises(typer.Exit):
        _managed_create_call(model_slug=None, region=None)
    assert created == {}


@pytest.mark.parametrize("json_output", [False, True])
def test_managed_create_without_slug_never_prompts_non_interactive(monkeypatch, json_output):
    created = _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    monkeypatch.setattr(deploy.render, "can_prompt", lambda: False)
    monkeypatch.setattr(deploy.render.questionary, "select", lambda *a, **k: pytest.fail("should not prompt"))
    _die_raises(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        _managed_create_call(model_slug=None, region=None, json_output=json_output)
    assert str(exc.value) == "Pass a model slug. See: jl deploy managed-models list"
    assert created == {}


def test_managed_create_slug_in_many_regions_non_interactive_asks_for_region(monkeypatch):
    _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-chennai-01"), ("glm-5.2", "india-noida-01")))
    monkeypatch.setattr(deploy.render, "can_prompt", lambda: False)
    _die_raises(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        _managed_create_call(region=None)
    assert str(exc.value) == "glm-5.2 is available in IN1, IN2. Pass --region."


def test_managed_models_are_priced_from_gpu_price(monkeypatch):
    _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01"), ("glm-5.2", "india-chennai-01")))
    result = deploy._fetch_managed_models(deploy.get_client())
    assert result.currency == "INR"
    # 8x H200 at 400/hr in IN2; no H200 price in IN1, so unknown there.
    assert [m.price_per_hour for m in result.models] == [3200.0, None]


def test_managed_create_confirm_shows_cost(monkeypatch):
    _install_managed_client(monkeypatch, _catalog(("glm-5.2", "india-noida-01")))
    prompts: list[str] = []
    monkeypatch.setattr(deploy.render, "confirm", lambda msg, skip=False: prompts.append(msg) or False)
    with pytest.raises(typer.Exit):
        _managed_create_call(max_workers=2)
    assert "cost=₹3200.00/hr per running worker, up to ₹6400.00/hr at 2 workers" in prompts[0]


# ── chat ───────────────────────────────────────────────────────────────────────


HEALTHY = {"total": 1, "healthy": 1, "provisioning": 0}
NO_WORKERS = {"total": 0, "healthy": 0, "provisioning": 0}
STARTING = {"total": 1, "healthy": 0, "provisioning": 1}
STARTING_MSG = (
    "This deployment is starting a worker. This can take a few minutes. Try again shortly. "
    "Check progress: jl deploy get dep1"
)
WOKEN_MSG = (
    "No worker was running, so this deployment is starting one now. This can take a few minutes. "
    "Try again shortly. Check progress: jl deploy get dep1"
)


def _chat_setup(monkeypatch, status="running", workers=HEALTHY):
    """Returns the chat_stream calls; ``.transport`` is the mock transport behind the real chat_submit."""
    calls = []

    def chat_stream(model, messages, **params):
        calls.append((model, messages, params))
        yield {"choices": [{"delta": {"content": "PONG"}, "finish_reason": "stop"}]}

    deployments = SimpleNamespace(
        get=lambda deployment_id, region=None: _deployment(status=status, workers=workers),
        openai_base_url=lambda deployment_id, region=None: "https://x/openai/dep1/v1",
    )
    client = _install_client(monkeypatch, deployments)
    transport = MagicMock(spec=Transport)
    client.models = Models(transport)
    client.models.chat_stream = chat_stream
    monkeypatch.setattr(deploy.render, "spinner", lambda *a, **k: nullcontext())
    monkeypatch.setattr(deploy.render, "chat_usage", lambda *a: None)
    monkeypatch.setattr("jarvislabs.cli.models._read_stdin", lambda **_: "")
    return SimpleNamespace(calls=calls, transport=transport)


def test_deploy_chat_sends_to_the_deployment(monkeypatch):
    sent = _chat_setup(monkeypatch)
    deploy.deploy_chat("dep1", "Say PONG", max_tokens=20)
    model, messages, params = sent.calls[0]
    assert model == "Qwen/Qwen3-0.6B"
    assert messages == [{"role": "user", "content": "Say PONG"}]
    assert params == {"base_url": "https://x/openai/dep1/v1", "max_tokens": 20}
    sent.transport.submit_stream.assert_not_called()


def test_deploy_chat_with_a_starting_worker_sends_nothing(monkeypatch):
    sent = _chat_setup(monkeypatch, workers=STARTING)
    errors = []
    monkeypatch.setattr(deploy.render, "error", errors.append)
    with pytest.raises(SystemExit) as exc:
        deploy.deploy_chat("dep1", "Say PONG")
    assert exc.value.code == 1
    assert errors == [STARTING_MSG]
    assert sent.calls == []
    sent.transport.submit_stream.assert_not_called()


def test_deploy_chat_with_no_workers_sends_one_wake_request(monkeypatch):
    sent = _chat_setup(monkeypatch, workers=NO_WORKERS)
    errors = []
    monkeypatch.setattr(deploy.render, "error", errors.append)
    with pytest.raises(SystemExit) as exc:
        deploy.deploy_chat("dep1", "Say PONG", max_tokens=20)
    assert exc.value.code == 1
    assert errors == [WOKEN_MSG]
    assert sent.calls == []
    sent.transport.submit_stream.assert_called_once_with(
        "chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": True,
        },
        base_url="https://x/openai/dep1/v1",
    )


@pytest.mark.parametrize(("workers", "message"), [(STARTING, STARTING_MSG), (NO_WORKERS, WOKEN_MSG)])
def test_deploy_chat_json_reports_no_worker_as_error(monkeypatch, workers, message):
    _chat_setup(monkeypatch, workers=workers)
    printed = []
    monkeypatch.setattr(deploy.render, "print_json", printed.append)
    with pytest.raises(SystemExit) as exc:
        deploy.deploy_chat("dep1", "Hi", json_output=True)
    assert exc.value.code == 1
    assert printed == [{"error": message}]


def test_deploy_chat_interactive_with_no_workers_exits_before_the_chat(monkeypatch):
    sent = _chat_setup(monkeypatch, workers=NO_WORKERS)
    monkeypatch.setattr(deploy.render, "can_prompt", lambda: True)
    monkeypatch.setattr(deploy.render, "error", lambda msg: None)
    monkeypatch.setattr(deploy, "interactive_chat", lambda *a: pytest.fail("chat started"))
    with pytest.raises(SystemExit):
        deploy.deploy_chat("dep1")
    sent.transport.submit_stream.assert_called_once()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": None, "image": ["photo.jpg"]}, "Pass a prompt with --image"),
        ({"prompt": None, "json_output": True}, "Pass a prompt."),
        ({"prompt": "Hi", "param": ["model=x"]}, "--param model is set by jl"),
    ],
)
def test_deploy_chat_checks_input_before_waking(monkeypatch, kwargs, message):
    sent = _chat_setup(monkeypatch, workers=NO_WORKERS)
    errors = []
    monkeypatch.setattr(deploy.render, "error", errors.append)
    monkeypatch.setattr(deploy.render, "print_json", lambda data: errors.append(data["error"]))
    with pytest.raises(SystemExit):
        deploy.deploy_chat("dep1", **kwargs)
    assert message in errors[0]
    sent.transport.submit_stream.assert_not_called()


def test_deploy_chat_needs_a_running_deployment(monkeypatch):
    _chat_setup(monkeypatch, status="starting")
    errors = []
    monkeypatch.setattr(deploy.render, "error", errors.append)
    with pytest.raises(SystemExit):
        deploy.deploy_chat("dep1", "Hi")
    assert "is starting; it must be running" in errors[0]


def test_deploy_chat_sends_image_and_params(monkeypatch, tmp_path):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    sent = _chat_setup(monkeypatch)
    deploy.deploy_chat("dep1", "Describe", image=[str(photo)], param=["temperature=0.5"])
    _, messages, params = sent.calls[0]
    assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert params == {"temperature": 0.5, "base_url": "https://x/openai/dep1/v1"}
