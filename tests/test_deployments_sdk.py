"""Unit tests for the Deployments SDK resource."""

from __future__ import annotations

import pytest

from jarvislabs import deployments as client_mod
from jarvislabs.constants import CHENNAI_REGION, INDIA_NOIDA_REGION
from jarvislabs.deployments import Deployments
from jarvislabs.exceptions import APIError, AuthError, JarvislabsError, NotFoundError, ValidationError
from jarvislabs.models import Deployment, DeploymentStatus, DeploymentSummary

NOIDA_URL = "https://serverlessn.jarvislabs.net/"


def _deps(mock_transport):
    return Deployments(mock_transport)


def _get_payload(status="running", **overrides):
    payload = {
        "deployment_id": "dep1",
        "name": "qwen",
        "status": status,
        "error_message": None,
        "region": INDIA_NOIDA_REGION,
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
        "env": {},
        "start_time": "2026-06-08T08:38:45.011322",
        "workers": {"total": 0, "healthy": 0, "provisioning": 0, "list": []},
        "queue_depth": 0,
    }
    payload.update(overrides)
    return payload


# ── create ─────────────────────────────────────────────────────────────────────


def _create_kwargs(**overrides):
    kwargs = {
        "name": "qwen",
        "region": "IN2",
        "framework": "vllm",
        "gpu": "L4",
        "model": "Qwen/Qwen3-0.6B",
        "gpus_per_worker": 1,
        "min_workers": 0,
        "max_workers": 2,
        "idle_timeout": 600,
        "wait_time": 60,
        "storage": 50,
        "wait": False,
    }
    kwargs.update(overrides)
    return kwargs


def test_create_payload_endpoint_and_base_url(mock_transport):
    mock_transport.request.return_value = {"deployment_id": "dep1"}
    dep_id = _deps(mock_transport).create(**_create_kwargs())
    assert dep_id == "dep1"
    args, kwargs = mock_transport.request.call_args
    assert args[0] == "POST"
    assert args[1] == "management/create"
    assert kwargs["base_url"] == NOIDA_URL
    payload = kwargs["json"]
    assert payload["region"] == INDIA_NOIDA_REGION
    assert payload["gpus_to_use"] == {"gpus": ["L4"]}
    assert payload["args"]["model"] == "Qwen/Qwen3-0.6B"
    assert "concurrent_requests" not in payload  # omitted when None
    assert "env" not in payload


def test_create_folds_arg_model_keeps_other_args(mock_transport):
    mock_transport.request.return_value = {"deployment_id": "dep1"}
    _deps(mock_transport).create(**_create_kwargs(args={"max-model-len": "8192"}, env={"HF_TOKEN": "x"}, concurrent=8))
    payload = mock_transport.request.call_args.kwargs["json"]
    assert payload["args"] == {"max-model-len": "8192", "model": "Qwen/Qwen3-0.6B"}
    assert payload["env"] == {"HF_TOKEN": "x"}
    assert payload["concurrent_requests"] == 8


def test_create_thin_cli_passes_arbitrary_gpu_through(mock_transport):
    mock_transport.request.return_value = {"deployment_id": "dep1"}
    _deps(mock_transport).create(**_create_kwargs(gpu="FOO"))
    assert mock_transport.request.call_args.kwargs["json"]["gpus_to_use"] == {"gpus": ["FOO"]}


def test_create_accepts_in1_region(mock_transport):
    # Serverless is now live in Chennai (IN1) — create routes there instead of rejecting.
    mock_transport.request.return_value = {"deployment_id": "dep1"}
    deps = _deps(mock_transport)
    assert deps.create(**_create_kwargs(region="IN1")) == "dep1"
    assert deps._region_cache["dep1"] == CHENNAI_REGION


def test_wait_until_running_times_out_on_unknown_status(mock_transport):
    # An unrecognized non-terminal status must not poll forever — the wall-clock
    # backstop raises with a clear message instead of hanging.
    mock_transport.request.return_value = {"status": "some_new_status"}
    with pytest.raises(JarvislabsError, match="Stopped waiting on deployment"):
        _deps(mock_transport).wait_until_running("dep1", region="IN2", timeout=0)


def test_create_rejects_eu_region(mock_transport):
    with pytest.raises(ValidationError):
        _deps(mock_transport).create(**_create_kwargs(region="EU1"))


def test_create_populates_cache(mock_transport):
    mock_transport.request.return_value = {"deployment_id": "dep1"}
    deps = _deps(mock_transport)
    deps.create(**_create_kwargs())
    assert deps._region_cache["dep1"] == INDIA_NOIDA_REGION


def test_create_no_id_raises(mock_transport):
    mock_transport.request.return_value = {}
    # A deploy-code failure raises a typed error, not the transport's APIError(0).
    with pytest.raises(JarvislabsError) as exc:
        _deps(mock_transport).create(**_create_kwargs())
    assert not isinstance(exc.value, APIError)


# ── poller (wait_until_running) ─────────────────────────────────────────────────


def _patch_sleep(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)


def test_wait_until_running_success(monkeypatch, mock_transport):
    _patch_sleep(monkeypatch)
    mock_transport.request.side_effect = [
        {"status": "starting"},
        {"status": "downloading_model"},
        {"status": "running"},
        _get_payload(),  # final GET
    ]
    deps = _deps(mock_transport)
    result = deps.wait_until_running("dep1", region="IN2")
    assert isinstance(result, Deployment)
    assert result.status == "running"
    assert deps._region_cache["dep1"] == INDIA_NOIDA_REGION


def test_wait_until_running_failed_surfaces_reason(monkeypatch, mock_transport):
    _patch_sleep(monkeypatch)
    mock_transport.request.side_effect = [
        {"status": "failed"},
        _get_payload(status="failed", error_message="download timeout"),
    ]
    with pytest.raises(JarvislabsError) as exc:
        _deps(mock_transport).wait_until_running("dep1", region="IN2")
    assert not isinstance(exc.value, APIError)
    assert "download timeout" in str(exc.value)


def test_wait_until_running_failed_reaped_reason_unavailable(monkeypatch, mock_transport):
    _patch_sleep(monkeypatch)
    mock_transport.request.side_effect = [
        {"status": "cleaning"},
        NotFoundError("gone"),  # GET 404 — already reaped
    ]
    with pytest.raises(JarvislabsError) as exc:
        _deps(mock_transport).wait_until_running("dep1", region="IN2")
    assert "reason unavailable" in str(exc.value)


def test_wait_until_running_deleting_is_not_success(monkeypatch, mock_transport):
    _patch_sleep(monkeypatch)
    mock_transport.request.side_effect = [{"status": "deleting"}]
    with pytest.raises(JarvislabsError) as exc:
        _deps(mock_transport).wait_until_running("dep1", region="IN2")
    assert "being deleted" in str(exc.value)


def test_wait_until_running_retries_transient_then_succeeds(monkeypatch, mock_transport):
    _patch_sleep(monkeypatch)
    mock_transport.request.side_effect = [
        APIError(0, "connection failed"),
        APIError(503, "down"),
        {"status": "running"},
        _get_payload(),
    ]
    result = _deps(mock_transport).wait_until_running("dep1", region="IN2")
    assert result.status == "running"


def test_wait_until_running_gives_up_after_max_transient(monkeypatch, mock_transport):
    _patch_sleep(monkeypatch)
    mock_transport.request.side_effect = APIError(0, "connection failed")
    with pytest.raises(APIError):
        _deps(mock_transport).wait_until_running("dep1", region="IN2")
    assert mock_transport.request.call_count == 5  # DEPLOYMENT_POLL_MAX_TRANSIENT_ERRORS


def test_wait_until_running_non_transient_raises_immediately(monkeypatch, mock_transport):
    _patch_sleep(monkeypatch)
    mock_transport.request.side_effect = APIError(400, "bad")
    with pytest.raises(APIError):
        _deps(mock_transport).wait_until_running("dep1", region="IN2")
    assert mock_transport.request.call_count == 1


# ── openai_base_url ─────────────────────────────────────────────────────────────


def test_openai_base_url_exact_string(mock_transport):
    url = _deps(mock_transport).openai_base_url("dep1", region="IN2")
    assert url == "https://serverlessn.jarvislabs.net/openai/dep1"
    assert not url.endswith("/v1")
    assert not url.endswith("/")
    mock_transport.request.assert_not_called()


# ── get / status fan-out ───────────────────────────────────────────────────────


def test_get_fast_path_with_region_skips_fan_out(mock_transport):
    mock_transport.request.return_value = _get_payload()
    deps = _deps(mock_transport)
    dep = deps.get("dep1", region="IN2")
    assert dep.deployment_id == "dep1"
    assert mock_transport.request.call_count == 1


def test_get_not_found_all_regions(mock_transport):
    mock_transport.request.side_effect = NotFoundError("nope")
    with pytest.raises(NotFoundError) as exc:
        _deps(mock_transport).get("dep1")
    assert "not found in any region" in str(exc.value)


def test_status_json_shape_includes_region(mock_transport):
    mock_transport.request.return_value = {"status": "running"}
    status = _deps(mock_transport).status("dep1", region="IN2")
    assert isinstance(status, DeploymentStatus)
    dumped = status.model_dump()
    assert dumped == {"deployment_id": "dep1", "region": "IN2", "status": "running"}


def test_get_uses_cache_fast_path(mock_transport):
    """A cache hit reads the cached region directly (no fan-out)."""
    mock_transport.request.return_value = _get_payload()
    deps = _deps(mock_transport)
    deps._region_cache["dep1"] = INDIA_NOIDA_REGION
    deps.get("dep1")
    assert mock_transport.request.call_count == 1
    assert mock_transport.request.call_args.kwargs["base_url"] == NOIDA_URL


# ── list ────────────────────────────────────────────────────────────────────────


def _list_item(deployment_id, start_time):
    # List items carry their own region; the SDK overrides it with the host region.
    return {
        "deployment_id": deployment_id,
        "name": deployment_id,
        "status": "running",
        "region": INDIA_NOIDA_REGION,
        "start_time": start_time,
        "framework": "vllm",
        "min_workers": 0,
        "max_workers": 1,
        "gpus_to_use": {"gpus": ["L4"]},
        "concurrent_requests": 1,
        "gpus_per_worker": 1,
        "error_message": None,
    }


def _summary(deployment_id, start_time, region):
    return DeploymentSummary(**{**_list_item(deployment_id, start_time), "region": region})


def test_list_merges_tags_and_sorts_newest_first(monkeypatch, mock_transport):
    def fake_fan_out(op, **kwargs):
        noida = [_summary("old", "2026-01-01T00:00:00", INDIA_NOIDA_REGION)]
        chennai = [_summary("new", "2026-06-01T00:00:00", CHENNAI_REGION)]
        return [(INDIA_NOIDA_REGION, noida), (CHENNAI_REGION, chennai)], []

    monkeypatch.setattr(client_mod, "search_serverless_regions", fake_fan_out)
    result = _deps(mock_transport).list()
    assert [d.deployment_id for d in result.deployments] == ["new", "old"]  # newest first
    assert result.deployments[0].model_dump()["region"] == "IN1"


def test_list_partial_failure_records_region_error(monkeypatch, mock_transport):
    def fake_fan_out(op, **kwargs):
        noida = [_summary("a", "2026-01-01T00:00:00", INDIA_NOIDA_REGION)]
        return [(INDIA_NOIDA_REGION, noida)], [(CHENNAI_REGION, "timeout")]

    monkeypatch.setattr(client_mod, "search_serverless_regions", fake_fan_out)
    result = _deps(mock_transport).list()
    assert len(result.deployments) == 1
    assert result.region_errors[0].model_dump() == {"region": "IN1", "error": "timeout"}


def test_list_all_regions_down_hard_fails(monkeypatch, mock_transport):
    monkeypatch.setattr(client_mod, "search_serverless_regions", lambda op, **k: ([], [(INDIA_NOIDA_REGION, "down")]))
    with pytest.raises(JarvislabsError) as exc:
        _deps(mock_transport).list()
    assert not isinstance(exc.value, APIError)


def test_list_caches_regions(mock_transport):
    mock_transport.request.return_value = {"deployments": [_list_item("a", "2026-01-01T00:00:00")]}
    deps = _deps(mock_transport)
    deps.list()
    assert deps._region_cache["a"] == INDIA_NOIDA_REGION


# ── update ──────────────────────────────────────────────────────────────────────


def test_update_empty_patch_raises_before_network(mock_transport):
    with pytest.raises(ValidationError):
        _deps(mock_transport).update("dep1", region="IN2")
    mock_transport.request.assert_not_called()


def test_update_only_sends_mutable_fields(mock_transport):
    # resolve GET, PATCH, re-fetch GET
    mock_transport.request.side_effect = [_get_payload(), {"updated": {}}, _get_payload(name="new")]
    _deps(mock_transport).update("dep1", name="new", region="IN2")
    patch_call = mock_transport.request.call_args_list[1]
    assert patch_call.args[0] == "PATCH"
    assert patch_call.kwargs["json"] == {"name": "new"}


def test_update_not_running_translated(mock_transport):
    mock_transport.request.side_effect = [_get_payload(status="starting"), APIError(409, "not running")]
    with pytest.raises(ValidationError) as exc:
        _deps(mock_transport).update("dep1", name="new", region="IN2")
    msg = str(exc.value)
    assert "recreate to rescale" in msg
    assert "starting" in msg  # current status interpolated


def test_update_resolves_then_refetches(mock_transport):
    mock_transport.request.side_effect = [_get_payload(), {"updated": {}}, _get_payload(idle_timeout=900)]
    dep = _deps(mock_transport).update("dep1", idle_timeout=900, region="IN2")
    assert dep.idle_timeout == 900


# ── delete ──────────────────────────────────────────────────────────────────────


def test_delete_resolves_then_deletes(mock_transport):
    mock_transport.request.side_effect = [_get_payload(), {"message": "Deleting deployment"}]
    assert _deps(mock_transport).delete("dep1", region="IN2") is True
    delete_call = mock_transport.request.call_args_list[1]
    assert delete_call.args[0] == "DELETE"
    assert delete_call.args[1] == "management/dep1"


def test_delete_404_after_resolve_swallowed(mock_transport):
    mock_transport.request.side_effect = [_get_payload(), NotFoundError("already gone")]
    assert _deps(mock_transport).delete("dep1", region="IN2") is True


def test_delete_unknown_id_raises_not_found(mock_transport):
    mock_transport.request.side_effect = NotFoundError("nope")
    with pytest.raises(NotFoundError):
        _deps(mock_transport).delete("dep1")


# ── path assertions ─────────────────────────────────────────────────────────────


def test_401_propagates_immediately(mock_transport):
    mock_transport.request.side_effect = AuthError("bad token")
    with pytest.raises(AuthError):
        _deps(mock_transport).list()


def test_all_ops_hit_only_management_paths(monkeypatch, mock_transport):
    """Across all ops the transport is only called with management/ paths."""
    _patch_sleep(monkeypatch)
    deps = _deps(mock_transport)

    mock_transport.request.return_value = {"deployment_id": "dep1"}
    deps.create(**_create_kwargs())
    mock_transport.request.side_effect = [{"status": "running"}, _get_payload()]
    deps.wait_until_running("dep1", region="IN2")
    mock_transport.request.side_effect = None
    mock_transport.request.return_value = {"deployments": []}
    deps.list()
    mock_transport.request.return_value = _get_payload()
    deps.get("dep1", region="IN2")
    mock_transport.request.return_value = {"status": "running"}
    deps.status("dep1", region="IN2")
    # update: safe-read GET, PATCH, re-fetch GET — all management/.
    mock_transport.request.side_effect = [_get_payload(), {"updated": {"name": "n"}}, _get_payload(name="n")]
    deps.update("dep1", name="n", region="IN2")
    # delete: safe-read GET then DELETE — all management/.
    mock_transport.request.side_effect = [_get_payload(), {"message": "Deleting deployment"}]
    deps.delete("dep1", region="IN2")

    for call in mock_transport.request.call_args_list:
        path = call.args[1]
        assert path.startswith("management/"), f"unexpected path: {path}"
        assert "/openai/" not in path
