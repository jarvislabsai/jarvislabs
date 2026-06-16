"""Unit tests for the Deployments SDK resource."""

from __future__ import annotations

import pytest

from jarvislabs import deployments as client_mod
from jarvislabs.constants import INDIA_NOIDA_REGION
from jarvislabs.deployments import Deployments
from jarvislabs.exceptions import (
    APIError,
    AuthError,
    JarvislabsError,
    NotFoundError,
    RegionResolutionError,
    ValidationError,
)
from jarvislabs.models import Deployment, DeploymentStatus

NOIDA_URL = "https://serverlessn.jarvislabs.net/"
CHENNAI_URL = "https://serverlessc.jarvislabs.net/"


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
    assert _deps(mock_transport).create(**_create_kwargs(region="IN1")) == "dep1"
    assert mock_transport.request.call_args.kwargs["base_url"] == CHENNAI_URL


def test_wait_until_running_times_out_on_unknown_status(mock_transport):
    # An unrecognized non-terminal status must not poll forever — the wall-clock
    # backstop raises with a clear message instead of hanging.
    mock_transport.request.return_value = {"status": "some_new_status"}
    with pytest.raises(JarvislabsError, match="Stopped waiting on deployment"):
        _deps(mock_transport).wait_until_running("dep1", region="IN2", timeout=0)


def test_create_rejects_eu_region(mock_transport):
    with pytest.raises(ValidationError):
        _deps(mock_transport).create(**_create_kwargs(region="EU1"))


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
    result = _deps(mock_transport).wait_until_running("dep1", region="IN2")
    assert isinstance(result, Deployment)
    assert result.status == "running"


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


def test_wait_until_running_vanished_mid_poll_reports_failure(monkeypatch, mock_transport):
    _patch_sleep(monkeypatch)
    # Seen alive once, then consistently 404s: the deployment failed and was
    # cleaned up between polls — report the failure, not a raw "not found".
    mock_transport.request.side_effect = [
        {"status": "downloading_model"},
        *[NotFoundError("gone")] * 5,
    ]
    with pytest.raises(JarvislabsError) as exc:
        _deps(mock_transport).wait_until_running("dep1", region="IN2")
    assert not isinstance(exc.value, NotFoundError)
    assert "failed: reason unavailable" in str(exc.value)


def test_wait_until_running_never_seen_404_raises_not_found(monkeypatch, mock_transport):
    _patch_sleep(monkeypatch)
    # 404 from the very first poll — the id is genuinely unknown; keep "not found".
    mock_transport.request.side_effect = NotFoundError("nope")
    with pytest.raises(NotFoundError):
        _deps(mock_transport).wait_until_running("dep1", region="IN2")


def test_wait_until_running_non_transient_raises_immediately(monkeypatch, mock_transport):
    _patch_sleep(monkeypatch)
    mock_transport.request.side_effect = APIError(400, "bad")
    with pytest.raises(APIError):
        _deps(mock_transport).wait_until_running("dep1", region="IN2")
    assert mock_transport.request.call_count == 1


# ── openai_base_url ─────────────────────────────────────────────────────────────


def test_openai_base_url_exact_string(mock_transport):
    # The /v1 form — paste-ready for any OpenAI client's base_url field.
    url = _deps(mock_transport).openai_base_url("dep1", region="IN2")
    assert url == "https://serverlessn.jarvislabs.net/openai/dep1/v1"
    assert not url.endswith("/")
    mock_transport.request.assert_not_called()


# ── get / status (region given vs. searched) ────────────────────────────────────


def test_get_with_region_goes_straight_there(mock_transport):
    mock_transport.request.return_value = _get_payload()
    dep = _deps(mock_transport).get("dep1", region="IN2")
    assert dep.deployment_id == "dep1"
    assert mock_transport.request.call_count == 1
    assert mock_transport.request.call_args.kwargs["base_url"] == NOIDA_URL


def test_get_without_region_searches_and_finds(mock_transport):
    # Chennai says "never heard of it", Noida has it — and the search already
    # returned the record, so no extra fetch happens.
    mock_transport.request.side_effect = [NotFoundError("nope"), _get_payload()]
    dep = _deps(mock_transport).get("dep1")
    assert dep.deployment_id == "dep1"
    assert mock_transport.request.call_count == 2
    assert dep.region == INDIA_NOIDA_REGION


def test_get_search_stops_at_first_region_that_has_it(mock_transport):
    # Found in Chennai (asked first) — the search returns immediately, no second request.
    mock_transport.request.return_value = _get_payload()
    dep = _deps(mock_transport).get("dep1")
    assert dep.deployment_id == "dep1"
    assert mock_transport.request.call_count == 1


def test_get_not_found_all_regions(mock_transport):
    mock_transport.request.side_effect = NotFoundError("nope")
    with pytest.raises(NotFoundError) as exc:
        _deps(mock_transport).get("dep1")
    assert "not found in any region" in str(exc.value)


def test_get_wrong_region_is_trusted_not_rescued(mock_transport):
    # An explicit --region is trusted: if the deployment isn't there, that's a
    # plain 404 — we don't search other regions behind the user's back.
    mock_transport.request.side_effect = NotFoundError("nope")
    with pytest.raises(NotFoundError):
        _deps(mock_transport).get("dep1", region="IN1")
    assert mock_transport.request.call_count == 1


def test_get_one_region_down_other_has_it_succeeds(mock_transport):
    # Chennai is unreachable but Noida has the deployment — partial outage
    # doesn't block a successful lookup.
    mock_transport.request.side_effect = [APIError(0, "timeout"), _get_payload()]
    dep = _deps(mock_transport).get("dep1")
    assert dep.deployment_id == "dep1"


def test_get_non_transient_error_raises_immediately(mock_transport):
    mock_transport.request.side_effect = APIError(400, "bad")
    with pytest.raises(APIError):
        _deps(mock_transport).get("dep1", region="IN2")
    assert mock_transport.request.call_count == 1


def test_get_unreachable_region_raises_resolution_error(mock_transport):
    # Not found in Chennai, Noida unreachable — can't honestly say "not found".
    mock_transport.request.side_effect = [NotFoundError("nope"), APIError(503, "down")]
    with pytest.raises(RegionResolutionError) as exc:
        _deps(mock_transport).get("dep1")
    msg = str(exc.value)
    assert "could not check" in msg and "IN2" in msg


def test_get_unreachable_then_404_raises_resolution_error(mock_transport):
    # Mirror ordering: Chennai unreachable, Noida says no — same honest answer.
    mock_transport.request.side_effect = [APIError(0, "down"), NotFoundError("nope")]
    with pytest.raises(RegionResolutionError) as exc:
        _deps(mock_transport).get("dep1")
    msg = str(exc.value)
    assert "could not check" in msg and "IN1" in msg


def test_status_json_shape_includes_region(mock_transport):
    mock_transport.request.return_value = {"status": "running"}
    status = _deps(mock_transport).status("dep1", region="IN2")
    assert isinstance(status, DeploymentStatus)
    dumped = status.model_dump()
    assert dumped == {"deployment_id": "dep1", "region": "IN2", "status": "running"}
    # With a region given, status uses the lightweight status endpoint.
    assert mock_transport.request.call_args.args[1] == "management/dep1/status"


def test_status_without_region_searches_and_reads_record(mock_transport):
    # The search returns the full record; status comes from it — no extra call.
    mock_transport.request.side_effect = [NotFoundError("nope"), _get_payload(status="starting")]
    status = _deps(mock_transport).status("dep1")
    assert status.model_dump() == {"deployment_id": "dep1", "region": "IN2", "status": "starting"}
    assert mock_transport.request.call_count == 2


def test_wait_until_running_hintless_reaped_reports_failure(mock_transport, monkeypatch):
    # The search proved the deployment existed; if it vanishes before the first
    # poll, that's a failure — not a confusing "not found".
    _patch_sleep(monkeypatch)
    mock_transport.request.side_effect = [
        NotFoundError("nope"),  # search: Chennai
        _get_payload(status="downloading_model"),  # search: Noida — it exists
        *[NotFoundError("gone")] * 5,  # reaped before the first poll
    ]
    with pytest.raises(JarvislabsError) as exc:
        _deps(mock_transport).wait_until_running("dep1")
    assert not isinstance(exc.value, NotFoundError)
    assert "failed: reason unavailable" in str(exc.value)


def test_wait_until_running_without_region_searches_then_polls(mock_transport, monkeypatch):
    _patch_sleep(monkeypatch)
    mock_transport.request.side_effect = [
        NotFoundError("nope"),  # search: Chennai
        _get_payload(status="downloading_model"),  # search: Noida — found
        {"status": "running"},  # poll
        _get_payload(),  # final GET
    ]
    result = _deps(mock_transport).wait_until_running("dep1")
    assert result.status == "running"
    polled = mock_transport.request.call_args_list[2]
    assert polled.args[1] == "management/dep1/status"
    assert polled.kwargs["base_url"] == NOIDA_URL


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


def test_list_merges_tags_and_sorts_newest_first(mock_transport):
    # Chennai (queried first) has the newer deployment, Noida the older one.
    mock_transport.request.side_effect = [
        {"deployments": [_list_item("new", "2026-06-01T00:00:00")]},  # Chennai
        {"deployments": [_list_item("old", "2026-01-01T00:00:00")]},  # Noida
    ]
    result = _deps(mock_transport).list()
    assert [d.deployment_id for d in result.deployments] == ["new", "old"]  # newest first
    assert result.deployments[0].model_dump()["region"] == "IN1"  # host region wins over payload's


def test_list_partial_failure_records_region_error(mock_transport):
    mock_transport.request.side_effect = [
        APIError(0, "timeout"),  # Chennai down
        {"deployments": [_list_item("a", "2026-01-01T00:00:00")]},  # Noida fine
    ]
    result = _deps(mock_transport).list()
    assert len(result.deployments) == 1
    assert result.region_errors[0].model_dump() == {"region": "IN1", "error": "timeout"}


def test_list_all_regions_down_hard_fails(mock_transport):
    mock_transport.request.side_effect = APIError(503, "down")
    with pytest.raises(JarvislabsError) as exc:
        _deps(mock_transport).list()
    assert not isinstance(exc.value, APIError)
    assert "IN1: down" in str(exc.value) and "IN2: down" in str(exc.value)


def test_list_empty_regions_is_not_an_error(mock_transport):
    # Both regions answered with nothing — that's a clean empty list, not a failure.
    mock_transport.request.return_value = {"deployments": []}
    result = _deps(mock_transport).list()
    assert result.deployments == [] and result.region_errors == []


def test_list_non_transient_error_reraises(mock_transport):
    mock_transport.request.side_effect = APIError(400, "bad request")
    with pytest.raises(APIError):
        _deps(mock_transport).list()


def test_list_region_404_propagates_loudly(mock_transport):
    # /list is a fixed route and can't legitimately 404 — if it ever does,
    # fail loudly instead of silently treating the region as empty.
    mock_transport.request.side_effect = NotFoundError("nope")
    with pytest.raises(NotFoundError):
        _deps(mock_transport).list()


def test_list_sorts_mixed_none_and_tz_aware_start_times(mock_transport):
    # Aware, naive, and missing timestamps must sort without crashing —
    # newest first, missing last.
    mock_transport.request.side_effect = [
        {"deployments": [_list_item("aware", "2026-06-01T00:00:00+00:00"), _list_item("missing", None)]},
        {"deployments": [_list_item("naive", "2026-01-01T00:00:00")]},
    ]
    result = _deps(mock_transport).list()
    assert [d.deployment_id for d in result.deployments] == ["aware", "naive", "missing"]


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


# ── logs ────────────────────────────────────────────────────────────────────────


RAW_STREAM = [
    ": deployment status is running",
    "",
    "data: [worker-12] INFO: server started",
    "",
    ": ping",
    "",
    "data: [worker-12] INFO: request handled",
    "",
    ": all streams ended",
    "",
]


def test_logs_parses_lines_and_notices(mock_transport):
    mock_transport.stream_lines.return_value = iter(RAW_STREAM)
    events = list(_deps(mock_transport).logs("dep1", region="IN2"))
    assert events == [
        ("notice", "deployment status is running"),
        ("log", "[worker-12] INFO: server started"),
        ("log", "[worker-12] INFO: request handled"),
        ("notice", "all streams ended"),
    ]


def test_logs_swallows_pings_and_blanks(mock_transport):
    mock_transport.stream_lines.return_value = iter([": ping", "", ": ping", ""])
    assert list(_deps(mock_transport).logs("dep1", region="IN2")) == []


def test_logs_params_and_path(mock_transport):
    mock_transport.stream_lines.return_value = iter([])
    list(_deps(mock_transport).logs("dep1", region="IN2", tail=50, follow=False))
    args, kwargs = mock_transport.stream_lines.call_args
    assert args[0] == "management/dep1/logs"
    assert kwargs["params"] == {"tail": 50, "follow": False}
    assert kwargs["base_url"] == NOIDA_URL


def test_logs_worker_param_only_when_given(mock_transport):
    mock_transport.stream_lines.return_value = iter([])
    list(_deps(mock_transport).logs("dep1", region="IN2", worker=12))
    assert mock_transport.stream_lines.call_args.kwargs["params"]["worker"] == 12

    mock_transport.stream_lines.return_value = iter([])
    list(_deps(mock_transport).logs("dep1", region="IN2"))
    assert "worker" not in mock_transport.stream_lines.call_args.kwargs["params"]


def test_logs_without_region_searches_first(mock_transport):
    # First region 404s, the second has it; the stream must go to the second.
    mock_transport.request.side_effect = [NotFoundError("nope"), _get_payload()]
    mock_transport.stream_lines.return_value = iter(["data: [worker-1] hi", ""])
    events = list(_deps(mock_transport).logs("dep1"))
    assert events == [("log", "[worker-1] hi")]
    assert mock_transport.stream_lines.call_args.kwargs["base_url"] == NOIDA_URL


def test_logs_unknown_id_raises_before_streaming(mock_transport):
    mock_transport.request.side_effect = [NotFoundError("nope"), NotFoundError("nope")]
    with pytest.raises(NotFoundError, match="not found in any region"):
        list(_deps(mock_transport).logs("ghost"))
    mock_transport.stream_lines.assert_not_called()


def test_logs_is_lazy_until_iterated(mock_transport):
    gen = _deps(mock_transport).logs("dep1", region="IN2")
    mock_transport.stream_lines.assert_not_called()
    mock_transport.stream_lines.return_value = iter([])
    list(gen)
    mock_transport.stream_lines.assert_called_once()


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
