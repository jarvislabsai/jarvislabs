from __future__ import annotations

from datetime import datetime

from jarvislabs.constants import INDIA_NOIDA_REGION
from jarvislabs.models import Deployment, DeploymentSummary

GET_PAYLOAD = {
    "deployment_id": "dep-abc",
    "name": "qwen",
    "status": "running",
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
    "env": {"HF_TOKEN": "hf_****"},
    "start_time": "2026-06-08T08:38:45.011322",
    "end_time": None,
    "created_at": "2026-06-08T08:38:45.011322+00:00",
    "updated_at": "2026-06-08T08:40:00+00:00",
    "workers": {
        "total": 2,
        "healthy": 1,
        "provisioning": 1,
        "list": [{"status": "healthy", "last_used": None}, {"status": "provisioning", "last_used": "soon"}],
    },
    "queue_depth": 0,
    "id": 123,  # internal column the API omits; extra="ignore" must drop it
    "user_id": "u1",
}

LIST_ITEM = {
    "deployment_id": "dep-abc",
    "name": "qwen",
    "status": "running",
    "region": INDIA_NOIDA_REGION,
    "start_time": "2026-06-08T08:38:45.011322",
    "framework": "vllm",
    "min_workers": 0,
    "max_workers": 2,
    "gpus_to_use": {"gpus": ["L4"]},
    "concurrent_requests": 4,
    "gpus_per_worker": 1,
    "error_message": None,
}


def test_deployment_parses_full_payload():
    dep = Deployment(**GET_PAYLOAD)
    assert dep.deployment_id == "dep-abc"
    assert dep.status == "running"
    assert dep.workers.total == 2
    assert dep.workers.healthy == 1
    assert len(dep.workers.list) == 2
    assert dep.workers.list[0].status == "healthy"
    assert dep.workers.list[0].last_used is None


def test_deployment_summary_parses_list_item():
    summary = DeploymentSummary(**LIST_ITEM)
    assert summary.deployment_id == "dep-abc"
    assert summary.gpus_to_use == {"gpus": ["L4"]}
    assert summary.error_message is None


def test_region_serializes_to_display_code():
    assert Deployment(**GET_PAYLOAD).model_dump()["region"] == "IN2"
    assert DeploymentSummary(**LIST_ITEM).model_dump()["region"] == "IN2"


def test_model_property_reads_args_model():
    assert Deployment(**GET_PAYLOAD).model == "Qwen/Qwen3-0.6B"


def test_model_property_none_without_args():
    dep = Deployment(deployment_id="d", status="running")
    assert dep.model is None


def test_start_time_parses_naive_and_tz_aware():
    naive = Deployment(deployment_id="d", status="running", start_time="2026-06-08T08:38:45.011322")
    assert isinstance(naive.start_time, datetime)
    assert naive.start_time.tzinfo is None

    aware = Deployment(deployment_id="d", status="running", start_time="2026-06-08T08:38:45+00:00")
    assert isinstance(aware.start_time, datetime)
    assert aware.start_time.tzinfo is not None
