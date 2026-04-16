from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from jarvislabs.cli import commands


def test_gpus_shows_eu1_single_gpu_note(monkeypatch):
    info_calls: list[str] = []
    table_calls: list[tuple[list[SimpleNamespace], str]] = []
    availability = [
        SimpleNamespace(gpu_type="H100", region="europe-01", num_free_devices=1, workload_type="container"),
        SimpleNamespace(gpu_type="L4", region="india-noida-01", num_free_devices=8, workload_type="container"),
    ]

    mock_client = SimpleNamespace(
        account=SimpleNamespace(
            gpu_availability=lambda: availability,
            currency=lambda: "USD",
        )
    )

    monkeypatch.setattr(commands, "get_client", lambda: mock_client)
    monkeypatch.setattr(commands.render, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(commands.render, "gpu_table", lambda gpus, currency: table_calls.append((gpus, currency)))
    monkeypatch.setattr(commands.render, "info", lambda msg: info_calls.append(msg))

    commands.gpus(json_output=False)

    assert table_calls == [(availability, "USD")]
    assert info_calls == ["EU1 H100/H200 availability is currently limited to single-GPU launches."]
