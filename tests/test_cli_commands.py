from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from jarvislabs.cli import commands


def test_gpus_shows_eu1_launch_limit_note(monkeypatch):
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
    assert info_calls == ["EU1 H100/H200 launches are currently limited to 1 GPU."]


def test_resources_json_sanitizes_cpu_meta_regions(monkeypatch):
    captured: dict = {}
    meta = SimpleNamespace(
        server_meta=[],
        cpu_meta={
            "region": "india-01",
            "combinations": [
                {
                    "vcpus": 4,
                    "ram_gb": 16,
                    "price": 0.0992,
                    "available": True,
                    "regions": {"india-noida-01": True, "india-01": True},
                }
            ],
        },
    )
    mock_client = SimpleNamespace(
        account=SimpleNamespace(
            resources=lambda: meta,
            gpu_availability_from=lambda meta: [],
            currency=lambda: "USD",
        )
    )

    monkeypatch.setattr(commands, "get_client", lambda: mock_client)
    monkeypatch.setattr(commands.render, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(commands.render, "print_json", lambda data: captured.update(data))

    commands.resources(json_output=True)

    assert "region" not in captured["cpu_meta"]
    assert captured["cpu_meta"]["combinations"][0]["regions"] == {"IN2": True}


def test_resources_non_json_renders_without_error(monkeypatch):
    """Exercise the real render path (tables + legend) so a signature break is caught."""
    from io import StringIO

    from rich.console import Console

    meta = SimpleNamespace(
        server_meta=[],
        cpu_meta={
            "combinations": [{"vcpus": 4, "ram_gb": 16, "price_per_hour": 0.10, "regions": {"india-noida-01": True}}]
        },
    )
    gpus = [
        SimpleNamespace(
            gpu_type="H100",
            region="europe-01",
            num_free_devices=1,
            effective_num_free_devices=1,
            workload_type="container",
            vram="80",
            ram_per_gpu=200,
            cpus_per_gpu=16,
            price_per_hour=2.99,
            spot_price=None,
        ),
    ]
    mock_client = SimpleNamespace(
        account=SimpleNamespace(
            resources=lambda: meta,
            gpu_availability_from=lambda meta: gpus,
            currency=lambda: "USD",
        )
    )
    monkeypatch.setattr(commands, "get_client", lambda: mock_client)
    monkeypatch.setattr(commands.render, "spinner", lambda *args, **kwargs: nullcontext())

    buf = StringIO()
    old_console = commands.render.stdout_console
    commands.render.stdout_console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    try:
        commands.resources(json_output=False)
    finally:
        commands.render.stdout_console = old_console
    output = buf.getvalue()

    assert "GPU Containers" in output
    assert "CPU VMs" in output
    assert "● available" in output
