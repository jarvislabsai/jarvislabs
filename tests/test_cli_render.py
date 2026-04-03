from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from jarvislabs.cli import render


def test_service_url_rows_maps_builtin_and_custom_ports():
    inst = SimpleNamespace(
        url="https://abc123.notebooksn.jarvislabs.net/lab?token=test",
        vs_url="https://abc1230.notebooksn.jarvislabs.net",
        endpoints=[
            "https://abc1231.notebooksn.jarvislabs.net",
            "https://abc1232.notebooksn.jarvislabs.net",
            "https://abc1233.notebooksn.jarvislabs.net",
        ],
        http_ports="7860,8080",
    )

    rows = render._service_url_rows(inst)

    assert [field for field, _ in rows] == [
        "URL",
        "Port 7007",
        "Port 6006",
        "Port 7860",
        "Port 8080",
    ]
    assert "abc123.notebooksn.jarvislabs.net" in rows[0][1]
    assert "abc1230.notebooksn.jarvislabs.net" in rows[1][1]
    assert "abc1231.notebooksn.jarvislabs.net" in rows[2][1]
    assert "abc1232.notebooksn.jarvislabs.net" in rows[3][1]
    assert "abc1233.notebooksn.jarvislabs.net" in rows[4][1]


# ── Helpers for instance_detail() tests ─────────────────────────────────────


def _capture_detail(inst, currency="USD"):
    """Render instance_detail() and return captured text."""
    buf = StringIO()
    old_console = render.stdout_console
    render.stdout_console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    try:
        render.instance_detail(inst, currency)
    finally:
        render.stdout_console = old_console
    return buf.getvalue()


def _make_inst(**overrides):
    """Create a minimal instance namespace with sensible defaults."""
    defaults = {
        "machine_id": 100,
        "name": "test-inst",
        "status": "Running",
        "gpu_type": "A100",
        "num_gpus": 1,
        "template": "pytorch",
        "storage_gb": 40,
        "region": "india-noida-01",
        "cost": 0.50,
        "ssh_command": "ssh root@1.2.3.4 -p 22",
        "public_ip": "1.2.3.4",
        "http_ports": "",
        "url": None,
        "vs_url": None,
        "endpoints": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── instance_detail(): Public IP display ─────────────────────────────────────


def test_instance_detail_shows_public_ip_for_running_vm():
    inst = _make_inst(template="vm", status="Running", public_ip="217.18.55.33")
    output = _capture_detail(inst)
    assert "Public IP" in output
    assert "217.18.55.33" in output


def test_instance_detail_hides_public_ip_for_paused_vm():
    inst = _make_inst(template="vm", status="Paused", public_ip="217.18.55.33")
    output = _capture_detail(inst)
    assert "Public IP" not in output


def test_instance_detail_hides_public_ip_for_running_container():
    inst = _make_inst(template="pytorch", status="Running", public_ip="5.6.7.8")
    output = _capture_detail(inst)
    assert "Public IP" not in output


def test_instance_detail_hides_public_ip_for_vm_with_no_ip():
    inst = _make_inst(template="vm", status="Running", public_ip=None)
    output = _capture_detail(inst)
    assert "Public IP" not in output


# ── instance_detail(): HTTP Ports hidden for VMs ─────────────────────────────


def test_instance_detail_hides_http_ports_for_vm():
    inst = _make_inst(template="vm", status="Running")
    output = _capture_detail(inst)
    assert "HTTP Ports" not in output


def test_instance_detail_shows_http_ports_for_container():
    inst = _make_inst(template="pytorch", status="Running", http_ports="7860,8080")
    output = _capture_detail(inst)
    assert "HTTP Ports" in output


# ── gpu_table(): workload_type split ────────────────────────────────────────


def _capture_gpu_table(gpus, currency="USD"):
    buf = StringIO()
    old_console = render.stdout_console
    render.stdout_console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    try:
        render.gpu_table(gpus, currency)
    finally:
        render.stdout_console = old_console
    return buf.getvalue()


def _make_gpu(**overrides):
    defaults = {
        "gpu_type": "H100",
        "region": "india-noida-01",
        "num_free_devices": 4,
        "price_per_hour": 3.50,
        "vram": "80",
        "cpus_per_gpu": 12,
        "ram_per_gpu": 96,
        "workload_type": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_gpu_table_splits_container_and_vm():
    gpus = [
        _make_gpu(workload_type="container", num_free_devices=8),
        _make_gpu(workload_type="vm", num_free_devices=2),
    ]
    output = _capture_gpu_table(gpus)
    assert "Containers" in output
    assert "VMs" in output


def test_gpu_table_container_only():
    gpus = [_make_gpu(workload_type="container")]
    output = _capture_gpu_table(gpus)
    assert "Containers" in output
    assert "VMs" not in output


def test_gpu_table_vm_only():
    gpus = [_make_gpu(workload_type="vm")]
    output = _capture_gpu_table(gpus)
    assert "VMs" in output
    assert "Containers" not in output


def test_gpu_table_null_appears_in_both():
    gpus = [_make_gpu(workload_type=None)]
    output = _capture_gpu_table(gpus)
    assert "Containers" in output
    assert "VMs" in output


def test_gpu_table_empty(capsys):
    render.gpu_table([], "USD")
    assert "No GPU data available" in capsys.readouterr().err
    assert "Containers" not in capsys.readouterr().out
