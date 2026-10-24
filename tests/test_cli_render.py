from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest
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
        "private_ip": None,
        "vpc_id": None,
        "http_ports": "",
        "url": None,
        "vs_url": None,
        "endpoints": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── instance_detail(): Connection details ────────────────────────────────────


def test_instance_detail_shows_ssh_command_when_available():
    inst = _make_inst(ssh_command="ssh root@1.2.3.4")
    output = _capture_detail(inst)
    assert "SSH" in output
    assert "ssh root@1.2.3.4" in output


def test_instance_detail_hides_ssh_command_when_unavailable():
    inst = _make_inst(status="Paused", ssh_command=None)
    output = _capture_detail(inst)
    assert "SSH" not in output


def test_instance_detail_shows_public_ip_for_running_vm():
    inst = _make_inst(template="vm", status="Running", public_ip="203.0.113.10")
    output = _capture_detail(inst)
    assert "Public IP" in output
    assert "203.0.113.10" in output


def test_instance_detail_hides_public_ip_when_unavailable():
    inst = _make_inst(template="vm", status="Paused", public_ip=None)
    output = _capture_detail(inst)
    assert "Public IP" not in output


def test_instance_detail_shows_public_ip_for_running_container():
    inst = _make_inst(template="pytorch", status="Running", public_ip="5.6.7.8")
    output = _capture_detail(inst)
    assert "Public IP" in output
    assert "5.6.7.8" in output


def test_instance_detail_hides_public_ip_for_vm_with_no_ip():
    inst = _make_inst(template="vm", status="Running", public_ip=None)
    output = _capture_detail(inst)
    assert "Public IP" not in output


# ── instance_detail(): Private IP and VPC display ──────────────────────────────────────


def test_instance_detail_shows_private_ip_for_running_container():
    inst = _make_inst(template="pytorch", status="Running", private_ip="10.11.2.10")
    output = _capture_detail(inst)
    assert "Private IP" in output
    assert "10.11.2.10" in output


def test_instance_detail_hides_private_ip_when_unavailable():
    inst = _make_inst(template="pytorch", status="Paused", private_ip=None)
    output = _capture_detail(inst)
    assert "Private IP" not in output


def test_instance_detail_shows_private_ip_and_vpc_for_paused_vpc_vm():
    inst = _make_inst(
        template="vm",
        status="Paused",
        public_ip=None,
        private_ip="10.0.0.2",
        vpc_id="vpc-abc123",
        ssh_command=None,
    )
    output = _capture_detail(inst)
    assert "Private IP" in output
    assert "10.0.0.2" in output
    assert "VPC" in output
    assert "vpc-abc123" in output


def test_instance_detail_hides_private_ip_for_paused_non_vpc_vm():
    inst = _make_inst(template="vm", status="Paused", public_ip=None, private_ip=None, ssh_command=None)
    output = _capture_detail(inst)
    assert "Private IP" not in output
    assert "VPC" not in output


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
    assert "GPU Containers" in output
    assert "GPU VMs" in output
    assert output.count("● available") == 1


def test_gpu_table_container_only():
    gpus = [_make_gpu(workload_type="container")]
    output = _capture_gpu_table(gpus)
    assert "GPU Containers" in output
    assert "GPU VMs" not in output


def test_gpu_table_uses_effective_free_devices_for_on_demand_availability():
    gpus = [
        _make_gpu(
            gpu_type="H100",
            workload_type="container",
            num_free_devices=0,
            effective_num_free_devices=7,
            spot_price=1.19,
        ),
        _make_gpu(gpu_type="A100-80GB", workload_type="container", num_free_devices=0, effective_num_free_devices=0),
    ]

    output = _capture_gpu_table(gpus)

    h100_row = next(line for line in output.splitlines() if "H100" in line)
    a100_row = next(line for line in output.splitlines() if "A100-80GB" in line)
    assert "●" in h100_row
    assert "○" in a100_row


def test_gpu_table_dims_spot_price_when_spot_capacity_is_not_free():
    gpus = [
        _make_gpu(
            gpu_type="H100",
            workload_type="container",
            num_free_devices=0,
            effective_num_free_devices=7,
            spot_price=1.19,
        ),
    ]

    output = _capture_gpu_table(gpus)

    # Spot price is still shown (not "—") when on-demand is available but no spot capacity is free.
    assert "$1.19" in output


def test_gpu_table_sorts_europe_last():
    gpus = [
        _make_gpu(gpu_type="H200", region="europe-01", workload_type="container"),
        _make_gpu(gpu_type="A100", region="india-noida-01", workload_type="container"),
    ]
    output = _capture_gpu_table(gpus)
    # EU row renders below the non-EU row even though it is listed first.
    assert output.index("A100") < output.index("H200")


def test_gpu_table_vm_only():
    gpus = [_make_gpu(workload_type="vm")]
    output = _capture_gpu_table(gpus)
    assert "GPU VMs" in output
    assert "GPU Containers" not in output
    assert "Spot" not in output


def test_gpu_table_null_appears_in_both():
    gpus = [_make_gpu(workload_type=None)]
    output = _capture_gpu_table(gpus)
    assert "GPU Containers" in output
    assert "GPU VMs" in output


def test_gpu_table_empty(capsys):
    render.gpu_table([], "USD")
    assert "No GPU data available" in capsys.readouterr().err
    assert "GPU Containers" not in capsys.readouterr().out


# ── Interactive prompts ──────────────────────────────────────────────────────

HEADINGS = ("NAME", "SLUG", "REGION")


def test_select_row_columns_line_up_with_long_values(monkeypatch):
    shown = []

    def fake_select(message, choices, **kwargs):
        shown.extend(choices)
        return SimpleNamespace(ask=lambda: choices[1].value, application=SimpleNamespace())

    monkeypatch.setattr(render.questionary, "select", fake_select)
    rows = [("GLM-5.2", "glm-5.2", "IN2"), ("Nemotron 3.5 Lightning", "nemotron-3.5-lightning-30b-a3b-bf16", "IN1")]
    assert render.select_row("Pick", HEADINGS, rows, ["a", "b"]) == "a"

    heading, *choices = shown
    lines = [choice.title for choice in choices]  # plain text, so questionary can highlight it
    # Every column starts at the same position in the heading and in each row.
    assert lines[0].index("glm-5.2") == heading.title.index("SLUG")
    assert lines[0].index("IN2") == lines[1].index("IN1") == heading.title.index("REGION")


def _capture_validators(monkeypatch):
    validators = []
    monkeypatch.setattr(
        render.questionary,
        "text",
        lambda message, default, validate, **k: validators.append(validate) or SimpleNamespace(ask=lambda: default),
    )
    return validators


@pytest.mark.parametrize(("text", "ok"), [("0", True), (" 100 ", True), ("-1", False), ("abc", False), ("", False)])
def test_ask_value_number_validation(monkeypatch, text, ok):
    validators = _capture_validators(monkeypatch)
    assert render.ask_value("min_workers", "Min workers:", 2) == 2
    assert (validators[0](text) is True) == ok


def test_ask_value_text_rejects_empty(monkeypatch):
    validators = _capture_validators(monkeypatch)
    assert render.ask_value("name", "Deployment name:", "glm") == "glm"
    assert validators[0]("  ") is not True
    assert validators[0]("mine") is True


def test_ask_value_keeps_flag_given_on_command_line(monkeypatch):
    monkeypatch.setattr(render.cli_options, "option_was_explicit", lambda param: True)
    monkeypatch.setattr(render.questionary, "text", lambda *a, **k: pytest.fail("should not prompt"))
    assert render.ask_value("max_workers", "Max workers:", 1) == 1


def test_select_row_with_nothing_to_choose(monkeypatch):
    errors = []
    monkeypatch.setattr(render, "error", errors.append)
    with pytest.raises(SystemExit):
        render.select_row("Pick", HEADINGS, [], [])
    assert errors == ["There is nothing to choose from."]


def test_user_named_rows_and_links_print_brackets(monkeypatch):
    buf = StringIO()
    monkeypatch.setattr(render, "stdout_console", Console(file=buf, color_system=None, width=200))
    render.ssh_keys_table([SimpleNamespace(key_id="k1", key_name="hello [/]", ssh_key="ssh-ed25519 AAAA")])
    render.scripts_table([SimpleNamespace(script_id=1, script_name="setup [gpu]")])
    render._print_detail_table([("URL", render._link_value("https://x.test/?filter[world]=1"))])
    out = buf.getvalue()
    assert "hello [/]" in out
    assert "setup [gpu]" in out
    assert "?filter[world]=1" in out


@pytest.mark.parametrize("show", [render.error, render.warning, render.info, render.success])
def test_message_text_is_not_read_as_markup(monkeypatch, show):
    buf = StringIO()
    monkeypatch.setattr(render, "console", Console(file=buf, color_system=None))
    show("echo 'hello [world]' [/]")
    assert "echo 'hello [world]' [/]" in buf.getvalue()


def test_confirm_text_is_not_read_as_markup(monkeypatch):
    buf = StringIO()
    console = Console(file=buf, color_system=None)
    monkeypatch.setattr(console, "input", lambda prompt, **k: console.print(prompt, end="") or "n")
    monkeypatch.setattr(render, "console", console)
    assert render.confirm("Delete deployment [/]?") is False
    assert "Delete deployment [/]?" in buf.getvalue()
