"""CLI tests for jl vpc commands and the --vpc-id instance flags."""

from __future__ import annotations

from contextlib import nullcontext
from io import StringIO
from unittest.mock import MagicMock

import pytest
import typer
from rich.console import Console

from jarvislabs.cli import commands, instance, render
from jarvislabs.models import Vpc, VpcIP

# ── Prompts ──────────────────────────────────────────────────────────────────


def _capture_confirm(monkeypatch, module):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(module.render, "confirm", fake_confirm)
    return captured


def test_vpc_create_prompt(monkeypatch):
    captured = _capture_confirm(monkeypatch, commands)

    with pytest.raises(typer.Exit):
        commands.vpc_create(name="training-net", cidr="10.50.0.0/24", region="in1")

    assert captured["msg"] == "Create VPC (name='training-net', cidr=10.50.0.0/24, region=IN1)?"


def test_vpc_delete_prompt(monkeypatch):
    captured = _capture_confirm(monkeypatch, commands)

    with pytest.raises(typer.Exit):
        commands.vpc_delete(vpc_id="vpc-aaaa")

    assert captured["msg"] == "Delete VPC vpc-aaaa?"


def test_vpc_create_json_passes_raw_region_to_sdk(monkeypatch):
    captured: dict = {}
    mock_client = MagicMock()
    mock_client.vpcs.create.return_value = Vpc(vpc_id="vpc-aaaa", region="india-chennai-01")
    monkeypatch.setattr(commands, "get_client", lambda: mock_client)
    monkeypatch.setattr(commands.render, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(commands.render, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(commands.render, "print_json", lambda data: captured.update({"data": data}))

    commands.vpc_create(name="n", cidr="10.50.0.0/24", region="IN1", yes=True, json_output=True)

    mock_client.vpcs.create.assert_called_once_with(name="n", cidr="10.50.0.0/24", region="IN1")
    assert captured["data"].vpc_id == "vpc-aaaa"


# ── --vpc-id on instance create/resume ───────────────────────────────────────


def _die_capture(monkeypatch):
    def fake_die(msg: str, code: int = 1):
        raise RuntimeError(msg)

    monkeypatch.setattr(instance.render, "die", fake_die)


def test_instance_create_vpc_id_requires_vm(monkeypatch):
    _die_capture(monkeypatch)

    with pytest.raises(RuntimeError, match="--vpc-id requires --vm"):
        instance.instance_create(gpu="A100", vpc_id="vpc-aaaa")


def test_instance_create_prompt_includes_vpc(monkeypatch):
    captured = _capture_confirm(monkeypatch, instance)

    with pytest.raises(typer.Exit):
        instance.instance_create(gpu="A100", vm=True, vpc_id="vpc-aaaa")

    assert "vpc=vpc-aaaa" in captured["msg"]


def test_instance_create_forwards_vpc_id_to_sdk(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(instance, "get_client", lambda: mock_client)
    monkeypatch.setattr(instance.render, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(instance.render, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(instance.render, "print_json", lambda data: None)

    instance.instance_create(gpu="A100", vm=True, vpc_id="vpc-aaaa", yes=True, json_output=True)

    assert mock_client.instances.create.call_args.kwargs["vpc_id"] == "vpc-aaaa"


def test_instance_resume_forwards_vpc_id_to_sdk(monkeypatch):
    mock_client = MagicMock()
    mock_client.instances.resume.return_value = MagicMock(machine_id=10)
    monkeypatch.setattr(instance, "get_client", lambda: mock_client)
    monkeypatch.setattr(instance.render, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(instance.render, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(instance.render, "print_json", lambda data: None)

    instance.instance_resume(machine_id=10, vpc_id="vpc-aaaa", yes=True, json_output=True)

    assert mock_client.instances.resume.call_args.kwargs["vpc_id"] == "vpc-aaaa"


def test_cpu_vm_create_infers_region_from_vpc(monkeypatch):
    mock_client = MagicMock()
    mock_client.vpcs.get.return_value = Vpc(vpc_id="vpc-aaaa", region="india-chennai-01")
    mock_client.instances.resolve_cpu_vm_plan.return_value = (2, 8, "india-chennai-01")
    monkeypatch.setattr(instance, "get_client", lambda: mock_client)
    monkeypatch.setattr(instance.render, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(instance.render, "confirm", lambda *args, **kwargs: False)

    with pytest.raises(typer.Exit):
        instance.instance_create(vm=True, cpu=True, vpc_id="vpc-aaaa")

    mock_client.vpcs.get.assert_called_once_with("vpc-aaaa")
    mock_client.instances.resolve_cpu_vm_plan.assert_called_once_with(vcpus=None, ram=None, region="india-chennai-01")


def test_instance_resume_prompt_includes_vpc(monkeypatch):
    captured = _capture_confirm(monkeypatch, instance)

    with pytest.raises(typer.Exit):
        instance.instance_resume(machine_id=10, vpc_id="vpc-aaaa")

    assert "vpc=vpc-aaaa" in captured["msg"]


# ── Rendering ────────────────────────────────────────────────────────────────


def _capture_render(fn, *args) -> str:
    buf = StringIO()
    old_console = render.stdout_console
    render.stdout_console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    try:
        fn(*args)
    finally:
        render.stdout_console = old_console
    return buf.getvalue()


def test_vpcs_table_renders_fields():
    vpcs = [
        Vpc(
            vpc_id="vpc-aaaa",
            name="training-net",
            region="india-chennai-01",
            cidr="10.50.0.0/24",
            gateway_ip="10.50.0.1",
            is_default=False,
            status="active",
        ),
        Vpc(vpc_id="vpc-bbbb", name="IN2 DEFAULT VPC", region="india-noida-01", is_default=True),
    ]

    out = _capture_render(render.vpcs_table, vpcs)

    assert "vpc-aaaa" in out
    assert "training-net" in out
    assert "IN1" in out
    assert "10.50.0.0/24" in out
    assert "✓" in out


def test_vpc_ips_table_hides_lsp_name():
    ips = [
        VpcIP(
            private_ip="10.50.0.2",
            machine_id=84512,
            status="attached",
            mac_address="42:a1:b2:c3:d4:e5",
            lsp_name="vpc-aaaa-10-50-0-2",
        ),
        VpcIP(private_ip="10.50.0.3", status="reserved"),
    ]

    out = _capture_render(render.vpc_ips_table, ips)

    assert "10.50.0.2" in out
    assert "84512" in out
    assert "attached" in out
    assert "reserved" in out
    assert "lsp" not in out.lower()


def test_vpcs_table_escapes_rich_markup_in_names():
    vpcs = [Vpc(vpc_id="vpc-aaaa", name="my [b]name", region="india-chennai-01")]

    out = _capture_render(render.vpcs_table, vpcs)

    assert "my [b]name" in out


def test_vpc_detail_renders_all_fields():
    vpc = Vpc(
        vpc_id="vpc-aaaa",
        name="training-net",
        region="india-chennai-01",
        cidr="10.50.0.0/24",
        gateway_ip="10.50.0.1",
        is_default=False,
        status="active",
    )

    out = _capture_render(render.vpc_detail, vpc)

    for expected in ("vpc-aaaa", "training-net", "IN1", "10.50.0.0/24", "10.50.0.1", "no", "active"):
        assert expected in out
