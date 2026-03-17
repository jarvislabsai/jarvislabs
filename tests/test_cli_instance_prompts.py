from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import typer

from jarvislabs.cli import instance


def test_instance_create_prompt_includes_storage_and_core_fields(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_create(
            vm=False,
            gpu="RTX5000",
            template="pytorch",
            storage=60,
            name="train-job",
            num_gpus=2,
            region=None,
            http_ports="",
            script_id=None,
            script_args="",
            fs_id=None,
        )

    assert captured["msg"] == "Create instance (gpu=2x RTX5000, template=pytorch, storage=60GB, name='train-job')?"


def test_instance_create_prompt_lists_script_fields_when_provided(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_create(
            vm=False,
            gpu="RTX5000",
            template="pytorch",
            storage=60,
            name="train-job",
            num_gpus=2,
            region=None,
            http_ports="",
            script_id="11",
            script_args="--foo bar",
            fs_id=7,
        )

    assert (
        captured["msg"]
        == "Create instance (gpu=2x RTX5000, template=pytorch, storage=60GB, name='train-job', script_id=11, script_args='--foo bar', fs_id=7)?"
    )


def test_instance_create_prompt_includes_region_when_provided(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_create(
            vm=False,
            gpu="RTX5000",
            template="pytorch",
            storage=60,
            name="train-job",
            num_gpus=2,
            region="IN2",
            http_ports="",
            script_id=None,
            script_args="",
            fs_id=None,
        )

    assert (
        captured["msg"]
        == "Create instance (gpu=2x RTX5000, template=pytorch, storage=60GB, name='train-job', region=IN2)?"
    )


def test_instance_create_prompt_includes_http_ports_when_provided(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_create(
            vm=False,
            gpu="RTX5000",
            template="pytorch",
            storage=60,
            name="train-job",
            num_gpus=2,
            region="IN2",
            http_ports="7860,8080",
            script_id=None,
            script_args="",
            fs_id=None,
        )

    assert (
        captured["msg"]
        == "Create instance (gpu=2x RTX5000, template=pytorch, storage=60GB, name='train-job', region=IN2, http_ports='7860,8080')?"
    )


def test_instance_create_prompt_normalizes_lowercase_region(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_create(
            vm=False,
            gpu="RTX5000",
            template="pytorch",
            storage=60,
            name="train-job",
            num_gpus=1,
            region="in2",
            http_ports="",
            script_id=None,
            script_args="",
            fs_id=None,
        )

    assert (
        captured["msg"]
        == "Create instance (gpu=1x RTX5000, template=pytorch, storage=60GB, name='train-job', region=IN2)?"
    )


def test_instance_create_passes_region_to_client(monkeypatch):
    mock_client = MagicMock()
    mock_client.instances.create.return_value = MagicMock(machine_id=123)
    monkeypatch.setattr(instance, "get_client", lambda: mock_client)
    monkeypatch.setattr(instance.render, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(instance.render, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(instance.render, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(instance.render, "instance_detail", lambda *args, **kwargs: None)

    instance.instance_create(
        vm=False,
        gpu="RTX5000",
        template="pytorch",
        storage=60,
        name="train-job",
        num_gpus=2,
        region="IN2",
        http_ports="7860,8080",
        script_id=None,
        script_args="",
        fs_id=None,
    )

    assert mock_client.instances.create.call_args.kwargs["region"] == "IN2"
    assert mock_client.instances.create.call_args.kwargs["http_ports"] == "7860,8080"


def test_instance_resume_prompt_defaults_to_current_configuration(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_resume(
            machine_id=123,
            gpu=None,
            num_gpus=None,
            storage=None,
            name=None,
            http_ports="",
            script_id=None,
            script_args=None,
            fs_id=None,
        )

    assert captured["msg"] == "Resume instance 123 with current configuration?"


def test_instance_resume_prompt_lists_all_requested_changes(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_resume(
            machine_id=42,
            gpu="H100",
            num_gpus=4,
            storage=120,
            name="new-name",
            http_ports="",
            script_id=None,
            script_args=None,
            fs_id=None,
        )

    assert captured["msg"] == "Resume instance 42 with gpu=H100, num_gpus=4, storage=120GB, name='new-name'?"


def test_instance_resume_prompt_includes_http_ports(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_resume(
            machine_id=42,
            gpu=None,
            num_gpus=None,
            storage=None,
            name=None,
            http_ports="7860,8080",
            script_id=None,
            script_args=None,
            fs_id=None,
        )

    assert captured["msg"] == "Resume instance 42 with http_ports='7860,8080'?"


def test_instance_resume_prompt_includes_script_changes(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_resume(
            machine_id=42,
            gpu=None,
            num_gpus=None,
            storage=None,
            name=None,
            http_ports="",
            script_id="9",
            script_args="--dry-run",
            fs_id=12,
        )

    assert captured["msg"] == "Resume instance 42 with script_id=9, script_args='--dry-run', fs_id=12?"


def test_instance_rename_prompt(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_rename(machine_id=5, name="new-name")

    assert captured["msg"] == "Rename instance 5 to 'new-name'?"


@pytest.mark.parametrize(
    ("fn", "args", "expected"),
    [
        (instance.instance_pause, {"machine_id": 7}, "Pause instance 7?"),
        (instance.instance_destroy, {"machine_id": 9}, "Destroy instance 9? This cannot be undone."),
    ],
)
def test_instance_pause_destroy_prompts(fn, args, expected, monkeypatch):
    """Pause/destroy validate instance existence before prompting."""
    from unittest.mock import MagicMock

    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    mock_client = MagicMock()
    monkeypatch.setattr(instance, "get_client", lambda: mock_client)
    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        fn(**args)

    assert captured["msg"] == expected


# ── --vm flag tests ──────────────────────────────────────────────────────


def test_create_vm_rejects_template_vm(monkeypatch):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(instance.render, "die", fake_die)

    with pytest.raises(SystemExit):
        instance.instance_create(
            gpu="A100",
            vm=False,
            template="vm",
            storage=40,
            name="test",
            num_gpus=1,
            region=None,
            http_ports="",
            script_id=None,
            script_args="",
            fs_id=None,
            yes=True,
            json_output=False,
        )

    assert captured["message"] == "Use --vm instead of --template vm."


def test_create_vm_rejects_http_ports(monkeypatch):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(instance.render, "die", fake_die)

    with pytest.raises(SystemExit):
        instance.instance_create(
            gpu="A100",
            vm=True,
            template="pytorch",
            storage=40,
            name="test",
            num_gpus=1,
            region=None,
            http_ports="8080",
            script_id=None,
            script_args="",
            fs_id=None,
            yes=True,
            json_output=False,
        )

    assert captured["message"] == "--http-ports is not supported with --vm. VMs are SSH-only."


def test_create_vm_rejects_template_with_vm(monkeypatch):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(instance.render, "die", fake_die)

    with pytest.raises(SystemExit):
        instance.instance_create(
            gpu="A100",
            vm=True,
            template="tensorflow",
            storage=40,
            name="test",
            num_gpus=1,
            region=None,
            http_ports="",
            script_id=None,
            script_args="",
            fs_id=None,
            yes=True,
            json_output=False,
        )

    assert captured["message"] == "--vm and --template cannot be used together."


def test_create_vm_sets_template_and_bumps_storage(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(instance.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        instance.instance_create(
            gpu="A100",
            vm=True,
            template="pytorch",
            storage=40,
            name="test",
            num_gpus=1,
            region=None,
            http_ports="",
            script_id=None,
            script_args="",
            fs_id=None,
            yes=False,
            json_output=False,
        )

    assert "template=vm" in captured["msg"]
    assert "storage=100GB" in captured["msg"]
