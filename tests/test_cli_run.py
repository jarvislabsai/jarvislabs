from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest
from typer.testing import CliRunner

from jarvislabs.cli import render, run, state
from jarvislabs.cli.app import app

runner = CliRunner()


def test_run_help_shows_direct_start_usage():
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "Start a run directly:" in result.output
    assert "jl run train.py --gpu L4" in result.output
    assert "--http-ports TEXT" in result.output
    assert "--requirements PATH" in result.output


def test_run_start_requires_existing_instance_for_now(monkeypatch):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(run.render, "die", fake_die)

    with pytest.raises(SystemExit) as exc:
        run.run_start(
            SimpleNamespace(args=["train.py"]),
            on=None,
            gpu=None,
            script=None,
            vm=False,
            template="pytorch",
            storage=40,
            name="jl-run",
            num_gpus=1,
            setup=None,
            requirements=None,
            pause=False,
            destroy=False,
            keep=False,
            follow=True,
        )

    assert exc.value.code == 1
    assert (
        captured["message"]
        == "Use --on <instance_id> to run on an existing instance, or --gpu <type> to create a fresh one."
    )


def test_run_start_prompt_includes_region_when_provided(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(run.render, "confirm", fake_confirm)

    with pytest.raises(click.exceptions.Exit):
        run.run_start(
            SimpleNamespace(args=["train.py"]),
            on=None,
            gpu="L4",
            region="IN2",
            script=None,
            vm=False,
            template="pytorch",
            storage=40,
            name="jl-run",
            num_gpus=1,
            http_ports="",
            setup=None,
            requirements=None,
            pause=False,
            destroy=False,
            keep=True,
            follow=True,
        )

    assert (
        captured["msg"]
        == "Create 1x L4 instance for jl run (template=pytorch, storage=40GB, name='jl-run', region=IN2)?"
    )


def test_run_start_prompt_includes_http_ports_when_provided(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(run.render, "confirm", fake_confirm)

    with pytest.raises(click.exceptions.Exit):
        run.run_start(
            SimpleNamespace(args=["train.py"]),
            on=None,
            gpu="L4",
            region="IN1",
            script=None,
            vm=False,
            template="pytorch",
            storage=40,
            name="jl-run",
            num_gpus=1,
            http_ports="7860",
            setup=None,
            requirements=None,
            pause=False,
            destroy=False,
            keep=True,
            follow=True,
        )

    assert (
        captured["msg"]
        == "Create 1x L4 instance for jl run (template=pytorch, storage=40GB, name='jl-run', region=IN1, http_ports='7860')?"
    )


def test_run_start_passes_region_to_client(monkeypatch):
    mock_client = MagicMock()
    mock_client.instances.create.return_value = MagicMock(machine_id=321)
    monkeypatch.setattr(run, "get_client", lambda: mock_client)
    monkeypatch.setattr(run.render, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(run.render, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(run.render, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(run.render, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(run.render, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "_start_managed_run", lambda *args, **kwargs: ("r_test", None))

    run.run_start(
        SimpleNamespace(args=["train.py"]),
        on=None,
        gpu="L4",
        region="EU1",
        script=None,
        vm=False,
        template="pytorch",
        storage=40,
        name="jl-run",
        num_gpus=1,
        http_ports="7860,8080",
        setup=None,
        requirements=None,
        pause=False,
        destroy=False,
        keep=True,
        follow=False,
    )

    assert mock_client.instances.create.call_args.kwargs["region"] == "EU1"
    assert mock_client.instances.create.call_args.kwargs["http_ports"] == "7860,8080"


def test_region_label_uses_ui_consistent_codes():
    assert render._region_label("india-01") == "IN1"
    assert render._region_label("india-noida-01") == "IN2"
    assert render._region_label("europe-01") == "EU1"
    assert render._region_label("future-01") == "future-01"


def test_run_start_rejects_region_with_on(monkeypatch):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(run.render, "die", fake_die)

    with pytest.raises(SystemExit):
        run.run_start(
            SimpleNamespace(args=["train.py"]),
            on=123,
            gpu=None,
            region="IN2",
            script=None,
            vm=False,
            template="pytorch",
            storage=40,
            name="jl-run",
            num_gpus=1,
            setup=None,
            requirements=None,
            pause=False,
            destroy=False,
            keep=False,
            follow=True,
        )

    assert captured["message"] == "--region is only supported with --gpu for fresh instances."


@pytest.mark.parametrize("flag", ["pause", "destroy"])
def test_run_start_rejects_json_lifecycle_for_fresh_runs(monkeypatch, flag):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(run.render, "die", fake_die)

    kwargs = {
        "pause": False,
        "destroy": False,
        "keep": False,
    }
    kwargs[flag] = True

    with pytest.raises(SystemExit):
        run.run_start(
            SimpleNamespace(args=["train.py"]),
            on=None,
            gpu="L4",
            region=None,
            script=None,
            vm=False,
            template="pytorch",
            storage=40,
            name="jl-run",
            num_gpus=1,
            http_ports="",
            setup=None,
            requirements=None,
            follow=True,
            json_output=True,
            **kwargs,
        )

    assert captured["message"] == (
        f"--{flag} is not supported with --json for fresh runs.\n\n"
        "--json is mainly meant for agent workflows. It prints the run details and returns right away, "
        f"so the run becomes detached from this CLI session. Because the CLI is no longer attached, it cannot "
        f"{flag} the instance when the run finishes.\n\n"
        "What to do instead:\n"
        f"  Agent workflow: use --keep --json, then have the agent watch the run and call jl {flag} <machine_id> when it is done.\n"
        f"  Human workflow: drop --json and use the default mode where the CLI stays attached to the run if you want it to apply --{flag} after the run finishes."
    )


# ── --vm flag tests for run_start ─────────────────────────────────────────


def test_run_start_vm_rejects_template_vm(monkeypatch):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(run.render, "die", fake_die)

    with pytest.raises(SystemExit):
        run.run_start(
            SimpleNamespace(args=["train.py"]),
            on=None,
            gpu="A100",
            region=None,
            script=None,
            vm=False,
            template="vm",
            storage=40,
            name="jl-run",
            num_gpus=1,
            http_ports="",
            setup=None,
            requirements=None,
            pause=False,
            destroy=False,
            keep=True,
            follow=False,
            yes=True,
            json_output=False,
        )

    assert captured["message"] == "Use --vm instead of --template vm."


def test_run_start_vm_rejects_http_ports(monkeypatch):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(run.render, "die", fake_die)

    with pytest.raises(SystemExit):
        run.run_start(
            SimpleNamespace(args=["train.py"]),
            on=None,
            gpu="A100",
            region=None,
            script=None,
            vm=True,
            template="pytorch",
            storage=40,
            name="jl-run",
            num_gpus=1,
            http_ports="8080",
            setup=None,
            requirements=None,
            pause=False,
            destroy=False,
            keep=True,
            follow=False,
            yes=True,
            json_output=False,
        )

    assert captured["message"] == "--http-ports is not supported with --vm. VMs are SSH-only."


def test_run_start_vm_rejects_template_conflict(monkeypatch):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(run.render, "die", fake_die)

    with pytest.raises(SystemExit):
        run.run_start(
            SimpleNamespace(args=["train.py"]),
            on=None,
            gpu="A100",
            region=None,
            script=None,
            vm=True,
            template="tensorflow",
            storage=40,
            name="jl-run",
            num_gpus=1,
            http_ports="",
            setup=None,
            requirements=None,
            pause=False,
            destroy=False,
            keep=True,
            follow=False,
            yes=True,
            json_output=False,
        )

    assert captured["message"] == "--vm and --template cannot be used together."


def test_run_start_vm_rejects_on_flag(monkeypatch):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(run.render, "die", fake_die)

    with pytest.raises(SystemExit):
        run.run_start(
            SimpleNamespace(args=["train.py"]),
            on=12345,
            gpu=None,
            region=None,
            script=None,
            vm=True,
            template="pytorch",
            storage=40,
            name="jl-run",
            num_gpus=1,
            http_ports="",
            setup=None,
            requirements=None,
            pause=False,
            destroy=False,
            keep=False,
            follow=True,
            yes=True,
            json_output=False,
        )

    assert captured["message"] == "--vm is only supported with --gpu for fresh instances."


def test_build_run_spec_for_python_file(monkeypatch, tmp_path):
    source = tmp_path / "train.py"
    source.write_text("print('hi')\n")

    spec = run._build_run_spec(str(source), ["--epochs", "5"])

    assert spec.target_kind == "file"
    assert spec.local_target == source
    assert spec.remote_target == f"/home/{source.stem}/{source.name}"
    assert spec.working_dir == f"/home/{source.stem}"
    assert spec.launch_command == f"python3 {source.name} --epochs 5"


def test_build_run_spec_for_bash_file(tmp_path):
    source = tmp_path / "train.sh"
    source.write_text("echo hi\n")

    spec = run._build_run_spec(str(source), ["--fast"])

    assert spec.target_kind == "file"
    assert spec.remote_target == "/home/train/train.sh"
    assert spec.working_dir == "/home/train"
    assert spec.launch_command == "bash train.sh --fast"


def test_build_run_spec_for_directory_uses_stable_home_path(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("print('hi')\n")

    spec = run._build_run_spec(str(project), ["python", "train.py"])

    assert spec.target_kind == "directory"
    assert spec.local_target == project
    assert spec.remote_target == "/home/project"
    assert spec.working_dir == "/home/project"
    assert spec.launch_command == "python train.py"


def test_build_run_spec_for_directory_with_script_option(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("print('hi')\n")

    spec = run._build_run_spec(str(project), ["--epochs", "5"], script_path="scripts/train.py")

    assert spec.target_kind == "directory"
    assert spec.remote_target == "/home/project"
    assert spec.working_dir == "/home/project"
    assert spec.launch_command == "python3 scripts/train.py --epochs 5"


def test_build_run_spec_for_directory_with_bash_script_option(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.sh").write_text("echo hi\n")

    spec = run._build_run_spec(str(project), ["--epochs", "5"], script_path="scripts/train.sh")

    assert spec.target_kind == "directory"
    assert spec.remote_target == "/home/project"
    assert spec.working_dir == "/home/project"
    assert spec.launch_command == "bash scripts/train.sh --epochs 5"


def test_build_run_spec_treats_non_path_target_as_raw_command():
    spec = run._build_run_spec("python", ["-c", "print('hi')"])

    assert spec.target_kind == "command"
    assert spec.local_target is None
    assert spec.remote_target is None
    assert spec.working_dir is None
    assert spec.launch_command == "python -c 'print('\"'\"'hi'\"'\"')'"


def test_build_run_spec_rejects_directory_without_command(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(run.render, "die", fake_die)

    with pytest.raises(SystemExit):
        run._build_run_spec(str(project), [])

    assert (
        captured["message"]
        == "Directory targets require --script <path> or a command after --. Example: jl run . --script train.py --gpu L4"
    )


def test_build_run_spec_rejects_script_option_for_file(monkeypatch, tmp_path):
    source = tmp_path / "train.py"
    source.write_text("print('hi')\n")
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(run.render, "die", fake_die)

    with pytest.raises(SystemExit):
        run._build_run_spec(str(source), [], script_path="other.py")

    assert captured["message"] == "--script can only be used with a directory target."


# ── _detect_requirements tests ────────────────────────────────────────────


def test_detect_requirements_finds_pyproject_with_project_table(tmp_path):
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'foo'\ndependencies = ['requests']\n")
    spec = run.RunSpec(
        target_kind="directory",
        local_target=project,
        remote_target="/home/myproject",
        working_dir="/home/myproject",
        launch_command="python3 train.py",
    )
    assert run._detect_requirements(spec) == "pyproject.toml"


def test_detect_requirements_skips_tool_only_pyproject(tmp_path):
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "pyproject.toml").write_text("[tool.ruff]\nline-length = 88\n")
    spec = run.RunSpec(
        target_kind="directory",
        local_target=project,
        remote_target="/home/myproject",
        working_dir="/home/myproject",
        launch_command="python3 train.py",
    )
    assert run._detect_requirements(spec) is None


def test_detect_requirements_falls_through_to_requirements_txt(tmp_path):
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "pyproject.toml").write_text("[tool.ruff]\nline-length = 88\n")
    (project / "requirements.txt").write_text("requests\n")
    spec = run.RunSpec(
        target_kind="directory",
        local_target=project,
        remote_target="/home/myproject",
        working_dir="/home/myproject",
        launch_command="python3 train.py",
    )
    assert run._detect_requirements(spec) == "requirements.txt"


def test_detect_requirements_finds_requirements_txt(tmp_path):
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "requirements.txt").write_text("torch\n")
    spec = run.RunSpec(
        target_kind="directory",
        local_target=project,
        remote_target="/home/myproject",
        working_dir="/home/myproject",
        launch_command="python3 train.py",
    )
    assert run._detect_requirements(spec) == "requirements.txt"


def test_detect_requirements_returns_none_when_no_files(tmp_path):
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "train.py").write_text("print('hi')\n")
    spec = run.RunSpec(
        target_kind="directory",
        local_target=project,
        remote_target="/home/myproject",
        working_dir="/home/myproject",
        launch_command="python3 train.py",
    )
    assert run._detect_requirements(spec) is None


def test_detect_requirements_skips_file_targets(tmp_path):
    source = tmp_path / "train.py"
    source.write_text("print('hi')\n")
    spec = run.RunSpec(
        target_kind="file",
        local_target=source,
        remote_target="/home/train/train.py",
        working_dir="/home/train",
        launch_command="python3 train.py",
    )
    assert run._detect_requirements(spec) is None


def test_detect_requirements_skips_command_targets():
    spec = run.RunSpec(
        target_kind="command",
        local_target=None,
        remote_target=None,
        working_dir=None,
        launch_command="python3 train.py",
    )
    assert run._detect_requirements(spec) is None


def test_detect_requirements_pyproject_takes_priority_over_requirements(tmp_path):
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'foo'\ndependencies = ['torch']\n")
    (project / "requirements.txt").write_text("requests\n")
    spec = run.RunSpec(
        target_kind="directory",
        local_target=project,
        remote_target="/home/myproject",
        working_dir="/home/myproject",
        launch_command="python3 train.py",
    )
    assert run._detect_requirements(spec) == "pyproject.toml"


def test_write_remote_text_streams_content_over_ssh(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(parts: list[str], *, input: str, text: bool, check: bool):
        captured["parts"] = parts
        captured["input"] = input
        captured["text"] = text
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    status = run._write_remote_text(["ssh", "root@example.com"], "/home/jl-runs/r1/meta.json", '{"ok": true}\n')

    assert status == 0
    assert captured["parts"] == [
        "ssh",
        "root@example.com",
        run._build_remote_command("mkdir -p /home/jl-runs/r1 && cat > /home/jl-runs/r1/meta.json"),
    ]
    assert captured["input"] == '{"ok": true}\n'
    assert captured["text"] is True
    assert captured["check"] is False


def test_prepare_remote_target_for_directory_replaces_stable_remote_path(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("print('hi')\n")
    inst = SimpleNamespace(machine_id=123, ssh_command="ssh -p 2222 root@example.com")
    captured: dict[str, object] = {}

    def fake_run_remote(ssh_parts: list[str], script: str):
        captured["prep_script"] = script
        return 0

    def fake_sync(ssh_command: str, source, dest: str):
        captured["sync"] = {
            "ssh_command": ssh_command,
            "source": source,
            "dest": dest,
        }
        return 0

    monkeypatch.setattr(
        run, "_ensure_remote_rsync", lambda machine_id, ssh_parts: captured.setdefault("rsync", machine_id)
    )
    monkeypatch.setattr(run, "_run_remote", fake_run_remote)
    monkeypatch.setattr(run, "_sync_directory_to_remote", fake_sync)
    monkeypatch.setattr(run.render, "info", lambda message: None)

    spec = run.RunSpec(
        target_kind="directory",
        local_target=project,
        remote_target="/home/project",
        working_dir="/home/project",
        launch_command="python3 train.py",
    )

    run._prepare_remote_target(inst, ["ssh", "root@example.com"], spec)

    assert captured["rsync"] == 123
    assert captured["prep_script"] == "mkdir -p /home/project"
    assert captured["sync"] == {
        "ssh_command": "ssh -p 2222 root@example.com",
        "source": project,
        "dest": "/home/project",
    }


def test_start_managed_run_for_file_launches_detached_and_saves_local_record(monkeypatch, tmp_path):
    source = tmp_path / "train.py"
    source.write_text("print('hi')\n")
    inst = SimpleNamespace(machine_id=123, ssh_command="ssh -p 2222 root@example.com")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        run, "_wait_for_ssh_ready", lambda machine_id: (inst, ["ssh", "-p", "2222", "root@example.com"])
    )
    monkeypatch.setattr(run, "_make_run_id", lambda: "r_test123")
    monkeypatch.setattr(run, "_run_remote", lambda ssh_parts, script: 0)
    monkeypatch.setattr(run, "_prepare_remote_target", lambda inst, ssh_parts, spec: captured.setdefault("spec", spec))
    monkeypatch.setattr(
        run,
        "_write_remote_wrapper",
        lambda ssh_parts, paths, spec: captured.setdefault("wrapper", paths.remote_wrapper),
    )
    monkeypatch.setattr(
        run,
        "_write_remote_metadata",
        lambda ssh_parts, inst, paths, spec: captured.setdefault("meta", paths.remote_meta),
    )
    monkeypatch.setattr(run, "_start_remote_run", lambda ssh_parts, paths: 4242)
    monkeypatch.setattr(run, "_save_local_run", lambda record: captured.setdefault("record", record))
    monkeypatch.setattr(run.render, "info", lambda message: None)
    monkeypatch.setattr(run.render, "success", lambda message: captured.setdefault("success", message))
    monkeypatch.setattr(state, "json_output", False)

    run_id, exit_code = run._start_managed_run(
        str(source),
        123,
        ["--epochs", "5"],
        follow=False,
        instance_origin="existing",
        lifecycle_policy="none",
    )

    spec = captured["spec"]
    assert spec.target_kind == "file"
    assert spec.remote_target == "/home/train/train.py"
    assert spec.launch_command == "python3 train.py --epochs 5"
    record = captured["record"]
    assert record.run_id == "r_test123"
    assert record.machine_id == 123
    assert record.remote_target == "/home/train/train.py"
    assert record.remote_log == "/home/jl-runs/r_test123/output.log"
    assert record.instance_origin == "existing"
    assert record.lifecycle_policy == "none"
    assert captured["success"] == "Started run r_test123 on instance 123 (launcher pid 4242)."
    assert run_id == "r_test123"
    assert exit_code is None


def test_start_managed_run_json_mode_returns_summary(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("print('hi')\n")
    inst = SimpleNamespace(machine_id=123, ssh_command="ssh -p 2222 root@example.com")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        run, "_wait_for_ssh_ready", lambda machine_id: (inst, ["ssh", "-p", "2222", "root@example.com"])
    )
    monkeypatch.setattr(run, "_make_run_id", lambda: "r_json")
    monkeypatch.setattr(run, "_run_remote", lambda ssh_parts, script: 0)
    monkeypatch.setattr(run, "_prepare_remote_target", lambda inst, ssh_parts, spec: None)
    monkeypatch.setattr(run, "_write_remote_wrapper", lambda ssh_parts, paths, spec: None)
    monkeypatch.setattr(run, "_write_remote_metadata", lambda ssh_parts, inst, paths, spec: None)
    monkeypatch.setattr(run, "_start_remote_run", lambda ssh_parts, paths: 4242)
    monkeypatch.setattr(run, "_save_local_run", lambda record: None)
    monkeypatch.setattr(run.render, "print_json", lambda payload: captured.setdefault("payload", payload))
    monkeypatch.setattr(state, "json_output", True)

    run_id, exit_code = run._start_managed_run(
        str(project),
        123,
        ["python", "train.py"],
        follow=False,
        instance_origin="fresh",
        lifecycle_policy="keep",
    )

    assert captured["payload"] == {
        "run_id": "r_json",
        "machine_id": 123,
        "launcher_pid": 4242,
        "remote_log": "/home/jl-runs/r_json/output.log",
        "remote_exit_code": "/home/jl-runs/r_json/exit_code",
        "target_kind": "directory",
        "remote_target": "/home/project",
        "command": (
            "command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; } && "
            'export PATH="$HOME/.local/bin:$PATH" && '
            "(test -d .venv || uv venv --system-site-packages --seed .venv) && "
            ". .venv/bin/activate && "
            'echo "[jl] No dependency file detected, using template packages" && '
            "python train.py"
        ),
        "instance_origin": "fresh",
        "lifecycle_policy": "keep",
    }
    assert run_id == "r_json"
    assert exit_code is None


def test_iter_local_runs_sorts_newest_first(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "_LOCAL_RUNS_ROOT", tmp_path)
    older = run.LocalRunRecord(
        run_id="r_old",
        machine_id=1,
        target_kind="file",
        local_target="/tmp/old.py",
        remote_target="/home/old/old.py",
        working_dir="/home/old",
        remote_log="/home/jl-runs/r_old/output.log",
        remote_pid="/home/jl-runs/r_old/pid",
        remote_exit_code="/home/jl-runs/r_old/exit_code",
        launch_command="python3 old.py",
        started_at="2026-03-08T10:00:00+00:00",
    )
    newer = run.LocalRunRecord(
        run_id="r_new",
        machine_id=2,
        target_kind="directory",
        local_target="/tmp/proj",
        remote_target="/home/proj",
        working_dir="/home/proj",
        remote_log="/home/jl-runs/r_new/output.log",
        remote_pid="/home/jl-runs/r_new/pid",
        remote_exit_code="/home/jl-runs/r_new/exit_code",
        launch_command="python3 train.py",
        started_at="2026-03-09T10:00:00+00:00",
        instance_origin="fresh",
        lifecycle_policy="keep",
    )
    (tmp_path / "r_old.json").write_text(json.dumps(run.asdict(older)))
    (tmp_path / "r_new.json").write_text(json.dumps(run.asdict(newer)))

    records = run._iter_local_runs()

    assert [record.run_id for record in records] == ["r_new", "r_old"]


def test_get_run_snapshot_reports_running(monkeypatch):
    record = run.LocalRunRecord(
        run_id="r_run",
        machine_id=123,
        target_kind="file",
        local_target="/tmp/train.py",
        remote_target="/home/train/train.py",
        working_dir="/home/train",
        remote_log="/home/jl-runs/r_run/output.log",
        remote_pid="/home/jl-runs/r_run/pid",
        remote_exit_code="/home/jl-runs/r_run/exit_code",
        launch_command="python3 train.py",
        started_at="2026-03-09T10:00:00+00:00",
    )
    monkeypatch.setattr(
        run,
        "_get_instance",
        lambda machine_id: SimpleNamespace(status="Running", ssh_command="ssh root@example.com", cost=0.79),
    )
    monkeypatch.setattr(run, "_ssh_parts_from_instance", lambda inst: ["ssh", "root@example.com"])
    monkeypatch.setattr(run, "_fetch_exit_code_path", lambda ssh_parts, path: None)

    snapshot = run._get_run_snapshot(record)

    assert snapshot.state == "running"
    assert snapshot.exit_code is None
    assert snapshot.instance_cost == 0.79


def test_get_run_snapshot_reports_instance_paused(monkeypatch):
    record = run.LocalRunRecord(
        run_id="r_paused",
        machine_id=123,
        target_kind="file",
        local_target="/tmp/train.py",
        remote_target="/home/train/train.py",
        working_dir="/home/train",
        remote_log="/home/jl-runs/r_paused/output.log",
        remote_pid="/home/jl-runs/r_paused/pid",
        remote_exit_code="/home/jl-runs/r_paused/exit_code",
        launch_command="python3 train.py",
        started_at="2026-03-09T10:00:00+00:00",
    )
    monkeypatch.setattr(
        run,
        "_get_instance",
        lambda machine_id: SimpleNamespace(status="Paused", ssh_command="ssh root@example.com", cost=0.05),
    )

    snapshot = run._get_run_snapshot(record)

    assert snapshot.state == "instance-paused"
    assert snapshot.instance_cost == 0.05


def test_run_logs_json_returns_content(monkeypatch):
    record = run.LocalRunRecord(
        run_id="r_logs",
        machine_id=123,
        target_kind="file",
        local_target="/tmp/train.py",
        remote_target="/home/train/train.py",
        working_dir="/home/train",
        remote_log="/home/jl-runs/r_logs/output.log",
        remote_pid="/home/jl-runs/r_logs/pid",
        remote_exit_code="/home/jl-runs/r_logs/exit_code",
        launch_command="python3 train.py",
        started_at="2026-03-09T10:00:00+00:00",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(run, "_resolve_run_ssh", lambda run_id: (record, ["ssh", "root@example.com"]))
    monkeypatch.setattr(
        run,
        "_tail_remote_log",
        lambda ssh_parts, remote_log, *, follow, tail: SimpleNamespace(stdout="hello\n", returncode=0),
    )
    monkeypatch.setattr(run, "_fetch_exit_code_path", lambda ssh_parts, path: 0)
    monkeypatch.setattr(run.render, "print_json", lambda payload: captured.setdefault("payload", payload))
    monkeypatch.setattr(state, "json_output", True)

    run.run_logs("r_logs", follow=False, tail=5, json_output=True)

    assert captured["payload"] == {
        "run_id": "r_logs",
        "machine_id": 123,
        "remote_log": "/home/jl-runs/r_logs/output.log",
        "content": "hello\n",
        "run_exit_code": 0,
    }


def test_run_logs_follow_shows_followups(monkeypatch):
    record = run.LocalRunRecord(
        run_id="r_logs",
        machine_id=123,
        target_kind="file",
        local_target="/tmp/train.py",
        remote_target="/home/train/train.py",
        working_dir="/home/train",
        remote_log="/home/jl-runs/r_logs/output.log",
        remote_pid="/home/jl-runs/r_logs/pid",
        remote_exit_code="/home/jl-runs/r_logs/exit_code",
        launch_command="python3 train.py",
        started_at="2026-03-09T10:00:00+00:00",
    )
    captured: list[str] = []

    monkeypatch.setattr(run, "_resolve_run_ssh", lambda run_id: (record, ["ssh", "root@example.com"]))
    monkeypatch.setattr(run, "_tail_remote_log", lambda ssh_parts, remote_log, *, follow, tail: 0)
    monkeypatch.setattr(run.render, "info", lambda message: captured.append(message))
    monkeypatch.setattr(run.render.console, "print", lambda message: None)
    monkeypatch.setattr(state, "json_output", False)

    with pytest.raises(SystemExit) as exc:
        run.run_logs("r_logs", follow=True, tail=None)

    assert exc.value.code == 0
    assert captured[0] == "Following logs for run r_logs on instance 123. Press Ctrl+C to stop streaming."
    assert "Showing the latest 20 log lines first." in captured[1]
    assert "Run ID: r_logs" in captured[2]


def test_run_list_does_not_refresh_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "_LOCAL_RUNS_ROOT", tmp_path)
    record = run.LocalRunRecord(
        run_id="r_saved",
        machine_id=123,
        target_kind="file",
        local_target="/tmp/train.py",
        remote_target="/home/train/train.py",
        working_dir="/home/train",
        remote_log="/home/jl-runs/r_saved/output.log",
        remote_pid="/home/jl-runs/r_saved/pid",
        remote_exit_code="/home/jl-runs/r_saved/exit_code",
        launch_command="python3 train.py",
        started_at="2026-03-09T10:00:00+00:00",
    )
    (tmp_path / "r_saved.json").write_text(json.dumps(run.asdict(record)))

    monkeypatch.setattr(
        run,
        "_get_run_snapshot",
        lambda record: (_ for _ in ()).throw(AssertionError("run list should not refresh by default")),
    )
    monkeypatch.setattr(run.render, "stdout_console", SimpleNamespace(print=lambda value: None))
    monkeypatch.setattr(state, "json_output", False)

    run.run_list(refresh=False, limit=None, status_filter=None)


def test_ensure_remote_rsync_rejects_missing_local_binary(monkeypatch):
    captured: dict[str, str] = {}

    def fake_die(message: str, code: int = 1) -> None:
        captured["message"] = message
        raise SystemExit(code)

    monkeypatch.setattr(run.shutil, "which", lambda name: None)
    monkeypatch.setattr(run.render, "die", fake_die)

    with pytest.raises(SystemExit):
        run._ensure_remote_rsync(123, ["ssh", "root@example.com"])

    assert captured["message"] == "rsync is required locally for directory runs. Install rsync and try again."


def test_run_stop_reports_completed_run(monkeypatch):
    record = run.LocalRunRecord(
        run_id="r_done",
        machine_id=123,
        target_kind="file",
        local_target="/tmp/train.py",
        remote_target="/home/train/train.py",
        working_dir="/home/train",
        remote_log="/home/jl-runs/r_done/output.log",
        remote_pid="/home/jl-runs/r_done/pid",
        remote_exit_code="/home/jl-runs/r_done/exit_code",
        launch_command="python3 train.py",
        started_at="2026-03-09T10:00:00+00:00",
    )
    captured: list[str] = []

    monkeypatch.setattr(run, "_load_local_run", lambda run_id: record)
    monkeypatch.setattr(
        run,
        "_get_run_snapshot",
        lambda record: run.RunStatusSnapshot(
            run_id=record.run_id,
            machine_id=record.machine_id,
            target_kind=record.target_kind,
            started_at=record.started_at,
            state="succeeded",
            instance_status="Running",
            exit_code=0,
            remote_log=record.remote_log,
            lifecycle_policy=record.lifecycle_policy,
        ),
    )
    monkeypatch.setattr(run.render, "info", lambda message: captured.append(message))
    monkeypatch.setattr(run.render.console, "print", lambda message: None)
    monkeypatch.setattr(state, "json_output", False)

    run.run_stop("r_done")

    assert captured[0] == "Run r_done already finished with state succeeded."


def test_run_stop_sends_term_to_running_process(monkeypatch):
    record = run.LocalRunRecord(
        run_id="r_live",
        machine_id=123,
        target_kind="file",
        local_target="/tmp/train.py",
        remote_target="/home/train/train.py",
        working_dir="/home/train",
        remote_log="/home/jl-runs/r_live/output.log",
        remote_pid="/home/jl-runs/r_live/pid",
        remote_exit_code="/home/jl-runs/r_live/exit_code",
        launch_command="python3 train.py",
        started_at="2026-03-09T10:00:00+00:00",
    )
    captured: dict[str, str] = {}

    monkeypatch.setattr(run, "_load_local_run", lambda run_id: record)
    monkeypatch.setattr(
        run,
        "_get_run_snapshot",
        lambda record: run.RunStatusSnapshot(
            run_id=record.run_id,
            machine_id=record.machine_id,
            target_kind=record.target_kind,
            started_at=record.started_at,
            state="running",
            instance_status="Running",
            exit_code=None,
            remote_log=record.remote_log,
            lifecycle_policy=record.lifecycle_policy,
        ),
    )
    monkeypatch.setattr(run, "_resolve_run_ssh", lambda run_id: (record, ["ssh", "root@example.com"]))
    monkeypatch.setattr(run, "_stop_remote_run", lambda ssh_parts, pid_file, exit_code_file: "stopped")
    monkeypatch.setattr(run.render, "success", lambda message: captured.setdefault("success", message))
    monkeypatch.setattr(run.render.console, "print", lambda message: None)
    monkeypatch.setattr(state, "json_output", False)

    run.run_stop("r_live")

    assert captured["success"] == "Stop signal sent to run r_live on instance 123."


def test_run_start_defaults_fresh_runs_to_pause(monkeypatch):
    inst = SimpleNamespace(machine_id=123, ssh_command="ssh -p 2222 root@example.com")
    captured: dict[str, object] = {}

    monkeypatch.setattr(run.render, "confirm", lambda prompt, skip=False: True)
    monkeypatch.setattr(run.render, "info", lambda message: captured.setdefault("info", []).append(message))
    monkeypatch.setattr(run.render, "success", lambda message: None)
    monkeypatch.setattr(run.render, "warning", lambda message: None)
    monkeypatch.setattr(run.render, "spinner", lambda message: __import__("contextlib").nullcontext())
    monkeypatch.setattr(
        run,
        "get_client",
        lambda: SimpleNamespace(instances=SimpleNamespace(create=lambda **kwargs: inst)),
    )

    def fake_start(
        target,
        machine_id,
        extra_args,
        *,
        follow,
        instance_origin,
        lifecycle_policy,
        script_path=None,
        setup_command=None,
        requirements_path=None,
    ):
        captured["start"] = {
            "target": target,
            "machine_id": machine_id,
            "extra_args": extra_args,
            "follow": follow,
            "instance_origin": instance_origin,
            "lifecycle_policy": lifecycle_policy,
            "script_path": script_path,
            "setup_command": setup_command,
            "requirements_path": requirements_path,
        }
        return "r_fresh", 0

    monkeypatch.setattr(run, "_start_managed_run", fake_start)
    monkeypatch.setattr(
        run, "_apply_lifecycle", lambda machine_id, policy: captured.setdefault("apply", (machine_id, policy))
    )

    run.run_start(
        SimpleNamespace(args=["train.py"]),
        on=None,
        gpu="L4",
        script=None,
        vm=False,
        template="pytorch",
        storage=40,
        name="jl-run",
        num_gpus=1,
        setup=None,
        requirements=None,
        pause=False,
        destroy=False,
        keep=False,
        follow=True,
    )

    assert captured["start"]["lifecycle_policy"] == "pause"
    assert captured["apply"] == (123, "pause")
    assert any("Defaulting to --pause" in message for message in captured["info"])


def test_compose_launch_command_bootstraps_uv_env():
    spec = run.RunSpec(
        target_kind="directory",
        local_target=None,
        remote_target="/home/project",
        working_dir="/home/project",
        launch_command="python3 train.py --epochs 5",
    )

    command = run._compose_launch_command(
        spec,
        setup_command="echo setup",
        requirements_name="requirements.txt",
    )

    assert command == (
        "command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; } && "
        'export PATH="$HOME/.local/bin:$PATH" && '
        "(test -d .venv || uv venv --system-site-packages --seed .venv) && "
        ". .venv/bin/activate && "
        "echo '[jl] Installing from' requirements.txt && uv pip install -r requirements.txt && "
        "echo setup && "
        "python3 train.py --epochs 5"
    )


def test_prepare_support_files_uploads_requirements_and_setup(monkeypatch, tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("transformers\n")
    inst = SimpleNamespace(machine_id=123, ssh_command="ssh root@example.com")
    spec = run.RunSpec(
        target_kind="directory",
        local_target=tmp_path,
        remote_target="/home/project",
        working_dir="/home/project",
        launch_command="python3 train.py",
    )
    uploads: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        run,
        "_upload_support_file",
        lambda inst, ssh_parts, source, destination, *, label: uploads.append((label, str(source), destination)),
    )

    requirements_name = run._prepare_support_files(
        inst,
        ["ssh", "root@example.com"],
        spec,
        requirements_path=requirements,
    )

    assert requirements_name == "requirements.txt"
    assert uploads == [
        ("requirements file", str(requirements), "/home/project/requirements.txt"),
    ]


def test_wait_for_ssh_ready_retries_until_remote_probe_succeeds(monkeypatch):
    inst = SimpleNamespace(status="Running", ssh_command="ssh -p 2222 root@example.com")
    attempts = {"count": 0}

    monkeypatch.setattr(run, "_get_instance", lambda machine_id: inst)
    monkeypatch.setattr(run, "_ssh_parts_from_instance", lambda inst: ["ssh", "root@example.com"])

    def fake_probe(ssh_parts, script):
        attempts["count"] += 1
        return SimpleNamespace(returncode=0 if attempts["count"] == 2 else 255)

    monkeypatch.setattr(run, "_run_remote_capture", fake_probe)
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)

    resolved, ssh_parts = run._wait_for_ssh_ready(123, timeout_s=1, poll_s=0)

    assert resolved is inst
    assert ssh_parts == ["ssh", "root@example.com"]
    assert attempts["count"] == 2
