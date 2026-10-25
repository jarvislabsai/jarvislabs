from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from jarvislabs.cli import commands, setup, state, update
from jarvislabs.cli.app import app

BUNDLED = "---\nname: jarvislabs\n---\n# New skill\n"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Point the cache and the skill paths at tmp_path, and keep the version fixed."""
    monkeypatch.delenv("JL_NO_UPDATE_CHECK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(update, "_CACHE_FILE", tmp_path / "cache" / "update_check.json")
    monkeypatch.setattr(update, "current_version", lambda: "0.2.18")
    monkeypatch.setattr(setup, "UNIVERSAL_PATH", tmp_path / ".agents" / "skills" / "jarvislabs" / "SKILL.md")
    monkeypatch.setattr(
        setup,
        "ADDITIONAL_AGENTS",
        {"claude-code": ("Claude Code", tmp_path / ".claude" / "skills" / "jarvislabs" / "SKILL.md")},
    )
    monkeypatch.setattr(setup, "_load_bundled_skill", lambda: BUNDLED)
    monkeypatch.setattr(update.render, "spinner", lambda msg: nullcontext())
    monkeypatch.setattr(state, "json_output", False)


def _write_skill(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _fake_pypi(monkeypatch, latest="0.2.19"):
    """Stub the PyPI request. latest=None makes it fail like a network error. Returns the list of calls."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if latest is None:
            raise httpx.ConnectError("offline")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"info": {"version": latest}})

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def _capture_console(monkeypatch):
    printed = []
    monkeypatch.setattr(update.render.console, "print", lambda msg="", **kw: printed.append(str(msg)))
    return printed


# ── Versions ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("latest", "expected"),
    [("0.2.19", True), ("0.3.0", True), ("0.2.18", False), ("0.2.9", False), (None, False), ("0.3.0rc1", False)],
)
def test_is_newer(latest, expected):
    assert update.is_newer(latest, "0.2.18") is expected


def test_is_newer_false_for_dev_current():
    assert update.is_newer("0.2.19", "0.2.18.dev3") is False


def test_fetch_latest_caches_the_result(monkeypatch):
    _fake_pypi(monkeypatch, latest="0.2.19")
    assert update._fetch_latest() == "0.2.19"
    assert update._read_cache()["latest_version"] == "0.2.19"


def test_fetch_latest_returns_none_on_network_error(monkeypatch):
    _fake_pypi(monkeypatch, latest=None)
    assert update._fetch_latest() is None
    assert update._read_cache() == {}


def test_update_cache_keeps_other_fields():
    update._write_cache({"notified_at": 1.0})
    update._update_cache(latest_version="0.2.19")
    assert update._read_cache() == {"notified_at": 1.0, "latest_version": "0.2.19"}


def test_latest_version_uses_fresh_cache_without_fetching(monkeypatch):
    update._write_cache({"checked_at": time.time(), "latest_version": "0.2.19"})
    calls = _fake_pypi(monkeypatch, latest="0.3.0")
    assert update.latest_version() == "0.2.19"
    assert calls == []


def test_latest_version_fetches_when_cache_is_old(monkeypatch):
    update._write_cache({"checked_at": time.time() - 2 * update._TTL, "latest_version": "0.2.18"})
    calls = _fake_pypi(monkeypatch, latest="0.2.19")
    assert update.latest_version() == "0.2.19"
    assert len(calls) == 1


def test_latest_version_falls_back_to_old_cache_when_pypi_fails(monkeypatch):
    update._write_cache({"checked_at": 0, "latest_version": "0.2.19"})
    _fake_pypi(monkeypatch, latest=None)
    assert update.latest_version() == "0.2.19"


def test_checks_disabled_by_env(monkeypatch):
    monkeypatch.setenv("JL_NO_UPDATE_CHECK", "1")
    update._write_cache({"checked_at": time.time(), "latest_version": "0.2.19"})
    calls = _fake_pypi(monkeypatch)
    assert update.latest_version() is None
    assert update.version_line() == "jl 0.2.18"
    assert calls == []


@pytest.mark.parametrize(
    ("cached", "expected"),
    [("0.2.19", "jl 0.2.18 (0.2.19 available, run jl update)"), ("0.2.18", "jl 0.2.18"), (None, "jl 0.2.18")],
)
def test_version_line(cached, expected):
    if cached:
        update._write_cache({"latest_version": cached})
    assert update.version_line() == expected


def test_version_flag_prints_version_line():
    update._write_cache({"latest_version": "0.2.19"})
    result = CliRunner().invoke(app, ["--version"])
    assert result.stdout.strip() == "jl 0.2.18 (0.2.19 available, run jl update)"


# ── Daily notice ─────────────────────────────────────────────────────────────


def test_print_notice_once_a_day(monkeypatch):
    monkeypatch.setattr(update, "_notice_enabled", lambda: True)
    update._write_cache({"checked_at": time.time(), "latest_version": "0.2.19"})
    printed = _capture_console(monkeypatch)

    update.print_notice()
    update.print_notice()

    assert len(printed) == 1
    assert "0.2.18 → 0.2.19" in printed[0]
    assert "jl update" in printed[0]


@pytest.mark.parametrize("latest", ["0.2.18", None])
def test_print_notice_silent_when_up_to_date(monkeypatch, latest):
    monkeypatch.setattr(update, "_notice_enabled", lambda: True)
    update._write_cache({"checked_at": time.time(), "latest_version": latest})
    printed = _capture_console(monkeypatch)
    update.print_notice()
    assert printed == []


def test_print_notice_silent_with_json(monkeypatch):
    monkeypatch.setattr(update, "_notice_enabled", lambda: True)
    monkeypatch.setattr(state, "json_output", True)
    update._write_cache({"checked_at": time.time(), "latest_version": "0.2.19"})
    printed = _capture_console(monkeypatch)
    update.print_notice()
    assert printed == []


def test_print_notice_never_raises(monkeypatch):
    monkeypatch.setattr(update, "_notice_enabled", lambda: True)
    update._write_cache({"checked_at": time.time(), "latest_version": "0.2.19"})

    def boom(*args, **kwargs):
        raise OSError("broken pipe")

    monkeypatch.setattr(update.render.console, "print", boom)
    update.print_notice()


@pytest.mark.parametrize("stderr", [SimpleNamespace(isatty=lambda: False), None])
def test_notice_disabled_without_terminal(monkeypatch, stderr):
    monkeypatch.setattr(update.sys, "stderr", stderr)
    assert update._notice_enabled() is False


@pytest.mark.parametrize(("checked_at", "started"), [(0, True), (time.time(), False)])
def test_start_background_check_only_when_cache_is_old(monkeypatch, checked_at, started):
    monkeypatch.setattr(update, "_notice_enabled", lambda: True)
    update._write_cache({"checked_at": checked_at})
    threads = []
    monkeypatch.setattr(update.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: threads.append(kw)))

    update.start_background_check()

    assert bool(threads) is started


@pytest.mark.parametrize(("command", "hooked"), [("gpus", True), ("status", False), ("update", False)])
def test_notice_hook_skips_status_and_update(monkeypatch, command, hooked):
    hooks = []
    monkeypatch.setattr(update, "start_background_check", lambda: hooks.append("start"))
    monkeypatch.setattr(update, "print_notice", lambda: hooks.append("notice"))
    model = SimpleNamespace(model_dump=lambda: {})
    client = SimpleNamespace(
        account=SimpleNamespace(
            user_info=lambda: model,
            balance=lambda: model,
            resource_metrics=lambda: model,
            currency=lambda: "USD",
            gpu_availability=lambda: [],
        )
    )
    monkeypatch.setattr(commands, "get_client", lambda: client)
    _fake_pypi(monkeypatch, latest="0.2.18")

    CliRunner().invoke(app, [command, "--json"] if command != "update" else [command, "--skills-only"])

    assert hooks == (["start", "notice"] if hooked else [])


# ── Install command ──────────────────────────────────────────────────────────


def test_install_command_uv_tool_uses_install_latest(tmp_path, monkeypatch):
    (tmp_path / "uv-receipt.toml").touch()
    monkeypatch.setattr(update.sys, "prefix", str(tmp_path))
    assert update.install_command() == ["uv", "tool", "install", "jarvislabs@latest"]


def test_install_command_pipx(tmp_path, monkeypatch):
    (tmp_path / "pipx_metadata.json").touch()
    monkeypatch.setattr(update.sys, "prefix", str(tmp_path))
    assert update.install_command() == ["pipx", "upgrade", "jarvislabs"]


def test_install_command_uv_pip(tmp_path, monkeypatch):
    monkeypatch.setattr(update.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(update, "_installer", lambda: "uv")
    assert update.install_command() == ["uv", "pip", "install", "--python", sys.executable, "--upgrade", "jarvislabs"]


def test_install_command_pip(tmp_path, monkeypatch):
    monkeypatch.setattr(update.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(update, "_installer", lambda: "pip")
    assert update.install_command() == [sys.executable, "-m", "pip", "install", "--upgrade", "jarvislabs"]


def test_command_text_uses_windows_quoting_on_windows(monkeypatch):
    cmd = [r"C:\Program Files\Python\python.exe", "-m", "pip"]
    monkeypatch.setattr(update.sys, "platform", "win32")
    assert update._command_text(cmd) == r'"C:\Program Files\Python\python.exe" -m pip'
    monkeypatch.setattr(update.sys, "platform", "linux")
    assert update._command_text(cmd) == r"'C:\Program Files\Python\python.exe' -m pip"


# ── Skills ───────────────────────────────────────────────────────────────────


def test_skills_current_none_when_not_installed():
    assert setup.skills_current() is None


def test_skills_current_detects_stale_copy():
    _write_skill(setup.UNIVERSAL_PATH, BUNDLED)
    assert setup.skills_current() is True
    _write_skill(setup.ADDITIONAL_AGENTS["claude-code"][1], "old")
    assert setup.skills_current() is False


def test_refresh_skills_rewrites_only_existing_stale_files():
    universal = _write_skill(setup.UNIVERSAL_PATH, "old")
    claude = setup.ADDITIONAL_AGENTS["claude-code"][1]

    assert setup.refresh_skills() == [universal]
    assert universal.read_text() == BUNDLED
    assert not claude.exists()
    assert setup.refresh_skills() == []


def test_refresh_skills_writes_symlinked_copy_once():
    universal = _write_skill(setup.UNIVERSAL_PATH, "old")
    claude_dir = setup.ADDITIONAL_AGENTS["claude-code"][1].parent
    claude_dir.parent.mkdir(parents=True)
    claude_dir.symlink_to(universal.parent)

    assert setup.refresh_skills() == [universal]
    assert (claude_dir / "SKILL.md").read_text() == BUNDLED


# ── jl update ────────────────────────────────────────────────────────────────


@pytest.fixture()
def runs(monkeypatch):
    """Record subprocess.run calls instead of running them."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(update.subprocess, "run", fake_run)
    monkeypatch.setattr(update, "_is_editable", lambda: False)
    monkeypatch.setattr(update, "install_command", lambda: ["uv", "tool", "install", "jarvislabs@latest"])
    monkeypatch.setattr(update.sys, "platform", "linux")
    return calls


def test_update_when_up_to_date_refreshes_skills(monkeypatch, runs):
    _fake_pypi(monkeypatch, latest="0.2.18")
    universal = _write_skill(setup.UNIVERSAL_PATH, "old")

    update.update()

    assert runs == []
    assert universal.read_text() == BUNDLED


def test_update_installs_then_refreshes_skills_with_new_jl(monkeypatch, runs):
    _fake_pypi(monkeypatch, latest="0.2.19")

    update.update()

    assert runs == [["uv", "tool", "install", "jarvislabs@latest"], [sys.argv[0], "update", "--skills-only"]]


def test_update_fails_when_installer_fails(monkeypatch, runs):
    _fake_pypi(monkeypatch, latest="0.2.19")
    monkeypatch.setattr(update.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1))
    with pytest.raises(SystemExit) as exc:
        update.update()
    assert exc.value.code == 1


def test_update_fails_when_installer_missing(monkeypatch, runs):
    _fake_pypi(monkeypatch, latest="0.2.19")

    def missing(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(update.subprocess, "run", missing)
    with pytest.raises(SystemExit):
        update.update()


def test_update_warns_when_new_jl_cannot_start(monkeypatch, runs):
    _fake_pypi(monkeypatch, latest="0.2.19")
    warnings = []
    monkeypatch.setattr(update.render, "warning", warnings.append)

    def run(cmd, **kwargs):
        if cmd[0] == sys.argv[0]:
            raise PermissionError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(update.subprocess, "run", run)
    update.update()

    assert warnings == ["Could not refresh the agent skills. Run jl update again."]


def test_update_on_windows_only_prints_command(monkeypatch, runs):
    _fake_pypi(monkeypatch, latest="0.2.19")
    monkeypatch.setattr(update.sys, "platform", "win32")
    printed = _capture_console(monkeypatch)

    update.update()

    assert runs == []
    assert any("uv tool install jarvislabs@latest" in msg for msg in printed)


def test_update_refuses_editable_install(monkeypatch, runs):
    monkeypatch.setattr(update, "_is_editable", lambda: True)
    calls = _fake_pypi(monkeypatch)
    with pytest.raises(SystemExit):
        update.update()
    assert calls == []
    assert runs == []


def test_update_fails_when_pypi_unreachable(monkeypatch, runs):
    _fake_pypi(monkeypatch, latest=None)
    with pytest.raises(SystemExit):
        update.update()
    assert runs == []


def test_update_skills_only_does_not_install(monkeypatch, runs):
    calls = _fake_pypi(monkeypatch)
    _write_skill(setup.UNIVERSAL_PATH, "old")

    update.update(skills_only=True)

    assert calls == []
    assert runs == []
    assert setup.UNIVERSAL_PATH.read_text() == BUNDLED


# ── jl status ────────────────────────────────────────────────────────────────


def test_status_json_includes_cli_block(monkeypatch, capsys):
    _fake_pypi(monkeypatch, latest="0.2.19")
    _write_skill(setup.UNIVERSAL_PATH, "old")
    model = SimpleNamespace(model_dump=lambda: {})
    client = SimpleNamespace(
        account=SimpleNamespace(
            user_info=lambda: model, balance=lambda: model, resource_metrics=lambda: model, currency=lambda: "USD"
        )
    )
    monkeypatch.setattr(commands, "get_client", lambda: client)

    commands.status(json_output=True)

    assert json.loads(capsys.readouterr().out)["cli"] == {
        "version": "0.2.18",
        "latest": "0.2.19",
        "update_available": True,
        "skill_current": False,
        "hint": "jl 0.2.19 is available. Run jl update.",
    }


@pytest.mark.parametrize(
    ("latest", "skill", "hint"),
    [
        ("0.2.19", "old", "jl 0.2.19 is available. Run jl update."),
        ("0.2.18", "old", "The jl agent skill is out of date. Run jl update."),
        ("0.2.18", None, "No jl agent skill is installed. Run jl setup to add one."),
        (
            "0.2.19",
            None,
            "jl 0.2.19 is available. Run jl update. No jl agent skill is installed. Run jl setup to add one.",
        ),
        ("0.2.18", BUNDLED, None),
    ],
)
def test_cli_status_hint(monkeypatch, latest, skill, hint):
    _fake_pypi(monkeypatch, latest=latest)
    if skill is not None:
        _write_skill(setup.UNIVERSAL_PATH, skill)
    assert update.cli_status()["hint"] == hint


def test_help_tells_agents_to_install_the_skill():
    result = CliRunner().invoke(app, ["--help"])
    assert "jl setup" in result.stdout
    assert "jl skill" in result.stdout
