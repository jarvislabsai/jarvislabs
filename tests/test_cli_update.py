from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from jarvislabs.cli import commands, setup, state, update

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
    monkeypatch.setattr(setup, "_read_bundled_skill", lambda: BUNDLED)
    monkeypatch.setattr(update.render, "spinner", lambda msg: nullcontext())
    monkeypatch.setattr(state, "json_output", False)


def _write_skill(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _fake_pypi(monkeypatch, latest="0.2.19"):
    calls = []

    def fake_fetch():
        calls.append(1)
        if latest is None:
            return None
        cache = update._read_cache()
        cache.update(checked_at=time.time(), latest_version=latest)
        update._write_cache(cache)
        return latest

    monkeypatch.setattr(update, "_fetch_latest", fake_fetch)
    return calls


# ── Versions ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("latest", "expected"),
    [("0.2.19", True), ("0.3.0", True), ("0.2.18", False), ("0.2.9", False), (None, False), ("0.3.0rc1", False)],
)
def test_is_newer(latest, expected):
    assert update.is_newer(latest, "0.2.18") is expected


def test_is_newer_false_for_dev_current():
    assert update.is_newer("0.2.19", "0.2.18.dev3") is False


def test_latest_version_uses_fresh_cache_without_fetching(monkeypatch):
    update._write_cache({"checked_at": time.time(), "latest_version": "0.2.19"})
    calls = _fake_pypi(monkeypatch, latest="0.3.0")
    assert update.latest_version() == "0.2.19"
    assert calls == []


def test_latest_version_fetches_when_cache_is_old(monkeypatch):
    update._write_cache({"checked_at": time.time() - 2 * update._TTL, "latest_version": "0.2.18"})
    calls = _fake_pypi(monkeypatch, latest="0.2.19")
    assert update.latest_version() == "0.2.19"
    assert calls == [1]


def test_latest_version_falls_back_to_old_cache_when_pypi_fails(monkeypatch):
    update._write_cache({"checked_at": 0, "latest_version": "0.2.19"})
    _fake_pypi(monkeypatch, latest=None)
    assert update.latest_version() == "0.2.19"


def test_latest_version_disabled_by_env(monkeypatch):
    monkeypatch.setenv("JL_NO_UPDATE_CHECK", "1")
    calls = _fake_pypi(monkeypatch)
    assert update.latest_version() is None
    assert update.cached_latest() is None
    assert calls == []


def test_fetch_latest_returns_none_on_network_error(monkeypatch):
    import httpx

    def boom(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", boom)
    assert update._fetch_latest() is None


# ── Daily notice ─────────────────────────────────────────────────────────────


def _capture_console(monkeypatch):
    printed = []
    monkeypatch.setattr(update.render.console, "print", lambda msg, **kw: printed.append(msg))
    return printed


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


def test_notice_disabled_without_terminal(monkeypatch):
    monkeypatch.setattr(update.sys, "stderr", SimpleNamespace(isatty=lambda: False))
    assert update._notice_enabled() is False


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


def test_update_installs_then_refreshes_skills_in_new_process(monkeypatch, runs):
    _fake_pypi(monkeypatch, latest="0.2.19")

    update.update()

    assert runs == [
        ["uv", "tool", "install", "jarvislabs@latest"],
        [sys.argv[0], "update", "--skills-only"],
    ]


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


def test_update_on_windows_only_prints_command(monkeypatch, runs):
    _fake_pypi(monkeypatch, latest="0.2.19")
    monkeypatch.setattr(update.sys, "platform", "win32")
    printed = _capture_console(monkeypatch)

    update.update()

    assert runs == []
    assert any("uv tool install jarvislabs@latest" in str(msg) for msg in printed)


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
    monkeypatch.setattr(commands.render, "spinner", lambda msg: nullcontext())

    commands.status(json_output=True)

    assert json.loads(capsys.readouterr().out)["cli"] == {
        "version": "0.2.18",
        "latest": "0.2.19",
        "update_available": True,
        "skill_current": False,
    }


@pytest.mark.parametrize(("command", "hooked"), [("gpus", True), ("status", False), ("update", False)])
def test_notice_hook_skips_status_and_update(monkeypatch, command, hooked):
    from typer.testing import CliRunner

    from jarvislabs.cli.app import app

    hooks = []
    monkeypatch.setattr(update, "start_background_check", lambda: hooks.append("start"))
    monkeypatch.setattr(update, "print_notice", lambda: hooks.append("notice"))
    monkeypatch.setattr(update, "update", lambda **kw: None)
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
