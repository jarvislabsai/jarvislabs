from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvislabs.cli import setup, state

# ── Helpers ──────────────────────────────────────────────────────────────────


def _fake_client(name="Test User", balance=42.50, currency="USD"):
    """Return a mock Client that passes user_info() validation."""
    client = MagicMock()
    client.account.user_info.return_value = SimpleNamespace(name=name, user_id="u_123")
    client.account.balance.return_value = SimpleNamespace(balance=balance, grants=0)
    client.account.resource_metrics.return_value = SimpleNamespace(running_instances=0, paused_instances=0)
    client.account.currency.return_value = currency
    return client


SKILL_CONTENT = "---\nname: jarvislabs\n---\n# Test skill content\n"


# ── _login_flow ──────────────────────────────────────────────────────────────


def test_login_flow_skips_when_existing_token_valid(monkeypatch):
    monkeypatch.setattr(setup, "load_config", lambda: {"auth": {"token": "existing"}})
    # User declines re-auth
    monkeypatch.setattr(setup.render, "confirm", lambda msg, skip=False: False)

    client = _fake_client()
    import jarvislabs.client

    monkeypatch.setattr(jarvislabs.client, "Client", lambda api_key: client)
    result = setup._login_flow(token=None)

    assert result is client


def test_login_flow_saves_new_token(monkeypatch):
    saved = {}
    monkeypatch.setattr(setup, "load_config", lambda: {})
    monkeypatch.setattr(setup, "save_config", lambda config: saved.update(config))
    monkeypatch.setattr(setup.render, "spinner", lambda msg: MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

    client = _fake_client()
    import jarvislabs.client

    monkeypatch.setattr(jarvislabs.client, "Client", lambda api_key: client)

    result = setup._login_flow(token="new-token-123")

    assert saved["auth"]["token"] == "new-token-123"
    assert result is client


def test_login_flow_returns_client(monkeypatch):
    monkeypatch.setattr(setup, "load_config", lambda: {"auth": {"token": "valid"}})
    monkeypatch.setattr(setup.render, "confirm", lambda msg, skip=False: False)

    client = _fake_client()
    import jarvislabs.client

    monkeypatch.setattr(jarvislabs.client, "Client", lambda api_key: client)

    result = setup._login_flow(token=None)
    assert result is client


def test_login_flow_reauths_when_user_confirms(monkeypatch):
    saved = {}
    monkeypatch.setattr(setup, "load_config", lambda: {"auth": {"token": "old-token"}})
    monkeypatch.setattr(setup, "save_config", lambda config: saved.update(config))
    monkeypatch.setattr(setup.render, "spinner", lambda msg: MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    # User accepts re-auth
    monkeypatch.setattr(setup.render, "confirm", lambda msg, skip=False: True)
    monkeypatch.setattr(setup.render.console, "input", lambda prompt, password=False: "new-token")

    client = _fake_client()
    import jarvislabs.client

    monkeypatch.setattr(jarvislabs.client, "Client", lambda api_key: client)

    result = setup._login_flow(token=None)
    assert saved["auth"]["token"] == "new-token"
    assert result is client


def test_login_flow_falls_back_when_saved_token_invalid(monkeypatch):
    """Invalid saved token should warn and prompt for a new one."""
    from jarvislabs.exceptions import JarvislabsError

    saved = {}
    monkeypatch.setattr(setup, "load_config", lambda: {"auth": {"token": "bad-token"}})
    monkeypatch.setattr(setup, "save_config", lambda config: saved.update(config))
    monkeypatch.setattr(setup.render, "spinner", lambda msg: MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    monkeypatch.setattr(setup.render.console, "input", lambda prompt, password=False: "fresh-token")

    call_count = 0
    client = _fake_client()

    def fake_client(api_key):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise JarvislabsError("invalid")
        return client

    import jarvislabs.client

    monkeypatch.setattr(jarvislabs.client, "Client", fake_client)

    result = setup._login_flow(token=None)
    assert saved["auth"]["token"] == "fresh-token"
    assert result is client


def test_login_flow_dies_on_empty_input(monkeypatch):
    """Pressing enter with no token should exit."""
    monkeypatch.setattr(setup, "load_config", lambda: {})
    monkeypatch.setattr(setup.render.console, "input", lambda prompt, password=False: "  ")
    monkeypatch.setattr(setup.render, "die", MagicMock(side_effect=SystemExit(1)))

    with pytest.raises(SystemExit):
        setup._login_flow(token=None)


def test_login_flow_dies_on_auth_failure(monkeypatch):
    """Bad new token should exit with auth failure message."""
    from contextlib import contextmanager

    from jarvislabs.exceptions import JarvislabsError

    @contextmanager
    def fake_spinner(msg):
        yield

    monkeypatch.setattr(setup, "load_config", lambda: {})
    monkeypatch.setattr(setup.render, "spinner", fake_spinner)
    monkeypatch.setattr(setup.render, "die", MagicMock(side_effect=SystemExit(1)))

    import jarvislabs.client

    monkeypatch.setattr(jarvislabs.client, "Client", MagicMock(side_effect=JarvislabsError("denied")))

    with pytest.raises(SystemExit):
        setup._login_flow(token="bad-token")


# ── _load_bundled_skill ──────────────────────────────────────────────────────


def test_load_bundled_skill_reads_file():
    content = setup._load_bundled_skill()
    assert "jarvislabs" in content
    assert "JarvisLabs" in content


# ── _parse_agents_flag ───────────────────────────────────────────────────────


def test_parse_agents_all():
    result = setup._parse_agents_flag("all")
    assert result == setup.ALL_AGENTS


def test_parse_agents_specific():
    result = setup._parse_agents_flag("claude-code,codex")
    assert result == ["claude-code", "codex"]


def test_parse_agents_invalid_dies(monkeypatch):
    monkeypatch.setattr(setup.render, "die", MagicMock(side_effect=SystemExit(1)))
    with pytest.raises(SystemExit):
        setup._parse_agents_flag("claude-code,invalid-agent")


# ── _install_skill ───────────────────────────────────────────────────────────


def test_install_skill_writes_to_correct_paths(tmp_path, monkeypatch):
    test_paths = {
        "claude-code": ("Claude Code", tmp_path / ".claude" / "skills" / "jarvislabs" / "SKILL.md"),
        "codex": ("Codex", tmp_path / ".agents" / "skills" / "jarvislabs" / "SKILL.md"),
    }
    monkeypatch.setattr(setup, "AGENT_PATHS", test_paths)

    installed = setup._install_skill(SKILL_CONTENT, ["claude-code", "codex"])

    assert len(installed) == 2
    for _label, path in installed:
        assert path.exists()
        assert path.read_text() == SKILL_CONTENT


def test_install_skill_creates_directories(tmp_path, monkeypatch):
    deep_path = tmp_path / "a" / "b" / "c" / "SKILL.md"
    test_paths = {"claude-code": ("Claude Code", deep_path)}
    monkeypatch.setattr(setup, "AGENT_PATHS", test_paths)

    setup._install_skill(SKILL_CONTENT, ["claude-code"])

    assert deep_path.exists()
    assert deep_path.read_text() == SKILL_CONTENT


def test_install_skill_overwrites_existing(tmp_path, monkeypatch):
    skill_path = tmp_path / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text("old content")

    test_paths = {"claude-code": ("Claude Code", skill_path)}
    monkeypatch.setattr(setup, "AGENT_PATHS", test_paths)

    setup._install_skill(SKILL_CONTENT, ["claude-code"])

    assert skill_path.read_text() == SKILL_CONTENT


# ── _select_agents_noninteractive ────────────────────────────────────────────


def test_select_agents_noninteractive_returns_all():
    result = setup._select_agents_noninteractive()
    assert result == setup.ALL_AGENTS


# ── _select_agents_interactive ───────────────────────────────────────────────


def test_select_agents_interactive_returns_selection(monkeypatch):
    monkeypatch.setattr(
        setup.questionary, "checkbox", lambda message, choices, **kw: MagicMock(ask=lambda: ["claude-code", "cursor"])
    )
    result = setup._select_agents_interactive()
    assert result == ["claude-code", "cursor"]


def test_select_agents_interactive_exits_on_ctrl_c(monkeypatch):
    import typer

    monkeypatch.setattr(setup.questionary, "checkbox", lambda message, choices, **kw: MagicMock(ask=lambda: None))
    with pytest.raises(typer.Exit):
        setup._select_agents_interactive()


# ── _skill_install_flow ──────────────────────────────────────────────────────


def test_skill_install_flow_skips_when_declined(monkeypatch):
    monkeypatch.setattr(state, "yes", False)
    monkeypatch.setattr(setup.render, "confirm", lambda msg, skip=False: False)

    result = setup._skill_install_flow(agents_flag=None)
    assert result == []


def test_skill_install_flow_with_agents_flag(tmp_path, monkeypatch):
    test_paths = {
        "claude-code": ("Claude Code", tmp_path / ".claude" / "skills" / "jarvislabs" / "SKILL.md"),
    }
    monkeypatch.setattr(setup, "AGENT_PATHS", test_paths)
    monkeypatch.setattr(state, "yes", True)

    result = setup._skill_install_flow(agents_flag="claude-code")
    assert len(result) == 1
    assert result[0][1].exists()


def test_skill_install_flow_with_yes_selects_all(tmp_path, monkeypatch):
    test_paths = {
        "claude-code": ("Claude Code", tmp_path / ".claude" / "SKILL.md"),
        "codex": ("Codex", tmp_path / ".agents" / "SKILL.md"),
    }
    monkeypatch.setattr(setup, "AGENT_PATHS", test_paths)
    monkeypatch.setattr(setup, "ALL_AGENTS", list(test_paths.keys()))
    monkeypatch.setattr(state, "yes", True)

    confirm_calls = iter([True])  # "Install agent skills?" -> yes
    monkeypatch.setattr(setup.render, "confirm", lambda msg, skip=False: next(confirm_calls))

    result = setup._skill_install_flow(agents_flag=None)
    assert len(result) == 2


# ── AGENT_PATHS ──────────────────────────────────────────────────────────────


def test_all_agent_paths_are_unique():
    """Each agent should have its own native path."""
    paths = [str(path) for _, path in setup.AGENT_PATHS.values()]
    assert len(paths) == len(set(paths))


def test_agent_paths_use_jarvislabs_name():
    """Skill directory should be named 'jarvislabs', not 'jl-gpu'."""
    for _label, path in setup.AGENT_PATHS.values():
        assert "jarvislabs" in str(path)
        assert "jl-gpu" not in str(path)
