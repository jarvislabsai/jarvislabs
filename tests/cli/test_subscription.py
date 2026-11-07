from __future__ import annotations

import sys
import webbrowser

import pytest
from typer.testing import CliRunner

from jarvislabs.cli import commands, subscription  # noqa: F401  (registers the commands)
from jarvislabs.cli.app import app
from jarvislabs.exceptions import APIError, NotFoundError

runner = CliRunner()

USAGE = {
    "plan": {"id": 2, "name": "Pro"},
    "period": {"start": "2026-10-20T09:00:00Z", "renews_at": "2026-11-20T09:00:00Z", "cancel_at_period_end": False},
    "use_balance": False,
    "renewal_pending": False,
    "windows": {
        "five_hour": {"percent_used": 42.5, "resets_at": "2026-11-05T16:05:00Z"},
        "weekly": {"percent_used": 10.0, "resets_at": "2026-11-09T00:00:00Z"},
        "monthly": {"percent_used": 3.2, "resets_at": "2026-11-20T09:00:00Z"},
    },
    "slots": {"in_use": 1, "limit": 4},
}


@pytest.fixture()
def opened(monkeypatch):
    """Record the pages `jl` opens in the browser instead of opening them."""
    pages: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *args, **kwargs: pages.append(url) or True)
    return pages


def test_usage_shows_the_plan_and_each_window(server):
    server.routes[("GET", "subscription/usage")] = USAGE

    result = runner.invoke(app, ["usage"])

    assert result.exit_code == 0, result.output
    for text in (
        "Pro",
        "renews 2026-11-20 09:00 UTC",
        "42.5% used, resets 2026-11-05 16:05 UTC",
        "10.0% used, resets 2026-11-09 00:00 UTC",
        "3.2% used, resets 2026-11-20 09:00 UTC",
        "1 of 4",
    ):
        assert text in result.output
    assert "Next: jl usage manage" in result.output


def test_usage_draws_a_bar_for_each_window(server):
    server.routes[("GET", "subscription/usage")] = USAGE

    result = runner.invoke(app, ["usage"])

    assert result.exit_code == 0, result.output
    assert "█" in result.output
    assert "░" in result.output
    # Three windows, each on its own row.
    assert result.output.count("used, resets") == 3


def test_usage_without_a_plan_points_at_subscribe(server):
    server.routes[("GET", "subscription/usage")] = NotFoundError("This account has no active subscription.")

    result = runner.invoke(app, ["usage"])

    assert result.exit_code == 0, result.output
    assert "No subscription" in result.output
    assert "Next: jl subscribe" in result.output
    assert "get a plan" in result.output


def test_status_leaves_plan_usage_to_jl_usage(server):
    server.routes[("GET", "subscription/usage")] = USAGE

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "42.5% used" not in result.output
    assert "Coding tools" in result.output
    assert "Ada" in result.output


def test_status_works_when_billing_is_unavailable(server):
    server.routes[("GET", "subscription/usage")] = APIError(503, "Billing is unavailable right now.")

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    # Usage is not status's job, so a billing outage doesn't clutter the overview.
    assert "Billing is unavailable" not in result.output
    assert "Ada" in result.output
    assert "Coding tools" in result.output


def test_use_balance_sets_spill_over(server):
    server.routes[("PUT", "subscription/use_balance")] = {"use_balance": True}

    result = runner.invoke(app, ["usage", "use-balance", "on"])

    assert result.exit_code == 0, result.output
    assert server.calls == [("PUT", "subscription/use_balance", {"enabled": True})]
    assert "Use balance is on" in result.output


def test_usage_with_a_plan_points_at_manage(server):
    server.routes[("GET", "subscription/usage")] = USAGE

    result = runner.invoke(app, ["usage"])

    assert result.exit_code == 0, result.output
    assert "Pro" in result.output
    assert "Next: jl usage manage" in result.output
    assert "manage or cancel" in result.output


def test_manage_opens_the_billing_page(server, opened):
    server.routes[("POST", "subscription/portal")] = {"url": "https://billing.example.com/session/test_1"}

    result = runner.invoke(app, ["usage", "manage"])

    assert result.exit_code == 0, result.output
    assert opened == ["https://billing.example.com/session/test_1"]
    assert "https://billing.example.com/session/test_1" in result.output


def test_subscribe_opens_the_subscribe_page(opened, monkeypatch):
    monkeypatch.setenv("JL_NO_UPDATE_CHECK", "1")

    result = runner.invoke(app, ["subscribe"])

    assert result.exit_code == 0, result.output
    assert opened == ["https://jarvislabs.ai/products/coding-plan"]
    assert "https://jarvislabs.ai/products/coding-plan" in result.output
    assert "Next: jl connect" in result.output
    assert "after paying, connect your tools" in result.output


@pytest.mark.parametrize(
    "session",
    [{"SSH_CONNECTION": "203.0.113.5 52144 10.0.0.2 22"}, {"DISPLAY": None, "WAYLAND_DISPLAY": None}],
    ids=["ssh", "linux-without-a-desktop"],
)
def test_subscribe_over_ssh_prints_the_url_only(opened, monkeypatch, session):
    monkeypatch.setenv("JL_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(sys, "platform", "linux")
    for name, value in session.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    result = runner.invoke(app, ["subscribe"])

    assert result.exit_code == 0, result.output
    # webbrowser would start a text browser in this terminal.
    assert opened == []
    assert "Open https://jarvislabs.ai/products/coding-plan in your browser." in result.output


@pytest.mark.parametrize(
    ("platform", "session"),
    [
        # VS Code's Remote-SSH and Codespaces terminals open pages in the user's own browser.
        ("linux", {"SSH_CONNECTION": "203.0.113.5 52144 10.0.0.2 22", "BROWSER": "/vscode/bin/helpers/browser.sh"}),
        ("darwin", {"DISPLAY": None}),
        ("win32", {"DISPLAY": None}),
    ],
    ids=["vscode-remote", "macos", "windows"],
)
def test_subscribe_opens_the_browser_where_there_is_one(opened, monkeypatch, platform, session):
    monkeypatch.setenv("JL_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(sys, "platform", platform)
    for name, value in session.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    result = runner.invoke(app, ["subscribe"])

    assert result.exit_code == 0, result.output
    assert opened == ["https://jarvislabs.ai/products/coding-plan"]
