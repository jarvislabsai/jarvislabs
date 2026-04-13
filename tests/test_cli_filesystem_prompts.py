from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import typer

from jarvislabs.cli import commands


def test_filesystem_create_prompt(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(commands.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        commands.filesystem_create(name="data", storage=120, region=None)

    assert captured["msg"] == "Create filesystem (name='data', storage=120GB)?"


def test_filesystem_create_prompt_with_region(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(commands.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        commands.filesystem_create(name="data", storage=120, region="IN2")

    assert captured["msg"] == "Create filesystem (name='data', storage=120GB, region=IN2)?"


def test_filesystem_edit_prompt(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(commands.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        commands.filesystem_edit(fs_id=7, storage=200)

    assert captured["msg"] == "Expand filesystem 7 to 200GB?"


def test_filesystem_remove_prompt(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(commands.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        commands.filesystem_remove(fs_id=9)

    assert captured["msg"] == "Remove filesystem 9?"


@pytest.mark.parametrize(
    ("region", "nudged"),
    [("india-01", True), ("india-noida-01", False)],
)
def test_filesystem_edit_nudges_on_in1(monkeypatch, region, nudged):
    """Resizing an IN1 filesystem should surface the migration hint, IN2 should not."""
    nudge_calls: list = []
    mock_client = MagicMock()
    mock_client.filesystems.list.return_value = [MagicMock(fs_id=7, region=region)]
    mock_client.filesystems.edit.return_value = 42
    monkeypatch.setattr(commands, "get_client", lambda: mock_client)
    monkeypatch.setattr(commands.render, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(commands.render, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(commands.render, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(commands.render, "in1_migration_hint", lambda: nudge_calls.append(1))

    commands.filesystem_edit(fs_id=7, storage=200, yes=True, json_output=False)

    assert bool(nudge_calls) is nudged
