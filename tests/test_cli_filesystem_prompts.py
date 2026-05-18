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


def test_filesystem_create_prompt_normalizes_region(monkeypatch):
    captured: dict[str, str] = {}

    def fake_confirm(msg: str, *, skip: bool = False) -> bool:
        captured["msg"] = msg
        return False

    monkeypatch.setattr(commands.render, "confirm", fake_confirm)

    with pytest.raises(typer.Exit):
        commands.filesystem_create(name="data", storage=120, region="india-chennai-01")

    assert captured["msg"] == "Create filesystem (name='data', storage=120GB, region=IN1)?"


def test_filesystem_create_json_normalizes_region(monkeypatch):
    captured: dict = {}
    mock_client = MagicMock()
    mock_client.filesystems.create.return_value = 123
    monkeypatch.setattr(commands, "get_client", lambda: mock_client)
    monkeypatch.setattr(commands.render, "confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(commands.render, "spinner", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(commands.render, "print_json", lambda data: captured.update(data))

    commands.filesystem_create(name="data", storage=120, region="india-chennai-01", yes=True, json_output=True)

    assert captured["region"] == "IN1"
    mock_client.filesystems.create.assert_called_once_with(fs_name="data", storage=120, region="india-chennai-01")


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
