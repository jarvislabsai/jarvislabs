"""Tests for CLI command structure — subgroups, aliases, and disambiguation."""

from __future__ import annotations

from typer.testing import CliRunner

from jarvislabs.cli import commands, instance, run, setup  # noqa: F401 — trigger registration
from jarvislabs.cli.app import app

runner = CliRunner()


def test_instance_subgroup_help():
    result = runner.invoke(app, ["instance", "--help"])
    assert result.exit_code == 0
    for cmd in ["list", "get", "create", "rename", "pause", "resume", "destroy", "ssh", "exec", "upload", "download"]:
        assert cmd in result.output, f"'{cmd}' not found in instance --help"


def test_alias_create_help():
    result = runner.invoke(app, ["create", "--help"])
    assert result.exit_code == 0
    assert "--gpu" in result.output


def test_alias_list_help():
    result = runner.invoke(app, ["list", "--help"])
    assert result.exit_code == 0


def test_alias_get_help():
    result = runner.invoke(app, ["get", "--help"])
    assert result.exit_code == 0
    assert "Instance ID" in result.output


def test_alias_exec_help():
    result = runner.invoke(app, ["exec", "--help"])
    assert result.exit_code == 0
    assert "Instance ID" in result.output


def test_disambiguation_create_filesystem():
    result = runner.invoke(app, ["create", "filesystem"])
    assert result.exit_code == 2
    assert "Did you mean: jl filesystem create" in result.output


def test_disambiguation_create_scripts():
    result = runner.invoke(app, ["create", "scripts"])
    assert result.exit_code == 2
    assert "Did you mean: jl scripts create" in result.output


def test_disambiguation_does_not_trigger_for_normal_args():
    """jl create --gpu H100 should NOT trigger disambiguation."""
    result = runner.invoke(app, ["create", "--gpu", "H100", "--help"])
    assert result.exit_code == 0
    assert "Did you mean" not in result.output
