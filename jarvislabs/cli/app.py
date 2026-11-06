"""Root Typer app — entry point for the `jl` CLI."""

from __future__ import annotations

from typing import Annotated

import typer
import typer.rich_utils as rich_utils

from jarvislabs.cli.render import die
from jarvislabs.client import Client
from jarvislabs.exceptions import JarvislabsError

# Typer dims every help line after the first by default, which makes the whole page look disabled.
# Description is body text; dim is for tertiary metadata only. Panel borders stay grey, not dim.
rich_utils.STYLE_HELPTEXT = ""
rich_utils.STYLE_OPTIONS_PANEL_BORDER = "bright_black"
rich_utils.STYLE_COMMANDS_PANEL_BORDER = "bright_black"

app = typer.Typer(
    name="jl",
    help=(
        "GPU cloud, model APIs, and serverless deployments from your terminal.\n\n"
        "Using a coding agent? jl setup installs the jl skill."
    ),
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)


@app.callback(invoke_without_command=True)
def _global_flags(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    # Imported here, not at the top: update.py imports `app` from this module.
    from jarvislabs.cli import update

    if version:
        typer.echo(update.version_line())
        raise typer.Exit()

    # If no subcommand was given (and --version wasn't handled above), show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    # `jl status` and `jl update` report updates themselves. After `jl update`, the notice would also
    # announce the version that was just replaced.
    if ctx.invoked_subcommand not in ("status", "update"):
        update.start_background_check()
        ctx.call_on_close(update.print_notice)


def get_client() -> Client:
    """Create a Client using the resolved token. Called lazily by commands that need it."""
    try:
        return Client()
    except JarvislabsError as e:
        die(str(e))


def main() -> None:
    """Entry point for `jl` command (wired via pyproject.toml [project.scripts])."""
    # Command modules import `app` from this module, so they can only load after it exists.
    from jarvislabs.cli import (  # noqa: F401
        commands,
        connect,
        deploy,
        instance,
        models,
        run,
        setup,
        subscription,
        update,
    )

    try:
        app()
    except JarvislabsError as e:
        die(str(e))
