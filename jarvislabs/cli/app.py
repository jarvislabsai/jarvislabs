"""Root Typer app — entry point for the `jl` CLI."""

from __future__ import annotations

from typing import Annotated

import click
import typer
import typer.rich_utils as rich_utils
from typer.main import get_command

from jarvislabs.cli.render import die
from jarvislabs.client import Client
from jarvislabs.exceptions import JarvislabsError

# Typer dims every help line after the first by default, which makes the whole page look disabled.
# Description is body text; dim is for tertiary metadata only. Panel borders stay grey, not dim.
rich_utils.STYLE_HELPTEXT = ""
# A default is help content too, so it is printed in body weight rather than dim.
rich_utils.STYLE_OPTION_DEFAULT = ""
rich_utils.STYLE_OPTIONS_PANEL_BORDER = "bright_black"
rich_utils.STYLE_COMMANDS_PANEL_BORDER = "bright_black"
# Typer colours Usage: yellow and switches green, but in jl yellow means warning and green means
# success. Help stays in body weight, with commands in the same accent as the option names.
rich_utils.STYLE_USAGE = "bold"
# Typer 0.28 renamed STYLE_METAVAR to STYLE_TYPES; users get whichever Typer uv resolves, so set both.
rich_utils.STYLE_METAVAR = "bold"
rich_utils.STYLE_TYPES = "bold"
rich_utils.STYLE_SWITCH = "bold cyan"

app = typer.Typer(
    name="jl",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "GPU cloud, model APIs, and serverless deployments from your terminal.\n\n"
        "[bold]Common workflows[/bold]\n"
        "  GPU     [bold cyan]jl gpus[/bold cyan] → [bold cyan]jl create --gpu A100[/bold cyan] → [bold cyan]jl run train.py --on <id>[/bold cyan]\n"
        "  Stop    [bold cyan]jl pause <id>[/bold cyan]  stop GPU billing, keep the disk\n"
        '  Models  [bold cyan]jl models list[/bold cyan] → [bold cyan]jl models chat <model> "Hello"[/bold cyan]\n'
        "  Serve   [bold cyan]jl deploy managed-models create[/bold cyan]\n\n"
        "Using a coding agent? [bold cyan]jl setup[/bold cyan] adds the jl skill; [bold cyan]jl connect[/bold cyan] adds our models.\n\n"
        "Use jl help <command> for details."
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


@app.command("help", hidden=True)
def help_command(
    command: Annotated[list[str] | None, typer.Argument(metavar="COMMAND...", help="Command to explain.")] = None,
) -> None:
    """Show help for jl or one of its commands."""
    target = get_command(app)
    for name in command or []:
        # A plain command has no subcommands, so a name after it is unknown too.
        found = target.commands.get(name) if isinstance(target, click.Group) else None
        if found is None:
            die(f"Unknown command {name}.")
        target = found
    click.echo(
        target.get_help(
            click.Context(
                target,
                info_name="jl" + (" " + " ".join(command) if command else ""),
                help_option_names=(app.info.context_settings or {}).get("help_option_names", ["--help"]),
            )
        )
    )


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
