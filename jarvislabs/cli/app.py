"""Root Typer app — entry point for the `jl` CLI."""

from __future__ import annotations

from typing import Annotated

import typer

from jarvislabs.cli.render import die
from jarvislabs.client import Client
from jarvislabs.exceptions import JarvislabsError

app = typer.Typer(
    name="jl",
    help=(
        "[bold cyan]⚡ Jarvislabs[/bold cyan]  [bold]GPU cloud[/bold]\n"
        "Create GPU instances, run jobs, and call models.\n\n"
        "[bold]Start here[/bold]\n"
        "  [bold cyan]jl gpus[/bold cyan]                      see GPUs and prices\n"
        "  [bold cyan]jl create --gpu A100[/bold cyan]         start a container (Jupyter, IDE)\n"
        "  [bold cyan]jl run train.py --gpu A100[/bold cyan]   run a script on a fresh GPU\n"
        "  [bold cyan]jl models list[/bold cyan]               pay-per-token models and prices\n\n"
        "[bold]Instances[/bold]\n"
        "  Containers  pre-configured with PyTorch, Jupyter and IDE\n"
        "  VMs         bare-metal SSH; add --vm\n\n"
        "[bold]More[/bold]\n"
        "  [bold cyan]jl deploy managed-models create[/bold cyan]   serve a ready-made model\n"
        "  [bold cyan]jl setup[/bold cyan]                          install the jl skill for coding agents\n\n"
        "Every command has --help, e.g. jl create --help."
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
