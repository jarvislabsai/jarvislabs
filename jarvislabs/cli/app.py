"""Root Typer app — entry point for the `jl` CLI."""

from __future__ import annotations

import click
import typer
from typer.core import TyperGroup


class DisambiguatingGroup(TyperGroup):
    """Suggest noun-first form when a user types 'jl <verb> <resource>'."""

    def resolve_command(self, ctx, args):  # type: ignore[override]
        cmd_name, cmd, remaining = super().resolve_command(ctx, args)
        # If the next arg is itself a registered subgroup, the user probably
        # meant "jl <resource> <verb>" (noun-first) instead of "jl <verb> <resource>".
        if cmd_name and remaining and remaining[0] in self.commands:
            maybe_resource = remaining[0]
            resource_cmd = self.commands.get(maybe_resource)
            if resource_cmd is not None and isinstance(resource_cmd, click.Group):
                click.echo(f"Did you mean: jl {maybe_resource} {cmd_name}", err=True)
                ctx.exit(2)
        return cmd_name, cmd, remaining


app = typer.Typer(
    name="jl",
    cls=DisambiguatingGroup,
    help=(
        "[bold cyan]⚡ JarvisLabs[/bold cyan] GPU Cloud CLI\n\n"
        "[dim]Two types of instances:[/dim]\n"
        "  [green]Containers[/green]  [dim]Pre-configured with PyTorch, Jupyter, and IDE. Default.[/dim]\n"
        "  [yellow]VMs[/yellow]          [dim]Bare-metal SSH access. Use --vm.[/dim]\n\n"
        "[dim]Quick examples:[/dim]\n"
        "  [bold]jl create --gpu A100[/bold]                       [rgb(100,100,100)]container instance[/rgb(100,100,100)]\n"
        "  [bold]jl create --gpu H100 --vm[/bold]                  [rgb(100,100,100)]VM instance[/rgb(100,100,100)]\n"
        "  [bold]jl run train.py --gpu L4[/bold]                   [rgb(100,100,100)]run on a fresh GPU[/rgb(100,100,100)]\n"
        "  [bold]jl run . --script train.py --on <id>[/bold]       [rgb(100,100,100)]upload project and run[/rgb(100,100,100)]\n"
        "  [bold]jl ssh <id>[/bold]                                [rgb(100,100,100)]SSH into an instance[/rgb(100,100,100)]\n"
        "  [bold]jl list[/bold]                                    [rgb(100,100,100)]list all instances[/rgb(100,100,100)]\n"
        "  [bold]jl gpus[/bold]                                    [rgb(100,100,100)]GPU availability and pricing[/rgb(100,100,100)]\n"
        "  [bold]jl pause <id>[/bold]                              [rgb(100,100,100)]stop billing, keep data[/rgb(100,100,100)]\n"
        "  [bold]jl destroy <id>[/bold]                            [rgb(100,100,100)]delete permanently[/rgb(100,100,100)]\n\n"
        "[dim]Instance commands are also available under[/dim] [bold]jl instance <verb>[/bold]\n"
        "[dim]Every command supports[/dim] [bold]--help[/bold] [dim]for details:[/dim]\n"
        "  [dim]jl create --help    jl run --help    jl instance --help[/dim]"
    ),
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)


@app.callback(invoke_without_command=True)
def _global_flags(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        from importlib.metadata import version as pkg_version

        typer.echo(f"jl {pkg_version('jarvislabs')}")
        raise typer.Exit()

    # If no subcommand was given (and --version wasn't handled above), show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def get_client():
    """Create a Client using the resolved token. Called lazily by commands that need it."""
    from jarvislabs.cli.render import die
    from jarvislabs.client import Client
    from jarvislabs.exceptions import JarvislabsError

    try:
        return Client()
    except JarvislabsError as e:
        die(str(e))


def main() -> None:
    """Entry point for `jl` command (wired via pyproject.toml [project.scripts])."""
    from jarvislabs.cli import commands, instance, run, setup  # noqa: F401
    from jarvislabs.cli.update_check import finish_update_check, start_update_check
    from jarvislabs.exceptions import JarvislabsError

    start_update_check()

    try:
        app()
    except JarvislabsError as e:
        from jarvislabs.cli.render import die

        die(str(e))
    finally:
        finish_update_check()
