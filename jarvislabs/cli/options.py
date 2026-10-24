from __future__ import annotations

from typing import Annotated

import click
import typer

from jarvislabs.cli import state

JsonOption = Annotated[bool, typer.Option("--json", help="Output as JSON.")]
YesOption = Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompts.")]


def apply_command_options(*, json_output: bool = False, yes: bool = False) -> None:
    state.json_output = json_output
    state.yes = yes


def parse_kv(pairs: list[str] | None, *, flag: str, redact: bool = False) -> dict[str, str]:
    """Parse repeatable ``key=value`` flags. Split on the first `=`; last key wins.

    ``redact`` keeps the offending value out of the error message (for secrets).
    """
    from jarvislabs.cli import render  # render imports this module

    result: dict[str, str] = {}
    for raw in pairs or []:
        if "=" not in raw:
            shown = "" if redact else f" {raw!r}"
            render.die(f"Invalid {flag}{shown}: expected key=value")
        key, value = raw.split("=", 1)
        result[key] = value
    return result


def option_was_explicit(name: str) -> bool:
    """Tell validation apart from Typer defaults for options with meaningful defaults."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    return ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
