from __future__ import annotations

from typing import Annotated

import typer

from jarvislabs.cli import state

JsonOption = Annotated[bool, typer.Option("--json", help="Output as JSON.")]
YesOption = Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompts.")]


def apply_command_options(*, json_output: bool = False, yes: bool = False) -> None:
    state.json_output = json_output
    state.yes = yes
