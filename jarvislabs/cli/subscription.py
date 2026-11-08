"""`jl usage` and `jl subscribe`: the account's Models API plan, from the terminal.

`jl usage` shows the plan and its usage, `use-balance` sets whether the wallet pays once the plan's
limits are reached, and `manage` opens the Stripe billing page. `jl subscribe` opens the console's
subscribe page. Pages are opened in the browser and their URL printed; without a desktop to open a
browser in (over SSH, or Linux without X11 or Wayland) and no BROWSER set, only the URL is printed.
"""

from __future__ import annotations

import os
import sys
import webbrowser
from typing import Annotated

import click
import typer
from rich.text import Text

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.constants import SUBSCRIBE_URL
from jarvislabs.exceptions import NotFoundError

subscription_app = typer.Typer(
    context_settings=cli_options.HELP_CONTEXT,
    name="usage",
    help="Your plan's usage and limits, use balance, and billing. With no command, shows usage.",
    invoke_without_command=True,
)
app.add_typer(subscription_app, rich_help_panel="Coding plan")


@subscription_app.callback()
def usage_show(ctx: typer.Context, json_output: cli_options.JsonOption = False) -> None:
    """Show your plan, its renewal date and limits."""
    if ctx.invoked_subcommand:
        return
    cli_options.apply_command_options(json_output=json_output)
    with render.spinner("Fetching subscription..."):
        usage = get_client().subscription.usage()

    if state.json_output:
        render.print_json({"usage": usage})
        return
    if usage is None:
        render.warning("No subscription", "Requests are charged to your wallet.")
        render.next_step("get a plan", ["jl subscribe"])
        return
    render.subscription_usage(usage)
    windows = usage.windows
    # A window under 80% still has room, so the wallet is not the next step yet.
    near_a_limit = any(w.percent_used >= 80 for w in (windows.five_hour, windows.weekly, windows.monthly))
    if not usage.use_balance and near_a_limit:
        render.next_step("pay from your wallet once a limit is reached", ["jl usage use-balance on"])


@subscription_app.command("use-balance")
def use_balance(
    setting: Annotated[str, typer.Argument(click_type=click.Choice(["on", "off"]), metavar="on|off")],
    json_output: cli_options.JsonOption = False,
) -> None:
    """Pay from the wallet once the plan's limits are reached (on), or refuse those requests (off)."""
    cli_options.apply_command_options(json_output=json_output)
    try:
        with render.spinner("Saving..."):
            on = get_client().subscription.set_use_balance(setting == "on")
    except NotFoundError:
        render.die(render.NO_SUBSCRIPTION)

    if state.json_output:
        render.print_json({"use_balance": on})
        return
    if on:
        render.success("Use balance is on: once the plan's limits are reached, requests are paid from your wallet.")
    else:
        render.success("Use balance is off: once the plan's limits are reached, requests are refused until they reset.")


@subscription_app.command("manage")
def manage() -> None:
    """Open your Stripe billing page, where you manage or cancel your subscription."""
    cli_options.apply_command_options()
    try:
        with render.spinner("Opening the billing page..."):
            url = get_client().subscription.portal_url()
    except NotFoundError:
        render.die(render.NO_SUBSCRIPTION)
    _open(url)


@app.command(rich_help_panel="Coding plan")
def subscribe() -> None:
    """Open the page where you subscribe to a Models API plan."""
    cli_options.apply_command_options()
    _open(SUBSCRIBE_URL)
    render.next_step("after paying, connect your tools", ["jl connect"])


def _open(url: str) -> None:
    # Without a desktop (over SSH, or Linux without X11 or Wayland), webbrowser falls back to a
    # text browser that takes over the terminal. BROWSER names one that works there, such as VS Code
    # Remote-SSH's and Codespaces', which opens the page in the user's own browser.
    no_desktop = not os.environ.get("BROWSER") and (
        os.environ.get("SSH_CONNECTION")
        or (
            sys.platform not in ("darwin", "win32")
            and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        )
    )
    if no_desktop:
        render.stdout_console.print(Text.assemble("Open ", render.link(url), " in your browser."), soft_wrap=True)
        return
    render.stdout_console.print(Text.assemble("Opening ", render.link(url)), soft_wrap=True)
    webbrowser.open(url)
