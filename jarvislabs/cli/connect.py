"""`jl connect`: point this machine's coding harnesses at the Models API.

connect checks the account's subscription (without one, it offers to subscribe or to pay
from the wallet, and remembers the wallet choice), gets this machine's key (named
"jl-<hostname>-<4 hex digits>", charged to the plan or to the wallet, and kept in config.toml
for every harness), reads the model catalog, and writes the Jarvislabs provider, every model
and the default model into each named harness's config. With no harness named, it asks with a
picker of the installed harnesses; with none installed, it names three with their install pages
and exits 0. jl never installs a harness. "Not now" at the pay question, or ticking nothing,
exits 0 and saves nothing; Ctrl-C at any question exits 130. It first prints a preview row per
harness and asks once; --dry-run stops after the preview, having created or revoked no key. The
key is created only after the user says yes. Running it again refreshes the models and the
default.

A harness that can't be connected is skipped, and the others go on: one that isn't installed
(its install page is named), or a config file jl can't read. Before the preview, a warning per
skipped harness says why. When a harness was skipped and none is connected, connect exits 1,
and a key it created in this run is revoked.

--yes answers yes to every yes/no question: the preview and the switch from wallet to plan. The
no-subscription choice and the cap on active keys aren't yes/no questions: under --yes they still
stop with a message, as every question does without a terminal.

When the key is replaced (by --rotate-key, a switch between wallet and plan, or because it was
revoked on the account), every connected harness gets the new one.

After writing, connect sends one tiny request with the key to prove the setup works, then ends
with a "Next:" line naming the command that starts each tool connected in this run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn

import click
import questionary
import typer
from rich.text import Text

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.cli.subscription import subscribe
from jarvislabs.client import Client
from jarvislabs.config import (
    Connection,
    Replaced,
    load_config,
    load_connection,
    save_config,
    save_connection,
    wallet_chosen,
)
from jarvislabs.constants import MODELS_API_URL, RECOMMENDED_MODEL
from jarvislabs.exceptions import APIError, JarvislabsError, NotFoundError
from jarvislabs.harnesses import (
    HARNESS_DOCS_URL,
    HARNESSES,
    Harness,
    Plan,
    Setup,
    apply,
    check_version,
    default_model,
    installed,
    plan_connect,
    plan_disconnect,
    snippet,
    vscode_running,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from jarvislabs.models import CreatedApiKey, ModelInfo

MACHINE_ID_FILE = Path("/etc/machine-id")
API_KEYS_URL = "https://jarvislabs.ai/settings/api-keys"
SWITCH_TO_PLAN = "You have a subscription now. Switch this machine's tools from your wallet to your plan?"
SWITCH_HINT = (
    "You have a subscription, but this machine pays from your wallet. "
    "Run jl connect in a terminal to switch it to your plan, or run jl connect again with --yes."
)
NO_SUBSCRIPTION_FOUND = "No subscription found. If you've just subscribed, give it a minute and run jl connect again."
NOT_NOW = "Not now"
# Cline's extension rewrites its settings while VS Code runs, so jl waits for it to quit.
VSCODE_RUNNING_HINT = (
    "Quit VS Code first: Cline rewrites its settings while it runs. Then run `jl connect cline-vscode`."
)


@app.command(rich_help_panel="Coding plan")
def connect(
    harnesses: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="HARNESS...",
            help="Harness IDs to connect; run jl connect --list for IDs and names.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            metavar="MODEL",
            help=f"Default model for the harnesses (default: {RECOMMENDED_MODEL}). IDs: jl models list.",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the changes and make none: no file or key.")
    ] = False,
    show_all: Annotated[bool, typer.Option("--all", help="Include tools that aren't installed in the picker.")] = False,
    list_tools: Annotated[bool, typer.Option("--list", help="List every tool ID and name, then exit.")] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Answer yes to every question: write the changes, and move this machine from the wallet to your plan.",
        ),
    ] = False,
    print_harness: Annotated[
        str | None,
        typer.Option(
            "--print",
            metavar="HARNESS",
            help="Print a harness's config to paste by hand, key filled in. IDs: jl connect --list.",
        ),
    ] = None,
    rotate_key: Annotated[
        bool,
        typer.Option(
            "--rotate-key", help="Revoke this machine's key and write a new one into every connected harness."
        ),
    ] = False,
    wallet: Annotated[
        bool, typer.Option("--wallet", help="Pay from the wallet when the account has no subscription, without asking.")
    ] = False,
) -> None:
    """Write the Jarvislabs provider into your coding tools.

    Each tool gets the Models API URL, this machine's key, and the model catalog, so it can use your plan's models. With no tool named, jl asks which of the installed ones. Undo one tool with jl disconnect <tool>.

    [bold]Examples:[/bold]

      # Pick the tools to connect from a list
      [bold cyan]jl connect[/bold cyan]

      # List every tool ID and name
      [bold cyan]jl connect --list[/bold cyan]

      # Connect one tool, or see how to install it
      [bold cyan]jl connect qwen[/bold cyan]

      # Connect OpenCode, with GLM-5.3 as its default model
      [bold cyan]jl connect opencode --model glm-5.3[/bold cyan]

      # Print Zed's config, to paste by hand
      [bold cyan]jl connect --print zed[/bold cyan]
    """
    cli_options.apply_command_options(yes=yes)
    if list_tools:
        _print_harnesses()
        raise typer.Exit()
    if print_harness and dry_run:
        render.die("--print writes no file, so it takes no --dry-run. Run it without --dry-run.")
    # --print also takes a coming-soon harness, which _named_harnesses refuses.
    printed = _printable(print_harness) if print_harness else None
    # --rotate-key alone rewrites only the connected harnesses, without the picker.
    only_rotate = rotate_key and not harnesses
    if not (printed or only_rotate or harnesses):
        # First, even with no one to ask, and before the subscription check: without a tool there's
        # nothing to connect or pay for yet.
        _stop_without_a_tool()
    named = [] if printed or only_rotate else _named_harnesses(harnesses)
    client = get_client()
    kept = load_connection()
    if only_rotate and kept and not kept.harnesses:
        # No harness would hold a new key: the one a logged-out disconnect kept is revoked instead.
        if dry_run:
            render.info(f"Would revoke this machine's key {kept.name}. No tools are connected.")
            return
        _revoke_unused_key(client, kept)
        render.success(f"Revoked this machine's key {kept.name}. No tools are connected, so jl made no new one.")
        return
    if not (printed or only_rotate) and os.environ.get("JL_FROM_INSTALLER"):
        # The heading belongs to the guided installer flow. A standalone connect starts at its picker,
        # the same way disconnect starts at its preview.
        render.rule("CONNECT TOOLS")
        render.console.print(Text("Sets up the Jarvislabs provider in your coding tools.", "dim"))
    with render.spinner("Checking your subscription..."):
        usage = client.subscription.usage()
    if usage is None:
        _continue_on_wallet(wallet, dry_run)
    kind: Literal["payg", "subscription"] = "payg" if usage is None else "subscription"
    with render.spinner("Fetching models..."):
        models = tuple(client.models.list())
    default = model or RECOMMENDED_MODEL
    if default not in {m.id for m in models}:
        render.die(f"Unknown model {default}. See jl models list.")
    # A headline and a hint per harness that isn't connected, printed once before the preview.
    skipped: list[tuple[str, str]] = []
    if printed or only_rotate:
        chosen = []
    else:
        if not named and kept and kept.kind == kind:
            _stop_if_up_to_date(kept, models, client)
        chosen = _ready(named or _pick_harnesses(show_all), skipped=skipped)

    if printed:
        connection, new_key = _machine_key(client, kind, rotate_key)
        if new_key:
            # Before the printed harness is recorded: printing writes none of its files.
            refresh_connected()
        setup = Setup(base_url=MODELS_API_URL, api_key=connection.token, models=models, default_model=default)
        render.stdout_console.print(snippet(printed, setup), end="", markup=False, highlight=False, soft_wrap=True)
        # Recorded, with nothing replaced, so status shows it and the key stays while it's in use.
        connection.harnesses.setdefault(printed.id, Replaced())
        save_connection(connection)
        return
    # The preview doesn't show the key, so it's planned with the saved one, or none yet.
    saved = load_connection()
    token = saved.token if saved else ""
    setup = Setup(base_url=MODELS_API_URL, api_key=token, models=models, default_model=default)
    plans = []
    for harness in chosen:
        try:
            plans.append(plan_connect(harness, setup))
        except JarvislabsError as e:
            skipped.append((f"{harness.name} skipped", str(e)))
    _summarize(skipped, connected=bool(plans) or not chosen)
    if plans:
        changes = {plan.preview.change for plan in plans}
        verb = "update" if changes == {"update"} else "set up" if "update" in changes else "add"
        many = not all(plan.harness.one_model for plan in plans)
        shared = f"{len(models)} models, default {default}" if many else f"default {default}"
        render.change_rows([_row(plan) for plan in plans], f"jl will {verb} the Jarvislabs provider", shared)
    if dry_run:
        if rotate_key and kept:
            connectable = {h.id for h in HARNESSES if h.support == "connect"}
            tools = {h.id for h in chosen} | (set(kept.harnesses) & connectable)
            s = "" if len(tools) == 1 else "s"
            render.info(f"Would replace this machine's key {kept.name} and update {len(tools)} tool{s}.")
        return
    # --rotate-key alone has no file change to preview: asking for it is the yes.
    if plans:
        render.console.print()
        if not render.confirm("Write these changes?", skip=yes, default=True):
            raise typer.Exit(1 if skipped else 0)

    # Created only now, so a dry run or a "no" leaves no key behind.
    connection, new_key = _machine_key(client, kind, rotate_key)
    if connection.token != setup.api_key:
        setup = dataclasses.replace(setup, api_key=connection.token)
        plans = [plan_connect(plan.harness, setup) for plan in plans]

    connected = []
    width = max((len(plan.harness.name) for plan in plans), default=0)
    for plan in plans:
        harness = plan.harness
        previous = connection.harnesses.get(harness.id)
        try:
            replaced = apply(plan, previous)
        except OSError as e:
            path = e.filename or plan.writes[0].path
            if new_key and not connection.harnesses:
                # Made for this run and no tool holds it: don't leave it live on the account.
                _revoke_unused_key(client, connection)
            render.die(f"Could not write {path}: {e.strerror or e}. Fix it, then run jl connect again.")
        connection.harnesses[harness.id] = replaced
        save_connection(connection)
        connected.append(harness)
        label = "added" if previous is None else "updated"
        backups = [b for b in replaced.backups if not previous or b not in previous.backups]
        notes = [f"your old config is at {', '.join(render.short_path(b) for b in backups)}"] if backups else []
        if harness.note:
            notes.append(harness.note.format(model=default))
        render.change_row(render.Row("ok", harness.name, label, note="\n".join(notes)), width, render.console)
    if new_key:
        # The old key is revoked: every connected harness still holding it gets the new one.
        refreshed = refresh_connected()
        if rotate_key:
            updated = len(connected) + refreshed
            render.success(f"Replaced this machine's key; updated {updated} tool{'' if updated == 1 else 's'}.")
    if plans and not connected:
        if new_key and not connection.harnesses:
            # Made for this run, and no harness holds it.
            _revoke_unused_key(client, connection)
        raise typer.Exit(1)
    _test_request(connection.token, default)
    if connected:
        render.hint_line("Add another tool: `jl connect`")
        render.next_step("start a tool in your project folder:", [_start_command(h) for h in connected])


@app.command(rich_help_panel="Coding plan")
def disconnect(
    harness_id: Annotated[
        str | None,
        typer.Argument(
            metavar="HARNESS",
            help="The harness ID; with none, jl asks which. IDs: jl connect --list.",
        ),
    ] = None,
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Remove jl's provider and key from your coding tools.

    Your account stays signed in; jl logout also signs you out. With no tool named, jl asks which of the connected ones. Disconnecting the last tool also revokes this machine's key. The backups jl made at connect stay, and their paths are printed.
    """
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    chosen = [_named_harness(harness_id)] if harness_id is not None else _picked_harnesses(load_connection())
    _disconnect_many(chosen, yes=yes)


def _named_harness(harness_id: str) -> Harness:
    """Return the harness named on the command line, or stop naming the ones jl knows."""
    harness = next((h for h in HARNESSES if h.id == harness_id), None)
    if harness is None:
        render.die(f"Unknown tool {harness_id}. Choose from: {', '.join(h.id for h in HARNESSES)}.")
    return harness


def _picked_harnesses(connection: Connection | None) -> list[Harness]:
    """Return the connected tools to disconnect: the only one, or the ones ticked in the picker.

    With no terminal, or in --json, jl stops with the error it gives when no tool is named. With no
    tool connected it says so and exits 0.

    Raises:
        typer.Exit: No tool connected, nothing ticked, or Ctrl-C at the picker (130).
    """
    if not _can_ask():
        # The error of the argument when it was required.
        raise click.MissingParameter(param_hint="'HARNESS'", param_type="argument")
    connected = [harness for harness in HARNESSES if connection and harness.id in connection.harnesses]
    if not connected:
        render.info("No tools are connected.")
        raise typer.Exit()
    if len(connected) == 1:
        return connected
    picked = render.answer(
        questionary.checkbox(
            "Disconnect which tool?",
            choices=[questionary.Choice(harness.name, harness.id) for harness in connected],
            use_search_filter=True,
            use_jk_keys=False,  # j and k must type into the filter
            # The hint of connect's picker: Space toggles, Enter continues, typing filters.
            instruction=(f"\n  Space toggles{render.MARKS['sep']}Enter continues{render.MARKS['sep']}type to filter"),
            style=render.PROMPT_STYLE,
        )
    )
    if not picked:
        render.info("No tools disconnected.")
        raise typer.Exit()
    return [harness for harness in connected if harness.id in picked]


def _disconnect_many(chosen: list[Harness], *, yes: bool) -> None:
    """Show one removal preview for every chosen tool, confirm once, then remove them."""
    if state.json_output:
        keep = _files_another_tool_uses(chosen, load_connection())[0]
        for harness in chosen:
            _disconnect(harness, preview=False, keep=keep)
        return
    connection = load_connection()
    if connection is not None and not connection.harnesses:
        # A key kept by a logged-out disconnect: no tool holds it, so this run revokes it.
        for harness in chosen:
            _disconnect(harness, preview=False)
        return
    keep = _files_another_tool_uses(chosen, connection)
    plans = []
    for harness in chosen:
        if connection is None or harness.id not in connection.harnesses:
            render.info(f"{harness.name} isn't connected.")
            continue
        record = connection.harnesses[harness.id]
        plans.append((harness, plan_disconnect(harness, record, keep[0]), record))
    if not plans:
        return
    render.change_rows(
        [_row(plan) for _, plan, _ in plans], "Remove jl's setup", f"{len(plans)} tool{'s' if len(plans) != 1 else ''}"
    )
    for harness, _, record in plans:
        if record.untracked:
            render.warning(
                f"{harness.name} was connected by an older jl, which didn't record the settings it replaced, "
                "so jl takes out only its own and puts none back."
            )
    warned = set()
    for _, plan, _ in plans:
        touched = {write.path for write in plan.writes}
        for other, path in keep[1]:
            if path in touched and (other.id, path) not in warned:
                warned.add((other.id, path))
                render.warning(
                    f"{other.name} shares {render.short_path(path)} and loses the Jarvislabs provider too. "
                    f"Run jl connect {other.id} afterwards."
                )
    names = [harness.name for harness, _, _ in plans]
    label = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    if not render.confirm(f"Disconnect {label}?", skip=yes):
        return
    for index, (harness, _, _) in enumerate(plans):
        _disconnect(harness, preview=False, spaced=index == 0, keep=keep[0])


def _config_paths(harness: Harness) -> set[Path]:
    """Return the files a harness's writer edits, without a key or catalog; empty when it has no writer."""
    if harness.writer is None:
        return set()
    setup = Setup(base_url=MODELS_API_URL, api_key="", models=(), default_model=None)
    return {edit.path for edit in harness.writer(setup)}


def _files_another_tool_uses(
    chosen: list[Harness], connection: Connection | None
) -> tuple[set[Path], list[tuple[Harness, Path]]]:
    """Return the files a connected tool that isn't being disconnected still writes, and each owner.

    Kilo's CLI and VS Code extension share one config: disconnecting one must empty the file, not
    delete it or the other's settings with it. The pairs are for the warning naming the tool.
    """
    if connection is None:
        return set(), []
    chosen_ids = {harness.id for harness in chosen}
    shared: list[tuple[Harness, Path]] = []
    for other in HARNESSES:
        if other.support != "connect" or other.id in chosen_ids or other.id not in connection.harnesses:
            continue
        shared += [(other, path) for path in _config_paths(other)]
    return {path for _, path in shared}, shared


def _disconnect(harness: Harness, *, spaced: bool = False, preview: bool = True, keep: Collection[Path] = ()) -> None:
    """Remove jl's setup from one harness, revoking this machine's key when it was the last.

    Args:
        harness: The tool to clean up.
        spaced: Put a blank line before the result. The standalone command does; logout's batch does not.
        preview: Show the removal row and ask before writing. False when a combined preview already asked.
        keep: Files another connected tool still writes, which are emptied rather than deleted.
    """
    connection = load_connection()
    if connection and not connection.harnesses:
        # Kept by a disconnect without a Jarvis key: no harness holds this key any more.
        _revoke_unused_key(get_client(), connection)
        if state.json_output:
            render.print_json({"harness": harness.id, "connected": False, "key_revoked": True})
        else:
            render.success(f"Revoked this machine's key {connection.name}.")
        return
    if connection is None or harness.id not in connection.harnesses:
        if state.json_output:
            render.print_json({"harness": harness.id, "connected": False})
        else:
            render.info(f"{harness.name} isn't connected.")
        return
    # A printed harness counts too: it may hold the key by hand.
    last = len(connection.harnesses) == 1
    client = None
    if last:
        # Without a Jarvis key (after jl logout), the files are still cleaned; the revoke waits for a login.
        with contextlib.suppress(JarvislabsError):
            client = Client()
    record = connection.harnesses[harness.id]
    plan = plan_disconnect(harness, record, keep)
    if not state.json_output and preview:
        render.change_rows([_row(plan)])
        if record.untracked:
            render.warning(
                f"{harness.name} was connected by an older jl, which didn't record the settings it replaced, "
                "so jl takes out only its own and puts none back."
            )
    if preview and plan.writes and not render.confirm(f"Disconnect {harness.name}?", skip=state.yes):
        return

    apply(plan, record)
    del connection.harnesses[harness.id]
    # Saved before the key is revoked: a failed revoke then leaves the key saved, never a harness holding it.
    save_connection(connection)
    if client:
        try:
            _revoke(client, connection.key_id)
        except JarvislabsError as e:
            render.die(f"Disconnected {harness.name}, but jl couldn't revoke this machine's key {connection.name}: {e}")
        save_connection(None)
    backups = [backup for backup in record.backups if backup.exists()]
    revoked = client is not None
    if state.json_output:
        files = [str(w.path) for w in plan.writes]
        render.print_json(
            {"harness": harness.id, "files": files, "backups": [str(b) for b in backups], "key_revoked": revoked}
        )
        return
    hints = []
    if revoked:
        hints.append(f"revoked this machine's key {connection.name}")
    elif last:
        hints.append(
            f"Run `jl setup`, then `jl disconnect {harness.id}` again to revoke this machine's key, "
            f"or revoke {connection.name} at {API_KEYS_URL}."
        )
    hints += [f"your config from before jl is still at {render.short_path(backup)}" for backup in backups]
    if spaced:
        render.console.print()
    render.success(f"{harness.name} disconnected", "\n".join(hints))


def _row(plan: Plan) -> render.Row:
    """Return a plan's row in the preview: its change, the files it writes, and what it replaces."""
    preview = plan.preview
    files = render.paths_label([write.path for write in plan.writes])
    note = preview.detail if files else ""
    return render.Row(preview.change, plan.harness.name, files or preview.detail, preview.replaces, note)


def _test_request(api_key: str, model: str) -> None:
    """Send one tiny request with this machine's key and show how long the reply took, or the server's error."""
    started = time.monotonic()
    try:
        with render.spinner(f"Testing {model}..."):
            Client(api_key=api_key).models.chat(model, [{"role": "user", "content": "Say OK."}], max_tokens=16)
    except JarvislabsError as e:
        render.error(
            f"Test request to {model} failed", f"{e}\nYour tools are set up; they work once requests go through."
        )
        raise typer.Exit(1) from None
    render.info(f"API test passed in {time.monotonic() - started:.1f} s")


def _print_harnesses() -> None:
    """List every tool ID, its name, and whether jl can connect it or only print its config."""
    rows = [{"id": harness.id, "name": harness.name, "support": harness.support} for harness in HARNESSES]
    if state.json_output:
        render.print_json(rows)
        return
    render.heading("Coding tools")
    for row in rows:
        label = "connect" if row["support"] == "connect" else "print only"
        render.stdout_console.print(f"  {row['id']:<16} {row['name']:<22} {label}")
    render.hint_line("connect: jl writes the config · print only: jl prints it to paste by hand")


def _start_command(harness: Harness) -> str:
    """Return how to start a harness here: its command, or for a VS Code extension, open VS Code."""
    return "open VS Code" if harness.kind == "vscode" else shlex.join(harness.launch)


def _named_harnesses(ids: list[str] | None) -> list[Harness]:
    """Return the harnesses named on the command line, or [] to ask with the picker."""
    if not ids:
        if state.json_output or not render.can_prompt():
            render.die("Name the tools to connect, e.g. jl connect opencode.")
        return []
    by_id = {h.id: h for h in HARNESSES}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        supported = ", ".join(h.id for h in HARNESSES if h.support == "connect")
        render.die(f"Unknown tool {', '.join(unknown)}. Choose from: {supported}.")
    for harness in (by_id[i] for i in ids):
        if harness.support == "coming_soon":
            render.die(f"{harness.name} is coming soon: see {harness.docs_url}")
    return [by_id[i] for i in ids]


def _stop_without_a_tool() -> None:
    """With no supported harness installed, name three with their install pages and exit 0.

    A last line points users of a coming-soon harness to its docs, and to --print where it works.

    Raises:
        typer.Exit: No supported harness is installed.
    """
    supported = [h for h in HARNESSES if h.support == "connect"]
    if any(installed(h) for h in supported):
        return
    # Three popular ones, so the list stays short; --list has the rest.
    offered = [h for i in ("opencode", "pi", "kilo") for h in supported if h.id == i]
    width = max(len(h.name) for h in offered)
    coming = [h for h in HARNESSES if h.support == "coming_soon"]

    body = Text()
    body.append("Install one, then run ", "dim")
    body.append("jl connect", render.ACCENT_BOLD)
    body.append(":\n", "dim")
    for harness in offered:
        body.append(f"{harness.name:<{width}}  ", "bold")
        body.append(harness.install_url, "accent")
        body.append("\n")
    if coming:
        names = ", ".join(h.name for h in coming[:-1])
        names = f"{names} or {coming[-1].name}" if names else coming[-1].name
        body.append(f"\nUsing {names}? jl doesn't set them up yet: ", "dim")
        body.append(HARNESS_DOCS_URL, "accent")
        body.append(".", "dim")
    body.append("\n\nJust installed one? Open a new terminal, then run ", "dim")
    body.append("jl connect", render.ACCENT_BOLD)
    body.append(".", "dim")

    render.console.print()
    render.warning("No supported coding tool found", body)
    raise typer.Exit()
    raise typer.Exit()


def _stop_if_up_to_date(connection: Connection | None, models: tuple[ModelInfo, ...], client: Client) -> None:
    """Say every installed tool is connected and current, and exit, when a refresh would change nothing.

    Args:
        connection: This machine's saved key and harnesses, from an earlier connect.
        models: The current catalog.
        client: The API client, to check the saved key is still active.

    Raises:
        typer.Exit: Every installed tool is connected, and no refresh changes a file.
    """
    supported = [harness for harness in HARNESSES if harness.support == "connect"]
    installed_tools = [harness for harness in supported if installed(harness)]
    if not connection or not installed_tools:
        return
    if not {harness.id for harness in installed_tools} <= set(connection.harnesses):
        return
    # A key revoked on the console leaves every tool broken: the normal flow makes a new one.
    if connection.key_id not in {key.id for key in client.account.list_api_keys()}:
        return
    try:
        if not all(_refreshes_to_nothing(harness, connection, models) for harness in installed_tools):
            return
    except JarvislabsError:
        # A config jl can't read is a change to plan, not a reason to stop before the picker.
        return
    names = [harness.name for harness in installed_tools]
    label = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    render.console.print()
    render.success(f"{label} {'is' if len(names) == 1 else 'are'} already connected")
    render.next_step("check or change them:", ["jl status", "jl disconnect <tool>"])
    raise typer.Exit()


def _pick_harnesses(show_all: bool = False) -> list[Harness]:
    """Ask which on-machine harnesses to connect: the installed ones, ticked.

    Tools that aren't installed, and coming-soon ones, are left out unless ``show_all``; one line
    names the flag. Nothing ticked says how to come back and exits 0.

    Args:
        show_all: List the tools that aren't installed and the coming-soon ones too.

    Returns:
        The harnesses picked.

    Raises:
        typer.Exit: Nothing ticked; with 130, Ctrl-C.
    """
    supported = [h for h in HARNESSES if h.support == "connect"]
    ticked = [h for h in supported if installed(h)]
    if len(ticked) == 1 and not show_all:
        render.info(f"{ticked[0].name} is the only installed tool.")
        return ticked
    choices: list[questionary.Choice] = [questionary.Choice(h.name, h.id, checked=True) for h in ticked]
    if show_all:
        groups = (
            ("Not installed: run the command to see where to get it", [h for h in supported if h not in ticked]),
            ("Coming soon: run the command for its docs", [h for h in HARNESSES if h.support == "coming_soon"]),
        )
        for heading, group in groups:
            if group:
                choices.append(questionary.Separator(heading))
                # questionary puts the reason in brackets.
                choices += [questionary.Choice(h.name, h.id, disabled=f"jl connect {h.id}") for h in group]
    picked = render.answer(
        questionary.checkbox(
            "Connect which tools?",
            choices=choices,
            use_search_filter=True,
            use_jk_keys=False,  # j and k must type into the filter
            # On its own line: Enter continues with the ticked rows, not the highlighted one. Replaces
            # questionary's own help, which names ctrl-a for both toggling and inverting.
            instruction=(f"\n  Space toggles{render.MARKS['sep']}Enter continues{render.MARKS['sep']}type to filter"),
            style=render.PROMPT_STYLE,
        )
    )
    if not picked:
        _leave_for_now()
    return [h for h in HARNESSES if h.id in picked]


def _install_hint(harness: Harness) -> str:
    """Return where to get a harness: its install page, or for a VS Code extension the command that installs it."""
    if harness.kind == "vscode":
        return " ".join(("code", "--install-extension", *harness.extension))
    return harness.install_url or ""


def _ready(chosen: list[Harness], *, skipped: list[tuple[str, str]]) -> list[Harness]:
    """Return the chosen harnesses that can be connected.

    A harness that isn't installed is left out, with a hint naming its install page (for a VS
    Code extension, the command that installs it); so is an installed one older than the version
    jl's config needs, and Cline's extension while VS Code runs, since Cline rewrites its
    settings underneath jl. With nothing left to connect, prints the skipped warnings and exits 1.

    Args:
        chosen: The harnesses to connect.
        skipped: A headline and a hint per harness left out, which this appends to.

    Returns:
        The harnesses to connect, in the order chosen.

    Raises:
        typer.Exit: None of the chosen harnesses can be connected.
    """
    ready = []
    for harness in chosen:
        if not installed(harness):
            if harness.kind == "vscode":
                reason, where = "the extension isn't installed", f"Install it with `{_install_hint(harness)}`"
            else:
                reason, where = "not installed", f"Install it from {_install_hint(harness)}"
            skipped.append((f"{harness.name} skipped: {reason}", f"{where}, then run `jl connect {harness.id}`."))
            continue
        if harness.id == "cline-vscode" and vscode_running():
            skipped.append((f"{harness.name} skipped: VS Code is running", VSCODE_RUNNING_HINT))
            continue
        try:
            check_version(harness)
        except JarvislabsError as e:
            skipped.append((f"{harness.name} skipped", str(e)))
            continue
        ready.append(harness)
    if not ready:
        _summarize(skipped, connected=False)
    return ready


def _summarize(skipped: list[tuple[str, str]], *, connected: bool) -> None:
    """Print a warning per harness that wasn't connected; with none connected, exit 1.

    Args:
        skipped: A headline naming each harness left out and why, and a hint saying what to do.
        connected: Whether this run connected any harness.

    Raises:
        typer.Exit: Nothing was connected.
    """
    for headline, hint in skipped:
        render.warning(headline, hint)
    if skipped:
        render.console.print()
    if not connected:
        raise typer.Exit(1)


def _printable(harness_id: str) -> Harness:
    harness = next((h for h in HARNESSES if h.id == harness_id and h.writer), None)
    if harness is None:
        choices = ", ".join(h.id for h in HARNESSES if h.writer)
        render.die(f"jl can't print a config for {harness_id}. Choose from: {choices}.")
    return harness


def _continue_on_wallet(wallet: bool, dry_run: bool) -> None:
    """Return when this machine pays from the wallet; else offer to subscribe, the wallet or "Not now".

    The wallet is chosen by ``wallet`` (--wallet), or by a saved wallet key, the last one this
    machine had, or an earlier "Continue on wallet": the user chose it before, so they aren't asked
    again. "Continue on wallet" is saved at once, except in a dry run. Subscribe now opens the
    subscribe page and exits. "Not now" says how to come back and exits 0, saving nothing, so the
    next run asks again. With --yes or no one to ask, connect stops: no choice is a plain yes.

    Args:
        wallet: --wallet was passed.
        dry_run: --dry-run was passed: the choice isn't saved.
    """
    if wallet or wallet_chosen():
        render.console.print()
        hint = Text("Requests are charged to your wallet. Run ", "dim")
        hint.append("jl subscribe", render.ACCENT_BOLD)
        hint.append(" for a plan.", "dim")
        render.warning("On wallet (no subscription)", hint)
        render.console.print()
        return
    if state.yes or not _can_ask():
        render.die(f"{NO_SUBSCRIPTION_FOUND} To pay from your wallet instead, run jl connect --wallet.")
    hint = Text("Just subscribed? Give it a minute, then run ", "dim")
    hint.append("jl connect", render.ACCENT_BOLD)
    hint.append(" again.", "dim")
    render.warning("No subscription found", hint)
    render.console.print()
    choice = render.answer(
        questionary.select(
            "How do you want to pay for requests?",
            # Wallet first: Enter must not open a purchase page by accident.
            choices=["Continue on wallet", "Subscribe now", NOT_NOW],
            instruction="(arrow keys, Enter)",
            style=render.PROMPT_STYLE,
        )
    )
    if choice == NOT_NOW:
        _leave_for_now()
    if choice == "Subscribe now":
        subscribe()
        raise typer.Exit()
    if dry_run:
        return
    # Saved now, so the next run doesn't ask again when this one stops before creating the key.
    saved = load_config()
    saved["pays_from_wallet"] = True
    save_config(saved)


def _leave_for_now() -> NoReturn:
    """Say how to come back to jl connect, and exit 0 without saving anything.

    Raises:
        typer.Exit: Always.
    """
    render.next_step("set up a coding tool when you want:", ["jl connect"])
    raise typer.Exit()


def _machine_key(client: Client, kind: Literal["payg", "subscription"], rotate: bool) -> tuple[Connection, bool]:
    """Return this machine's key, and whether it was just created.

    Creates a key of ``kind`` when none is saved, or when the saved one is no longer active on
    the account (revoked on the console, or by a replace that failed). A saved key is replaced
    when ``rotate`` is set, when it's a wallet key and the user agrees to switch to the plan
    (always with --yes; with no one to ask, the wallet key stays), or when it's a subscription key
    and the account has no plan. Replacing revokes the saved key first, since the new key
    takes its name.

    Args:
        client: The client holding the Jarvis key.
        kind: The kind the account can use now: "subscription" with a plan, else "payg".
        rotate: Replace the saved key even when its kind is right.
    """
    connection = load_connection()
    replaced_key = None
    if connection and connection.key_id not in {key.id for key in client.account.list_api_keys()}:
        render.info(f"This machine's key {connection.name} was revoked, so jl creates a new one.")
    elif connection:
        switch = connection.kind != kind and (kind == "payg" or _agrees_to_switch())
        if not (switch or rotate):
            return connection, False
        kind = kind if switch else connection.kind
        _revoke(client, connection.key_id)
        replaced_key = connection.name
    try:
        key = _create_key(client, kind)
    except JarvislabsError as e:
        if replaced_key:
            raise JarvislabsError(
                f"{e} jl had already revoked {replaced_key}, so your connected tools hold a dead key: "
                "run jl connect to write a new one."
            ) from e
        raise
    harnesses = connection.harnesses if connection else {}
    connection = Connection(key_id=key.id, name=key.name, kind=key.kind, token=key.token, harnesses=harnesses)
    # Saved at once: the backend never shows the secret again.
    save_connection(connection)
    return connection, True


def _agrees_to_switch() -> bool:
    """Return whether the user agrees to switch to the plan: asked in a terminal, or by --yes.

    With no one to ask and no --yes, prints how to switch and returns False.
    """
    if state.yes:
        render.info("You have a subscription now, so jl switches this machine's tools to your plan.")
        return True
    if _can_ask():
        return render.confirm(SWITCH_TO_PLAN)
    render.info(SWITCH_HINT)
    return False


def _create_key(client: Client, kind: Literal["payg", "subscription"]) -> CreatedApiKey:
    """Create this machine's key, of ``kind``, on the account.

    At the cap on active keys, offers to revoke one of jl's keys first. A key of another kind
    is revoked at once, and connect stops.
    """
    key = _create_making_room(client, kind)
    if key.kind != kind:
        # A key of another kind than the one asked for must never reach a harness.
        try:
            _revoke(client, key.id)
        except JarvislabsError as error:
            render.die(f"jl couldn't revoke the key {key.name} it just created ({error}). Revoke it at {API_KEYS_URL}.")
        render.die(
            "Subscriptions aren't available yet. Run jl connect again later."
            if kind == "subscription"
            else f"The server created a {key.kind} key instead of a wallet key, so jl revoked it."
        )
    return key


def _create_making_room(client: Client, kind: Literal["payg", "subscription"]) -> CreatedApiKey:
    name = _key_name()
    # A full account can need both: a free slot (429), then the name freed (409).
    for _ in range(2):
        try:
            with render.spinner("Creating this machine's key..."):
                return client.account.create_api_key(name, kind)
        except APIError as error:
            if error.status_code == 409:
                _revoke_lost_key(client, name)
            elif error.status_code == 429:
                _free_a_key_slot(client, error.message)
            else:
                raise
    with render.spinner("Creating this machine's key..."):
        return client.account.create_api_key(name, kind)


def _revoke_lost_key(client: Client, name: str) -> None:
    """Revoke the account's key named ``name``, whose secret this machine lost."""
    for key in client.account.list_api_keys():
        if key.name == name:
            _revoke(client, key.id)
            render.info(f"Revoked the key {name}, whose secret this machine no longer has.")


def _free_a_key_slot(client: Client, cap_message: str) -> None:
    """Revoke this machine's lost key, or the jl key the user picks, to get under the cap on active keys.

    A key with this machine's name is revoked without asking. Otherwise this stops with the
    account's keys listed when there's no jl key to offer, with --yes or no one to ask (which key
    to revoke isn't a plain yes), or when the user picks none.
    """
    keys = client.account.list_api_keys()
    if any(key.name == _key_name() for key in keys):
        # Always a lost key: jl creates one only when no live key is saved. It frees a slot and the name.
        _revoke_lost_key(client, _key_name())
        return
    ours = [key for key in keys if key.name.startswith("jl-")]
    rows = {
        key.id: (
            key.name,
            "plan" if key.kind == "subscription" else "wallet",
            f"{key.created_at:%Y-%m-%d}" if key.created_at else "-",
        )
        for key in keys
    }
    give_up = "\n".join(
        [
            cap_message,
            *(f"  {name} ({pays}, created {created})" for name, pays, created in rows.values()),
            f"Revoke one at {API_KEYS_URL}, then run jl connect again.",
        ]
    )
    if not ours or state.yes or not _can_ask():
        render.die(give_up)
    render.warning(cap_message)
    render.info(
        "Revoking another machine's key disconnects the coding tools on that machine until it runs jl connect again."
    )
    key_id = render.select_row(
        "Revoke one of jl's keys to make room for this machine's key:",
        ("NAME", "MACHINE", "PAYS FROM", "CREATED"),
        [
            *((name, _key_machine(name), *rest) for name, *rest in (rows[key.id] for key in ours)),
            ("Keep every key", "", "", ""),
        ],
        [*(key.id for key in ours), ""],
    )
    if not key_id:
        render.die(give_up)
    _revoke(client, key_id)


def _key_machine(name: str) -> str:
    """Return the hostname of the machine a jl key belongs to, from the key's name.

    Args:
        name: The key's name, "jl-<hostname>-<4 hex digits>", or "jl-<hostname>" from an older jl.

    Returns:
        The hostname, e.g. "DESKTOP-1A2B3C" for "jl-DESKTOP-1A2B3C-c2a1".
    """
    return re.sub(r"^jl-|-[0-9a-f]{4}$", "", name)


def _revoke_unused_key(client: Client, connection: Connection) -> None:
    """Revoke this machine's key, which no harness holds any more, and forget it."""
    try:
        _revoke(client, connection.key_id)
    except JarvislabsError as e:
        render.die(f"jl couldn't revoke this machine's key {connection.name}: {e}")
    save_connection(None)


def _can_ask() -> bool:
    """Return whether there's a person to ask: in a terminal, without --json.

    --yes is checked by each question, since some take it as the answer and some still stop.
    """
    return render.can_prompt() and not state.json_output


def _revoke(client: Client, key_id: str) -> None:
    """Revoke an API key on the account; one already gone counts as revoked."""
    with contextlib.suppress(NotFoundError):
        client.account.revoke_api_key(key_id)


def _key_name() -> str:
    """Return this machine's key name, "jl-<hostname>-<4 hex digits>".

    The digits come from a hash of the machine ID: they tell apart machines with the same
    hostname, and stay the same when jl's config is lost.
    """
    # Key names allow letters, digits, spaces, "-" and "_"; macOS hostnames end in ".local".
    host = re.sub(r"[^A-Za-z0-9_-]", "-", socket.gethostname().split(".")[0])
    return f"jl-{host}-{hashlib.sha256(_machine_id().encode()).hexdigest()[:4]}"


def _machine_id() -> str:
    """Return the OS's ID for this machine, or a random ID saved in config.toml."""
    machine_id = _os_machine_id()
    if machine_id:
        return machine_id
    saved = load_config()
    if "machine_id" not in saved:
        saved["machine_id"] = uuid.uuid4().hex
        save_config(saved)
    return saved["machine_id"]


def _os_machine_id() -> str | None:
    """Return the ID the OS gives this machine, or None when it can't be read.

    Reads ``MachineGuid`` from the registry on Windows, ``IOPlatformUUID`` from ``ioreg`` on
    macOS, and ``/etc/machine-id`` elsewhere.
    """
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                return str(winreg.QueryValueEx(key, "MachineGuid")[0]) or None
        if sys.platform == "darwin":
            ioreg = ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"]
            output = subprocess.run(ioreg, capture_output=True, text=True, check=True).stdout
            match = re.search(r'"IOPlatformUUID" = "([^"]+)"', output)
            return match.group(1) if match else None
        return MACHINE_ID_FILE.read_text().strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _refresh_plan(
    harness: Harness, connection: Connection, models: tuple[ModelInfo, ...]
) -> tuple[Plan, str | None, str | None]:
    """Return the plan that refreshes a harness's models, and the default it keeps or falls back to."""
    current = default_model(harness)
    default = RECOMMENDED_MODEL if current and current not in {m.id for m in models} else None
    setup = Setup(base_url=MODELS_API_URL, api_key=connection.token, models=models, default_model=default)
    return plan_connect(harness, setup), current, default


def _unchanged(plan: Plan) -> bool:
    """Return whether writing a plan would change no file."""
    return all(write.path.exists() and write.path.read_text(encoding="utf-8") == write.text for write in plan.writes)


def _refreshes_to_nothing(harness: Harness, connection: Connection, models: tuple[ModelInfo, ...]) -> bool:
    """Return whether refreshing a harness's models would change no file."""
    return _unchanged(_refresh_plan(harness, connection, models)[0])


def refresh_connected() -> int:
    """Rewrite the models in this machine's connected harnesses from the current catalog.

    The harness's default model stays as it is, unless it is a Jarvislabs model the catalog no
    longer has: then it becomes the recommended model, with a warning. Prints a line per
    harness whose config changed. Problems are warnings, not errors, because jl update calls
    this after updating jl.

    Returns:
        The number of harnesses whose config changed.
    """
    connection = load_connection()
    if not connection:
        return 0
    connected = [h for h in HARNESSES if h.support == "connect" and h.id in connection.harnesses]
    if not connected:
        return 0
    try:
        models = tuple(Client().models.list())
    except JarvislabsError as e:
        render.warning(f"Could not refresh the models in your coding tools: {e}")
        return 0
    refreshed = 0
    for harness in connected:
        try:
            plan, current, default = _refresh_plan(harness, connection, models)
        except JarvislabsError as e:
            render.warning(str(e))
            continue
        if _unchanged(plan):
            continue
        # Recorded, so disconnect also takes out the entries of models added now.
        connection.harnesses[harness.id] = apply(plan, connection.harnesses.get(harness.id))
        save_connection(connection)
        refreshed += 1
        # A one-model harness holds no model list to count; a replaced default gets the warning below.
        render.success(
            f"Refreshed {harness.name}." if harness.one_model else f"Refreshed {harness.name}: {len(models)} models."
        )
        if default:
            render.warning(f"{current} is no longer offered, so {harness.name}'s default model is now {default}.")
    return refreshed
