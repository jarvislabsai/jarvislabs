"""jl setup — authenticate and install agent skills."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import questionary
import typer
from rich.markup import escape
from rich.text import Text

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app
from jarvislabs.config import load_config, save_config

if TYPE_CHECKING:
    from jarvislabs.client import Client

# Universal path — always installed, covers most agents
UNIVERSAL_PATH = Path("~/.agents/skills/jarvislabs/SKILL.md")
UNIVERSAL_AGENTS = [
    "Codex",
    "Cursor",
    "OpenCode",
    "Amp",
    "Cline",
    "Gemini CLI",
    "GitHub Copilot",
    "Kimi Code CLI",
    "Warp",
]

# Additional agent-specific paths (agents with their own skill directories)
ADDITIONAL_AGENTS: dict[str, tuple[str, Path]] = {
    "claude-code": ("Claude Code", Path("~/.claude/skills/jarvislabs/SKILL.md")),
}

WORDMARK = """\
    ▄▄▄▄▄▖                       ▄▄▖        ▐██          ██▌
    ▀▀▀▜█▌                       ▝▀▘        ▐██          ██▌
       ▐█▌ ▄█████▙▖ ▗▟██ ██▌  ██ ██▌▗▟█████ ▐██ ▗█████▙▖ ██▙████▄ ▗█████▙
██▖    ▐█▌▗██▘  ▜█▙ ██▛  ██▌ ▗██ ██▌▝██▙▄▄▖ ▐██ ██▘  ▜██ ██▛  ▝██▖▜██▄▄▄
▜█▙▖  ▄██▘▝██▄ ▗▟██ ██▌  ██▙▄██▛ ██▌  ▝▀▀██▌▐██ ██▄  ▟██ ▜█▙▖ ▄██▘  ▀▀▜██
 ▀▜████▛▘  ▀████▜██ ██▘  ████▀▘  ▜█▌ █████▛▘▐██ ▝▜███▀██  ▀████▛▘ ▐█████▀
"""


# ── Login ────────────────────────────────────────────────────────────────────


def _ask_key() -> str:
    """Ask for the Jarvis key, drawing one * per character typed.

    A script with no terminal reads a line of stdin instead, so a piped key still works.
    """
    if not render.can_prompt():
        return render.console.input("[yellow]?[/yellow] Jarvis key: ", password=True).strip()
    return render.answer(questionary.password("Jarvis key:", style=render.PROMPT_STYLE)).strip()


def _login_flow(token: str | None) -> Client:
    """Authenticate and save token. Returns authenticated Client. Dies on failure."""
    from jarvislabs.client import Client
    from jarvislabs.exceptions import JarvislabsError

    config = load_config()
    existing = config.get("auth", {}).get("token")

    # Already authenticated — ask to re-authenticate
    if existing and not token:
        # The header ends without a blank line, and a login line brings none of its own.
        render.console.print()
        try:
            client = Client(api_key=existing)
            info = client.account.user_info()
            render.success(f"Logged in as {info.name}")
            if state.yes or not render.confirm("Re-authenticate with a different account?", optional=True):
                return client
        except JarvislabsError:
            render.warning("Saved Jarvis key is invalid.")

    # Prompt for token; a typed key that fails is asked for again, up to 3 tries.
    typed = not token
    if typed:
        render.rule("SIGN IN")
        render.info(Text.assemble("Get your Jarvis key at: ", render.link("https://jarvislabs.ai/settings/api-keys")))
        render.console.print()
    for attempt in range(1, 4):
        if typed:
            token = _ask_key()
        if not token:
            render.die("No Jarvis key provided.")
        try:
            with render.spinner("Authenticating..."):
                client = Client(api_key=token)
                info = client.account.user_info()
            break
        except JarvislabsError as e:
            if not typed or attempt == 3:
                render.die(f"Authentication failed: {e}")
            render.warning(f"Authentication failed: {e}. Try again.")

    config.setdefault("auth", {})["token"] = token
    save_config(config)
    if not typed:
        # A --token login has no SIGN IN rule to open the block, so the blank line comes here.
        render.console.print()
    render.success(f"Logged in as {info.name}")
    return client


# ── Skill loading ────────────────────────────────────────────────────────────


def _load_bundled_skill() -> str:
    """Load SKILL.md bundled with the package."""
    import importlib.resources

    try:
        return (importlib.resources.files("jarvislabs.skills") / "SKILL.md").read_text(encoding="utf-8")
    except Exception:
        render.die("Could not find SKILL.md. Try reinstalling jarvislabs.")


def _installed_skill_paths() -> list[Path]:
    """Skill files that exist on disk, once each (~/.claude/skills/jarvislabs can be a symlink to ~/.agents)."""
    paths: dict[Path, Path] = {}
    for path in (p.expanduser() for p in [UNIVERSAL_PATH, *(p for _label, p in ADDITIONAL_AGENTS.values())]):
        if path.is_file():
            paths.setdefault(path.resolve(), path)
    return list(paths.values())


def _stale_skill_paths(content: str) -> list[Path]:
    return [p for p in _installed_skill_paths() if p.read_text(encoding="utf-8", errors="replace") != content]


def skills_current() -> bool | None:
    """True if every installed skill file matches the bundled one. None if no skill is installed."""
    if not _installed_skill_paths():
        return None
    return not _stale_skill_paths(_load_bundled_skill())


def refresh_skills() -> list[Path]:
    """Overwrite installed skill files that differ from the bundled one. Never creates new files."""
    content = _load_bundled_skill()
    refreshed = []
    for path in _stale_skill_paths(content):
        try:
            _install_skill_to_path(content, path)
        except OSError as e:
            render.warning(f"Could not update the agent skill at {path}: {e.strerror or e}")
            continue
        refreshed.append(path)
        # No wrapping: agents read this path, and a long path would be split across lines.
        render.console.print(
            f"[green]✓[/green] Refreshed agent skill: {escape(str(path))}", soft_wrap=True, highlight=False
        )
    return refreshed


# ── Agent selection ──────────────────────────────────────────────────────────


def _agent_on_machine(key: str) -> bool:
    """Return whether an additional agent is on this machine: Claude Code's folder, or its command, is."""
    return key == "claude-code" and ((Path.home() / ".claude").exists() or shutil.which("claude") is not None)


def _select_additional_agents_interactive() -> list[str]:
    """Return the additional agent-specific paths to install for.

    With one agent there is no question: it is installed when it is on the machine. A picker with a
    single row is worse than no picker. More agents get the multi-select, each detected one ticked.
    """
    if not ADDITIONAL_AGENTS:
        return []

    detected = [key for key in ADDITIONAL_AGENTS if _agent_on_machine(key)]
    if len(ADDITIONAL_AGENTS) == 1:
        return detected

    choices = [
        questionary.Choice(title=label, value=key, checked=key in detected)
        for key, (label, _) in ADDITIONAL_AGENTS.items()
    ]
    question = questionary.checkbox(
        "Extra agent folders (optional):",
        choices=choices,
        instruction=f"\n  Space toggles{render.MARKS['sep']}Enter continues",
        style=render.PROMPT_STYLE,
    )
    return render.answer(question)


def _parse_agents_flag(agents: str) -> list[str]:
    """Parse --agents flag value into validated list of additional agent keys."""
    if agents == "all":
        return list(ADDITIONAL_AGENTS.keys())
    agent_list = [a.strip() for a in agents.split(",")]
    valid_keys = {*ADDITIONAL_AGENTS.keys(), "universal"}
    invalid = [a for a in agent_list if a not in valid_keys]
    if invalid:
        render.die(f"Unknown agents: {', '.join(invalid)}. Valid: {', '.join(ADDITIONAL_AGENTS.keys())}, all")
    # Filter out "universal" since it's always installed
    return [a for a in agent_list if a != "universal"]


# ── Skill installation ───────────────────────────────────────────────────────


def _install_skill_to_path(content: str, path: Path) -> Path:
    """Write skill file to a single path. Returns resolved path."""
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return resolved


def _skill_install_flow(agents_flag: str | None) -> list[tuple[str, Path]]:
    """Run the skill install flow.

    Returns:
        A pair per skill file installed: the agents it serves, and its path.
    """
    render.rule("AGENT SKILLS")
    render.info("Skills teach your coding agents how to use jl.")
    render.console.print()

    # Yes in a terminal, where the installer tells users to press Enter at every question; no for a script.
    asked = "Install agent skills?"
    if not agents_flag and not render.confirm(asked, skip=state.yes, default=render.can_prompt(), optional=True):
        return []

    # Determine additional agent-specific installs (validate before any writes)
    if agents_flag:
        additional = _parse_agents_flag(agents_flag)
    elif state.yes:
        # --yes auto-selects all additional agents
        additional = list(ADDITIONAL_AGENTS.keys())
    else:
        additional = _select_additional_agents_interactive()

    content = _load_bundled_skill()
    installed: list[tuple[str, Path]] = []

    # Always install universal (.agents/skills)
    universal_resolved = _install_skill_to_path(content, UNIVERSAL_PATH)
    agents_covered = ", ".join(UNIVERSAL_AGENTS)
    installed.append((agents_covered, universal_resolved))

    for key in additional:
        label, path = ADDITIONAL_AGENTS[key]
        resolved = _install_skill_to_path(content, path)
        installed.append((label, resolved))

    names = [*UNIVERSAL_AGENTS, *(label for label, _ in installed[1:])]
    summary = f"for {', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else f"for {names[0]}"
    render.success("Agent skills installed")
    # Hung under its indent: ten agent names wrap, and a wrap at column 0 reads as a new block.
    render._hang(Text("  "), Text(summary, "dim"), render.console)
    render.console.print(Text("  " + render.MARKS["sep"].join(render.short_path(path) for _, path in installed), "dim"))

    return installed


# ── Display ──────────────────────────────────────────────────────────────────


def _show_account_status(client: Client) -> None:
    """Fetch and display account status panel."""
    with render.spinner("Fetching account info..."):
        info = client.account.user_info()
        bal = client.account.balance()
        metrics = client.account.resource_metrics()
        currency = client.account.currency()

    sym = render.currency_symbol(currency)
    render.console.print()
    render.account_status(info, bal, metrics, sym)


def _show_next_steps() -> None:
    """Print four commands to try and where the full walkthrough lives, before the next action."""
    render.rule("GET STARTED")
    rows = (
        ("jl connect", "add Jarvislabs models to your coding tools"),
        ("jl gpus", "GPUs and prices"),
        ("jl run train.py --gpu A100", "run a script on a GPU"),
        ("jl status", "balance and connected tools"),
    )
    width = max(len(command) for command, _ in rows)
    for command, detail in rows:
        render.console.print(
            Text.assemble("  ", (command.ljust(width), render.ACCENT_BOLD), "  ", (detail, "dim")), highlight=False
        )


def _show_header() -> None:
    """Print the wordmark when there is room and someone to see it, else the one-line header.

    The wordmark needs an interactive terminal at least 80 columns wide, and no --yes.
    """
    if not state.yes and render.can_prompt() and render.console.width >= 80:
        render.console.print()
        # No blank line here: each block after the wordmark prints its own first.
        render.console.print(WORDMARK, markup=False, highlight=False, end="")
    else:
        render.heading("Jarvislabs CLI setup")


# ── Command ──────────────────────────────────────────────────────────────────


@app.command(rich_help_panel="Account")
def setup(
    token: Annotated[str | None, typer.Option("--token", "-t", help="Jarvis key (skips interactive prompt).")] = None,
    agents: Annotated[
        str | None,
        typer.Option(
            "--agents", help="Comma-separated additional agents: claude-code,all (.agents/skills always installed)"
        ),
    ] = None,
    yes: cli_options.YesOption = False,
) -> None:
    """Set up the Jarvislabs CLI: authenticate and install agent skills."""
    cli_options.apply_command_options(yes=yes)

    _show_header()

    # Step 1: Authentication
    client = _login_flow(token)

    # Step 2: Skill installation
    _skill_install_flow(agents)

    # Show account status, then the short next steps.
    # The install scripts run jl connect next, so they would only sit between the two.
    _show_account_status(client)
    if not os.environ.get("JL_FROM_INSTALLER"):
        _show_next_steps()
        render.next_step("see all commands:", ["jl --help"])
