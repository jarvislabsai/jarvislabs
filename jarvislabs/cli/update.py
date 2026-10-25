"""Version checks against PyPI, the daily update notice, and `jl update`.

The latest version is cached in ~/.cache/jl/update_check.json and fetched from PyPI at most once a day.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Annotated

import typer
from platformdirs import user_cache_dir

from jarvislabs.cli import render, setup, state
from jarvislabs.cli.app import app

_PKG = "jarvislabs"
_PYPI_URL = f"https://pypi.org/pypi/{_PKG}/json"
_CACHE_FILE = Path(user_cache_dir("jl")) / "update_check.json"
_TTL = 86_400
_TIMEOUT = 3.0
# The background PyPI check and the notice both update the cache in the same process.
_cache_lock = threading.Lock()


# ── Cache ────────────────────────────────────────────────────────────────────


def _read_cache() -> dict:
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_cache(data: dict) -> None:
    """Write the cache atomically (temp file + rename). Never raises."""
    tmp = None
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_CACHE_FILE.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, _CACHE_FILE)
    except Exception:
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _update_cache(**fields) -> None:
    with _cache_lock:
        _write_cache({**_read_cache(), **fields})


def _cache_is_fresh(cache: dict) -> bool:
    return time.time() - cache.get("checked_at", 0) < _TTL


# ── Versions ─────────────────────────────────────────────────────────────────


def _checks_disabled() -> bool:
    return bool(os.environ.get("JL_NO_UPDATE_CHECK") or os.environ.get("CI"))


def current_version() -> str:
    return version(_PKG)


def _parse_version(v: str | None) -> tuple[int, ...] | None:
    """Parse 'X.Y.Z' into (X, Y, Z). Returns None for anything else, including dev builds."""
    if not v:
        return None
    try:
        parts = tuple(int(x) for x in v.strip().split("."))
        return parts if len(parts) >= 2 else None
    except ValueError:
        return None


def is_newer(latest: str | None, current: str) -> bool:
    latest_t, current_t = _parse_version(latest), _parse_version(current)
    return bool(latest_t and current_t and latest_t > current_t)


def _fetch_latest() -> str | None:
    """Ask PyPI for the latest version and cache it. Returns None on any failure."""
    import httpx

    try:
        resp = httpx.get(_PYPI_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        latest = resp.json()["info"]["version"]
    except Exception:
        return None
    _update_cache(checked_at=time.time(), latest_version=latest)
    return latest


def latest_version() -> str | None:
    """Latest version, from the cache if it is less than a day old, otherwise from PyPI."""
    if _checks_disabled():
        return None
    cache = _read_cache()
    if _cache_is_fresh(cache):
        return cache.get("latest_version")
    return _fetch_latest() or cache.get("latest_version")


def cli_status() -> dict:
    """Version and skill state for `jl status`."""
    current, latest = current_version(), latest_version()
    return {
        "version": current,
        "latest": latest,
        "update_available": is_newer(latest, current),
        "skill_current": setup.skills_current(),
    }


def status_hint(cli: dict) -> str | None:
    """The line `jl status` prints when jl or the agent skill needs attention."""
    hints = []
    if cli["update_available"]:
        hints.append(f"jl {cli['latest']} is available. Run jl update.")
    elif cli["skill_current"] is False:
        hints.append("The jl agent skill is out of date. Run jl update.")
    if cli["skill_current"] is None:
        hints.append("No jl agent skill is installed. Run jl setup to add one.")
    return " ".join(hints) or None


def _update_message(current: str, latest: str) -> str:
    return f"[yellow]Update available:[/yellow] {current} → {latest}. Run: [bold]jl update[/bold]"


def version_line() -> str:
    """`jl --version` output. Mentions a newer version if the cache knows one (no network call)."""
    current = current_version()
    latest = None if _checks_disabled() else _read_cache().get("latest_version")
    available = f" ({latest} available, run jl update)" if is_newer(latest, current) else ""
    return f"jl {current}{available}"


# ── Daily notice ─────────────────────────────────────────────────────────────


def _notice_enabled() -> bool:
    return not _checks_disabled() and sys.stderr is not None and sys.stderr.isatty()


def start_background_check() -> None:
    """Refresh the cache from PyPI in a background thread if it is more than a day old."""
    # The update check must never break the command the user ran.
    with contextlib.suppress(Exception):
        if _notice_enabled() and not _cache_is_fresh(_read_cache()):
            threading.Thread(target=_fetch_latest, daemon=True).start()


def print_notice() -> None:
    """Print the update notice after a command, at most once a day."""
    with contextlib.suppress(Exception):
        if not _notice_enabled() or state.json_output:
            return
        cache = _read_cache()
        current, latest = current_version(), cache.get("latest_version")
        if not is_newer(latest, current) or time.time() - cache.get("notified_at", 0) < _TTL:
            return
        render.console.print(f"\n{_update_message(current, latest)}", highlight=False)
        _update_cache(notified_at=time.time())


# ── jl update ────────────────────────────────────────────────────────────────


def _installer() -> str:
    try:
        return (distribution(_PKG).read_text("INSTALLER") or "").strip().lower()
    except Exception:
        return ""


def _is_editable() -> bool:
    try:
        direct_url = json.loads(distribution(_PKG).read_text("direct_url.json") or "{}")
    except Exception:
        return False
    return bool(direct_url.get("dir_info", {}).get("editable"))


def install_command() -> list[str]:
    """The command that installs the latest jl, based on how jl was installed."""
    prefix = Path(sys.prefix)
    if (prefix / "uv-receipt.toml").is_file():
        # `uv tool upgrade` keeps a version pin from the original install; `install @latest` replaces it.
        return ["uv", "tool", "install", f"{_PKG}@latest"]
    if (prefix / "pipx_metadata.json").is_file():
        return ["pipx", "upgrade", _PKG]
    if _installer() == "uv":
        return ["uv", "pip", "install", "--python", sys.executable, "--upgrade", _PKG]
    return [sys.executable, "-m", "pip", "install", "--upgrade", _PKG]


def _command_text(cmd: list[str]) -> str:
    """The command as the user's shell expects it."""
    return subprocess.list2cmdline(cmd) if sys.platform == "win32" else shlex.join(cmd)


@app.command(rich_help_panel="Account")
def update(
    skills_only: Annotated[bool, typer.Option("--skills-only", hidden=True)] = False,
) -> None:
    """Update jl to the latest version and refresh the installed agent skills."""
    if skills_only:
        setup.refresh_skills()
        return

    if _is_editable():
        render.die("jl is installed from source. Update it with git pull.")

    current = current_version()
    with render.spinner("Checking for updates..."):
        latest = _fetch_latest()
    if latest is None:
        render.die("Could not reach PyPI to check for updates. Try again later.")

    if not is_newer(latest, current):
        render.success(f"jl {current} is up to date.")
        setup.refresh_skills()
        return

    cmd = install_command()
    command_text = _command_text(cmd)
    render.console.print(f"Updating jl {current} → {latest}", highlight=False)

    if sys.platform == "win32":
        # Windows locks the running jl.exe, so jl cannot safely replace itself.
        render.info("jl cannot update itself on Windows. Close jl, run this command, then run jl update again:")
        render.console.print(f"  {command_text}", markup=False, highlight=False)
        return

    render.info(f"Running: {command_text}")
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError:
        render.die(f"{cmd[0]} was not found. Run this command yourself: {command_text}")
    if result.returncode != 0:
        render.die(f"Update failed. Run this command yourself to see the details: {command_text}")

    # This process still has the old code loaded, so the new version refreshes the skills. Run the jl
    # launcher, not `python -m jarvislabs`: -m imports from the current directory first.
    try:
        refreshed = subprocess.run([sys.argv[0], "update", "--skills-only"]).returncode == 0
    except OSError:
        refreshed = False
    if not refreshed:
        render.warning("Could not refresh the agent skills. Run jl update again.")
    render.success(f"Updated jl to {latest}.")
