"""What each coding harness is, and the changes `jl connect` makes to its config files.

A harness is a coding agent, such as OpenCode. HARNESSES lists the ones jl knows. A harness
that jl connects has a writer: a function that takes a Setup (the Models API base URL, this
machine's key, the model catalog and the default model) and returns the config files to edit
and the value to set at each key path in them. Paths are worked out when the writer runs, from
the user's home and environment.

plan_connect reads the current files and returns a Plan: each file's new text, the values it
overwrites, and a preview of the changes: whether they add, update or change, and what they
replace. It writes nothing. apply writes a Plan's files, owner-only, after saving a copy of each file that existed before jl's first connect as
"<config>.jl-backup", and returns the harness's record: the values to put back, and a hash of
each value jl wrote. plan_disconnect uses that record to take out only jl's own values: a value
that no longer matches its hash, and doesn't name Jarvislabs, is the user's and stays. In a list
shared with the user's entries, each entry jl wrote has its own hash, so an entry the user added
or changed stays.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef,import-not-found]

import json5
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from jarvislabs.config import Replaced, Written, write_private
from jarvislabs.config_edit import KeyPath, edit_config, read_block, replace_block
from jarvislabs.constants import MODELS_API_URL, MODELS_DEFAULT_MAX_TOKENS, RECOMMENDED_MODEL
from jarvislabs.exceptions import JarvislabsError
from jarvislabs.models import ModelInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

# The provider ID jl writes into each harness; models are picked as "jarvislabs/<model>".
PROVIDER_ID = "jarvislabs"
# Harnesses whose backend endpoint isn't public yet stay behind this flag. It is read on every
# call, so a test or a preview shell can set it without reloading jl.
PREVIEW_FLAG = "JL_PREVIEW_HARNESSES"
# The Anthropic Messages endpoint (Claude Code appends /v1/messages) and the Responses endpoint
# (Codex appends /responses). Both are jarvislabs.net hosts; the env override is for preview
# builds and tests that stand up a local server.
ANTHROPIC_API_URL = "https://models.jarvislabs.net"
RESPONSES_API_URL = "https://models.jarvislabs.net/v1"
ANTHROPIC_URL_VAR = "JL_ANTHROPIC_URL"
RESPONSES_URL_VAR = "JL_RESPONSES_URL"
# Reading a harness's config fails on bad syntax (ValueError, YAMLError) and on a shape the writer
# doesn't expect, such as a string where an object belongs (TypeError, AttributeError, KeyError).
CONFIG_ERRORS = (ValueError, TypeError, AttributeError, KeyError, YAMLError)
# Cline has a single OpenAI-compatible provider, which jl takes over.
_CLINE_SLOT = ("providers", "openai-compatible")


@dataclass(frozen=True)
class Setup:
    """The input for connecting one harness.

    Attributes:
        base_url: The Models API's OpenAI-compatible base URL.
        api_key: The secret of this machine's key.
        models: Every model in the catalog.
        default_model: The ID of the model to set as the harness's default, or None to leave the
            harness's default as it is. Writers set no default-model value when it is None.
    """

    base_url: str
    api_key: str
    models: tuple[ModelInfo, ...]
    default_model: str | None


@dataclass(frozen=True)
class ConfigEdit:
    """One config file of a harness, and the values jl sets in it.

    Attributes:
        path: The file, which may not exist yet.
        fmt: The file's format, as edit_config reads it, or "block" for a line-based file
            whose jl block replace_block swaps.
        values: The value to set at each key path; for "block", the block's lines at ().
        keep: Key paths in values that disconnect leaves in place, because the harness needs them
            once it shares the file, such as Cline's schema version.
        items: jl's own entries in each list in values that also holds the user's entries, such
            as Qwen Code's modelProviders.jarvislabs. Disconnect takes out only these.
        delete: Key paths to remove, for entries jl wrote before that this setup no longer has,
            such as a model the catalog dropped.
    """

    path: Path
    fmt: Literal["jsonc", "toml", "yaml", "block"]
    values: Mapping[KeyPath, object]
    keep: tuple[KeyPath, ...] = ()
    items: Mapping[KeyPath, Sequence[object]] = dataclasses.field(default_factory=dict)
    delete: tuple[KeyPath, ...] = ()


@dataclass(frozen=True)
class Harness:
    """A coding harness jl knows.

    Attributes:
        id: The name used on the command line, e.g. "opencode".
        name: The display name.
        kind: "cli" for a program on PATH, "vscode" for a VS Code extension.
        support: "connect" when jl writes its config, "coming_soon" when it doesn't yet.
        launch: The command that starts the harness in the current folder.
        writer: Returns the config edits for a Setup; None for a harness jl neither connects nor prints.
        docs_url: The page explaining how to set the harness up by hand.
        note: A line printed after the harness connects, with "{model}" standing for the default
            model; None for no line.
        one_model: True when the harness holds only the default model, not every model.
        extension: For a VS Code extension, its ID, then the flags its install command needs.
        install_url: The harness's own install page, for a CLI harness.
        min_version: The oldest version, as "<major>.<minor>.<patch>", that reads the config jl writes;
            None when every version does.
        preview: True while the endpoint it needs isn't public. jl connects it only with
            ``JL_PREVIEW_HARNESSES=1``; otherwise it is listed as coming soon.
    """

    id: str
    name: str
    kind: Literal["cli", "vscode"]
    support: Literal["connect", "coming_soon"]
    launch: tuple[str, ...]
    writer: Callable[[Setup], tuple[ConfigEdit, ...]] | None = None
    docs_url: str | None = None
    note: str | None = None
    one_model: bool = False
    extension: tuple[str, ...] = ()
    install_url: str | None = None
    min_version: str | None = None
    preview: bool = False


@dataclass(frozen=True)
class FileWrite:
    """A config file's new contents.

    Attributes:
        path: The file.
        text: Its new text, or None to remove the file.
    """

    path: Path
    text: str | None


@dataclass(frozen=True)
class Preview:
    """What a plan does to a harness, for its row in the preview.

    Attributes:
        change: "add" for new entries, "update" for jl's own entries from an earlier connect, "change"
            when the writes replace or override settings of the user's, "remove" for a disconnect.
        replaces: What the writes replace or override of the user's, e.g. "replaces glm-4.6"; "" for
            nothing.
        detail: Anything else to say, e.g. "default model only"; "" for nothing.
    """

    change: Literal["add", "update", "change", "remove"]
    replaces: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Plan:
    """The changes connecting a harness makes, before any is written.

    Attributes:
        harness: The harness the changes are for.
        writes: Each config file's new contents.
        replaced: The values the writes overwrite.
        preview: What the writes do, for the preview: whether they add, update or change, and what they
            replace.
    """

    harness: Harness
    writes: tuple[FileWrite, ...]
    replaced: Replaced
    preview: Preview


def installed(harness: Harness) -> bool:
    """Return whether a harness is on this machine.

    Args:
        harness: The harness to look for.

    Returns:
        For a CLI harness, whether launchable finds it; for a VS Code extension, whether it's in
        VS Code's extensions folder.
    """
    if harness.kind == "vscode":
        # VS Code unpacks each extension into ~/.vscode/extensions/<id>-<version>[-<platform>], and keeps
        # an uninstalled one's folder there, listed in .obsolete, until its next start.
        folder = Path.home() / ".vscode" / "extensions"
        obsolete = _obsolete_extensions(folder)
        return any(path.name not in obsolete for path in folder.glob(f"{harness.extension[0]}-*"))
    return launchable(harness)


def _obsolete_extensions(folder: Path) -> set[str]:
    """Return the extension folders VS Code lists in .obsolete, which it removes at its next start."""
    try:
        data = json.loads((folder / ".obsolete").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return set(data) if isinstance(data, dict) else set()


def launchable(harness: Harness) -> bool:
    """Return whether the command that starts a harness is on this machine.

    Args:
        harness: The harness to look for.

    Returns:
        Whether its launch command is on PATH, or in a folder a tool's own installer puts on PATH
        for new terminals only.
    """
    return _find(harness.launch[0]) is not None


def check_version(harness: Harness) -> None:
    """Refuse a harness older than its min_version, as its --version prints it.

    A version jl can't read (a development build, or a program that doesn't start) isn't
    refused, and a pre-release counts as the release it leads to.

    Args:
        harness: An installed CLI harness.

    Raises:
        JarvislabsError: The harness is older than its min_version.
    """
    if not harness.min_version:
        return
    found = re.search(r"\bv?(\d+)\.(\d+)\.(\d+)", _output([harness.launch[0], "--version"]))
    if found and tuple(map(int, found.groups())) < tuple(map(int, harness.min_version.split("."))):
        version = ".".join(found.groups())
        raise JarvislabsError(
            f"{harness.name} {version} is too old for jl's setup: update it to {harness.min_version} or later, "
            f"then run jl connect {harness.id}."
        )


def preview_enabled() -> bool:
    """Return whether the preview endpoints are switched on: ``JL_PREVIEW_HARNESSES=1``."""
    return os.environ.get(PREVIEW_FLAG, "").strip().lower() not in ("", "0", "false", "no")


def connectable(harness: Harness) -> bool:
    """Return whether jl can connect this harness now, counting a preview one with the flag on."""
    return harness.support == "connect" or (harness.preview and preview_enabled())


def anthropic_base_url() -> str:
    """Return the Anthropic Messages base URL: the preview override, else the Models API host."""
    return os.environ.get(ANTHROPIC_URL_VAR, ANTHROPIC_API_URL).rstrip("/")


def responses_base_url() -> str:
    """Return the OpenAI Responses base URL: the preview override, else the Models API host."""
    return os.environ.get(RESPONSES_URL_VAR, RESPONSES_API_URL).rstrip("/")


def vscode_running() -> bool:
    """Return whether VS Code is running, checked cheaply per OS.

    Cline's extension rewrites its own settings while it runs, so jl must not write underneath
    it. A check that can't run (no pgrep or tasklist) counts as not running: it mustn't stop a
    connect.

    Returns:
        True when VS Code is running.
    """
    if sys.platform == "win32":
        argv = ["tasklist", "/FI", "IMAGENAME eq Code.exe", "/NH"]
    elif sys.platform == "darwin":
        argv = ["pgrep", "-f", "Visual Studio Code.app"]
    else:
        argv = ["pgrep", "-x", "code"]
    try:
        found = subprocess.run(argv, capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    # tasklist exits 0 whether or not it found Code.exe; pgrep exits 0 only when it did.
    return b"Code.exe" in found.stdout if sys.platform == "win32" else found.returncode == 0


def _unreadable(path: Path, error: Exception, command: str) -> JarvislabsError:
    """Return the error for a config file jl can't read, naming the file and what to do.

    A parse error's own message is useful; the internals of a wrong shape are not, so those keep
    only the fact that the file doesn't look the way jl expects.
    """
    detail = str(error) if isinstance(error, (ValueError, YAMLError)) else "jl expected a different shape there"
    return JarvislabsError(f"Can't read {path}: {detail}. Fix the file, then run jl {command} again.")


def plan_connect(harness: Harness, setup: Setup) -> Plan:
    """Return the changes that point a harness at the Models API, without writing them.

    Reads the harness's current config files. Values jl sets replace what is at their key
    paths; everything else in the files is kept, comments included.

    Args:
        harness: A harness whose support is "connect".
        setup: The base URL, key, models and default model to write.

    Returns:
        The new text of each config file, the values it overwrites, and the preview.

    Raises:
        JarvislabsError: A config file isn't valid in its format.
    """
    assert harness.writer is not None, f"{harness.id} has no writer"
    edits = harness.writer(setup)
    writes, replaced, written, block_found, overridden = [], {}, {}, False, []
    for edit in edits:
        text = edit.path.read_text(encoding="utf-8") if edit.path.exists() else ""
        if edit.fmt == "block":
            # Nothing to record as replaced: removing jl's block brings back the user's own lines.
            writes.append(FileWrite(edit.path, _edited(text, edit)))
            block_found = block_found or read_block(text) is not None
            overridden += _overridden_lines(text, str(edit.values[()]))
            written[edit.path] = Written("block")
            continue
        try:
            current = _parse(text, edit.fmt)
            new_text = _edited(text, edit)
        except CONFIG_ERRORS as e:
            raise _unreadable(edit.path, e, "connect") from e
        writes.append(FileWrite(edit.path, new_text))
        old = {path: value for path in edit.values if (value := _lookup(current, path)) is not _MISSING}
        hashes = {path: _hash(value) for path, value in edit.values.items() if path not in (*edit.keep, *edit.items)}
        parents = {_holder(current, path) for path in edit.values if path not in old} - {()}
        items = {path: tuple(_hash(item) for item in ours) for path, ours in edit.items.items()}
        written[edit.path] = Written(edit.fmt, hashes, tuple(sorted(parents)), edit.keep, items)
        if old:
            replaced[edit.path] = old
    preview = _preview(harness, setup, replaced, {edit.path: edit.values for edit in edits}, block_found, overridden)
    return Plan(harness, tuple(writes), Replaced(replaced, written=written), preview)


def plan_disconnect(harness: Harness, replaced: Replaced, keep: Collection[Path] = ()) -> Plan:
    """Return the changes that take jl's entries out of a harness's config files, without writing them.

    Each value jl wrote is removed, or replaced by the value it overwrote, but only while it
    is still jl's: the value jl last wrote, or one naming Jarvislabs. A value the user changed
    since stays, as does everything else in the files. From a list shared with the user's
    entries, only the entries jl last wrote go, unchanged. An object that held only jl's entries
    goes, unless it existed before jl. A file jl created is removed when nothing else is left
    in it. Only the files recorded in ``replaced`` are read, so a harness whose config was
    printed has nothing to change. A record from an older jl (``untracked``) also reads the files
    jl's writer names now, and takes out only the values that are plainly jl's.

    Args:
        harness: The connected harness.
        replaced: The harness's record from apply.
        keep: Files another connected tool also writes, which are emptied rather than deleted.

    Returns:
        The new text of each file that changes, or None for a file to remove, and the preview.

    Raises:
        JarvislabsError: A config file isn't valid in its format.
    """
    writes: list[FileWrite] = []
    restored: list[str] = []
    files = {**_untracked(harness, replaced), **replaced.written} if replaced.untracked else replaced.written
    for path, written in files.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if written.fmt == "block":
            new_text = replace_block(text, None)
            empty = not new_text.strip()
        else:
            try:
                new_text = _without_jl(text, path, written, replaced.values.get(path, {}), restored)
                # What jl keeps for the harness doesn't count: alone, it's still only jl's.
                fmt = cast("Literal['jsonc', 'toml', 'yaml']", written.fmt)
                empty = not _parse(edit_config(new_text, fmt, {}, written.kept), fmt)
            except CONFIG_ERRORS as e:
                raise _unreadable(path, e, "disconnect") from e
        if path in replaced.created and empty and path not in keep:
            writes.append(FileWrite(path, None))
        elif new_text != text:
            writes.append(FileWrite(path, new_text))
    if len(writes) == 1:
        parts = ["delete the file jl created" if writes[0].text is None else "remove the Jarvislabs entries"]
    else:
        parts = [
            f"delete {write.path.name}"
            if write.text is None
            else f"remove the Jarvislabs entries from {write.path.name}"
            for write in writes
        ]
    if restored:
        parts.append("put back your " + ", ".join(dict.fromkeys(restored)))
    detail = ", ".join(parts) if parts else "jl wrote no files, so only its record goes"
    return Plan(harness, tuple(writes), Replaced(), Preview("remove", detail=detail))


def apply(plan: Plan, previous: Replaced | None = None) -> Replaced:
    """Write a plan's config files, each atomically and readable only by its owner.

    Before writing an existing file that has no "<config>.jl-backup" next to it, saves a
    copy of it there, also owner-only, unless jl created the file. A backup is never
    overwritten. A write without text removes its file.

    Args:
        plan: The changes from plan_connect or plan_disconnect.
        previous: The harness's record from an earlier connect, whose created files get no
            backup; None when the harness isn't connected.

    Returns:
        The harness's record. On the first connect: the values the writes overwrote, the
        backups saved and the files created. On a later connect, the record keeps what the
        earlier connects overwrote, except values the user set since: the values from before
        this connect, which disconnect puts back.
    """
    backups, created = [], []
    for write in plan.writes:
        backup = write.path.with_name(write.path.name + ".jl-backup")
        if write.text is None:
            write.path.unlink(missing_ok=True)
            continue
        text = write.text
        if not write.path.exists():
            created.append(write.path)
        elif not backup.exists() and not (previous and write.path in previous.created):
            write_private(backup, write.path.read_bytes())
            backups.append(backup)
        write_private(write.path, text.encode("utf-8"))
    now = dataclasses.replace(plan.replaced, backups=tuple(backups), created=tuple(created))
    return _merged(previous, now) if previous else now


def default_model(harness: Harness) -> str | None:
    """Return the Jarvislabs model a harness's config names as its default.

    Runs the writer with a marker as the default model to learn where the harness keeps it and
    in what form (e.g. "jarvislabs/<model>"), then reads that value back from the file.

    Args:
        harness: The harness to read.

    Returns:
        The model ID, or None when the default is missing, another provider's, or unreadable,
        or when the harness has no writer (its config isn't in a file jl reads).
    """
    if harness.writer is None:
        return None
    marker = "\0"
    # The real base URL, so that a base URL set beside the default (Aider's openai-api-base) must be ours.
    for edit in harness.writer(Setup(base_url=MODELS_API_URL, api_key="", models=(), default_model=marker)):
        # Searched value by value inside objects too: Cline's default is in a slot jl writes whole.
        leaves = _leaves(edit.values)
        for path, value in leaves.items():
            if not (isinstance(value, str) and marker in value):
                continue
            text = edit.path.read_text(encoding="utf-8") if edit.path.exists() else ""
            current: object
            try:
                if edit.fmt == "block":
                    value, current = _block_word(value, read_block(text) or "", marker)
                else:
                    data = _parse(text, edit.fmt)
                    current = _lookup(data, path)
                    # The default is ours only if the values set beside it are there too: at the top of the
                    # edit, e.g. Goose's active_provider, or next to it, e.g. Cline's baseUrl. The blank Setup
                    # leaves values such as the key empty.
                    beside = [
                        (p, v)
                        for p, v in leaves.items()
                        if (p in edit.values or p[:-1] == path[:-1]) and isinstance(v, str) and v and marker not in v
                    ]
                    if any(_lookup(data, p) != v for p, v in beside):
                        return None
            except (ValueError, YAMLError):
                return None
            prefix, suffix = value.split(marker)
            middle = current[len(prefix) : len(current) - len(suffix)] if isinstance(current, str) else ""
            return middle if middle and current == prefix + middle + suffix else None
    return None


def snippet(harness: Harness, setup: Setup) -> str:
    """Return the config to paste into a harness by hand, with the key and models filled in.

    Args:
        harness: A harness with a writer.
        setup: The base URL, key, models and default model to fill in.

    Returns:
        Each config file's path and the entries to add to it, then the key when no file holds it.
    """
    assert harness.writer is not None, f"{harness.id} has no writer"
    text = "\n".join(f"Add this to {edit.path}:\n{_edited('', edit)}" for edit in harness.writer(setup))
    if setup.api_key not in text:
        text += f"\nThen give {harness.name} your key: {setup.api_key}\n"
    return text


def _windows_path() -> str | None:
    """Return the Path Windows' registry gives a new terminal, or None off Windows or when unreadable.

    A terminal keeps the PATH it started with, and an installer such as Node.js's adds its folders
    to the registry only, so a terminal opened before the install sees neither the tool nor node.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:  # pragma: no cover - os.name says Windows but there is no registry
        return None
    folders = []
    for root, key_name in (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ):
        try:
            with winreg.OpenKey(root, key_name) as key:
                value = str(winreg.QueryValueEx(key, "Path")[0])
        except OSError:
            continue
        folders.append(_expand_windows_vars(value))
    return os.pathsep.join(folders) if folders else None


def _expand_windows_vars(value: str) -> str:
    """Expand the %VARS% in a registry Path, as Windows does, from this process's environment."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1).upper()
        return next((env_value for env, env_value in os.environ.items() if env.upper() == name), match.group(0))

    return re.sub(r"%([^%]+)%", replace, value)


def _path_env(name: str, program: str) -> dict[str, str] | None:
    """Return the environment a program needs when it was found off this process's PATH, else None.

    On Windows, the registry's Path is put first too: a terminal keeps the PATH it started with, so a
    tool installed since then (and the node its launcher needs) is missing from this one.
    """
    registry_path = _windows_path()
    if shutil.which(name) is None and os.path.isabs(program):
        # Found off PATH: a Node.js script there (Cline's, under nvm) finds the node beside it.
        path = os.path.dirname(program) + os.pathsep + (registry_path or os.environ.get("PATH", ""))
        return {**os.environ, "PATH": path}
    if registry_path:
        # After this window's own PATH, so a node chosen in this window still comes first.
        return {**os.environ, "PATH": os.environ.get("PATH", "") + os.pathsep + registry_path}
    return None


_MISSING = object()


def _output(argv: list[str]) -> str:
    # What a program prints, for jl to read; "" when it can't run. Decoded as UTF-8, which Go programs
    # (Crush) print, rather than in Windows' ANSI code page.
    program = _find(argv[0]) or argv[0]
    try:
        output = subprocess.run(
            [program, *argv[1:]], capture_output=True, timeout=30, env=_path_env(argv[0], program)
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return output.decode("utf-8", errors="replace")


def _find(program: str) -> str | None:
    """Return the full path of a program: on PATH, or in a folder a tool's own installer uses.

    The installers add these folders to the shell profile, so a terminal opened before the install
    doesn't have them on PATH yet: OpenCode's and Kilo's scripts install into ~/.opencode/bin and
    ~/.kilo/bin, pi's into ~/.pi/agent/bin when no folder it prefers is on PATH, Kimi Code's into
    ~/.kimi-code/bin, and npm under nvm into the Node.js version's own bin folder, newest first. On
    Windows, the registry's Path (the one a new terminal gets) is searched too, before those folders.
    """
    found = shutil.which(program)
    if found:
        return found
    # Checked before the folders below, which need Path.home().
    registry_path = _windows_path()
    if registry_path and (found := shutil.which(program, path=registry_path)):
        return found
    xdg = os.environ.get("XDG_CONFIG_HOME")
    # nvm's own rule for its folder: $NVM_DIR, else $XDG_CONFIG_HOME/nvm, else ~/.nvm.
    nvm = Path(os.environ.get("NVM_DIR") or (Path(xdg) / "nvm" if xdg else Path.home() / ".nvm")).expanduser().resolve()
    versions = sorted(
        (nvm / "versions" / "node").glob("v*/bin"),
        key=lambda folder: [int(n) for n in re.findall(r"\d+", folder.parent.name)],
        reverse=True,
    )
    home = Path.home()
    folders = [
        home / ".opencode" / "bin",
        home / ".kilo" / "bin",
        home / ".pi" / "agent" / "bin",
        home / ".kimi-code" / "bin",
        *versions,
    ]
    # path=, not a path in program: only then does Windows try .exe and .cmd (PATHEXT) on Python 3.10.
    return shutil.which(program, path=os.pathsep.join(map(str, folders)))


def _holder(data: object, path: KeyPath) -> KeyPath:
    """Return the deepest object data already has on the way to path; () for the top."""
    parent = path[:-1]
    while parent and not isinstance(_lookup(data, parent), dict):
        parent = parent[:-1]
    return parent


def _untracked(harness: Harness, replaced: Replaced) -> dict[Path, Written]:
    """Return what jl wrote for a record from an older jl, which didn't record it.

    Every key path jl's writer sets now, where a value counts as jl's only when it names Jarvislabs,
    or sits beside one that does in the same object (Aider's key and model, beside its base URL).
    In a shared list, the entries naming Jarvislabs; and a file the older jl created that is named
    for Jarvislabs (Junie's model files) is jl's whole.
    """
    if harness.writer is None:
        return {}
    files: dict[Path, Written] = {}
    setup = Setup(base_url=MODELS_API_URL, api_key="", models=(), default_model=RECOMMENDED_MODEL)
    for edit in harness.writer(setup):
        if edit.fmt == "block":
            files[edit.path] = Written("block")
            continue
        # A file that can't be read is reported when disconnect reads it.
        data = _current(edit.path, (), edit.fmt)
        current = {p: _lookup(data, p) for p in edit.values if p not in (*edit.keep, *edit.items)}
        named = {p[:-1] for p, v in current.items() if v is not _MISSING and _is_jl_value(None, edit.path, p, v)}
        # A hash that matches the value now marks it as jl's; "" marks one disconnect decides by its name.
        hashes = {p: _hash(value) if p[:-1] in named else "" for p, value in current.items()}
        items = {
            key: tuple(_hash(entry) for entry in entries if PROVIDER_ID in json.dumps(entry).lower())
            for key in edit.items
            if isinstance(entries := _lookup(data, key), list)
        }
        files[edit.path] = Written(edit.fmt, hashes, (), edit.keep, items)
    for path in replaced.created:
        data = _current(path, ())
        if path not in files and PROVIDER_ID in path.name and isinstance(data, dict):
            files[path] = Written("jsonc", {(key,): "" for key in data})
    return files


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _is_jl_value(written: Written | None, path: Path, key_path: KeyPath, value: object) -> bool:
    """Return whether a config value is jl's: what jl last wrote there, or anything naming Jarvislabs.

    A value naming Jarvislabs (its provider, base URL or key path, or a file named for it) is
    jl's even when the user edited it, so disconnect never leaves jl's provider behind. So is a
    value inside one jl wrote whole, such as the key a refresh sets in Cline's slot.
    """
    if written and written.hashes.get(key_path) == _hash(value):
        return True
    if written and any(key_path[: len(p)] == p for p in written.hashes if len(p) < len(key_path)):
        return True
    return PROVIDER_ID in (path.name + json.dumps([key_path, value], default=str)).lower()


def _merged(previous: Replaced, now: Replaced) -> Replaced:
    """Return a harness's record after a reconnect: ``previous`` updated by ``now``, the new connect's.

    A value the new connect overwrote that isn't jl's was set by the user since the last
    connect, so it's the one to put back; one the user removed has nothing to put back. Every
    path jl has written stays recorded, so disconnect also takes out entries a later connect
    stopped writing, such as a model the catalog dropped.
    """
    values = {path: dict(paths) for path, paths in previous.values.items()}
    written = dict(previous.written)
    for path, new in now.written.items():
        before = now.values.get(path, {})
        # An older jl didn't record what it wrote, so what this connect overwrote may be its own.
        guess = previous.untracked and path not in previous.written
        for key_path in new.hashes:
            if key_path not in before:
                values.get(path, {}).pop(key_path, None)
            elif not guess and not _is_jl_value(previous.written.get(path), path, key_path, before[key_path]):
                values.setdefault(path, {})[key_path] = before[key_path]
        old = written.get(path)
        if old:
            # Parents are the user's only on jl's first write to the file; later ones may be jl's own.
            kept = tuple({*old.kept, *new.kept})
            new = Written(new.fmt, {**old.hashes, **new.hashes}, old.parents, kept, {**old.items, **new.items})
        written[path] = new
    return Replaced(
        {path: paths for path, paths in values.items() if paths},
        tuple(dict.fromkeys(previous.backups + now.backups)),
        tuple(dict.fromkeys(previous.created + now.created)),
        written,
        previous.untracked,
    )


def _without_jl(text: str, path: Path, written: Written, values: Mapping[KeyPath, object], restored: list[str]) -> str:
    """Return a config file's text with jl's values taken out and the ones they overwrote put back.

    Appends the key path of each value put back to ``restored``.
    """
    fmt = cast("Literal['jsonc', 'toml', 'yaml']", written.fmt)
    data = _parse(text, fmt)
    ours = [
        key_path
        for key_path in written.hashes
        if (value := _lookup(data, key_path)) is not _MISSING and _is_jl_value(written, path, key_path, value)
    ]
    # A path inside another of jl's goes with it; setting it after its parent's restore would undo that.
    ours = [key_path for key_path in ours if not any(key_path[: len(p)] == p for p in ours if len(p) < len(key_path))]
    restore = {key_path: values[key_path] for key_path in ours if key_path in values}
    restored += [".".join(key_path) for key_path in restore]
    deleted = [key_path for key_path in ours if key_path not in restore]
    for key_path, hashes in written.items.items():
        entries = _lookup(data, key_path)
        if not isinstance(entries, list):
            continue
        left = [entry for entry in entries if _hash(entry) not in hashes]
        if left:
            restore[key_path] = left
        else:
            deleted.append(key_path)
    text = edit_config(text, fmt, restore, deleted)
    # Deleting jl's last entry from an object deletes the object too; put back one the user had.
    data = _parse(text, fmt)
    gone = [parent for parent in written.parents if _lookup(data, parent) is _MISSING]
    return edit_config(text, fmt, {parent: {} for parent in sorted(gone, key=len)}) if gone else text


def _parse(text: str, fmt: Literal["jsonc", "toml", "yaml"]) -> object:
    if not text.strip():
        return {}
    if fmt == "jsonc":
        return json5.loads(text)
    if fmt == "toml":
        return tomllib.loads(text)
    return YAML(typ="safe").load(text)


def _lookup(data: object, path: KeyPath) -> object:
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return _MISSING
        data = data[key]
    return data


def _leaves(values: Mapping[KeyPath, object]) -> dict[KeyPath, object]:
    # Each value that isn't an object, at its full key path: {("a",): {"b": 1}} gives {("a", "b"): 1}.
    leaves: dict[KeyPath, object] = {}
    for path, value in values.items():
        if isinstance(value, dict):
            leaves.update(_leaves({(*path, key): item for key, item in value.items()}))
        else:
            leaves[path] = value
    return leaves


def _edited(text: str, edit: ConfigEdit) -> str:
    if edit.fmt == "block":
        return replace_block(text, str(edit.values[()]))
    return edit_config(text, edit.fmt, edit.values, edit.delete)


def _block_word(block: str, text: str, marker: str) -> tuple[str, str | None]:
    """Return the shell word holding marker in block, and the word in its place in text.

    Lines are compared as shell words, so quoting doesn't matter: the word comes from the first
    line of text whose other words match the marker's line.
    """
    want = next(shlex.split(line) for line in block.splitlines() if marker in line)
    i = next(i for i, word in enumerate(want) if marker in word)
    for line in text.splitlines():
        words = shlex.split(line)
        if len(words) == len(want) and words[:i] + words[i + 1 :] == want[:i] + want[i + 1 :]:
            return want[i], words[i]
    return want[i], None


def _preview(
    harness: Harness,
    setup: Setup,
    replaced: Mapping[Path, Mapping[KeyPath, object]],
    new: Mapping[Path, Mapping[KeyPath, object]],
    block_found: bool,
    overridden: list[str],
) -> Preview:
    # A jl block already in the file is an earlier connect's.
    verb: Literal["add", "update"] = "update" if block_found else "add"
    old = []
    for file, values in replaced.items():
        for path, value in values.items():
            if _jl_own(file, path, value, setup):
                verb = "update"
            elif value is None or value == new[file][path] or isinstance(value, list):
                # Nothing there, unchanged, or a list jl adds its entries to, keeping the rest.
                continue
            elif path == _CLINE_SLOT:
                url = _lookup(value, ("settings", "baseUrl"))
                old.append(f"the OpenAI-compatible provider at {url} (Cline holds only one)")
            else:
                # A replaced text value, such as the old default model, is shown as is; anything else, and
                # a key that jl's key replaces (Aider's openai-api-key), by its key path.
                key = new[file][path] == setup.api_key
                old.append(value if isinstance(value, str) and not key else ".".join(path))
    parts = [f"replaces {', '.join(old)}"] if old else []
    parts += [f"overrides your {', '.join(overridden)}"] if overridden else []
    detail = "default model only" if harness.one_model else ""
    return Preview("change" if parts else verb, ", ".join(parts), detail)


def _overridden_lines(text: str, block: str) -> list[str]:
    # The user's own lines setting what jl's block sets too (Crush's "model large" and "model small"),
    # which the block overrides. Assumes the user's lines come before the block, where jl adds it; a line
    # after it would win over the block instead.
    kinds = {tuple(line.split()[:2]) for line in block.splitlines()} & {("model", "large"), ("model", "small")}
    outside = replace_block(text, None).splitlines()
    # Shown as "model large <model>", without a trailing comment.
    return [" ".join(line.split()[:3]) for line in outside if tuple(line.split()[:2]) in kinds]


def _jl_own(file: Path, path: KeyPath, value: object, setup: Setup) -> bool:
    # What an earlier connect wrote is updated, not replaced: a provider entry or file named for jl, or
    # a value pointing at the Models API (Cline's slot, an entry list jl is in).
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return PROVIDER_ID in path or PROVIDER_ID in file.name or setup.base_url in text


def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _limits(model: ModelInfo) -> tuple[int, int]:
    # The catalog always has a context length; the fallback only keeps such a model usable.
    context = model.context_length or MODELS_DEFAULT_MAX_TOKENS
    # A harness that compacts (OpenCode) sends the whole conversation plus the model's full
    # advertised output budget as max_tokens, and the Models API refuses prompt_tokens +
    # max_tokens over the context window. Compaction keeps a 15% buffer, so an output budget
    # of a tenth of the window always fits (0.85 + 0.10 < 1); a flat 32768 doesn't on the
    # smaller windows.
    return context, min(model.max_output_tokens or MODELS_DEFAULT_MAX_TOKENS, context // 10)


def _default_effort(model: ModelInfo) -> str | None:
    """Return the effort a harness that takes one fixed value should pin for the model.

    The catalog's own default, which the Models API applies to a request that sends none anyway,
    so pinning it changes nothing on the wire while making the effort explicit for harnesses with
    no effort picker; a model that offers exactly one effort gets it (the API's default for it is
    unstated); a model that reasons with a toggle only, none. An effort outside the model's own
    list is never returned: the API refuses it.
    """
    efforts = model.thinking.supported_efforts
    if model.thinking.default_effort in efforts:
        return model.thinking.default_effort
    return efforts[0] if len(efforts) == 1 else None


def _opencode(setup: Setup) -> tuple[ConfigEdit, ...]:
    # OpenCode reads opencode.jsonc after opencode.json, so a .jsonc file wins when both exist.
    folder = _xdg_config_home() / "opencode"
    path = folder / "opencode.jsonc" if (folder / "opencode.jsonc").exists() else folder / "opencode.json"
    return (ConfigEdit(path, "jsonc", _opencode_values(setup)),)


def _kilo(setup: Setup) -> tuple[ConfigEdit, ...]:
    # Kilo reads kilo.jsonc after config.json and kilo.json, and keeps a .jsonc file's comments when it
    # rewrites it. Without per-model limits Kilo sets the context to 0 and turns compaction off.
    return (ConfigEdit(_xdg_config_home() / "kilo" / "kilo.jsonc", "jsonc", _kilo_values(setup)),)


def _pi(setup: Setup) -> tuple[ConfigEdit, ...]:
    folder = Path.home() / ".pi" / "agent"
    provider = {
        "name": "Jarvislabs",
        "baseUrl": setup.base_url,
        "api": "openai-completions",
        "apiKey": setup.api_key,
        # pi takes an unknown base URL for OpenAI and sends store, the developer role and
        # max_completion_tokens; the Models API reads max_tokens, and caps output at 1024 without it.
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
        "models": [_pi_model(model) for model in setup.models],
    }
    edits = [ConfigEdit(folder / "models.json", "jsonc", {("providers", PROVIDER_ID): provider})]
    if setup.default_model:
        # pi keeps the default provider and model apart, in settings.json.
        default: dict[KeyPath, object] = {("defaultProvider",): PROVIDER_ID, ("defaultModel",): setup.default_model}
        edits.append(ConfigEdit(folder / "settings.json", "jsonc", default))
    return tuple(edits)


def _pi_model(model: ModelInfo) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": model.id,
        "name": model.name or model.id,
        "reasoning": model.thinking.supported,
        "input": _input(model),
        "contextWindow": _limits(model)[0],
        "maxTokens": _limits(model)[1],
    }
    efforts = model.thinking.supported_efforts
    if model.thinking.supported and efforts:
        # pi sends its thinking level as the effort, medium by default, and the Models API refuses an
        # effort the model doesn't list. null takes a level out of pi's picker, so pi moves up to the
        # next one; xhigh and max need naming to be offered at all.
        levels = ("minimal", "low", "medium", "high", "xhigh", "max")
        entry["thinkingLevelMap"] = {level: level if level in efforts else None for level in levels}
    return entry


def _cline_data_dir() -> Path:
    """Return Cline's data folder, as Cline resolves it: CLINE_DATA_DIR, then CLINE_DIR/data, then ~/.cline/data."""
    if data_dir := os.environ.get("CLINE_DATA_DIR", "").strip():
        return Path(data_dir).expanduser()
    cline_dir = os.environ.get("CLINE_DIR", "").strip()
    return (Path(cline_dir) if cline_dir else Path.home() / ".cline").expanduser() / "data"


def _kimi(setup: Setup) -> tuple[ConfigEdit, ...]:
    # Kimi Code reads ~/.kimi-code/config.toml, or the same file under KIMI_CODE_HOME.
    folder = Path(os.environ.get("KIMI_CODE_HOME") or Path.home() / ".kimi-code").expanduser()
    path = folder / "config.toml"
    aliases = {f"{PROVIDER_ID}/{model.id}" for model in setup.models}
    # Each model is its own table, so an alias jl wrote for a model the catalog dropped has to be
    # named for deletion; the next connect would otherwise leave it in the picker for ever.
    stale = tuple(("models", alias) for alias in _kimi_aliases(path) if alias not in aliases)
    values: dict[KeyPath, object] = {
        ("providers", PROVIDER_ID): {
            "type": "openai",
            "base_url": setup.base_url,
            "api_key": setup.api_key,
        }
    }
    for model in setup.models:
        alias = f"{PROVIDER_ID}/{model.id}"
        entry: dict[str, object] = {
            "provider": PROVIDER_ID,
            "model": model.id,
            # Required by Kimi Code; a model without one would fail at startup.
            "max_context_size": _limits(model)[0],
        }
        if model.name:
            entry["display_name"] = model.name
        capabilities = _kimi_capabilities(model)
        if capabilities:
            entry["capabilities"] = capabilities
        if efforts := model.thinking.supported_efforts:
            # default_effort is ignored without support_efforts beside it, so the two go together;
            # a model that offers no efforts gets neither, and sends no effort at all.
            entry["support_efforts"] = efforts
            if effort := _default_effort(model):
                entry["default_effort"] = effort
        values[("models", alias)] = entry
    if setup.default_model:
        # An alias from models above, so Kimi Code finds it at startup.
        values[("default_model",)] = f"{PROVIDER_ID}/{setup.default_model}"
    return (ConfigEdit(path, "toml", values, delete=stale),)


def _kimi_aliases(path: Path) -> list[str]:
    """Return the model aliases jl's provider holds now; empty for a missing or unreadable file."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    models = data.get("models")
    if not isinstance(models, dict):
        return []
    return [
        alias for alias, entry in models.items() if isinstance(entry, dict) and entry.get("provider") == PROVIDER_ID
    ]


def _kimi_capabilities(model: ModelInfo) -> list[str]:
    """Return Kimi Code's capability tags for a catalog model, in a stable order.

    Kimi Code infers some capabilities from the model name; these add what the catalog knows for
    models it doesn't recognise, and never remove anything.
    """
    tags = []
    if model.capabilities.tool_calling:
        tags.append("tool_use")
    if model.thinking.supported:
        # A model that reasons without a toggle is always thinking; one with a toggle can be turned off.
        tags.append("thinking" if model.thinking.supports_toggle else "always_thinking")
    for tag, modality in (("image_in", "image"), ("video_in", "video"), ("audio_in", "audio")):
        if modality in model.input_modalities:
            tags.append(tag)
    return tags


def _cline(setup: Setup) -> tuple[ConfigEdit, ...]:
    path = _cline_data_dir() / "settings" / "providers.json"
    settings = (*_CLINE_SLOT, "settings")
    if not setup.default_model:
        # Cline's slot holds only the default model, so a refresh (after a key rotation, say) sets just
        # the key, and only while the slot is still jl's.
        if _current(path, (*settings, "baseUrl")) != setup.base_url:
            return ()
        return (ConfigEdit(path, "jsonc", {(*settings, "apiKey"): setup.api_key}),)
    model = _model(setup, setup.default_model)
    context, output = _limits(model)
    capabilities = ["streaming"]
    if model.capabilities.tool_calling:
        capabilities.append("tools")
    if model.thinking.supported:
        capabilities.append("reasoning")
    if "image" in model.input_modalities:
        capabilities.append("vision")
    entry = {
        "settings": {
            "provider": _CLINE_SLOT[1],
            "apiKey": setup.api_key,
            "baseUrl": setup.base_url,
            "model": model.id,
            "contextWindow": context,
            "maxTokens": output,
            "capabilities": capabilities,
        },
        # Cline's schema wants an ISO time with milliseconds and a "Z".
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "tokenSource": "manual",
    }
    values: dict[KeyPath, object] = {_CLINE_SLOT: entry, ("lastUsedProvider",): _CLINE_SLOT[1]}
    if _current(path, ("version",)) is None:
        # Cline treats a file without its schema's version as empty, and its next save drops every provider.
        values["version",] = 1
    return (ConfigEdit(path, "jsonc", values, keep=(("version",),)),)


def _cline_extension(setup: Setup) -> tuple[ConfigEdit, ...]:
    """Cline's VS Code extension keeps its provider in globalState.json and its key in secrets.json.

    It reads providers.json only while none of its own settings names a provider, so jl writes its
    state too. Both are flat JSON files shared with Cline's other keys, which are kept.
    """
    edits = _cline(setup)
    data = _cline_data_dir()
    state, secrets = data / "globalState.json", data / "secrets.json"
    key: dict[KeyPath, object] = {("openAiApiKey",): setup.api_key}
    if not setup.default_model:
        # A refresh (a key rotation, say) sets just the key, and only while the extension still
        # points at jl, as Cline's providers.json slot does.
        if _current(state, ("openAiBaseUrl",)) != setup.base_url:
            return edits
        return (*edits, ConfigEdit(secrets, "jsonc", key))
    return (
        *edits,
        ConfigEdit(
            state,
            "jsonc",
            {
                # Cline shows its sign-in screen until this is set.
                ("welcomeViewCompleted",): True,
                # The legacy provider ID Cline's state transform folds "openai-compatible" back to.
                ("planModeApiProvider",): "openai",
                ("actModeApiProvider",): "openai",
                ("openAiBaseUrl",): setup.base_url,
                ("planModeOpenAiModelId",): setup.default_model,
                ("actModeOpenAiModelId",): setup.default_model,
            },
        ),
        # apply writes every file owner-only, which is the mode Cline gives its secrets file.
        ConfigEdit(secrets, "jsonc", key),
    )


def _qwen(setup: Setup) -> tuple[ConfigEdit, ...]:
    path = Path.home() / ".qwen" / "settings.json"
    key_env = "JARVISLABS_API_KEY"
    ours = []
    for model in setup.models:
        entry: dict[str, object] = {
            "id": model.id,
            "name": model.name or model.id,
            # Qwen Code never reads a key from a provider entry, only from the env var it names.
            "envKey": key_env,
            "baseUrl": setup.base_url,
            "generationConfig": {
                "contextWindowSize": _limits(model)[0],
                # The Models API caps output at 1024 tokens when a request doesn't set max_tokens.
                "samplingParams": {"max_tokens": _limits(model)[1]},
                "modalities": {"image": "image" in model.input_modalities},
            },
        }
        if efforts := model.thinking.supported_efforts:
            # The capabilities name the efforts the /effort picker offers and the one to send
            # without a choice; generationConfig's reasoning is dropped once samplingParams is
            # set, which jl always sets. A model without efforts gets no capabilities at all:
            # an empty ladder makes Qwen Code discard the declaration.
            reasoning: dict[str, object] = {"profile": "openai-effort", "efforts": efforts}
            if effort := _default_effort(model):
                reasoning["defaultEffort"] = effort
            entry["capabilities"] = {"reasoning": reasoning}
        ours.append(entry)
    # A custom provider loads only once providerProtocol names its protocol; ours speaks OpenAI.
    # security.auth.selectedType stays "openai" though: a custom id there crashes Qwen Code.
    values: dict[KeyPath, object] = {
        ("env", key_env): setup.api_key,
        ("providerProtocol", PROVIDER_ID): "openai",
        # jl's own provider list, which holds the user's entries too, if they added any.
        ("modelProviders", PROVIDER_ID): _others(path, ("modelProviders", PROVIDER_ID), "baseUrl", setup.base_url)
        + ours,
    }
    items: dict[KeyPath, Sequence[object]] = {("modelProviders", PROVIDER_ID): ours}
    delete: tuple[KeyPath, ...] = ()
    # An earlier jl wrote its entries into the shared modelProviders.openai list, where they
    # shadow these (Qwen Code keeps the first entry of a protocol+id+baseUrl), so a refresh
    # moves them out; the user's own entries stay.
    openai = ("modelProviders", "openai")
    entries = _current(path, openai)
    stale = (
        [entry for entry in entries if isinstance(entry, dict) and entry.get("baseUrl") == setup.base_url]
        if isinstance(entries, list)
        else ()
    )
    if stale:
        items[openai] = stale
        if kept := _others(path, openai, "baseUrl", setup.base_url):
            values[openai] = kept
        else:
            # Nothing of the user's is left in the list: the key goes rather than an empty array.
            delete = (openai,)
    if setup.default_model:
        values["security", "auth", "selectedType"] = "openai"
        values["model", "name"] = setup.default_model
    return (ConfigEdit(path, "jsonc", values, items=items, delete=delete),)


def _copilot(setup: Setup) -> tuple[ConfigEdit, ...]:
    folder = Path.home() / ".copilot"
    path = folder / "providers.json"
    # Copilot CLI takes "$VAR" in this file literally, so the key goes in as is.
    provider = {
        "name": PROVIDER_ID,
        "type": "openai",
        "wireApi": "completions",
        "baseUrl": setup.base_url,
        "apiKey": setup.api_key,
    }
    models = []
    for model in setup.models:
        entry: dict[str, object] = {
            "id": model.id,
            "provider": PROVIDER_ID,
            "name": model.name or model.id,
            "maxPromptTokens": _limits(model)[0],
            "maxOutputTokens": _limits(model)[1],
        }
        if efforts := model.thinking.supported_efforts:
            # The ladder the picker's Reasoning column cycles through; without it the column is
            # hidden and no effort is ever sent. Copilot itself never sends a default.
            entry["capabilities"] = {"supports": {"reasoning_effort": efforts}}
        models.append(entry)
    # Providers and models are lists shared with the user's own, which are kept.
    values: dict[KeyPath, object] = {
        ("providers",): [*_others(path, ("providers",), "name", PROVIDER_ID), provider],
        ("models",): _others(path, ("models",), "provider", PROVIDER_ID) + models,
    }
    edits = [ConfigEdit(path, "jsonc", values, items={("providers",): [provider], ("models",): models})]
    if setup.default_model:
        # Without a model here, Copilot CLI refuses a custom provider.
        edits.append(
            ConfigEdit(folder / "settings.json", "jsonc", {("model",): f"{PROVIDER_ID}/{setup.default_model}"})
        )
    return tuple(edits)


def _junie(setup: Setup) -> tuple[ConfigEdit, ...]:
    folder = Path.home() / ".junie"
    edits = [
        ConfigEdit(
            # Each file is one model; its name, without ".json", is the profile picked as "custom:<name>".
            folder / "models" / f"{PROVIDER_ID}-{model.id}.json",
            "jsonc",
            {
                ("id",): model.id,
                ("displayName",): model.name or model.id,
                # Junie appends no path to the base URL.
                ("baseUrl",): setup.base_url + "/chat/completions",
                ("apiType",): "OpenAICompletion",
                ("apiKey",): setup.api_key,
                ("maxContextLength",): _limits(model)[0],
                # The Models API caps output at 1024 tokens when a request doesn't set max_tokens.
                ("extraBody",): {"max_tokens": _limits(model)[1]},
            },
        )
        for model in setup.models
    ]
    if setup.default_model:
        profile = f"custom:{PROVIDER_ID}-{setup.default_model}"
        # The model Junie starts with, also set when the user picks one in Junie; without it, Junie opens its
        # "Select model" screen. First, so that jl status reads it back. Junie's earlier builds read config.json.
        edits.append(ConfigEdit(folder / "settings.json", "jsonc", {("modelForLaunch",): profile}))
        edits.append(ConfigEdit(folder / "config.json", "jsonc", {("model",): profile}))
    return tuple(edits)


def _others(path: Path, key: KeyPath, field: str, ours: str) -> list[object]:
    # The entries of a JSON config's list that aren't jl's: those whose field isn't ours.
    current = _current(path, key)
    if not isinstance(current, list):
        return []
    return [entry for entry in current if not (isinstance(entry, dict) and entry.get(field) == ours)]


def _current(path: Path, key: KeyPath, fmt: Literal["jsonc", "toml", "yaml"] = "jsonc") -> object:
    # The value at a key path in a config now; None when it or the file is missing, or the file is
    # unreadable, which plan_connect then reports.
    try:
        value = _lookup(_parse(path.read_text(encoding="utf-8"), fmt), key) if path.exists() else None
    except (ValueError, YAMLError):
        return None
    return None if value is _MISSING else value


def _model(setup: Setup, model_id: str) -> ModelInfo:
    # A model the catalog doesn't list, such as default_model's marker, gets the fallback limits.
    return next((model for model in setup.models if model.id == model_id), ModelInfo(id=model_id))


def _input(model: ModelInfo) -> list[str]:
    # The input kinds harnesses take; the catalog also lists ones they don't, such as video.
    return [kind for kind in ("text", "image") if kind in model.input_modalities]


def _opencode_values(setup: Setup) -> dict[KeyPath, object]:
    # OpenCode 2.2.3+ reads the providers entry, whose capabilities carry each model's real
    # reasoning efforts: OpenCode builds the picker's effort variants from them, and without them
    # it offers low, medium and high, of which the Models API refuses any the model doesn't list.
    # The V1 provider entry stays beside it for OpenCode installs that predate this one, and Kilo.
    models = {}
    for model in setup.models:
        context, output = _limits(model)
        reasoning: dict[str, object] = {"supported": model.thinking.supported}
        if efforts := model.thinking.supported_efforts:
            reasoning["effort"] = efforts
        models[model.id] = {
            "name": model.name or model.id,
            "capabilities": {
                "tools": bool(model.capabilities.tool_calling),
                "input": _input(model),
                "output": ["text"],
                "reasoning": reasoning,
            },
            "limit": {"context": context, "output": output},
        }
    values: dict[KeyPath, object] = {
        ("providers", PROVIDER_ID): {
            "name": "Jarvislabs",
            "package": "@opencode/ai/providers/openai-compatible",
            "settings": {"baseURL": setup.base_url, "apiKey": setup.api_key},
            "models": models,
        },
        ("provider", PROVIDER_ID): _v1_provider(setup),
    }
    if setup.default_model:
        values["model",] = f"{PROVIDER_ID}/{setup.default_model}"
    return values


def _kilo_values(setup: Setup) -> dict[KeyPath, object]:
    # Kilo, an OpenCode fork, reads only the V1 provider entry, which cannot express an effort
    # ladder — but a model's options.reasoningEffort does reach its wire as a fixed effort.
    values: dict[KeyPath, object] = {("provider", PROVIDER_ID): _v1_provider(setup, fixed_effort=True)}
    if setup.default_model:
        values["model",] = f"{PROVIDER_ID}/{setup.default_model}"
    return values


def _v1_provider(setup: Setup, fixed_effort: bool = False) -> dict[str, object]:
    # The V1 provider entry, without an effort ladder: it predates the capabilities that carry
    # one. Kilo maps a model's options.reasoningEffort onto the wire, so it gets the model's
    # default effort pinned there; variants would inject junk into the request body, never those.
    models = {}
    for model in setup.models:
        context, output = _limits(model)
        entry: dict[str, object] = {
            "name": model.name or model.id,
            "tool_call": bool(model.capabilities.tool_calling),
            "limit": {"context": context, "output": output},
            "modalities": {
                "input": _input(model),
                "output": ["text"],
            },
        }
        if fixed_effort and (effort := _default_effort(model)):
            entry["options"] = {"reasoningEffort": effort}
        models[model.id] = entry
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Jarvislabs",
        "options": {"baseURL": setup.base_url, "apiKey": setup.api_key},
        "models": models,
    }


def _claude_code(setup: Setup) -> tuple[ConfigEdit, ...]:
    # Claude Code keeps a custom gateway in settings.json's env block, the same file on every OS.
    # The tiers are Claude names; pointing all of them at our default keeps /model on our models.
    values: dict[KeyPath, object] = {
        ("env", "ANTHROPIC_BASE_URL"): anthropic_base_url(),
        ("env", "ANTHROPIC_AUTH_TOKEN"): setup.api_key,
    }
    if setup.default_model:
        values[("env", "ANTHROPIC_MODEL")] = setup.default_model
        values[("env", "ANTHROPIC_SMALL_FAST_MODEL")] = setup.default_model
        values[("env", "CLAUDE_CODE_SUBAGENT_MODEL")] = setup.default_model
        # Every alias Claude Code can pick (--model opus/sonnet/haiku/fable, subagents, /compact).
        for tier in ("OPUS", "SONNET", "HAIKU", "FABLE"):
            values[("env", f"ANTHROPIC_DEFAULT_{tier}_MODEL")] = setup.default_model
        # One extra row in /model with our model's own name, so the picker shows Jarvislabs and
        # not only Claude's aliases renamed to our id.
        model = next((m for m in setup.models if m.id == setup.default_model), None)
        values[("env", "ANTHROPIC_CUSTOM_MODEL_OPTION")] = setup.default_model
        values[("env", "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME")] = (
            model.name if model and model.name else setup.default_model
        )
        values[("env", "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION")] = "Served by Jarvislabs"
    return (ConfigEdit(Path.home() / ".claude" / "settings.json", "jsonc", values),)


def _codex(setup: Setup) -> tuple[ConfigEdit, ...]:
    # Codex reads $CODEX_HOME/config.toml, else ~/.codex/config.toml, and only speaks the
    # Responses API. The bearer token is inline so the user doesn't have to export a variable.
    folder = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    values: dict[KeyPath, object] = {
        ("model_provider",): PROVIDER_ID,
        ("model_providers", PROVIDER_ID): {
            "name": "Jarvislabs",
            "base_url": responses_base_url(),
            "wire_api": "responses",
            "experimental_bearer_token": setup.api_key,
        },
    }
    if setup.default_model:
        values[("model",)] = setup.default_model
        if model := next((m for m in setup.models if m.id == setup.default_model), None):
            values[("model_context_window",)] = _limits(model)[0]
    edits = [ConfigEdit(folder / "config.toml", "toml", values)]
    # The catalog tells Codex each model's own reasoning efforts, for its picker and its
    # requests; a custom provider's model otherwise gets none. Its entries each need the base
    # instructions, which replace the prompt Codex assembles itself, so the stock text is read
    # from the installed binary — and when that fails, no catalog is written, which leaves
    # Codex's own default prompt and no efforts, exactly as before.
    if any(model.thinking.supported_efforts for model in setup.models) and (instructions := _codex_instructions()):
        catalog = folder / "models.json"
        ours = [_codex_entry(model, instructions) for model in setup.models]
        # The file is shared — DeepSeek's own setup script writes it too — so only jl's own
        # earlier entries (marked) and entries for models of ours go; the user's stay.
        slugs = {model.id for model in setup.models}
        current = _current(catalog, ("models",))
        kept = [
            entry
            for entry in (current if isinstance(current, list) else ())
            if not (
                isinstance(entry, dict)
                and (entry.get("description") == _CODEX_MARKER or entry.get("slug") in slugs)
            )
        ]
        values[("model_catalog_json",)] = str(catalog)
        edits.append(ConfigEdit(catalog, "jsonc", {("models",): [*kept, *ours]}, items={("models",): ours}))
    return tuple(edits)


# jl's mark on its entries in Codex's catalog file, which the user's own setup may hold entries
# in too; it names the model in Codex's picker as well.
_CODEX_MARKER = "Served by Jarvislabs"
_CODEX_EFFORT_DESCRIPTIONS = {
    "low": "Lighter reasoning",
    "medium": "Balanced reasoning",
    "high": "Deep reasoning",
    "xhigh": "Deeper reasoning",
    "max": "Maximum reasoning",
}


def _codex_entry(model: ModelInfo, instructions: str) -> dict[str, object]:
    """Return one catalog entry: the fields Codex requires, plus the model's own efforts."""
    entry: dict[str, object] = {
        "slug": model.id,
        "display_name": model.name or model.id,
        "description": _CODEX_MARKER,
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,
        "support_verbosity": True,
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "experimental_supported_tools": [],
        # The entry's instructions replace the prompt Codex would assemble, so they carry its own.
        "base_instructions": instructions,
        # Without this, Codex asks the Responses API for a reasoning summary the Models API doesn't send.
        "default_reasoning_summary": "none",
        # Required even for a model that offers no efforts.
        "supported_reasoning_levels": [],
    }
    if efforts := model.thinking.supported_efforts:
        entry["supported_reasoning_levels"] = [
            {"effort": effort, "description": _CODEX_EFFORT_DESCRIPTIONS.get(effort, f"{effort} reasoning")}
            for effort in efforts
        ]
        if effort := _default_effort(model):
            entry["default_reasoning_level"] = effort
    return entry


def _codex_instructions() -> str | None:
    """Return the base instructions of Codex's own models, read from the installed binary.

    They change with the version the user runs, so reading them beats shipping a copy: the text
    always matches. Any failure returns None, and the caller then writes no catalog. The command
    runs under an empty CODEX_HOME, so it prints the stock catalog rather than the user's.
    """
    program = _find("codex")
    if not program:
        return None
    with tempfile.TemporaryDirectory() as scratch:
        try:
            run = subprocess.run(
                [program, "debug", "models"],
                capture_output=True,
                timeout=30,
                env={**(_path_env("codex", program) or os.environ), "CODEX_HOME": scratch},
            )
        except (OSError, subprocess.SubprocessError):
            return None
    try:
        models = json.loads(run.stdout.decode("utf-8", errors="replace")).get("models")
    except ValueError:
        return None
    if not isinstance(models, list):
        return None
    entries = [
        entry
        for entry in models
        if isinstance(entry, dict) and isinstance(entry.get("base_instructions"), str) and entry["base_instructions"]
    ]
    # DeepSeek's own setup ships this model's text for the same purpose; any built-in's does.
    for entry in entries:
        if entry.get("slug") == "gpt-5.6-sol":
            return entry["base_instructions"]
    return entries[0]["base_instructions"] if entries else None


def _zed(setup: Setup) -> tuple[ConfigEdit, ...]:
    # Zed keeps its settings in %APPDATA%\Zed on Windows, the XDG config folder on Linux, and
    # ~/.config/zed on macOS. Its docs say keys don't belong in this file: Zed asks for the key.
    if sys.platform == "win32":
        folder = Path(os.environ["APPDATA"]) / "Zed"
    elif sys.platform == "darwin":
        folder = Path.home() / ".config" / "zed"
    else:
        folder = _xdg_config_home() / "zed"
    models = [
        {
            "name": model.id,
            "display_name": model.name or model.id,
            "max_tokens": _limits(model)[0],
            "capabilities": {
                "tools": bool(model.capabilities.tool_calling),
                "images": "image" in model.input_modalities,
            },
        }
        for model in setup.models
    ]
    values: dict[KeyPath, object] = {
        ("language_models", "openai_compatible", "Jarvislabs"): {"api_url": setup.base_url, "available_models": models}
    }
    if setup.default_model:
        values["agent", "default_model", "provider"] = "Jarvislabs"
        values["agent", "default_model", "model"] = setup.default_model
    return (ConfigEdit(folder / "settings.json", "jsonc", values),)


def _goose(setup: Setup) -> tuple[ConfigEdit, ...]:
    folder = (
        Path(os.environ["APPDATA"]) / "Block" / "goose" / "config"
        if sys.platform == "win32"
        else _xdg_config_home() / "goose"
    )
    provider_models = []
    for model in setup.models:
        entry: dict[str, object] = {
            "name": model.id,
            "context_limit": _limits(model)[0],
            "supports_vision": "image" in model.input_modalities,
        }
        if effort := _default_effort(model):
            # Goose sends request_params verbatim with every request for the model, whichever of
            # them the user picks; a model without efforts gets no request_params at all.
            entry["request_params"] = {"reasoning_effort": effort}
        provider_models.append(entry)
    provider = {
        "name": PROVIDER_ID,
        "engine": "openai",
        "display_name": "Jarvislabs",
        "base_url": setup.base_url + "/chat/completions",
        "models": provider_models,
        # Use the context limits above, not a model list fetched from /models.
        "dynamic_models": False,
        "supports_streaming": True,
        # Goose reads a key named by api_key_env from its keyring, and falls back to secrets.yaml only
        # when the keyring holds no goose entry; a header in jl's own file works either way.
        "requires_auth": False,
        "headers": {"Authorization": f"Bearer {setup.api_key}"},
    }
    edits = (
        ConfigEdit(
            folder / "custom_providers" / f"{PROVIDER_ID}.json", "jsonc", {(k,): v for k, v in provider.items()}
        ),
    )
    if not setup.default_model:
        return edits
    default: dict[KeyPath, object] = {
        ("active_provider",): PROVIDER_ID,
        ("providers", PROVIDER_ID, "enabled"): True,
        ("providers", PROVIDER_ID, "model"): setup.default_model,
        ("providers", PROVIDER_ID, "configured"): True,
    }
    return (*edits, ConfigEdit(folder / "config.yaml", "yaml", default))


def _aider(setup: Setup) -> tuple[ConfigEdit, ...]:
    # Aider has no provider entry: one default model plus metadata, so other models keep their limits.
    metadata: dict[KeyPath, object] = {}
    for model in setup.models:
        context, output = _limits(model)
        metadata[(f"openai/{model.id}",)] = {
            "max_input_tokens": context,
            "max_output_tokens": output,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    limits = ConfigEdit(Path.home() / ".aider.model.metadata.json", "jsonc", metadata)
    # Aider reads each model's reasoning effort from .aider.model.settings.yml, a list the user
    # may hold entries in too, so jl's sit in a marked block that leaves theirs alone. Aider
    # flattens extra_body onto the request, and takes the entry of whichever model it runs.
    entries = "\n".join(
        f"- name: openai/{model.id}\n  extra_params:\n    extra_body:\n      reasoning_effort: {effort}"
        for model in setup.models
        if (effort := _default_effort(model))
    )
    efforts = ConfigEdit(Path.home() / ".aider.model.settings.yml", "block", {(): entries}) if entries else None
    conf = Path.home() / ".aider.conf.yml"
    settings: dict[KeyPath, object] = {("openai-api-base",): setup.base_url, ("openai-api-key",): setup.api_key}
    if setup.default_model:
        settings["model",] = f"openai/{setup.default_model}"
        write_conf = True
    else:
        # Aider's base URL and key serve every openai/ model, so a refresh (jl update, a key rotation)
        # leaves them alone once the user points Aider elsewhere.
        write_conf = _current(conf, ("openai-api-base",), "yaml") == setup.base_url
    edits = [ConfigEdit(conf, "yaml", settings)] if write_conf else []
    edits.append(limits)
    if efforts:
        edits.append(efforts)
    return tuple(edits)


def _crush(setup: Setup) -> tuple[ConfigEdit, ...]:
    lines = [
        f"provider add {PROVIDER_ID} --type completions --name Jarvislabs"
        f" --base-url {shlex.quote(setup.base_url)} --api-key {shlex.quote(setup.api_key)}"
    ]
    for model in setup.models:
        context, output = _limits(model)
        images = "true" if "image" in model.input_modalities else "false"
        lines.append(
            f"model add {PROVIDER_ID}/{shlex.quote(model.id)} --name {shlex.quote(model.name or model.id)}"
            f" --context-window {context} --default-max-tokens {output} --supports-images {images}"
        )
    path = _xdg_config_home() / "crush" / "crushrc"
    if setup.default_model:
        default = shlex.quote(setup.default_model)
        # The slot lines carry the default model's effort: model add takes no provider options,
        # and a provider-level one would send a single effort for every model, which the Models
        # API refuses for any model whose own list lacks it. Switching models in Crush drops the
        # slot's options, and the API's own default then applies.
        options = ""
        model = next((m for m in setup.models if m.id == setup.default_model), None)
        if model and (effort := _default_effort(model)):
                options = f" --provider-options {shlex.quote(json.dumps({'reasoning_effort': effort}))}"
        lines += [f"model large {PROVIDER_ID}/{default}{options}", f"model small {PROVIDER_ID}/{default}{options}"]
    else:
        # The block is rewritten whole, so a refresh carries over the default the last connect set.
        block = read_block(path.read_text(encoding="utf-8")) if path.exists() else None
        lines += [line for line in (block or "").splitlines() if line.startswith(("model large ", "model small "))]
    return (ConfigEdit(path, "block", {(): "\n".join(lines)}),)


# The extension reads providers.json only while no provider is picked in its own settings.
_CLINE_NOTE = (
    "Already use Cline in VS Code? In Cline's settings, set API Provider to OpenAI Compatible and Model ID to {model}."
)

HARNESSES: tuple[Harness, ...] = (
    Harness(
        id="opencode",
        name="OpenCode",
        kind="cli",
        support="connect",
        launch=("opencode",),
        writer=_opencode,
        install_url="https://opencode.ai/docs/#install",
    ),
    Harness(
        id="kilo",
        name="Kilo",
        kind="cli",
        support="connect",
        launch=("kilo",),
        writer=_kilo,
        install_url="https://kilo.ai/cli",
    ),
    Harness(
        id="pi",
        name="pi",
        kind="cli",
        support="connect",
        launch=("pi",),
        writer=_pi,
        install_url="https://pi.dev/docs/latest/quickstart",
    ),
    Harness(
        id="kimi",
        name="Kimi Code",
        kind="cli",
        support="connect",
        launch=("kimi",),
        writer=_kimi,
        install_url="https://moonshotai.github.io/kimi-code/en/guides/getting-started.html",
    ),
    Harness(
        id="cline",
        name="Cline",
        kind="cli",
        support="connect",
        launch=("cline", "-i"),
        writer=_cline,
        one_model=True,
        note=_CLINE_NOTE,
        install_url="https://docs.cline.bot/getting-started/installing-cline#cli",
    ),
    # Kilo's extension reads the same config file as its CLI; Cline's keeps its own state.
    Harness(
        id="kilo-vscode",
        name="Kilo (VS Code)",
        kind="vscode",
        support="connect",
        launch=("code", "."),
        writer=_kilo,
        # The current extension ships on the pre-release channel only.
        extension=("kilocode.kilo-code", "--pre-release"),
    ),
    Harness(
        id="cline-vscode",
        name="Cline (VS Code)",
        kind="vscode",
        support="connect",
        launch=("code", "."),
        writer=_cline_extension,
        one_model=True,
        extension=("saoudrizwan.claude-dev",),
    ),
    Harness(
        id="qwen",
        name="Qwen Code",
        kind="cli",
        support="connect",
        launch=("qwen",),
        writer=_qwen,
        install_url="https://qwenlm.github.io/qwen-code-docs/en/users/quickstart/",
        # The providerProtocol mapping is new in Qwen Code 0.19.3; older builds skip an unmapped
        # custom provider, so jl's models never reach the picker.
        min_version="0.19.3",
    ),
    Harness(
        id="copilot",
        name="Copilot CLI",
        kind="cli",
        support="connect",
        launch=("copilot",),
        writer=_copilot,
        install_url="https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli",
    ),
    Harness(
        id="junie",
        name="Junie CLI",
        kind="cli",
        support="connect",
        launch=("junie",),
        writer=_junie,
        install_url="https://junie.jetbrains.com/docs/junie-cli.html",
    ),
    Harness(
        id="goose",
        name="Goose",
        kind="cli",
        support="connect",
        launch=("goose", "session"),
        writer=_goose,
        install_url="https://goose-docs.ai/docs/getting-started/installation/",
    ),
    Harness(
        id="aider",
        name="Aider",
        kind="cli",
        support="connect",
        launch=("aider",),
        writer=_aider,
        install_url="https://aider.chat/docs/install.html",
    ),
    Harness(
        id="crush",
        name="Crush",
        kind="cli",
        support="connect",
        launch=("crush",),
        writer=_crush,
        install_url="https://github.com/charmbracelet/crush#installation",
        # --type completions in jl's crushrc is new in Crush 0.99.0; older ones ignore jl's provider.
        min_version="0.99.0",
    ),
)

# Each coming-soon harness's manual setup page is <HARNESS_DOCS_URL>/<id>.
HARNESS_DOCS_URL = "https://docs.jarvislabs.ai/models-api/harnesses"


def _coming_soon(
    harness_id: str,
    name: str,
    kind: Literal["cli", "vscode"],
    command: tuple[str, ...],
    writer: Callable[[Setup], tuple[ConfigEdit, ...]] | None = None,
    preview: bool = False,
    one_model: bool = False,
) -> Harness:
    docs_url = f"{HARNESS_DOCS_URL}/{harness_id}"
    return Harness(
        id=harness_id,
        name=name,
        kind=kind,
        support="coming_soon",
        launch=command,
        writer=writer,
        docs_url=docs_url,
        preview=preview,
        one_model=one_model,
    )


# Claude Code and Codex wait for the Models API's Anthropic and Responses endpoints, so they are
# preview-only until those are public. Cursor, VS Code Copilot and Zed keep the key in an app
# database or the OS keychain (phase 2).
HARNESSES += (
    # Behind JL_PREVIEW_HARNESSES until /v1/messages and /v1/responses are public. Both hold one
    # model: the gateway name is passed as the model, not picked from a catalog in the file.
    _coming_soon("claude-code", "Claude Code", "cli", ("claude",), writer=_claude_code, preview=True, one_model=True),
    _coming_soon("codex", "Codex", "cli", ("codex",), writer=_codex, preview=True, one_model=True),
    _coming_soon("cursor", "Cursor", "cli", ("cursor", ".")),
    _coming_soon("vscode-copilot", "VS Code Copilot", "vscode", ("code", ".")),
    # A writer, though jl doesn't write Zed's key to the keychain yet: jl connect --print shows the settings.
    _coming_soon("zed", "Zed", "cli", ("zed", "."), writer=_zed),
)
