"""Config file R/W, token storage, atomic writes.

Config lives at ~/.config/jl/config.toml (XDG-compliant via platformdirs).
Token precedence: explicit arg > JL_API_KEY env var > config file > error.

The [connect] table holds this machine's subscription key and, per connected harness,
the values `jl connect` overwrote in that harness's config files, the backups it saved of
them, the files it created, and what it wrote to each file (see Connection).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w
from platformdirs import user_config_dir

from jarvislabs.exceptions import JarvislabsError

CONFIG_DIR_NAME = "jl"
CONFIG_FILE = "config.toml"
ENV_VAR = "JL_API_KEY"


def config_dir() -> Path:
    """Platform-appropriate config directory (e.g. ~/.config/jl/)."""
    return Path(user_config_dir(CONFIG_DIR_NAME))


def config_path() -> Path:
    return config_dir() / CONFIG_FILE


def load_config() -> dict:
    """Read config.toml. Returns empty dict if file doesn't exist or is malformed."""
    path = config_path()
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def write_private(path: Path, data: bytes) -> None:
    """Write bytes to a file atomically, readable and writable only by its owner.

    Creates the parent folder if needed. The data goes to a temp file in the same
    folder, which is fsynced, set to mode 0o600 and renamed over ``path``, so a
    reader sees the old file or the new one, never a partial write. On Windows
    the mode only controls the read-only flag. The temp file is removed if any
    step fails. A symlinked file is written through: its target is replaced, not
    the link, so a dotfiles-managed config stays live.

    Args:
        path: The file to write.
        data: The file's new contents.
    """
    # Follow a link so the user's dotfiles stay the real config instead of being replaced by a copy.
    target = path.resolve() if path.is_symlink() else path
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def save_config(config: dict) -> None:
    """Atomically write config dict to config.toml with 0o600 permissions."""
    write_private(config_path(), tomli_w.dumps(config).encode())


@dataclass(frozen=True)
class Written:
    """What `jl connect` wrote to one config file, so `jl disconnect` can take out only that.

    Attributes:
        fmt: The file's format: "jsonc", "toml", "yaml", or "block" for a file whose jl block
            is marked by comments.
        hashes: Per key path jl set, a hash of the value it last wrote there. A value that no
            longer matches its hash was changed by the user.
        parents: Objects holding a key path jl added that existed before jl, such as an empty
            "provider": {}. Removing jl's entries would otherwise remove an object left empty.
        kept: Key paths jl set that disconnect leaves, because the harness needs them once it
            shares the file, such as Cline's schema version. They have no hash.
        items: Per key path holding a list that jl shares with the user's own entries, such as
            Qwen Code's modelProviders.jarvislabs, a hash of each entry jl last wrote there. Such a
            path has no hash of its own: disconnect takes out only the entries still matching.
    """

    fmt: Literal["jsonc", "toml", "yaml", "block"]
    hashes: dict[tuple[str, ...], str] = field(default_factory=dict)
    parents: tuple[tuple[str, ...], ...] = ()
    kept: tuple[tuple[str, ...], ...] = ()
    items: dict[tuple[str, ...], tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Replaced:
    """The values `jl connect` overwrote in one harness's config files, and their backups.

    Attributes:
        values: Per config file, each key path jl set that already held a value, and
            that value. A path that held nothing isn't listed.
        backups: The copies of the original files jl saved, as "<config>.jl-backup".
        created: The config files that didn't exist until jl wrote them, so have no backup.
        written: Per config file, what jl wrote to it.
        untracked: True when the harness was first connected by an older jl, which saved no
            ``written``. The values that jl listed can't be told from its own, so they're dropped
            when its record is loaded.
    """

    values: dict[Path, dict[tuple[str, ...], object]] = field(default_factory=dict)
    backups: tuple[Path, ...] = ()
    created: tuple[Path, ...] = ()
    written: dict[Path, Written] = field(default_factory=dict)
    untracked: bool = False


@dataclass
class Connection:
    """This machine's key for the Models API, and the harnesses connected with it.

    Attributes:
        key_id: The key's ID on the backend.
        name: The key's name, "jl-<hostname>-<4 hex digits of the machine ID's hash>".
        kind: Who pays for the key's requests: "subscription" (the plan) or "payg" (the wallet).
        token: The key's secret, which the backend shows only when it creates the key.
        harnesses: Per connected harness ID, what connecting it replaced.
    """

    key_id: str
    name: str
    kind: Literal["payg", "subscription"]
    token: str
    harnesses: dict[str, Replaced] = field(default_factory=dict)


def load_connection() -> Connection | None:
    """Return this machine's key and connected harnesses, from config.toml's [connect] table.

    Returns:
        The connection, or None when no key is saved.
    """
    table = load_config().get("connect")
    if not table:
        return None
    try:
        return _connection_from(table)
    except (KeyError, TypeError, ValueError) as e:
        raise JarvislabsError(
            f"Can't read this machine's saved connection in {config_path()}: {e}. "
            "Edit the file, or remove its [connect] table to start over."
        ) from e


def _connection_from(table: dict) -> Connection:
    """Build the connection from config.toml's [connect] table, or raise when the table is incomplete."""
    harnesses = {}
    for harness_id, record in table.get("harnesses", {}).items():
        values: dict[Path, dict[tuple[str, ...], object]] = {}
        for entry in record.get("replaced", []):
            values.setdefault(Path(entry["file"]), {})[tuple(entry["path"])] = json.loads(entry["value"])
        backups = tuple(Path(p) for p in record.get("backups", []))
        created = tuple(Path(p) for p in record.get("created", []))
        written = {
            Path(entry["file"]): Written(
                entry["format"],
                {tuple(path): digest for path, digest in zip(entry["paths"], entry["hashes"], strict=True)},
                tuple(tuple(path) for path in entry["parents"]),
                tuple(tuple(path) for path in entry["kept"]),
                # A record saved before shared lists were tracked entry by entry has neither key.
                {
                    tuple(path): tuple(digests)
                    for path, digests in zip(entry.get("lists", []), entry.get("items", []), strict=True)
                },
            )
            for entry in record.get("written", [])
        }
        # A printed record has nothing to track; an older jl's connect saved backups or created files.
        untracked = record.get("untracked", "written" not in record and bool(backups or created or values))
        if "written" not in record:
            values = {}
        harnesses[harness_id] = Replaced(values, backups, created, written, untracked)
    return Connection(table["id"], table["name"], table["kind"], table["token"], harnesses)


def wallet_chosen() -> bool:
    """Return whether this machine pays from the wallet: by its saved key, the last one it had, or the user's choice."""
    config = load_config()
    return config.get("connect", {}).get("kind") == "payg" or bool(config.get("pays_from_wallet"))


def save_connection(connection: Connection | None) -> None:
    """Save this machine's key and connected harnesses to config.toml's [connect] table.

    Rewrites config.toml, keeping its other tables. Deleting the table of a wallet key leaves
    ``pays_from_wallet = true`` behind, so the wallet choice outlasts the key; saving a key
    removes it, since the key's kind then holds the choice.

    Args:
        connection: What to save, or None to delete the table.
    """
    config = load_config()
    old = config.pop("connect", None)
    if connection is None and old and old.get("kind") == "payg":
        config["pays_from_wallet"] = True
    if connection is not None:
        config.pop("pays_from_wallet", None)
        config["connect"] = {
            "id": connection.key_id,
            "name": connection.name,
            "kind": connection.kind,
            "token": connection.token,
            "harnesses": {
                harness_id: {
                    # Values are stored as JSON, since TOML has no null.
                    "replaced": [
                        {"file": str(path), "path": list(key_path), "value": json.dumps(value)}
                        for path, paths in replaced.values.items()
                        for key_path, value in paths.items()
                    ],
                    "backups": [str(path) for path in replaced.backups],
                    "created": [str(path) for path in replaced.created],
                    "written": [
                        {
                            "file": str(path),
                            "format": written.fmt,
                            "paths": [list(key_path) for key_path in written.hashes],
                            "hashes": list(written.hashes.values()),
                            "parents": [list(key_path) for key_path in written.parents],
                            "kept": [list(key_path) for key_path in written.kept],
                            "lists": [list(key_path) for key_path in written.items],
                            "items": [list(digests) for digests in written.items.values()],
                        }
                        for path, written in replaced.written.items()
                    ],
                    "untracked": replaced.untracked,
                }
                for harness_id, replaced in connection.harnesses.items()
            },
        }
    save_config(config)


def resolve_token(explicit: str | None = None) -> str | None:
    """Return the API token using precedence: explicit > env > config file.

    Returns None if no token found anywhere (caller decides how to handle).
    """
    if explicit:
        return explicit

    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return from_env

    config = load_config()
    return config.get("auth", {}).get("token")
