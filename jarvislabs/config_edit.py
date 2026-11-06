"""Edit harness config files as text, keeping the user's comments and layout.

edit_config sets and deletes values in JSON with comments (JSONC), TOML and YAML. A key
path is a tuple of object keys from the top of the file, e.g. ("provider", "jarvislabs").
Comments, blank lines, trailing commas and indentation outside the edited values stay as
they were, so deleting what an earlier edit added gives back the text from before it.

replace_block swaps jl's marked block in a line-based file, such as Crush's crushrc, and
read_block returns its lines.

Each function takes the file's text and returns text; none reads or writes files.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import tomlkit
from json5.dumper import ModelDumper, dumps
from json5.loader import ModelLoader, loads
from json5.model import DoubleQuotedString, JSONObject
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.util import load_yaml_guess_indent
from tomlkit.container import Container
from tomlkit.items import AoT, Comment as TomlComment, Null, Table, Whitespace

KeyPath = tuple[str, ...]

_BLOCK_START = "# >>> jl >>>"
_BLOCK_END = "# <<< jl <<<"
_BLOCK = re.compile(rf"^{re.escape(_BLOCK_START)}$.*?^{re.escape(_BLOCK_END)}$\n?", re.MULTILINE | re.DOTALL)


def edit_config(
    text: str,
    fmt: Literal["jsonc", "toml", "yaml"],
    set_values: Mapping[KeyPath, object],
    delete: Iterable[KeyPath] = (),
) -> str:
    """Return the config text with values set and deleted at key paths.

    A path whose parent objects are missing gets them created. A deleted path that is
    missing is skipped, and a parent object the deletion leaves empty is removed with it.
    Empty text is read as an empty object.

    Args:
        text: The config file's current text.
        fmt: The file's format: "jsonc" (also plain JSON), "toml" or "yaml".
        set_values: The value to write at each key path, replacing any value there.
        delete: The key paths to remove, applied after the sets.

    Returns:
        The edited text.
    """
    if fmt == "jsonc":
        return _edit_jsonc(text, set_values, delete)
    if fmt == "toml":
        return _edit_toml(text, set_values, delete)
    return _edit_yaml(text, set_values, delete)


def replace_block(text: str, block: str | None) -> str:
    """Return the text with jl's marked block replaced, added or removed.

    The block sits between the lines "# >>> jl >>>" and "# <<< jl <<<". A missing block is
    added at the end of the text.

    Args:
        text: The file's current text.
        block: The lines to put between the markers, or None to remove the block and its
            markers.

    Returns:
        The edited text.
    """
    new = "" if block is None else _BLOCK_START + "\n" + block.rstrip("\n") + "\n" + _BLOCK_END + "\n"
    if _BLOCK.search(text):
        return _BLOCK.sub(lambda _: new, text, count=1)
    if text and not text.endswith("\n"):
        text += "\n"
    return text + new


def read_block(text: str) -> str | None:
    """Return the lines of jl's marked block, without the markers.

    Args:
        text: The file's text.

    Returns:
        The lines between the markers, or None when the text has no jl block.
    """
    match = _BLOCK.search(text)
    return "\n".join(match[0].splitlines()[1:-1]) if match else None


def _nest(path: KeyPath, value: object) -> object:
    for key in reversed(path):
        value = {key: value}
    return value


def _set_target(root: Any, path: KeyPath, value: object) -> tuple[Any, str, object]:
    """Return the mapping, key and value to assign so that path holds value.

    The value is wrapped in objects for the parents the path is missing.
    """
    node = root
    for i, key in enumerate(path[:-1]):
        if key not in node:
            return node, key, _nest(path[i + 1 :], value)
        node = node[key]
    return node, path[-1], value


def _delete_target(root: Any, path: KeyPath) -> tuple[Any, str] | None:
    """Return the mapping and key to delete for path, or None when path is missing.

    Climbs to the highest parent that holds nothing but path, so parents the deletion
    would leave empty go with it.
    """
    chain = []
    node = root
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        chain.append((node, key))
        node = node[key]
    while len(chain) > 1 and len(chain[-1][0]) == 1:
        chain.pop()
    return chain[-1]


# JSONC, on json-five's model: each node keeps the whitespace and comments around it in
# wsc_before and wsc_after. An object keeps what follows "{" in leading_wsc, and what
# follows a trailing comma in trailing_comma.wsc_after.


def _edit_jsonc(text: str, set_values: Mapping[KeyPath, object], delete: Iterable[KeyPath]) -> str:
    model = loads(text if text.strip() else "{}\n", loader=ModelLoader())
    match = re.search(r"^([ \t]+)\S", text, re.MULTILINE)
    unit = match[1] if match else "  "
    for path, value in set_values.items():
        _jsonc_set(model.value, path, value, unit)
    for path in delete:
        _jsonc_delete(model.value, path)
    return dumps(model, dumper=ModelDumper())  # type: ignore[arg-type]  # the stubs name only the plain dumper


def _jsonc_index(obj: JSONObject, key: str) -> int | None:
    for i, k in enumerate(obj.keys):
        if getattr(k, "characters", getattr(k, "name", None)) == key:
            return i
    return None


def _jsonc_node(value: object, unit: str, depth: int) -> Any:
    # json-five's own value-to-node conversion writes single-quoted strings, which JSONC rejects.
    text = json.dumps(value, indent=unit).replace("\n", "\n" + unit * depth)
    return loads(text, loader=ModelLoader()).value


def _split_closing(wsc: list[Any]) -> tuple[list[Any], list[Any]]:
    """Split whitespace and comments into the comments and the whitespace that ends them."""
    if wsc and isinstance(wsc[-1], str):
        return wsc[:-1], wsc[-1:]
    return wsc, []


def _jsonc_set(obj: Any, path: KeyPath, value: object, unit: str) -> None:
    for depth, key in enumerate(path, start=1):
        j = _jsonc_index(obj, key)
        if j is None:
            _jsonc_insert(obj, key, _nest(path[depth:], value), unit, depth)
            return
        if depth == len(path):
            new = _jsonc_node(value, unit, depth)
            new.wsc_before, new.wsc_after = obj.values[j].wsc_before, obj.values[j].wsc_after
            obj.values[j] = new
            return
        obj = obj.values[j]


def _jsonc_insert(obj: JSONObject, key: str, value: object, unit: str, depth: int) -> None:
    """Append key to obj, its keys sitting depth levels deep.

    Comments after the old last pair stay where they were, so the new comma goes before
    them: '"a": 1 // note' becomes '"a": 1, // note' followed by the new pair.
    """
    k = DoubleQuotedString(key, raw_value=json.dumps(key))
    new = _jsonc_node(value, unit, depth)
    new.wsc_before = [" "]
    if not obj.keys:
        comments, _ = _split_closing(obj.leading_wsc)
        obj.leading_wsc = []
        new.wsc_after = ["\n" + unit * (depth - 1)]
    elif obj.trailing_comma:
        comments, obj.trailing_comma.wsc_after = _split_closing(obj.trailing_comma.wsc_after)
    else:
        comments, new.wsc_after = _split_closing(obj.values[-1].wsc_after)
        obj.values[-1].wsc_after = []
    k.wsc_before = [*comments, "\n" + unit * depth]
    obj.keys.append(k)
    obj.values.append(new)


def _jsonc_delete(root: JSONObject, path: KeyPath) -> None:
    chain = []
    obj: Any = root
    for key in path:
        j = _jsonc_index(obj, key) if isinstance(obj, JSONObject) else None
        if j is None:
            return
        chain.append((obj, j))
        obj = obj.values[j]
    while len(chain) > 1 and len(chain[-1][0].keys) == 1:
        chain.pop()
    obj, j = chain[-1]
    # Comments that followed the previous pair's comma go back after that pair.
    kept, _ = _split_closing(obj.keys[j].wsc_before)
    if len(obj.keys) == 1:
        obj.leading_wsc = []
    elif j < len(obj.keys) - 1:
        obj.keys[j + 1].wsc_before = kept + obj.keys[j + 1].wsc_before
    elif obj.trailing_comma:
        obj.trailing_comma.wsc_after = kept + obj.trailing_comma.wsc_after
    else:
        obj.values[j - 1].wsc_after = kept + obj.values[j].wsc_after
    del obj.keys[j]
    del obj.values[j]


# TOML, on tomlkit: a table's body ends with the blank lines and comments that follow it
# in the file, up to the next table header. A table jl adds takes over those trailing
# lines from the table before it, so they stay above the next header; deleting it hands
# them back.


def _edit_toml(text: str, set_values: Mapping[KeyPath, object], delete: Iterable[KeyPath]) -> str:
    doc = tomlkit.parse(text)
    # A dotted key (providers.jl.type = "x") makes tomlkit render later keys under the wrong
    # table once jl adds one. Refuse instead of corrupting the user's file.
    if _toml_has_dotted_keys(text):
        raise ValueError("jl can't edit dotted keys; write them as [tables] first")
    for path, value in set_values.items():
        _toml_assign(*_set_target(doc, path, value))
    for path in delete:
        target = _delete_target(doc, path)
        if target:
            _toml_delete(*target)
    return tomlkit.dumps(doc)


def _toml_has_dotted_keys(text: str) -> bool:
    """Return whether the text sets a key with a dot in it, such as providers.jl.type = "x".

    Read from the text, not the parsed document: tomlkit turns dotted keys into ordinary tables.
    A quoted key may hold a dot without being dotted, and a table header is not a key.
    """
    in_multiline = False
    for line in text.splitlines():
        if (line.count(chr(34) * 3) + line.count(chr(39) * 3)) % 2:
            in_multiline = not in_multiline
            continue
        if in_multiline:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "[")) or "=" not in stripped:
            continue
        key = re.sub(r'"[^"]*"|' + chr(39) + r"[^" + chr(39) + r"]*" + chr(39), "", stripped.split("=", 1)[0])
        if "." in key:
            return True
    return False


def _toml_body(item: Container | Table) -> list[tuple[Any, Any]]:
    return item.body if isinstance(item, Container) else item.value.body


def _toml_tail(table: Table) -> list[tuple[Any, Any]]:
    """Return the body of the last table nested in table, which holds its trailing lines."""
    body = table.value.body
    items = [item for _, item in body if not isinstance(item, Null)]
    while items and isinstance(items[-1], Table):
        body = items[-1].value.body
        items = [item for _, item in body if not isinstance(item, Null)]
    return body


def _toml_pop_trailing(body: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    trailing: list[tuple[Any, Any]] = []
    while body and body[-1][0] is None and isinstance(body[-1][1], (Whitespace, TomlComment, Null)):
        trailing.insert(0, body.pop())
    return [entry for entry in trailing if not isinstance(entry[1], Null)]


def _toml_previous(container: Container | Table, key: str) -> Any:
    previous = None
    for k, item in _toml_body(container):
        if k is not None and k.key == key:
            return previous
        if not isinstance(item, Null):
            previous = item
    return previous


def _toml_assign(container: Container | Table, key: str, value: object) -> None:
    old = container.get(key)
    if not isinstance(value, Mapping):
        last_pair = [k for k, item in _toml_body(container) if k is not None and not isinstance(item, (Table, AoT))]
        if old is None and last_pair and isinstance(container, Container):
            # Right after the last top-level key-value pair, not after the comments above the
            # first table. tomlkit has no public insert-at-position, hence its exact version pin.
            container._insert_after(last_pair[-1], key, value)
        else:
            container[key] = value
        return
    if isinstance(old, Table):
        trailing, indent = _toml_pop_trailing(_toml_tail(old)), old.trivia.indent
        container[key] = value
        tail = _toml_tail(container[key])
        # The new table already ends with the blank line that separated the old one from the next
        # table; copying that blank line too would add one more on every rewrite.
        while trailing and tail and isinstance(trailing[-1][1], Whitespace) and isinstance(tail[-1][1], Whitespace):
            trailing.pop()
        tail.extend(trailing)
        container[key].trivia.indent = indent
        return
    previous = _toml_previous(container, key) if old is None else None
    container[key] = value
    if isinstance(previous, Table):
        _toml_tail(container[key]).extend(_toml_pop_trailing(_toml_tail(previous)))
        container[key].trivia.indent = "\n"


def _toml_delete(container: Container | Table, key: str) -> None:
    item = container[key]
    previous = _toml_previous(container, key)
    if isinstance(item, Table) and isinstance(previous, Table):
        trailing = _toml_pop_trailing(_toml_tail(item))
        previous_tail = _toml_tail(previous)
        _toml_pop_trailing(previous_tail)
        previous_tail.extend(trailing)
    body = _toml_body(container)
    place = next(i for i, (k, _) in enumerate(body) if k is not None and k.key == key)
    del container[key]
    if not isinstance(item, Table) or not isinstance(container, Container):
        return
    # The last table's blank line separates it from what is above; with nothing below it is a
    # trailing blank line the file didn't have. Older tables keep theirs for the one after, and a
    # table's own body is not the end of the file.
    rest = [entry for entry in body[place + 1 :] if not isinstance(entry[1], Null)]
    if rest:
        return
    before = place - 1
    while before >= 0 and isinstance(body[before][1], Null):
        before -= 1
    if before >= 0 and isinstance(body[before][1], Whitespace):
        leftover = body[before][1].s[:-1]
        if leftover:
            body[before] = (None, Whitespace(leftover))
        else:
            body.pop(before)


# YAML, on ruamel.yaml's round-trip mode. Indentation is one setting per file, so it is
# read from the file first; an end-of-line comment keeps its distance from the value.


def _edit_yaml(text: str, set_values: Mapping[KeyPath, object], delete: Iterable[KeyPath]) -> str:
    yaml = YAML()
    yaml.preserve_quotes = True
    doc, sequence, offset = load_yaml_guess_indent(text, yaml=yaml)
    if offset is None:
        yaml.indent(mapping=_yaml_mapping_indent(text))
    else:
        yaml.indent(mapping=_yaml_mapping_indent(text), sequence=sequence, offset=offset)
    if doc is None:
        doc = CommentedMap()
    lines = text.splitlines()
    for path, value in set_values.items():
        node, key, value = _set_target(doc, path, value)
        _yaml_keep_comment_gap(yaml, lines, node, key, value)
        node[key] = value
    for path in delete:
        target = _delete_target(doc, path)
        if target:
            del target[0][target[1]]
    out = io.StringIO()
    yaml.dump(doc, out)
    return out.getvalue()


def _yaml_mapping_indent(text: str) -> int:
    parent = None
    for line in text.splitlines():
        content = line.lstrip(" ")
        if not content or content.startswith("#"):
            continue
        indent = len(line) - len(content)
        if parent is not None and indent > parent and not content.startswith("-"):
            return indent - parent
        parent = indent if line.rstrip().endswith(":") else None
    return 2


def _yaml_keep_comment_gap(yaml: YAML, lines: list[str], node: Any, key: str, value: object) -> None:
    """Move the comment after key's value so its gap from the value stays the same.

    ruamel keeps the comment's column instead, which shifts it when the value's length changes.
    """
    comment = node.ca.items.get(key, [None] * 3)[2] if isinstance(node, CommentedMap) else None
    if comment is None or key not in node or isinstance(value, (Mapping, list)):
        return
    line, column = node.lc.value(key)
    old = lines[line][column : comment.column]
    rendered = io.StringIO()
    yaml.dump(value, rendered)
    comment.column = column + len(rendered.getvalue().splitlines()[0]) + len(old) - len(old.rstrip())
