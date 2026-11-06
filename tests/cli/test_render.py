"""The shared output helpers: marks, headlines with hints, rows of changes, and the next-step line."""

from __future__ import annotations

import io
import re
import sys

import pytest
from rich.console import Console

from jarvislabs.cli import render, state

COLOUR = re.compile(r"\x1b\[[0-9;]*(3[0-9]|9[0-7]|38;)")


@pytest.fixture(autouse=True)
def human_output(monkeypatch):
    """Output for a person, as an earlier command's --json may have left it otherwise."""
    monkeypatch.setattr(state, "json_output", False)


@pytest.fixture()
def out(monkeypatch):
    """The stderr and stdout consoles, 40 columns wide and writing to the returned buffer, with no terminal."""
    buffer = io.StringIO()
    for name in ("console", "stdout_console"):
        monkeypatch.setattr(render, name, Console(file=buffer, width=40, theme=render.theme))
    return buffer


@pytest.fixture()
def coloured(monkeypatch):
    """The stderr console as a colour terminal, 80 columns wide, writing to the returned buffer."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=80, theme=render.theme, force_terminal=True, color_system="standard")
    monkeypatch.setattr(render, "console", console)
    return buffer


def test_a_hint_wraps_hanging_under_its_headline(out):
    render.warning("No subscription found", hint="Just subscribed? Give it a minute, then run `jl connect` again.")

    # 40 columns: the hint wraps at 38, two spaces in, with nothing trailing.
    assert out.getvalue().splitlines() == [
        "! No subscription found",
        "  Just subscribed? Give it a minute,",
        "  then run jl connect again.",
    ]


def test_a_command_in_a_hint_is_in_the_accent(coloured):
    render.warning("Goose skipped: not installed", hint="Then run `jl connect goose`.")

    # Cyan (36) starts at the command and ends after it.
    assert re.search(r"\x1b\[[0-9;]*36mjl connect goose\x1b\[0m", coloured.getvalue())


def test_info_and_hints_are_not_faint(coloured):
    # Sanctioned dim sites are tertiary only: borders, legends, IDs, paths, unavailable rows and
    # confirm choices. Instructions and hints are never faint.
    assert render.theme.styles["info"].dim is None
    assert render.theme.styles["hint"].dim is None
    assert render.theme.styles["hint"].color.name == "default"

    render.info("Resolving changes")
    render.hint_line("run `jl connect`")

    out = coloured.getvalue()
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert "Resolving changes" in plain
    assert "run jl connect" in plain
    # SGR 2 is dim: instructions are never the faintest text on screen.
    assert "\x1b[2m" not in out


def test_an_error_colours_its_mark_not_its_message(coloured):
    render.error("Test request failed")

    line = coloured.getvalue()
    assert re.sub(r"\x1b\[[0-9;]*m", "", line).endswith("Test request failed\n")
    assert "\x1b[1;31m" in line.split("Test request")[0]


def test_without_colour_marks_and_words_still_carry_the_state(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    buffer = io.StringIO()
    monkeypatch.setattr(render, "console", Console(file=buffer, theme=render.theme, force_terminal=True))

    render.success("OpenCode connected")
    render.error("Test request failed", hint="Run `jl connect` again.")

    assert not COLOUR.search(buffer.getvalue())
    assert re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue()).splitlines() == [
        "✓ OpenCode connected",
        "✗ Test request failed",
        "  Run jl connect again.",
    ]


@pytest.mark.parametrize(
    ("platform", "env", "encoding", "ascii_marks"),
    [
        # PowerShell 5.1's own window: no Windows Terminal, no VS Code.
        ("win32", {}, "utf-8", True),
        ("win32", {"WT_SESSION": "1"}, "utf-8", False),
        ("win32", {"TERM_PROGRAM": "vscode"}, "utf-8", False),
        # A Windows redirect under Python's default code page.
        ("win32", {"WT_SESSION": "1"}, "cp1252", True),
        ("linux", {}, "utf-8", False),
    ],
)
def test_a_legacy_windows_console_or_a_non_utf8_stream_gets_ascii_marks(
    monkeypatch, platform, env, encoding, ascii_marks
):
    monkeypatch.setattr(sys, "platform", platform)
    for var in ("WT_SESSION", "TERM_PROGRAM"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding=encoding))

    marks = render.choose_marks()

    assert marks == (render.ASCII_MARKS if ascii_marks else render.UNICODE_MARKS)


def test_ascii_marks_print_in_place_of_unicode_ones(out, monkeypatch):
    monkeypatch.setattr(render, "MARKS", render.ASCII_MARKS)

    render.success("Logged in as Ada")
    render.error("Test request failed")

    assert out.getvalue().splitlines() == ["+ Logged in as Ada", "x Test request failed"]


def test_change_rows_align_mark_name_and_files_with_notes_under_the_files(out):
    render.change_rows(
        [
            render.Row("add", "OpenCode", "~/.config/opencode/opencode.json"),
            render.Row("change", "Kilo", "~/kilo.jsonc", warning="replaces anthropic/claude-sonnet-4"),
            render.Row("remove", "pi", "~/.pi/agent/models.json", note="delete the file jl created"),
        ]
    )

    # The name column is the longest name, OpenCode, and two spaces: the files start at column 12 of 40, so a
    # longer line wraps under them.
    assert out.getvalue().splitlines() == [
        "+ OpenCode  ~/.config/opencode/opencode.",
        "            json",
        "~ Kilo      ~/kilo.jsonc",
        "            replaces",
        "            anthropic/claude-sonnet-4",
        "- pi        ~/.pi/agent/models.json",
        "            delete the file jl created",
    ]


def test_next_step_names_the_commands_to_type(out):
    render.next_step("start a tool in this folder:", ["opencode", "cline -i"])

    # The command comes first, the explanation under it, with a blank line before.
    assert out.getvalue().splitlines() == ["", "Next: opencode · cline -i", "      start a tool in this folder"]


def test_a_heading_opens_a_phase_after_a_blank_line(out):
    render.heading("Connect your coding tools", "to the API")

    assert out.getvalue().splitlines() == ["", "Connect your coding tools  to the API"]


def test_a_long_heading_note_wraps_under_the_title(out):
    render.heading("Connect tools", "to the Jarvislabs Models API, then start one")

    lines = out.getvalue().splitlines()
    assert lines[0] == ""
    assert lines[1] == "Connect tools  to the Jarvislabs Models"
    assert all(line.startswith(" " * len("Connect tools  ")) for line in lines[2:])
    assert " ".join(" ".join(lines[1:]).split()) == "Connect tools to the Jarvislabs Models API, then start one"


def test_a_long_change_rows_note_wraps_under_the_title(out):
    render.change_rows(
        [render.Row("add", "OpenCode", "~/x")],
        title="Connect tools",
        note="to the Jarvislabs Models API, then start one",
    )

    lines = out.getvalue().splitlines()
    assert lines[0] == "Connect tools  to the Jarvislabs Models"
    assert lines[1].startswith(" " * len("Connect tools  "))
    assert " ".join(" ".join(lines[0:2]).split()) == "Connect tools to the Jarvislabs Models API, then start one"


def test_next_step_uses_the_ascii_separator_when_marks_are_ascii(out, monkeypatch):
    monkeypatch.setattr(render, "MARKS", render.ASCII_MARKS)

    render.next_step("start a tool in this folder:", ["opencode", "cline -i"])

    assert out.getvalue().splitlines()[-2:] == ["Next: opencode | cline -i", "      start a tool in this folder"]


def test_an_empty_wrapped_line_has_no_trailing_spaces(out):
    render._hang(render.Text("xx "), render.Text("first\n\nsecond"), render.console)

    assert out.getvalue().splitlines() == ["xx first", "", "   second"]


def test_the_confirmation_mark_is_the_accent_colour_bold(coloured, monkeypatch):
    monkeypatch.setattr(render.console, "input", lambda prompt, **k: render.console.print(prompt, end="") or "n")

    assert render.confirm("Continue?") is False
    assert re.search(r"\x1b\[1;36m\? \x1b\[0m", coloured.getvalue())


def test_short_path_shows_the_home_folder_as_a_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert render.short_path(tmp_path / ".config" / "kilo" / "kilo.jsonc") == "~/.config/kilo/kilo.jsonc"
    assert render.short_path(tmp_path.parent / "elsewhere") == (tmp_path.parent / "elsewhere").as_posix()


def test_paths_label_uses_posix_separators(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    label = render.paths_label(
        [tmp_path / ".config" / "kilo" / "kilo.jsonc", tmp_path / ".config" / "kilo" / "settings.json"]
    )

    assert label == "~/.config/kilo/kilo.jsonc, settings.json"
