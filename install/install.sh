#!/bin/sh
# Installs jl on macOS and Linux: curl -fsSL https://jarvislabs.ai/install.sh | sh
#
# Installs uv if missing, installs jl with `uv tool install`, puts it on PATH,
# then runs `jl setup` and `jl connect`.
#
# Environment:
#   JL_INSTALL_SPEC  what to install (default: jarvislabs); CI passes a wheel path.
#   JL_SKIP_SETUP    set to skip `jl setup` and `jl connect`.

set -eu

# Everything runs from main, called on the last line, so a download cut short runs nothing.
main() {
    spec="${JL_INSTALL_SPEC:-jarvislabs}"
    # The PATH this window started with, before the script adds uv's and jl's folders to it.
    start_path=":$PATH:"

    if ! command -v uv >/dev/null 2>&1; then
        echo "Installing uv..."
        # A uv release jl tested, from Astral's version-pinned installer; it moves with jl releases.
        # Quiet: uv's own advice to add its folder to PATH is the script's job; quiet also hides uv's errors.
        curl -LsSf https://astral.sh/uv/0.13.6/install.sh | sh -s -- --quiet || { echo "Installing uv failed." >&2; return 1; }
        # Where uv's installer puts uv; its rarer $XDG_DATA_HOME/../bin fallback isn't covered.
        PATH="${UV_INSTALL_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}:$PATH"
    fi

    echo "Installing jl..."
    # uv's package list and progress are noise before the product appears. Keep them for a failure,
    # and for JL_VERBOSE=1; otherwise the install is one line and one check.
    log="$(mktemp)"
    if ! uv tool install --upgrade "$spec" >"$log" 2>&1; then
        cat "$log" >&2
        rm -f "$log"
        echo "Installing jl failed." >&2
        return 1
    fi
    if [ -n "${JL_VERBOSE:-}" ]; then cat "$log"; fi
    rm -f "$log"
    echo "✓ jl installed"
    bin="$(uv tool dir --bin)"
    # uv can report the bin folder with ".." in it (e.g. ~/.local/share/../bin); resolve it, or the
    # comparisons below miss that this is the jl the window already runs.
    bin="$(cd "$bin" && pwd)"
    if uv tool update-shell >/dev/null 2>&1; then
        next_step="jl is installed. Open a new terminal, then:"
    else
        # Say it now: when setup and connect run, this is the only PATH message left before them.
        echo "Add $bin to your PATH to run jl from a new terminal."
        next_step=""
    fi
    PATH="$bin:$PATH"
    # An older jl earlier on the PATH would run in a new terminal instead of the one just installed.
    start="${start_path#:}"
    earlier="$(PATH="${start%:}" command -v jl 2>/dev/null || true)"
    if [ -n "$earlier" ] && [ "$earlier" != "$bin/jl" ]; then
        echo "Another jl at $earlier comes first on your PATH: remove it, or open a new terminal."
    fi

    status=0
    setup_done=0
    setup_and_connect || status=$?
    # When setup and connect ran, the product's own Next: is the ending. The closing lines are for a
    # skipped or failed setup, so the user still knows how to come back.
    if [ "$setup_done" -eq 0 ]; then
        case "$start_path" in
            # Nothing to say about PATH: typing jl in this window already finds it.
            *":$bin:"*) next_step="jl is installed. Next:" ;;
        esac
        if [ -n "$next_step" ]; then
            echo "$next_step"
        fi
        echo "  jl status    see your coding tools"
        echo "  jl connect   connect another one"
    fi
    return "$status"
}

# Called in an || list, where set -e is off, so each failure returns by hand.
setup_and_connect() {
    if [ -n "${JL_SKIP_SETUP:-}" ]; then
        return
    fi
    # Under `curl | sh` stdin is the script, so jl's prompts read the terminal instead.
    if ! (: </dev/tty) 2>/dev/null; then
        echo "No terminal to prompt in. Run \`jl setup\`, then \`jl connect\`."
        return
    fi
    # Ctrl-C in a tool jl starts is the tool's, but the terminal sends it to this shell too, and dash
    # would exit on it. A handler, not '' (ignore), which jl would inherit and then never see Ctrl-C.
    # jl's own exit code still stops the script.
    trap : INT
    # The terminal's own device rather than /dev/tty: macOS's kqueue can't watch /dev/tty, so
    # prompt_toolkit in jl connect aborts there. ps prints "?" or "??" with no controlling terminal.
    t="/dev/$(ps -o tty= -p $$ | tr -d ' ')"
    [ -r "$t" ] || t=/dev/tty
    # Marks jl setup's run so it hands off to jl connect instead of printing its guide.
    JL_FROM_INSTALLER=1 jl setup <"$t" || return
    jl connect <"$t" || return
    setup_done=1
}

main "$@"
