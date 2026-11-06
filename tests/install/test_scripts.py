from __future__ import annotations

import contextlib
import os
import re
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / "install"


def test_install_scripts_pin_the_uv_installer():
    sh = (SCRIPTS / "install.sh").read_text()
    ps1 = (SCRIPTS / "install.ps1").read_text()

    # Astral's version-pinned installers, the same uv release in both scripts.
    assert re.findall(r"https://astral\.sh/uv/[^\s'\"]*", sh) == ["https://astral.sh/uv/0.13.6/install.sh"]
    assert re.findall(r"https://astral\.sh/uv/[^\s'\"]*", ps1) == ["https://astral.sh/uv/0.13.6/install.ps1"]


def test_uvs_path_advice_doesnt_print_in_install_sh():
    sh = (SCRIPTS / "install.sh").read_text()

    # Quiet still adds uv's folder to the shell profiles, but hides uv's own errors, so jl names a failure.
    assert (
        "curl -LsSf https://astral.sh/uv/0.13.6/install.sh | sh -s -- --quiet"
        ' || { echo "Installing uv failed." >&2; return 1; }\n'
    ) in sh


def test_install_scripts_mark_their_setup_run():
    sh = (SCRIPTS / "install.sh").read_text()
    ps1 = (SCRIPTS / "install.ps1").read_text()

    # On jl setup's command line only, so neither jl connect nor the user's shell sees it.
    assert '\n    JL_FROM_INSTALLER=1 jl setup <"$t" || return\n    jl connect <"$t"\n' in sh
    # irm | iex runs in the user's own session, so the marker is removed again whatever jl setup does.
    assert (
        "\n    $env:JL_FROM_INSTALLER = '1'\n"
        "    try { jl setup } finally { Remove-Item Env:JL_FROM_INSTALLER -ErrorAction SilentlyContinue }\n"
    ) in ps1
    assert ps1.index("Remove-Item Env:JL_FROM_INSTALLER") < ps1.index("\n    jl connect\n")


def test_installer_ctrl_c_stops_quietly():
    ps1 = (SCRIPTS / "install.ps1").read_text()

    # jl exits 130 when Ctrl-C answers one of its questions: the user stopped it, so no error block.
    setup = ps1.index("    try { jl setup }")
    stop = ps1.index("    if ($LASTEXITCODE -eq 130) { return }\n", setup)
    assert stop < ps1.index("throw '`jl setup` failed.", setup)
    # And the user stopped it, so the window ends on jl's own words, not on the next steps.
    assert stop < ps1.rindex("Show-JlNext")


def test_install_ps1_says_what_to_run_next():
    ps1 = (SCRIPTS / "install.ps1").read_text()

    assert (
        "function Show-JlNext {\n"
        "    Write-Host 'jl is installed. Next:'\n"
        "    Write-Host '  jl status    see your coding tools'\n"
        "    Write-Host '  jl connect   connect another one'\n"
        "}\n"
    ) in ps1
    # On every path that installed jl, including the one that skips setup.
    assert "if ($env:JL_SKIP_SETUP) { Show-JlNext; return }" in ps1
    # Last, whatever jl connect printed.
    assert "\n    jl connect\n    Show-JlNext\n}" in ps1
    # irm | iex runs in the user's session, so the normal next step needs no new terminal.
    assert "Open a new terminal, then:" not in ps1


def run_in_a_terminal(argv: list[str], env: dict[str, str], ctrl_c_after: bytes | None = None) -> tuple[str, int]:
    """Run ``argv`` with a terminal of its own, as `curl | sh` does in a terminal window.

    Presses Ctrl-C once the output holds ``ctrl_c_after``: the terminal sends SIGINT to every
    program in the foreground, the script and the jl it runs alike. Returns what was printed, and
    the exit code.
    """
    import pty

    pid, terminal = pty.fork()
    if pid == 0:
        os.execve(argv[0], argv, env)
    output = b""
    with contextlib.suppress(OSError):  # the terminal closes when the script exits
        while chunk := os.read(terminal, 1024):
            output += chunk
            if ctrl_c_after and ctrl_c_after in output:
                os.write(terminal, b"\x03")
                ctrl_c_after = None
    _, status = os.waitpid(pid, 0)
    return output.decode(), os.waitstatus_to_exitcode(status)


def fake_install(
    tmp_path: Path, jl: str, update_shell: int = 0, unnormalized: bool = False
) -> tuple[list[str], dict[str, str]]:
    """Set up install.sh's run with uv installed and ``jl`` as jl's script; return its argv and env.

    uv's tool folder, where jl lands, isn't on the PATH the script starts with. ``update_shell`` is
    the exit code of `uv tool update-shell`. ``unnormalized`` makes `uv tool dir --bin` report the
    folder with ".." in it, as uv does on Linux.
    """
    fakes, jl_bin = tmp_path / "fakes", tmp_path / "jl-bin"
    fakes.mkdir()
    jl_bin.mkdir()
    shown = f"{jl_bin}/../{jl_bin.name}" if unnormalized else jl_bin
    (fakes / "uv").write_text(
        f'#!/bin/sh\n[ "$1 $2" = "tool dir" ] && echo {shown}\n'
        f'[ "$1 $2" = "tool update-shell" ] && exit {update_shell}\nexit 0\n'
    )
    (jl_bin / "jl").write_text("#!/bin/sh\n" + jl)
    for program in (fakes / "uv", jl_bin / "jl"):
        program.chmod(0o755)
    env = {"HOME": str(tmp_path), "PATH": f"{fakes}:/usr/bin:/bin", "JL_BIN": str(jl_bin)}
    return [str(SCRIPTS / "install.sh")], env


# The user's /bin/sh: dash on Ubuntu, bash on macOS.
SHELLS = [shell for shell in ("/bin/dash", "/bin/bash") if Path(shell).exists()]
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="install.sh is for macOS and Linux")

# What the installer ends on, so a new user knows what jl can do next.
NEXT_STEPS = ["  jl status    see your coding tools", "  jl connect   connect another one"]


@posix_only
@pytest.mark.parametrize("on_path", [False, True], ids=["not-on-path", "on-path"])
def test_the_installer_says_to_open_a_new_terminal(tmp_path, on_path):
    script, env = fake_install(tmp_path, 'echo "jl $1 ran"\n')
    if on_path:
        env["PATH"] = env["JL_BIN"] + ":" + env["PATH"]

    output, code = run_in_a_terminal(["/bin/sh", *script], env)

    assert code == 0
    assert "jl connect ran" in output
    # Last: what the user types next is jl, in this window or in a new one.
    first = "jl is installed. Next:" if on_path else "jl is installed. Open a new terminal, then:"
    assert output.splitlines()[-3:] == [first, *NEXT_STEPS]
    # The jl this window runs is the one just installed.
    assert "Another jl at" not in output


@posix_only
def test_a_uv_bin_path_with_dots_does_not_look_like_another_jl(tmp_path):
    # uv on Linux reports the bin folder as ~/.local/share/../bin. The user already has ~/.local/bin
    # on PATH, so a string compare would call the jl just installed someone else's.
    script, env = fake_install(tmp_path, 'echo "jl $1 ran"\n', unnormalized=True)
    env["PATH"] = env["JL_BIN"] + ":" + env["PATH"]

    output, code = run_in_a_terminal(["/bin/sh", *script], env)

    assert code == 0
    assert "Another jl at" not in output
    assert output.splitlines()[-3] == "jl is installed. Next:"


@posix_only
def test_the_installer_warns_about_another_jl_earlier_on_the_path(tmp_path):
    script, env = fake_install(tmp_path, 'echo "jl $1 ran"\n')
    earlier = tmp_path / "earlier-bin"
    earlier.mkdir()
    (earlier / "jl").write_text("#!/bin/sh\necho old jl\n")
    (earlier / "jl").chmod(0o755)
    env["PATH"] = f"{earlier}:{env['PATH']}"

    output, code = run_in_a_terminal(["/bin/sh", *script], env)

    assert code == 0
    assert f"Another jl at {earlier / 'jl'} comes first on your PATH: remove it, or open a new terminal." in output
    assert "jl connect ran" in output


def test_install_ps1_warns_about_another_jl_earlier_on_the_path():
    ps1 = (SCRIPTS / "install.ps1").read_text()

    # Against the PATH the window started with, before jl's folder goes in front of it.
    assert "$startPath = $env:Path" in ps1
    assert "(Resolve-Path (uv tool dir --bin)).Path" in ps1
    assert "$env:Path = $startPath\n    $earlier = Get-Command jl -ErrorAction SilentlyContinue" in ps1
    assert (
        'Write-Host "Another jl at $($earlier.Source) comes first on your PATH: remove it, or open a new terminal."'
    ) in ps1
    # Before jl runs, so the warning is the first thing the user sees about it.
    assert ps1.index("$earlier = Get-Command jl") < ps1.index("$env:JL_SKIP_SETUP")


@posix_only
def test_the_installer_says_what_to_run_next_when_setup_is_skipped(tmp_path):
    script, env = fake_install(tmp_path, 'echo "jl $1 ran"\n')
    env["JL_SKIP_SETUP"] = "1"

    output, code = run_in_a_terminal(["/bin/sh", *script], env)

    assert code == 0
    assert "jl setup ran" not in output
    assert output.splitlines()[-3:] == ["jl is installed. Open a new terminal, then:", *NEXT_STEPS]


@posix_only
def test_the_installer_says_to_add_jls_folder_when_uv_cant(tmp_path):
    script, env = fake_install(tmp_path, 'echo "jl $1 ran"\n', update_shell=1)

    output, code = run_in_a_terminal(["/bin/sh", *script], env)

    assert code == 0
    assert output.splitlines()[-3:] == [
        f"Add {env['JL_BIN']} to your PATH to run jl from a new terminal.",
        *NEXT_STEPS,
    ]
    assert "Open a new terminal" not in output


@posix_only
@pytest.mark.parametrize("shell", SHELLS)
def test_jl_reads_the_terminals_device_not_dev_tty(tmp_path, shell):
    # macOS's kqueue can't watch /dev/tty, so prompt_toolkit in jl connect aborts there.
    jl = 'echo "jl $1 on $(tty)"\n'
    script, env = fake_install(tmp_path, jl)

    output, code = run_in_a_terminal([shell, *script], env)

    assert code == 0
    assert re.search(r"jl setup on /dev/pts/\d+", output)
    assert re.search(r"jl connect on /dev/pts/\d+", output)


@posix_only
@pytest.mark.parametrize("shell", SHELLS)
def test_ctrl_c_in_a_tool_jl_started_doesnt_stop_the_installer(tmp_path, shell):
    # jl setup stands in for a tool jl started that keeps going on Ctrl-C, as Aider does.
    jl = 'if [ "$1" = setup ]; then trap "echo tool got Ctrl-C" INT; echo ready; sleep 1; fi\necho "jl $1 ran"\n'
    script, env = fake_install(tmp_path, jl)

    output, code = run_in_a_terminal([shell, *script], env, ctrl_c_after=b"ready")

    assert "tool got Ctrl-C" in output
    assert "jl connect ran" in output
    assert code == 0


@posix_only
@pytest.mark.parametrize("shell", SHELLS)
def test_ctrl_c_at_a_jl_question_stops_the_installer(tmp_path, shell):
    # jl exits 130 when Ctrl-C answers one of its questions.
    script, env = fake_install(tmp_path, 'echo "jl $1 ran"\n[ "$1" = setup ] && exit 130\nexit 0\n')

    output, code = run_in_a_terminal([shell, *script], env)

    assert "jl setup ran" in output
    assert "jl connect ran" not in output
    assert code == 130
