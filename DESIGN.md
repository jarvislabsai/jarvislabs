# Output design

The rules behind `jl`'s human output: help, setup, connect, status, logout, colour and spacing.
Written 2026-11-06, after the coding-plan surface was built. This is the file to check against when
adding a command or changing a screen.

## The layers

| Layer | Job |
|---|---|
| Top-level `--help` | Map: one-line definition, common workflows, grouped commands. Not a tutorial. |
| `jl <command> --help` | Contract: every flag, default, example and caveat needed to type it. |
| `jl help <command>` | Shortcut to the same subcommand help. |
| Docs | Tutorials, concepts, long workflows, troubleshooting, policy. |
| Errors | The immediate fix for the command that just failed. |

Rule: **anything needed to type the command belongs in `--help`; anything needed to decide how to use
the platform belongs in docs.** Do not duplicate a tutorial in both; one will go stale.

## Rules

### Help

1. Top-level help is a map. One noun-phrase definition, then the command panels. No feature tour.
2. Four common workflows at most, as sequences: `GPU`, `Stop`, `Models`, `Serve`. They add intent and
   order; the panels add the index.
3. One line for coding agents: `Using a coding agent? jl setup installs the jl skill.`
4. Every command has `--help` and `-h`, including sub-Typers. `jl help <command>` works.
5. Subcommand help carries the concepts and examples: container vs VM in `jl create`, examples in
   `jl create` and `jl deploy`, the run lifecycle in `jl run`.
6. Help body is default foreground. Dim is not a help style. Panel borders are grey, not dim.
7. Headings are bold default; commands are bold cyan. Do not colour a heading cyan.

### Setup

8. Wordmark on any interactive setup with ≥80 columns, unless `--yes`. No version line under it.
9. Phase rules: `SIGN IN`, `AGENT SKILLS`, `GET STARTED`. Uppercase, bold default, over a grey line.
10. The body sits directly under its heading — no blank line between them. A blank line goes before
    the next block.
11. Skills: one line of context, one yes/no. A single extra agent is installed when detected; no
    one-row picker. The picker returns when more agents are added.
12. One success row for the skill install, naming every agent it covers, with the written paths on a
    dim line. Do not repeat the sentence per path.
13. Account panel, then `Next: jl --help` with `see all commands` under it. No subscription CTA from
    standalone setup; `jl connect` is the installer's job.

### Connect

14. Every `jl connect` opens with `CONNECT TOOLS` and one concrete line (what is written, not a
    tautology): pay, pick, preview and write are a multi-step flow whether the installer runs it or
    the user types it after setup. `--print` and `--rotate-key` have no phase and skip the heading;
    per-step headings inside connect would be ceremony.
15. Picker lists installed tools only. `--all` reveals the catalog. One installed tool means no
    picker; connect it and point at `--all`.
16. Preview with the `+ add` / `~ update` legend, then one read gate: `Write these changes?`.
17. Result rows name each tool and say `added` or `updated`. No anonymous counts. Bookkeeping
    that happens while writing (a lost key revoked so a new one can be made) is a dim line, so
    the rows stay the loudest thing after the gate.
18. A no-change run names the tools: `OpenCode and pi are already connected`. Never "up to date" for
    connection state; that means software versions.
19. One `API test passed` line, an `Add another tool: jl connect` hint, then one `Next:`.
20. Wallet billing is a yellow warning: `On wallet (no subscription)` plus the consequence
    (`Requests are charged to your wallet`) and the plan command. No subscription must not be quiet.

### Status

21. Account, then Coding tools. Plan usage is its own command, `jl usage`; status never repeats it.
22. Hide tools that are not installed; `--all` shows them, with a hint naming the count.
23. Never print an empty panel. If nothing is shown, say `No coding tools installed.`
24. A connected tool that has been uninstalled is a warning row with `jl disconnect <id>`.
25. An update or an out-of-date agent skill is a warning. A missing skill is a hint
    (`jl setup` to add one): it is an option not taken, not attention needed, and must not
    read as a failure right after leaving setup.
26. End on `Next:` with a command; no summary paragraph after it.

### Logout

27. `jl logout` is a full cleanup: remove jl's provider entries, revoke the machine key, remove the
    account key. One confirmation; `--yes` for scripts.
28. `--keep-tools` is the explicit account-only logout. It warns that the machine key remains.

### Next

29. The command comes first, the explanation on a dim line underneath:
    ```text
    Next: jl --help
          see all commands
    ```
30. Exactly one blank line before `Next:`; it is the last block on screen.

## Colour

The palette is semantic. One accent, four state colours, default body, grey chrome.

| Role | Colour | Applied to |
|---|---|---|
| Brand / action | cyan | commands, links, surface titles (Account, Coding tools), `Next:` command |
| Success | green | the `✓` mark only. Not the message; success is frequent and can be long |
| Warning | yellow | the `!` and the short message; wallet, invalid key, changes |
| Failure | red | the `✗` and the short message |
| Content | default foreground | descriptions, explanations, status text |
| Chrome | grey (`bright_black`) | panel borders, rule lines |
| Tertiary | dim | IDs, paths, timestamps, the explanation under `Next:` |

Rules:

31. Green on success marks, not paragraphs. Colour the smallest unit that carries the state.
32. Warnings and errors may tint their short message; they are rare and need attention.
33. Headings are never cyan; cyan means act here. If headings and commands share cyan, neither reads.
34. No decorative hues. Orange is warning territory. Magenta/green panel borders were removed.
35. Named ANSI colours only; no `rgb(...)` or hex. The user's theme owns light/dark.
36. Use bold and spacing before saturation. If it only works because it is bright, it does not work.
37. Marks and words carry state together; colour is never the only signal. `NO_COLOR` keeps both marks
    and plain text.
38. No emoji as UI chrome. The brand mark is the wordmark and the cyan `.ai`, not `⚡`.
39. Links are accent, underlined and clickable where the terminal supports OSC 8. The underline is
    the signal in error messages and anywhere the terminal can't make the link clickable.

## Space

40. Blank line **before** a rule, none after; the body is tight to its heading.
41. One blank line between blocks, panels and outcomes. Never two.
42. One screen for a successful setup or a two-tool connect. If it needs more, the detail belongs
    behind `--all`, `--wide`, `get` or `--help`.
43. `Next:` is separated from the block above; nothing follows it.

## Voice

44. Say what happens, not what the command is called: “Sets up the Jarvislabs provider in your coding
    tools”, not “Connect the tools”.
45. Name the tools/agents. “9 agents” and “2 tools” make the user ask which.
46. One idea per block; one action per hint. Hints are at most two lines.
47. Multi-step onboarding (`setup`, `connect`) ends with what changed and one next action. An overview
    (`status`) may suggest the next inspect/act command. A one-shot cleanup (`disconnect`, `logout`) ends
    on its result; it does not need a Next block.
48. A phase heading is for a multi-step flow. Do not put one on a one-shot command; its preview or
    result is the heading.
49. No artificial delay, typewriter effect, or press-Enter gate. A question is the only read gate.

## Automated pass

Existing coverage:

- `tests/cli/test_connect.py` — picker, preview, added/updated rows, next step, line budget.
- `tests/cli/test_setup.py`, `test_setup_header.py` — wordmark, phases, skills, guide gone.
- `tests/cli/test_render.py` — next step, success mark-only colour, NO_COLOR, ASCII marks.
- `tests/cli/test_app.py` — help wrapping, new commands, `-h`/help surface.
- `tests/cli/test_disconnect.py` — logout cleanup and `--keep-tools`.

Checks still to add (the “automated pass”):

1. Static lint: banned output tokens — `rgb(`, `⚡`, `"bold accent"`, `time.sleep`, `[dim]` outside the
   allowlist (borders, IDs, paths, legends, unavailable rows, `Next:` hint).
2. Static lint: no `dim` in `typer.rich_utils.STYLE_HELPTEXT` or panel borders.
3. Help tests: top-level definition + workflows; `-h` on root and every sub-Typer; `jl help <cmd>`;
   examples in `create`/`deploy`/`run`.
4. pty snapshots for setup, connect, status and logout with a fake transport: exact final block, no
   SGR 2 on body, no empty panels, line budget.
5. A light/dark theme render check, at 80 and 100 columns, for the wordmark and panel titles.
6. A check that every state-changing command’s help has at least one example.

Until that exists, review new output by hand against this file; the spirit is subtraction first,
then weight, then space, then colour.
