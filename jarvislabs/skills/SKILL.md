---
name: jarvislabs
description: Agent guide for running and monitoring GPU experiments with the jl CLI on JarvisLabs.ai.
---

# JarvisLabs GPU CLI (`jl`) — Agent Guide

## Getting Started

Verify auth with `jl status --json` before doing anything. If not logged in, use `jl setup --token <token> --yes`.

Before starting a run, check the project for existing setup files: use `--requirements` for `requirements.txt`, `--setup-file` for `setup.sh`, or `--setup "uv sync"` for `pyproject.toml`-only projects.

Use `--help` on any command to discover flags (e.g., `jl run --help`, `jl run logs --help`). If something goes wrong, use `jl run logs`, `jl run status`, and `jl instance exec` to diagnose — don't guess.

## Mental Model

- `jl instance` = machine lifecycle and low-level access
- `jl run` = managed job with log file + pid tracking (use `--no-follow` for detached/agent workflows)
- `jl instance exec` = quick commands and debugging (nvidia-smi, ps, grep, df)

## Important: `run_id` Is Locally Tracked

`jl run logs`, `jl run status`, `jl run stop`, and `jl run list` depend on local records under `~/.jl/runs/`. Start and monitor runs from the same machine. If the local record is missing, the `run_id` alone is not enough.

## Run Modes

### File Mode

```bash
jl run train.py --on <machine_id> --no-follow --json --yes
```

Uploads the file, creates working directory, runs `python3 <file>` (or `bash` for `.sh`).

### Directory Mode (best for real projects)

```bash
jl run . --script train.py --on <machine_id> --no-follow --json --yes
```

Rsyncs the whole directory, runs inside it. `--script` is only valid with a directory target. Requires `rsync` installed locally.

### Command Mode

```bash
jl run --on <machine_id> --no-follow --json --yes -- python train.py --epochs 5
```

No upload. Runs the raw command. Good when code already exists on the instance. Note: `--requirements` and `--setup-file` are not available in command mode — only `--setup` works.

## Regions & GPU Constraints

Valid region codes: `IN1`, `IN2`, `EU1`. If `--region` is omitted, the CLI auto-selects based on availability.

Code-backed constraints:
- `EU1` supports only `H100` and `H200`
- `EU1` supports only 1 or 8 GPUs per instance
- `EU1` requires at least 100 GB storage (auto-bumped if lower)
- `vm` is only available in `IN2` and `EU1`
- `vm` requires at least one SSH key registered
- `vm` uses a 100 GB minimum storage floor

Run `jl gpus --json` to check current availability and pricing.

## Ports & Exposing Services

Container instances expose these default HTTP ports (each gets its own HTTPS URL):

| Port | Service |
|---|---|
| 8889 | JupyterLab |
| 7007 | IDE |
| 6006 | Available on generic templates (pytorch, etc.) — some specialized templates use this port |

VM instances (`--template vm`) get SSH-only access — no HTTP ports or web URLs. Use `ssh_command` from `jl instance get <id> --json`.

For SSH on any instance type, always use the `ssh_command` field from instance details — the port varies by region and architecture.

### When to use default ports

If the user wants to expose a FastAPI app, Gradio UI, Streamlit dashboard, or any HTTP service — **first check if port 6006 is free** (run `jl instance exec <id> -- ss -tlnp` or try binding). On generic templates like pytorch, 6006 is typically unused. Bind the service to `0.0.0.0:6006` and it's accessible via the `endpoints[0]` URL.

```bash
# Service on default port 6006 — no --http-ports needed
jl instance exec <id> -- sh -lc 'uvicorn main:app --host 0.0.0.0 --port 6006 &'
```

### When to use custom ports (`--http-ports`)

Use `--http-ports` when:
- Port 6006 is already in use by the template
- Multiple services need to be exposed simultaneously
- The service requires a specific port number

```bash
# Expose two custom ports at creation
jl instance create --gpu RTX5000 --http-ports "7860,8080" --yes --json

# Or with jl run (fresh instance mode)
jl run . --script train.py --gpu RTX5000 --http-ports "7860,8080" --no-follow --keep --json --yes
```

Rules:
- Comma-separated port numbers (e.g. `"7860,8080"`)
- Maximum 10 custom ports
- Port range: 0–65535
- Reserved ports `8889`, `6006`, `7007`, `22` cannot be specified (they're always available)
- Custom ports can be set at creation or changed on resume with `--http-ports`

### Finding service URLs

Run `jl instance get <id> --json`. Port URLs are in `endpoints` (port 6006 first, then custom ports in order) and `vs_url` (port 7007). Bind your service to `0.0.0.0:<port>` inside the instance.

## Fresh Instance Inline

Skip `jl instance create` when you know the GPU:

```bash
jl run . --script train.py --gpu RTX5000 --no-follow --keep --json --yes
```

Creates instance, uploads, starts detached. `--keep` is required with `--no-follow` — the CLI will error without it. Additional flags: `--template` (default: pytorch), `--storage` (default: 40GB), `--num-gpus` (default: 1), `--region`, `--http-ports`.

Use separate `jl instance create` when you need to inspect GPU availability, reuse machines across runs, or attach filesystems/scripts beforehand.

## Setup Chain

For file and directory modes, these run before your main command (chained with `&&`):

1. `uv` installed if missing → `.venv` created if missing → `.venv` activated
2. `uv pip install -r requirements.txt` (if `--requirements` provided)
3. `bash setup.sh` (if `--setup-file` provided)
4. Shell command (if `--setup` provided)
5. Main script runs

```bash
jl run . --script train.py --on <id> --requirements requirements.txt --setup "pip install flash-attn" --no-follow --json --yes
```

## The Monitoring Pattern

The main monitoring primitive — gives both recent logs AND completion status in one call:

```bash
jl run logs <run_id> --tail 50
```

Always use `--tail N` — without it, the entire log file is returned and can be enormous.

The output has a **header** and **footer** that tell you the run state:

**While running:**
```
--- run r_abc | machine 123 | running ---

step=100 loss=2.31
step=200 loss=2.11

--- still running | log: /home/jl-runs/r_abc/output.log ---
```

**When succeeded:**
```
--- run r_abc | machine 123 | succeeded (exit 0) ---

step=900 loss=0.12
step=1000 loss=0.11

--- succeeded | exit code: 0 | log: /home/jl-runs/r_abc/output.log ---
```

**When failed:**
```
--- run r_abc | machine 123 | failed (exit 1) ---

Traceback (most recent call last):
  File "train.py", line 5, in <module>
ModuleNotFoundError: No module named 'transformers'

--- failed | exit code: 1 | log: /home/jl-runs/r_abc/output.log ---
```

If the instance is paused, missing, or SSH is unavailable, `jl run logs` fails before printing a log header. Use `jl run status <run_id>` or `jl run list --refresh` to inspect those states.

### The Agent Loop

1. **Start detached:** `jl run ... --no-follow --json --yes` — extract `run_id` and `machine_id` from the JSON response
2. **Early check (catch fast failures):** `sleep 15 && jl run logs <run_id> --tail 30` — look at the header and footer. If it says `failed (exit N)`, the script crashed immediately (import error, syntax error, pip failure). Read the error in the log body, fix, and retry without waiting further.
3. **Steady-state polling:** `sleep 120 && jl run logs <run_id> --tail 50`
4. Read the log body for loss values, errors, or progress
5. Check the footer:
   - `still running` → repeat step 3
   - `succeeded | exit code: 0` → done, download results
   - `failed | exit code: N` → read the error, fix, start a new run

Cadence: 60-120s (short experiments), 180-300s (long training), 300-600s (very long runs).

## Quick System Checks

```bash
jl instance exec <id> -- nvidia-smi
jl instance exec <id> -- ps -ef
jl instance exec <id> -- df -h
```

Output streams directly to terminal. The exit code of the remote command is propagated.

For pipes or shell syntax, wrap in `sh -lc`:

```bash
jl instance exec <id> -- sh -lc 'grep "loss" /path/to/log | tail -5'
```

## Stopping a Run

```bash
jl run stop <run_id> --json
```

Kills the entire process group (training script + all child processes).

## File Persistence

The remote home directory (typically `/home/` on containers, `/home/<user>/` on VMs) persists across pause/resume cycles. Everything else (system packages, `/tmp`, `/root`, etc.) is ephemeral and lost on pause.

- When working directly on the instance, always place files, directories, virtual environments, and any other artifacts under `/home/`. `/root/` and other system paths are lost on pause — only `/home/` persists across pause/resume cycles.
- Save all outputs, checkpoints, and results under the home directory
- Uploaded directories go to `<home>/<directory_name>/`, uploaded files go to `<home>/<file_stem>/<file>`
- Run metadata (logs, pid, exit code) is stored separately under `<home>/jl-runs/<run_id>/`
- `.venv` created inside the project directory persists across restarts
- System-level installs (`apt-get`, global pip) do NOT persist — use `--setup` or `--requirements` to reinstall on each run

## Error Handling

When `--json` is active, many CLI validation and API failures are emitted as `{"error": "..."}` to stdout.

But do not assume every non-zero JSON command returns that shape. Some commands, such as `jl instance exec --json`, still return their normal structured payload with a non-zero `exit_code`.

Agent rule:
- First inspect the JSON shape
- If it has an `error` key, treat it as a CLI failure
- Otherwise inspect command-specific fields such as `exit_code`, `state`, or `run_exit_code`

## Anti-Patterns

- Do not use `jl run logs --follow` — blocks forever, will timeout. `--json` is also incompatible with `--follow`.
- Do not omit `--no-follow` when starting runs — default is `--follow` which blocks
- Do not read full logs — always use `--tail N`
- Do not poll every few seconds — wasteful and noisy
- Do not use lifecycle flags (`--keep`, `--pause`, `--destroy`) with `--on <machine_id>` — they only apply to fresh instances
- Do not forget to pause/destroy instances after experiments — they cost money

## Command Reference

### Account

| Command | Description |
|---|---|
| `jl setup --token <token> --yes` | Authenticate + install agent skills |
| `jl logout` | Remove saved token |
| `jl status --json` | Balance, running/paused counts |
| `jl gpus --json` | GPU availability and pricing |
| `jl templates --json` | Framework templates |

### Instance Lifecycle

| Command | Description |
|---|---|
| `jl instance create --gpu H100 --storage 100 --template pytorch --yes --json` | Create instance |
| `jl instance list --json` | List all instances |
| `jl instance get <id> --json` | Instance details (SSH, URLs, status) |
| `jl instance pause <id> --yes --json` | Pause (stops billing, keeps storage) |
| `jl instance resume <id> --yes --json` | Resume (can change GPU with `--gpu`, expand storage with `--storage`, change ports with `--http-ports`) |
| `jl instance destroy <id> --yes --json` | Permanently delete |
| `jl instance rename <id> --name "new-name" --yes --json` | Rename |

### Instance Operations

| Command | Description |
|---|---|
| `jl instance exec <id> -- <command>` | Run command (use `--` separator), output streams directly |
| `jl instance ssh <id>` | Interactive SSH |
| `jl instance ssh <id> --print-command` | Print SSH command |
| `jl instance upload <id> ./local /remote` | Upload file or directory |
| `jl instance download <id> /remote ./local` | Download (`-r` for directories) |

### Managed Runs

| Command | Description |
|---|---|
| `jl run <target> --on <id> --no-follow --json --yes` | Start detached run |
| `jl run <target> --gpu <gpu> --no-follow --keep --json --yes` | Fresh instance + detached run |
| `jl run logs <run_id> --tail 50` | Recent logs with run state in header/footer |
| `jl run status <run_id> --json` | Run state only |
| `jl run stop <run_id> --json` | Kill the run (process group) |
| `jl run list --refresh --json` | All runs with live status (without `--refresh`, state shows as `saved`) |
| `jl run list --limit 5 --json` | Most recent N runs |
| `jl run list --status running --json` | Filter by state (auto-refreshes) |
| `jl run list --machine <id> --json` | Filter by instance |

### Infrastructure

| Command | Description |
|---|---|
| `jl ssh-key list/add/remove` | Manage SSH keys |
| `jl scripts list/add/update/remove` | Manage startup scripts |
| `jl filesystem list/create/edit/remove` | Manage persistent filesystems |

## Typical Agent Workflow

```bash
# 1. Check GPUs and create instance
jl gpus --json
jl instance create --gpu RTX5000 --storage 50 --yes --json

# 2. Start detached run
jl run . --script train.py --on <machine_id> --no-follow --json --yes

# 3. Early check (catch import/syntax/pip failures fast)
sleep 15 && jl run logs <run_id> --tail 30

# 4. Steady-state monitoring (repeat until footer shows succeeded or failed)
sleep 120 && jl run logs <run_id> --tail 50

# 5. Download results (use /home/<user>/ for VMs instead of /home/)
jl instance download <machine_id> /home/results ./results -r

# 6. Cleanup — respect user preference: pause (default), destroy, or keep running
jl instance pause <machine_id> --yes --json
```
