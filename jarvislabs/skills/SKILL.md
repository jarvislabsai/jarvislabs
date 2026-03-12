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

No upload. Runs the raw command. Good when code already exists on the instance.

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

The main monitoring primitive — gives both logs AND completion status in one call:

```bash
jl run logs <run_id> --tail 50 --json
```

Response fields:
- `run_id`, `machine_id` — identifiers
- `content` — recent log lines (loss values, errors, progress)
- `run_exit_code` — `null` = still running, `0` = succeeded, non-zero = failed

Note: `--json` is not supported with `--follow`. Without `--tail`, the entire log file is returned (can be very large — always use `--tail N`).

Without `--json`, the output shows raw logs with a header/footer indicating run state:

```
--- run r_abc | machine 123 | running ---
step=100 loss=2.31
step=200 loss=2.11
--- still running | log: <home>/jl-runs/r_abc/output.log ---
```

### The Agent Loop

1. Start detached: `jl run ... --no-follow --json --yes` — extract `run_id` from the JSON response
2. Wait and check: `sleep 120 && jl run logs <run_id> --tail 50 --json`
3. Inspect `content` for loss/errors/progress
4. If `run_exit_code` is `null` → still running, repeat step 2
5. If `run_exit_code` is `0` → succeeded, download results
6. If `run_exit_code` is non-zero → failed, read error, fix, start a new run

Cadence: 60-120s (short experiments), 180-300s (long training), 300-600s (very long runs).

## Quick System Checks

```bash
jl instance exec <id> --json -- nvidia-smi
jl instance exec <id> --json -- ps -ef
jl instance exec <id> --json -- df -h
```

For pipes or shell syntax, wrap in `sh -lc`:

```bash
jl instance exec <id> --json -- sh -lc 'grep "loss" /path/to/log | tail -5'
```

## Stopping a Run

```bash
jl run stop <run_id> --json
```

Kills the entire process group (training script + all child processes).

## File Persistence

The remote home directory (typically `/home/` on containers, `/home/<user>/` on VMs) persists across pause/resume cycles. Everything else (system packages, `/tmp`, `/root`, etc.) is ephemeral and lost on pause.

- Save all outputs, checkpoints, and results under the home directory
- Uploaded directories go to `<home>/<directory_name>/`, uploaded files go to `<home>/<file_stem>/<file>`
- Run metadata (logs, pid, exit code) is stored separately under `<home>/jl-runs/<run_id>/`
- `.venv` created inside the project directory persists across restarts
- System-level installs (`apt-get`, global pip) do NOT persist — use `--setup` or `--requirements` to reinstall on each run

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
| `jl instance resume <id> --yes --json` | Resume (can change GPU with `--gpu`, expand storage with `--storage`) |
| `jl instance destroy <id> --yes --json` | Permanently delete |
| `jl instance rename <id> --name "new-name" --yes` | Rename |

### Instance Operations

| Command | Description |
|---|---|
| `jl instance exec <id> -- <command>` | Run command (use `--` separator) |
| `jl instance exec <id> --json -- <command>` | Exec with structured output |
| `jl instance ssh <id>` | Interactive SSH |
| `jl instance ssh <id> --print-command` | Print SSH command |
| `jl instance upload <id> ./local /remote` | Upload file or directory |
| `jl instance download <id> /remote ./local` | Download (`-r` for directories) |

### Managed Runs

| Command | Description |
|---|---|
| `jl run <target> --on <id> --no-follow --json --yes` | Start detached run |
| `jl run <target> --gpu <gpu> --no-follow --keep --json --yes` | Fresh instance + detached run |
| `jl run logs <run_id> --tail 50 --json` | Recent logs + exit code |
| `jl run logs <run_id> --tail 50` | Recent logs with run state header/footer |
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

# 3. Monitor periodically (repeat until run_exit_code is not null)
sleep 180 && jl run logs <run_id> --tail 50 --json

# 4. Download results
jl instance download <machine_id> /home/results ./results -r

# 5. Cleanup — respect user preference: pause (default), destroy, or keep running
jl instance pause <machine_id> --yes --json
```
