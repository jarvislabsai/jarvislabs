---
name: jarvislabs
description: Agent guide for running and monitoring GPU experiments with the jl CLI on JarvisLabs.ai.
---

# JarvisLabs GPU CLI (`jl`) — Agent Guide

## Getting Started

Verify auth with `jl status --json` before doing anything. If not logged in, use `jl setup --token <token> --yes`. You can also authenticate via `export JL_API_KEY="..."`.

Use `--help` on any command to discover flags (e.g., `jl run --help`, `jl instance create --help`). If something goes wrong, use `jl run logs`, `jl run status`, and `jl instance exec` to diagnose — don't guess.

## Mental Model

- `jl instance` = GPU machine. Create, pause, resume, destroy, SSH into, upload/download files.
- `jl run` = managed job on an instance. Uploads code, sets up a Python environment, runs your script in the background with log tracking.
- `jl instance exec` = run a quick command on an instance (nvidia-smi, ps, df) and low-level instance access. No environment setup, no tracking.

## Instances

### Creating

```bash
jl instance create --gpu L4 --storage 40 --yes --json
```

`--gpu` is required. Run `jl instance create --help` for all available flags.

### Lifecycle

Instances have three states that matter: **Running** (billing active), **Paused** (compute billing stopped, storage billing continues, data persists), **Destroyed** (everything deleted).

```bash
jl instance pause <id> --yes --json      # stop compute billing, keep data
jl instance resume <id> --yes --json     # restart a paused instance
jl instance destroy <id> --yes --json    # permanently delete
```

Resume caveats:
- Resume is **region-locked** — an instance always resumes in its original region. GPU swaps are only possible within that region.
- Resume may return a **new machine_id**. Always use the returned ID for subsequent operations.
- Run `jl instance resume --help` for all available flags (GPU swap, storage expansion, rename, etc.).

SSH, exec, upload, and download only work on **Running** instances.

### Regions & GPUs

Valid region codes: `IN1`, `IN2`, `EU1`. If `--region` is omitted, the CLI picks a region based on GPU availability.

| Constraint | Detail |
|---|---|
| EU1 | H100 and H200 only, 1 or 8 GPUs, 100 GB minimum storage (auto-bumped) |
| VM template | IN2 and EU1 only, requires at least one SSH key, 100 GB minimum storage |

Run `jl gpus --json` to check current availability and pricing.

### Ports & Services

Container instances expose default HTTP ports (each gets its own HTTPS URL):

| Port | Service |
|---|---|
| 8889 | JupyterLab (`url` field) |
| 7007 | IDE (`vs_url` field) |
| 6006 | Available on generic templates like pytorch (`endpoints[0]`) |

VM instances (`--template vm`) get SSH-only access. Use `ssh_command` from `jl instance get <id> --json`.

To expose a service (FastAPI, Gradio, etc.), bind to `0.0.0.0:6006` — it's accessible via `endpoints[0]` on generic templates. Use `--http-ports "7860,8080"` at creation or resume to expose custom ports. Custom port URLs appear in `endpoints` after the default 6006 entry.

Run `jl instance get <id> --json` to find all service URLs (`url`, `vs_url`, `endpoints`).

## Managed Runs

### How `jl run` works

`jl run` uploads your code to an instance, sets up a Python environment, and runs your script in the background with log and exit code tracking. You need either `--on <machine_id>` (existing instance) or `--gpu <type>` (creates a fresh instance).

`run_id` is tracked locally under `~/.jl/runs/`. All run management commands (`logs`, `status`, `stop`, `list`) depend on these local records. Start and monitor runs from the same machine.

### Run targets

| Target | What happens |
|---|---|
| `train.py` | Uploads the file, creates working dir, runs `python3 train.py` (or `bash` for `.sh`) |
| `.` or `./project` with `--script train.py` | Rsyncs the directory, runs inside it. Requires `rsync` locally. |
| No target, command after `--` | No upload, no environment setup. Runs the raw command directly. |

Only `.py` and `.sh` file targets are supported. For other file types, use a directory target or `jl instance upload` + `jl instance exec`.

### Environment & setup

For file and directory targets, `jl run` creates a `.venv` inside the project working directory on the remote machine. This venv:
- **Inherits template packages** — on a pytorch template, `import torch` works without installing anything.
- **Has pip available** — `pip install X` in `--setup` installs into the venv.
- **Persists across runs** — on the same instance with the same target, the venv and installed packages are reused.

**Auto-detection (directory targets only):** The CLI checks the local directory before upload. If `pyproject.toml` with a `[project]` table exists, dependencies are installed from it. Otherwise, if `requirements.txt` exists, it's installed. If neither exists, no packages are installed — template packages are enough for simple scripts.

**`--requirements <file>`** overrides auto-detection. The specified file is uploaded and installed instead.

**`--setup <command>`** runs a shell command after dependency installation, before your script. Use for quick one-off installs (`--setup "pip install flash-attn"`) or system commands (`--setup "apt-get install -y libsndfile1"`).

Setup chain (runs before your script, chained with `&&`):

1. `uv` installed if missing
2. `.venv` created if it doesn't exist (with template package visibility and pip)
3. `.venv` activated
4. Dependencies installed (auto-detected `pyproject.toml` or `requirements.txt`, or explicit `--requirements`)
5. `--setup` command (if provided)
6. Main script runs

```bash
# Directory with requirements.txt → auto-detected, no flag needed
jl run . --script train.py --on <id> --no-follow --json --yes

# Override auto-detection with a custom file
jl run . --script train.py --on <id> --requirements custom-reqs.txt --no-follow --json --yes

# Single file target → no auto-detection, pass --requirements if needed
jl run train.py --on <id> --requirements requirements.txt --no-follow --json --yes
```

**Command mode** — when you pass a raw command after `--` with no file or directory target. Useful when code already exists on the instance (e.g., the agent wrote files directly via `exec`, or a previous run left scripts on the remote). No upload, no venv — the command runs on system Python with template packages. You still get `jl run` log tracking (`logs`, `status`, `stop`), which is the main advantage over `jl instance exec`. Only `--setup` is available as a hook. `--requirements` is not supported in command mode.

```bash
jl run --on <id> --no-follow --json --yes -- python3 /home/train/train.py
jl run --on <id> --no-follow --json --yes -- torchrun --nproc_per_node=2 train.py
```

### Running on an existing instance

```bash
jl run train.py --on <machine_id> --no-follow --json --yes
jl run . --script train.py --on <machine_id> --requirements requirements.txt --no-follow --json --yes
```

Lifecycle flags (`--keep`, `--pause`, `--destroy`) are not allowed with `--on` — the instance is not touched after the run.

### Running on a fresh instance

```bash
jl run . --script train.py --gpu L4 --no-follow --keep --json --yes
```

Creates a new instance, uploads code, runs the script. Additional flags: `--template` (default: pytorch), `--storage` (default: 40GB), `--num-gpus` (default: 1), `--region`, `--http-ports`.

**Lifecycle rules for fresh instances:**

- **With `--json` or `--no-follow`:** `--keep` is required. The CLI rejects `--pause` and `--destroy` because it returns immediately and cannot apply lifecycle actions later. Use `--keep` and have the agent pause or destroy the instance after the run completes.
- **Without `--json` or `--no-follow` (human mode):** the CLI stays attached, streams logs, and applies lifecycle when the run finishes. Default lifecycle is `--pause`.

Use separate `jl instance create` when you need to inspect GPU availability, reuse machines across runs, or attach filesystems/scripts.

## Monitoring & Control

### Reading logs

The primary monitoring command:

```bash
jl run logs <run_id> --tail 50
```

Always use `--tail N` — without it, the entire log file is returned and can be enormous.

The output includes a **header** and **footer** with run state (in non-follow, non-JSON mode):

```
--- run r_abc | machine 123 | running ---

step=100 loss=2.31
step=200 loss=2.11

--- still running | log: /home/jl-runs/r_abc/output.log ---
```

When done, the footer shows the final state:
```
--- succeeded | exit code: 0 | log: /home/jl-runs/r_abc/output.log ---
```

Or on failure:
```
--- failed | exit code: 1 | log: /home/jl-runs/r_abc/output.log ---
```

If the instance is paused, missing, or SSH is unavailable, `jl run logs` fails before printing any output. Use `jl run status <run_id> --json` to check those states.

### The agent monitoring loop

1. **Start detached:** `jl run ... --no-follow --json --yes` — extract `run_id` and `machine_id` from JSON
2. **Early check (catch fast failures):** `sleep 15 && jl run logs <run_id> --tail 30` — if footer says `failed`, fix and retry immediately
3. **Steady-state polling:** `sleep 120 && jl run logs <run_id> --tail 50`
4. Read log body for loss values, errors, or progress
5. Check footer:
   - `still running` → repeat step 3
   - `succeeded | exit code: 0` → download results
   - `failed | exit code: N` → read error, fix, start a new run

Cadence: 60-120s (short experiments), 180-300s (long training), 300-600s (very long runs).

### Checking status

```bash
jl run status <run_id> --json
```

Returns run state, machine_id, exit_code, lifecycle_policy, launch_command, and more. Without `--refresh`, `jl run list` shows state as `"saved"` (a sentinel, not a real run state). Use `--refresh` or `--status` to get live state.

### Stopping a run

```bash
jl run stop <run_id> --json
```

Kills the entire process group (training script + all child processes). Escalates to SIGKILL if the process doesn't exit after TERM.

### System checks via exec

```bash
jl instance exec <id> -- nvidia-smi
jl instance exec <id> -- ps -ef
jl instance exec <id> -- df -h
```

Exit code of the remote command is propagated. For pipes or shell syntax, wrap in `sh -lc`:

```bash
jl instance exec <id> -- sh -lc 'grep "loss" /path/to/log | tail -5'
```

## File Transfer & Persistence

### Upload and download

```bash
jl instance upload <id> ./local /remote          # upload file or directory
jl instance download <id> /remote ./local         # download file
jl instance download <id> /remote ./local -r      # download directory
```

Default destinations: upload without dest → remote home directory. Download without dest → `./<basename>` in current local directory.

### What persists across pause/resume

The remote home directory (`/home/` on containers, `/home/<user>/` on VMs) persists. Everything else is ephemeral.

**Persists:**
- Files and directories under the home directory
- `.venv` created by `jl run` (inside the project working directory)
- Attached filesystems (mounted at `/home/jl_fs/`)
- Run metadata under `<home>/jl-runs/<run_id>/`

**Lost on pause:**
- System-level installs (`apt-get`, global pip packages outside the home directory)
- Files outside the home directory (`/tmp`, `/root`, etc.)

Use `--setup` for system-level reinstalls (e.g., `apt-get`). Python packages in the venv persist. For recurring system setup, use startup scripts (`jl scripts add`).

### Remote file paths

- Uploaded directories: `<home>/<directory_name>/`
- Uploaded files (via `jl run`): `<home>/<file_stem>/<file>` (e.g., `train.py` → `/home/train/train.py`)
- Uploaded files (via `jl instance upload`): `<home>/<filename>`
- Run metadata: `<home>/jl-runs/<run_id>/`

### Filesystems

- **Region-bound:** A filesystem created in IN2 is only visible to IN2 instances. Use `jl filesystem list --json` to see each filesystem's region.
- **ID changes on edit:** Expanding a filesystem (`jl filesystem edit`) may return a new `fs_id`. Always use the returned ID.
- The CLI validates that `fs_id` exists before creating/resuming, but does **not** validate that the filesystem's region matches the instance's region. Ensure they match yourself.

## Agent Workflow (End-to-End)

```bash
# 1. Check GPUs and create instance
jl gpus --json
jl instance create --gpu L4 --storage 50 --yes --json

# 2. Start detached run
jl run . --script train.py --on <machine_id> --requirements requirements.txt --no-follow --json --yes

# 3. Early check (catch import/syntax/pip failures fast)
sleep 15 && jl run logs <run_id> --tail 30

# 4. Steady-state monitoring (repeat until footer shows succeeded or failed)
sleep 120 && jl run logs <run_id> --tail 50

# 5. Download results (use /home/<user>/ for VMs instead of /home/)
jl instance download <machine_id> /home/results ./results -r

# 6. Cleanup
jl instance pause <machine_id> --yes --json
```

For fresh instances without a pre-created instance:

```bash
# Creates instance inline, runs detached — agent must clean up after
jl run . --script train.py --gpu L4 --no-follow --keep --json --yes
# ... monitor with jl run logs ...
jl instance pause <machine_id> --yes --json
```

## Error Handling

When `--json` is active, CLI validation and API failures are emitted as `{"error": "..."}` to stdout.

Not all non-zero exits use that shape. `jl instance exec --json` returns its own structured payload with `stdout`, `stderr`, and `exit_code` fields.

Agent rule:
- First inspect the JSON shape
- If it has an `error` key, treat it as a CLI failure
- Otherwise inspect command-specific fields (`exit_code`, `state`, `run_exit_code`)

## Anti-Patterns

- Do not use `jl run logs --follow` — blocks forever, will timeout. `--json` is also incompatible with `--follow`.
- Do not omit `--no-follow` when starting runs — default is `--follow` which blocks the agent.
- Do not read full logs without `--tail N` — can return megabytes of output.
- Do not poll every few seconds — use 60-600s intervals based on expected run duration.
- Do not use lifecycle flags (`--keep`, `--pause`, `--destroy`) with `--on` — they are rejected. Only for fresh instances.
- Do not use `--pause` or `--destroy` with `--json` for fresh instances — rejected. Use `--keep --json` and clean up yourself.
- Do not forget to pause/destroy instances after experiments — they cost money.

## Command Reference

### Account & Resources

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
| `jl instance pause <id> --yes --json` | Pause (stops compute billing, keeps data) |
| `jl instance resume <id> --yes --json` | Resume (supports `--gpu`, `--storage`, `--num-gpus`, `--name`, `--http-ports`, `--fs-id`) |
| `jl instance destroy <id> --yes --json` | Permanently delete |
| `jl instance rename <id> --name "new-name" --yes --json` | Rename |

### Instance Operations (require Running status)

| Command | Description |
|---|---|
| `jl instance exec <id> -- <command>` | Run command remotely (use `--` separator) |
| `jl instance exec <id> --json -- <command>` | Capture stdout, stderr, exit_code as JSON |
| `jl instance ssh <id>` | Interactive SSH session |
| `jl instance ssh <id> --print-command` | Print the SSH command |
| `jl instance upload <id> ./local [/remote]` | Upload file or directory |
| `jl instance download <id> /remote [./local] [-r]` | Download file or directory |

### Managed Runs

| Command | Description |
|---|---|
| `jl run <target> --on <id> --no-follow --json --yes` | Start run on existing instance |
| `jl run <target> --gpu <gpu> --no-follow --keep --json --yes` | Fresh instance + run (agent must clean up) |
| `jl run logs <run_id> --tail 50` | Recent logs with run state in header/footer |
| `jl run status <run_id> --json` | Run state, exit_code, machine_id, launch_command |
| `jl run stop <run_id> --json` | Kill the run (process group) |
| `jl run list --refresh --json` | All runs with live status (`saved` without `--refresh`) |
| `jl run list --limit 5 --json` | Most recent N runs |
| `jl run list --status running --json` | Filter by state (auto-refreshes) |
| `jl run list --machine <id> --json` | Filter by instance |

### Infrastructure

| Command | Description |
|---|---|
| `jl ssh-key list/add/remove` | Manage SSH keys |
| `jl scripts list/add/update/remove` | Manage startup scripts |
| `jl filesystem list/create/edit/remove` | Manage persistent filesystems |
