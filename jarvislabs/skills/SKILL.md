---
name: jarvislabs
description: Agent guide for running and monitoring GPU experiments with the jl CLI on JarvisLabs.ai.
---

# JarvisLabs GPU CLI (`jl`) — Agent Guide

## Getting Started

Verify auth with `jl status --json` before doing anything. If not logged in, use `jl setup --token <token> --yes`. You can also authenticate via `export JL_API_KEY="..."`.

Use `--help` on any command to discover flags (e.g., `jl run --help`, `jl create --help`). If something goes wrong, use `jl run logs`, `jl run status`, and `jl exec` to diagnose — don't guess.

## Mental Model

- Machine commands (`jl create/list/pause/resume/destroy/ssh/exec/upload/download`) = GPU instance lifecycle and access.
- `jl run` = managed job on an instance. Uploads code, sets up a Python environment, runs your script in the background with log tracking.
- `jl exec` = run a quick command on an instance (nvidia-smi, ps, df) and low-level instance access. No environment setup, no tracking.

## Instances

### Creating

```bash
jl create --gpu L4 --storage 40 --yes --json
```

`--gpu` is required. Run `jl create --help` for all available flags.

### Lifecycle

Instances have three states that matter: **Running** (billing active), **Paused** (compute billing stopped, storage billing continues, data persists), **Destroyed** (everything deleted).

```bash
jl pause <id> --yes --json      # stop compute billing, keep data
jl resume <id> --yes --json     # restart a paused instance
jl destroy <id> --yes --json    # permanently delete
```

Resume caveats:
- Resume is **region-locked** — an instance always resumes in its original region. GPU swaps are only possible within that region.
- Resume may return a **new machine_id**. Always use the returned ID for subsequent operations.
- Run `jl resume --help` for all available flags (GPU swap, storage expansion, rename, etc.).

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

VM instances (`jl create --gpu ... --vm`) get SSH-only access. VMs require at least one SSH key registered (`jl ssh-key add`). Use `ssh_command` from `jl get <id> --json`.

To expose a service (FastAPI, Gradio, etc.), bind to `0.0.0.0:6006` — it's accessible via `endpoints[0]` on generic templates. Use `--http-ports "7860,8080"` at creation or resume to expose custom ports. Custom port URLs appear in `endpoints` after the default 6006 entry.

Run `jl get <id> --json` to find all service URLs (`url`, `vs_url`, `endpoints`).

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

Only `.py` and `.sh` file targets are supported. For other file types, use a directory target or `jl upload` + `jl exec`.

Pass script arguments after `--`:
```bash
jl run train.py --on <id> --json --yes -- --epochs 50 --lr 0.001
```

### Environment & setup

`jl run` creates a `.venv` on the remote machine. Template packages (torch, etc.) are visible by default — no need to install them.

**How dependencies get installed:**

- **Directory targets** — auto-detected. If your directory has `requirements.txt` or `pyproject.toml` (with `[project]`), deps are installed automatically. No flag needed.
- **Single file targets** — no auto-detection. Pass `--requirements requirements.txt` if you need extra packages.
- **`--requirements <file>`** — overrides auto-detection. Uploads and installs the specified file instead.
- **`--setup <command>`** — runs a shell command before your script (e.g., `--setup "pip install flash-attn"`).

```bash
# Directory — auto-detects requirements.txt
jl run . --script train.py --on <id> --json --yes

# Single file — pass requirements explicitly
jl run train.py --on <id> --requirements requirements.txt --json --yes

# Extra setup command
jl run . --script train.py --on <id> --setup "pip install flash-attn" --json --yes
```

**Command mode** — when you pass a raw command after `--` with no file or directory target. Useful when code already exists on the instance (e.g., the agent wrote files directly via `exec`, or a previous run left scripts on the remote). No upload, no venv — the command runs on system Python with template packages. You still get `jl run` log tracking (`logs`, `status`, `stop`), which is the main advantage over `jl exec`. Only `--setup` is available as a hook. `--requirements` is not supported in command mode.

```bash
jl run --on <id> --json --yes -- python3 /home/train/train.py
jl run --on <id> --json --yes -- torchrun --nproc_per_node=2 train.py
```

### Running on an existing instance

```bash
jl run train.py --on <machine_id> --json --yes
jl run . --script train.py --on <machine_id> --requirements requirements.txt --json --yes
```

Lifecycle flags (`--keep`, `--pause`, `--destroy`) are not allowed with `--on` — the instance is not touched after the run.

### Running on a fresh instance

```bash
jl run . --script train.py --gpu L4 --keep --json --yes
```

Creates a new instance, uploads code, runs the script. Additional flags: `--vm` (VM instead of container), `--template` (default: pytorch), `--storage` (default: 40GB), `--num-gpus` (default: 1), `--region`, `--http-ports`.

**Lifecycle rules for fresh instances:**

- **With `--json` or `--no-follow`:** `--keep` is required. The CLI rejects `--pause` and `--destroy` because it returns immediately and cannot apply lifecycle actions later. Use `--keep` and have the agent pause or destroy the instance after the run completes.
- **Without `--json` or `--no-follow` (human mode):** the CLI stays attached, streams logs, and applies lifecycle when the run finishes. Default lifecycle is `--pause`.

Use separate `jl create` when you need to inspect GPU availability, reuse machines across runs, or attach filesystems/scripts.

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

1. **Start detached:** `jl run ... --json --yes` — extract `run_id` and `machine_id` from JSON
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
jl exec <id> -- nvidia-smi
jl exec <id> -- ps -ef
jl exec <id> -- df -h
```

Exit code of the remote command is propagated. For pipes or shell syntax, wrap in `sh -lc`:

```bash
jl exec <id> -- sh -lc 'grep "loss" /path/to/log | tail -5'
```

## File Transfer & Persistence

### Upload and download

```bash
jl upload <id> ./local /remote          # upload file or directory
jl download <id> /remote ./local         # download file
jl download <id> /remote ./local -r      # download directory
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
- Uploaded files (via `jl upload`): `<home>/<filename>`
- Run metadata: `<home>/jl-runs/<run_id>/`

### Filesystems

- **Region-bound:** A filesystem created in IN2 is only visible to IN2 instances. Use `jl filesystem list --json` to see each filesystem's region.
- **ID changes on edit:** Expanding a filesystem (`jl filesystem edit`) may return a new `fs_id`. Always use the returned ID.
- The CLI validates that `fs_id` exists before creating/resuming, but does **not** validate that the filesystem's region matches the instance's region. Ensure they match yourself.

## Agent Workflow (End-to-End)

```bash
# 1. Check GPUs and create instance
jl gpus --json
jl create --gpu L4 --storage 50 --yes --json

# 2. Start detached run
jl run . --script train.py --on <machine_id> --requirements requirements.txt --json --yes

# 3. Early check (catch import/syntax/pip failures fast)
sleep 15 && jl run logs <run_id> --tail 30

# 4. Steady-state monitoring (repeat until footer shows succeeded or failed)
sleep 120 && jl run logs <run_id> --tail 50

# 5. Download results (use /home/<user>/ for VMs instead of /home/)
jl download <machine_id> /home/results ./results -r

# 6. Cleanup
jl pause <machine_id> --yes --json
```

For fresh instances without a pre-created instance:

```bash
# Creates instance inline, runs detached — agent must clean up after
jl run . --script train.py --gpu L4 --keep --json --yes
# ... monitor with jl run logs ...
jl pause <machine_id> --yes --json
```

## Error Handling

When `--json` is active, CLI validation and API failures are emitted as `{"error": "..."}` to stdout.

Not all non-zero exits use that shape. `jl exec --json` returns its own structured payload with `stdout`, `stderr`, and `exit_code` fields.

Agent rule:
- First inspect the JSON shape
- If it has an `error` key, treat it as a CLI failure
- Otherwise inspect command-specific fields (`exit_code`, `state`, `run_exit_code`)

## Anti-Patterns

- Do not use `jl run logs --follow` — blocks forever, will timeout. `--json` is also incompatible with `--follow`.
- Always use `--json` when starting runs — it returns immediately. Without `--json`, the CLI streams logs and blocks.
- Do not read full logs without `--tail N` — can return megabytes of output.
- Do not poll every few seconds — use 60-600s intervals based on expected run duration.
- Do not use lifecycle flags (`--keep`, `--pause`, `--destroy`) with `--on` — they are rejected. Only for fresh instances.
- Do not use `--pause` or `--destroy` with `--json` for fresh instances — rejected. Use `--keep --json` and clean up yourself.
- Do not use `jl exec` for long-running tasks — it blocks until the command finishes. Use `jl run` which runs in the background with log tracking.
- Do not trust `jl run list` without `--refresh` — state shows as `"saved"` (stale). Use `--refresh` or `--status` for live state.
- Do not assume `machine_id` is stable after `jl resume` — it may return a new ID. Always use the returned ID.
- Do not forget to pause/destroy instances after experiments — they cost money.

## Command Discovery

Every command supports `--help` for full flag details:

```bash
jl create --help       jl run --help       jl ssh-key --help
jl resume --help       jl run logs --help  jl filesystem --help
```
