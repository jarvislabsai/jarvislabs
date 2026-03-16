# jarvislabs

[![Python](https://img.shields.io/pypi/pyversions/jarvislabs)](https://pypi.org/project/jarvislabs/)
[![License](https://img.shields.io/pypi/l/jarvislabs)](https://github.com/jarvislabsai/jarvislabs/blob/main/LICENSE)

CLI and Python SDK for managing GPU instances on [JarvisLabs.ai](https://jarvislabs.ai).

> **Beta release.** The v0.2 rewrite is in pre-release. Install with `--pre` to get it.

See the docs on [Jarvislabs Docs](https://docs.jarvislabs.ai/cli/)
## Installation

### As a CLI tool (recommended)

```bash
uv tool install --pre jarvislabs
```

To upgrade:

```bash
uv tool upgrade --pre jarvislabs
```

### As a library

```bash
pip install --pre jarvislabs
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install --pre jarvislabs
```

Requires Python 3.11+.

## Authentication

Get your API key at [jarvislabs.ai/settings/api-keys](https://jarvislabs.ai/settings/api-keys).

```bash
jl setup
```

Or set an environment variable:

```bash
export JL_API_KEY="your_api_key"
```

## CLI Quick Start

```bash
# See available GPUs and pricing
jl gpus

# Create an instance
jl instance create --gpu A100 --name "my-instance"

# Create an instance and expose a custom HTTP port
jl instance create --gpu RTX5000 --http-ports 7860

# SSH into it
jl instance ssh <machine_id>

# Pause when done (stops compute billing, data persists)
jl instance pause <machine_id>

# Resume later — optionally with different hardware
jl instance resume <machine_id> --gpu H100

# Destroy when no longer needed
jl instance destroy <machine_id>
```

### Managed Runs

Run scripts on GPU instances without manual setup. Code is uploaded, a virtual environment is created, and logs are tracked automatically.

For humans, the default mode stays attached to the run, streams logs, and can auto-pause or auto-destroy the instance after the run finishes. For agents, `--json` is meant for detached workflows and returns immediately, so use `--keep` and have the agent pause or destroy the instance after the run.

```bash
# Run a training script on a fresh GPU (instance auto-pauses when done)
jl run train.py --gpu RTX5000

# Start a long-running web app on a fresh GPU and expose port 8000
jl run app.py --gpu RTX5000 --http-ports 8000 --keep --no-follow

# Pass script arguments
jl run train.py --gpu RTX5000 -- --epochs 50 --lr 0.001

# Sync a project directory and run a script inside it
jl run . --script train.py --gpu A100 --requirements requirements.txt

# Run on an existing instance
jl run train.py --on <machine_id>

# Check on a run
jl run logs <run_id> --follow
jl run status <run_id>
jl run stop <run_id>
```

### More Commands

```bash
jl status                   # Account info and balance
jl templates                # Available framework templates
jl instance list            # List all instances
jl instance exec <id> -- nvidia-smi   # Run a command remotely
jl instance upload <id> ./data        # Upload files
jl instance download <id> /home/results.csv  # Download files
jl ssh-key add ~/.ssh/id_ed25519.pub --name "my-key"
jl scripts add ./setup.sh --name "install-deps"
jl filesystem create --name "datasets" --storage 200
jl instance get <id>                  # Shows Jupyter + exposed port URLs
```

Every command supports `--help`, `--json` (machine-readable output), and `--yes` (skip confirmations).

## Python SDK

```python
from jarvislabs import Client

with Client() as client:
    # Create a GPU instance (blocks until running)
    inst = client.instances.create(gpu_type="A100", name="my-run")
    print(f"SSH: {inst.ssh_command}")
    print(f"URL: {inst.url}")

    # When done
    client.instances.pause(inst.machine_id)
```

```python
from jarvislabs import Client

with Client() as client:
    # List and filter instances
    running = [i for i in client.instances.list() if i.status == "Running"]

    # Check GPU availability and pricing
    for gpu in client.account.gpu_availability():
        print(f"{gpu.gpu_type}: {gpu.num_free_devices} free, ${gpu.price_per_hour}/hr")

    # Manage filesystems
    fs_id = client.filesystems.create(fs_name="data", storage=100)

    # Manage startup scripts
    client.scripts.add(script="#!/bin/bash\npip install wandb", name="setup")
```

## Development

```bash
uv pip install -e ".[dev]"
uv run ruff format . && uv run ruff check --fix .
uv run pytest
```

## License

[MIT](LICENSE)
