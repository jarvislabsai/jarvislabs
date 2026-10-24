---
name: jarvislabs
description: How to use the jl CLI on JarvisLabs.ai. Covers GPU instances and training runs, pay-per-token models (Token Factory), and serverless model deployments, including managed models. Read it before you run jl commands for a user.
---

# JarvisLabs CLI (`jl`)

`jl` is the command-line tool for JarvisLabs.ai. With it you can rent GPU machines, run training jobs on them, call hosted models, and serve a model behind your own API endpoint.

This guide explains how each part works. It does not list every flag. To see the flags of a command, run `jl <command> --help`.

## Before you start

**Log in.** Run `jl status`. If you are not logged in, run `jl setup --token <token> --yes`, or set the `JL_API_KEY` environment variable. `JL_API_KEY` takes priority over the saved token.

**Use `--json` in scripts.** With `--json`, a command prints machine-readable JSON. When a command fails, it prints `{"error": "<message>"}` and exits with code 1. `jl exec --json` is different. Read its `exit_code`. Commands that ask for confirmation, such as `create`, `pause` and `destroy`, also need `--yes` when you use `--json`.

**Pass every value on the command line.** If you leave out an ID, some commands open a list to pick from. Without `--yes`, actions first ask "yes or no". Both need a person at a terminal, so in an agent or a script the command fails or does nothing. Always pass the ID, and pass `--yes` for actions that you mean to do.

**Do not block on long commands.** `jl create`, `jl resume` and `jl exec` wait until they finish. For long work, start a job with `jl run ... --json`, which returns at once. Create deployments with `--detach`, and read deployment logs with `--no-follow`.

**When something fails, read the state first.** Use `jl get`, `jl run status`, `jl run logs`, `jl deploy get` or `jl deploy logs` before you change anything.

## Core products

`jl` covers these JarvisLabs products.

**GPU instances.** A GPU instance is a machine with one or more GPUs that you rent by the hour. You can install and run anything on it, such as training, fine-tuning, notebooks or your own server. You pay while it runs, and a small amount for its disk while it is paused. Commands: `jl create`, `jl run`, `jl exec` and the other instance commands.

**Token Factory.** Models that JarvisLabs hosts for you, behind an OpenAI-compatible API. You pay per token. Commands: `jl models`.

**Serverless deployments** (beta). A private OpenAI-compatible endpoint for one model, on GPU workers that start and stop with your traffic. There are two kinds:

- **Managed models** are ready to deploy. JarvisLabs has already tested and tuned the GPU, the serving engine and its settings, so the model is optimized to run fast. You choose only the scaling.
- **Custom deployments** run any Hugging Face or Ollama model, with the engine, GPU and settings that you choose.

You pay for each running worker by the hour, and for storage until you delete the deployment. Commands: `jl deploy`.

### How to choose

Do what the user asks for. If the request is open, this is what each product is for:

- To call a model with no setup, use Token Factory. Run `jl models list` to see which models it has.
- For a private endpoint with its own GPUs and your own scaling, use a deployment. Use a managed model if the model is in `jl deploy managed-models list`, and a custom deployment if it is not. A deployment takes some minutes to start, and it costs money while workers run.
- For training, fine-tuning, scripts or any other program, use a GPU instance.

If the choice changes what the user pays and they did not say what they want, ask them.

## GPU instances

### Kinds of instance

- **GPU container** (`jl create --gpu <type>`). The default kind. It starts from a template (`pytorch` unless you pass `--template`). It has JupyterLab, a browser IDE, and HTTPS URLs for web services. Only its home folder is kept when you pause it.
- **GPU VM** (`jl create --gpu <type> --vm`). A full virtual machine that you reach over SSH only. It keeps its whole disk when you pause it.
- **CPU VM** (`jl create --vm --cpu`). For work that needs no GPU. If you leave out `--vcpus` and `--ram`, you get the smallest plan. If you set one of them, set both.

Some options work only with some kinds. For example, `--spot`, `--template` and `--http-ports` work only on GPU containers, and CPU VMs cannot attach filesystems. The CLI rejects a wrong combination with a clear message.

### Before you create one

**Add an SSH key first.** Run `jl ssh-key list`, and if it is empty, add one with `jl ssh-key add <public-key-file> --name <name>`. VMs cannot be created without a key. A container can be created without one, but then `jl ssh`, `jl exec`, `jl upload`, `jl download` and `jl run` do not work on it. The matching private key must be on the machine where you run `jl`, and it must work without a passphrase prompt. A key that you add later works on an existing instance only after you pause and resume it.

**Check availability and price.** `jl gpus` shows GPU types, prices and free GPUs for containers and VMs. `jl cpus` shows CPU VM plans. `jl resources` shows both. In `jl gpus --json`:

- `effective_num_free_devices` is how many GPUs a normal (on-demand) instance can get.
- `num_free_devices` is how many GPUs are free right now, without taking any back from spot instances.
- `spot_num_free_devices` is how many GPUs a spot instance can get. Spot is offered only in rows that have a `spot_price`.
- `workload_type` is `"container"`, `"vm"`, or `null`, which means the row is for both.

**Pick a region, or let the CLI pick.** Instance regions are `IN1`, `IN2` and `EU1`, and each region has different GPU types. If you leave out `--region`, the CLI prefers a region that has free GPUs of the type you asked for.

**Keep enough balance.** Creating or resuming an instance needs enough balance. When the balance runs out, JarvisLabs pauses running instances.

### Pause, resume and destroy

An instance is in one of three main states:

- **Running.** You pay for the GPU (or CPU) and the disk. SSH, `exec`, `upload`, `download` and `jl run` work only in this state.
- **Paused.** The GPU is released and you pay only for the disk. Your files are kept, as described in "Files, and what a pause keeps".
- **Destroyed.** The instance and its disk are gone. This cannot be undone.

Use `jl pause <id>`, `jl resume <id>` and `jl destroy <id>`.

**`jl resume` may return a new `machine_id`.** Always read the ID from the output and use it from then on. Other things to know about resume:

- An instance always resumes in its original region. You can change the GPU type or count on resume, but only to a GPU that is free in that region.
- You can make the disk bigger on resume (`--storage`), but not smaller.
- To resize a CPU VM, pass both `--vcpus` and `--ram`.

**Spot instances are cheaper but can be taken back.** When an on-demand user needs the GPU, JarvisLabs pauses your spot instance. Your files are kept. A resume is on-demand unless you pass `--spot` again.

### Running commands, SSH and web services

`jl exec <id> -- <command>` runs one command on the instance and returns its exit code. It waits until the command ends, so use `jl run` for long jobs. If the command uses a pipe, `&&`, `cd` or a variable, wrap it in `sh -lc '...'`. Otherwise your local shell handles those parts instead of the instance.

```bash
jl exec <id> -- sh -lc 'cd /home/project && tail -5 train.log'
```

`jl ssh <id>` opens an interactive shell. To get the SSH command as text, use `jl ssh <id> --json` or `jl get <id> --json` (the `ssh_command` field).

GPU containers have HTTPS URLs for web services. `jl get <id> --json` shows them while the instance runs:

- `url` is JupyterLab.
- `vs_url` is the browser IDE.
- `endpoints[0]` is port 6006. To expose your own app (FastAPI, Gradio and so on), make it listen on `0.0.0.0:6006`.
- To expose more ports, pass `--http-ports "7860,8080"` on create or resume. Their URLs follow in `endpoints`, in the same order.

VMs have no HTTPS URLs. Use SSH.

### Files, and what a pause keeps

In this guide, `<home>` means `/home` on containers and `/home/<user>` on VMs.

`jl upload <id> <local> [<remote>]` copies a file or a folder to the instance. Without a remote path, it goes to `<home>/<name>`. `jl download <id> <remote> [<local>]` copies back to the current folder. Add `-r` to download a folder.

What a pause keeps:

- **On a container**, only `<home>` is kept. That includes your files, the Python environments that `jl run` creates, and attached filesystems. Everything outside `<home>` is reset on resume: `apt` installs, global `pip` installs, `/tmp` and `/root`. To set these up again on every start, pass `--setup "<command>"` to `jl run`, or attach a startup script.
- **On a VM**, the whole disk is kept.

### Filesystems, startup scripts and private networks

**Filesystems** (`jl filesystem`) are shared disks that exist apart from any instance. Attach one with `--fs-id` on create or resume. It appears at `/home/jl_fs`. A filesystem works only with instances in its own region. Its size can only grow, and `jl filesystem edit` can return a new `fs_id`, so use the ID that it returns. You pay for a filesystem's full size for as long as it exists, even when no instance uses it.

**Startup scripts** (`jl scripts`) run every time a GPU container starts. They do not run on VMs. Attach one with `--script-id` on create or resume. They are the place for system setup that a container loses on pause.

**Private networks** (`jl vpc`) connect VMs to each other over private IP addresses. They are for VMs only. Attach a VM with `--vpc-id` on create or resume.

## Running jobs with `jl run`

### How a run works

`jl run` copies your code to an instance, sets up a Python environment, and starts your script in the background. It saves the output to a log file and records the exit code. You can then check the job with `jl run logs`, `jl run status` and `jl run stop`.

There are three ways to say what to run:

| You pass | What `jl run` does |
|---|---|
| A file: `train.py` or `run.sh` | Copies it to `<home>/train.py` and runs it there. Uses the shared environment `<home>/.venv`. |
| A folder and `--script`: `. --script train.py` | Syncs the folder to `<home>/<folder name>/` and runs the script inside it. Uses a separate environment in `<home>/<folder name>/.venv`. Needs `rsync` on your machine. |
| Only a command after `--` | Copies nothing and runs the command from `~`. If `<home>/.venv` exists, its `python` and `pip` come first on the PATH. |

Things to know about these modes:

- File targets must end in `.py` or `.sh`.
- Arguments for your script go after `--`: `jl run train.py --on <id> --json --yes -- --epochs 50`.
- A folder target syncs the remote copy to match your local folder. Files that your job wrote inside that remote folder are deleted on the next `jl run` of the same folder. Write outputs to another folder, or download them before you run again.
- Two file targets with the same name, such as `a/train.py` and `b/train.py`, overwrite each other on the instance. Use folder targets for real projects.
- Command mode still gives you logs, status and stop, which `jl exec` does not. It starts in `~`, so use absolute paths or `sh -lc 'cd /path && ...'`.

### Python environment and dependencies

The environments that `jl run` creates can see the packages of the template, such as PyTorch. Do not install those again. The environments are kept when you pause the instance.

- A folder target installs its dependencies by itself. It uses `pyproject.toml` if that file has a `[project]` section, and otherwise `requirements.txt`.
- A file target installs nothing by itself. Pass `--requirements requirements.txt` if the script needs packages.
- `--setup "<command>"` runs a shell command before your script, for example `--setup "apt-get install -y ffmpeg"`.

Setup runs as part of the background job, so a failed install does not make `jl run` fail at start. The error shows up in the logs, and the run ends as `failed`. Check the logs once, about 15 seconds after the start, to catch it early.

### On an existing instance, or on a new one

`jl run ... --on <machine_id>` uses an instance that already exists. `jl run` does not pause or destroy it afterwards.

`jl run ... --gpu <type>` creates a new GPU instance for the run. When you use `--json`, you must also pass `--keep`. The CLI returns at once and cannot pause or destroy the instance later, so you must do that yourself when the run ends. If the run fails to start after the instance was created, the instance keeps running. Find it with `jl list` and clean it up.

Create the instance yourself with `jl create` when you need a CPU VM, a filesystem, a startup script, or a check of availability first. `jl run --gpu` cannot do these.

`jl run --help` shows only the main options. `jl run start --help` shows all of them.

### Watching a run

Start the run with `--json --yes`, and keep the `run_id` and `machine_id` from the output. Then:

1. After about 15 seconds, read the logs once with `jl run logs <run_id> --tail 50`. This catches import errors, syntax errors and failed installs.
2. After that, check every 1 to 10 minutes, based on how long the job takes. Do not check every few seconds.
3. The last line of `jl run logs` says `still running`, `succeeded | exit code: 0`, or `failed | exit code: N`. It is printed on stderr. With `--json`, read `run_exit_code` instead, which is `null` while the run is still going.
4. When the run succeeds, download the results. When it fails, read the error, fix the cause, and start a new run.
5. When you are done, pause or destroy the instance, or tell the user that it is still running.

Always pass `--tail` to `jl run logs`. Without it, you get the whole log, which can be very large.

`jl run status <run_id> --json` shows the state, the exit code and the instance status. `jl run stop <run_id>` stops the script and all its child processes. `jl run list` shows `saved` as the state until you add `--refresh`.

## Token Factory (`jl models`)

`jl models list --json` lists the models with their context length and their prices per million tokens. `jl models get <model_id> --json` shows one model in full, including whether it accepts images (`input_modalities`) and which thinking controls it has (`thinking`).

### Try a model from the terminal

```bash
jl models chat <model_id> "Your question" --max-tokens 2000
```

- Always pass the model ID and a prompt. Without them, the command opens a picker or an interactive chat, which needs a terminal.
- The answer goes to stdout. The thinking, the token count and the cost go to stderr. Each call is a new conversation.
- `--thinking`, `--no-thinking` and `--effort` work only on models that support them, and the effort levels differ per model. Check `thinking` in `jl models get <model_id> --json`.
- `--image <file or URL>` attaches an image, for models that accept images.
- `--max-tokens` defaults to 32768, and thinking tokens count toward it. With `--json`, check `finish_reason` in the response, because a cut-off reply still exits with code 0.
- Piped text is added after the prompt. For large input, use `< file` instead of a pipe.

### Use it in your own code

The API is OpenAI-compatible, so any OpenAI client or library works. Set the base URL to `https://models.jarvislabs.net/v1`, use the JarvisLabs API key, and pass the model ID as `model`. `jl models get <model_id>` prints a curl and a Python example.

Two things to know:

- **Always set `max_tokens`.** Without it, the reply stops at 1024 tokens, and thinking counts toward that limit.
- **Thinking settings go in `chat_template_kwargs`.** In the OpenAI Python SDK, send them through `extra_body`. The key names differ per model, and `jl models get <model_id> --json` shows them under `thinking.template_parameters`. The thinking text comes back in `reasoning_content`.

You pay per token, at the prices in `jl models list`. Thinking tokens count as output tokens.

## Serverless deployments (`jl deploy`, beta)

### How a deployment works

A deployment serves one model on GPU workers. A worker is a machine with one or more GPUs that runs the model and answers requests. These settings control the workers:

- **Min workers.** These always run, even with no traffic. With 0, the deployment can scale down to no workers.
- **Max workers.** The most workers that can run at the same time.
- **Concurrent requests.** How many requests one worker handles at the same time. When all workers are full, a new one starts, up to max workers.
- **Idle timeout.** A worker above the minimum stops after this many seconds without requests.
- **Wait time.** How many seconds a request waits for a free worker before it fails.

https://docs.jarvislabs.ai/serverless explains each setting. A deployment has no SSH and no pause or resume. Deployments run in `IN1` and `IN2`.

**Status `running` does not mean a worker is ready.** With min workers 0, a deployment can be `running` with no workers. Look at `workers.healthy` in `jl deploy get <id> --json` to see how many workers can answer now.

### Managed models and custom deployments

- **Managed models.** `jl deploy managed-models list` shows them, and `jl deploy managed-models create <slug>` deploys one. Each scaling setting has a default. Always pass the slug, because without it the command opens a picker. Names must be unique among your managed deployments, so pass `--name` to deploy the same model twice.
- **Custom deployments.** `jl deploy create` needs the model (a Hugging Face repo ID or an Ollama tag), the framework, the GPU, the storage and the scaling settings. See `jl deploy create --help`. Pass engine options as `--arg key=value`, without the leading `--` of the key. A gated Hugging Face model needs `--env HF_TOKEN=<token>`.

### Create and test

```bash
jl deploy managed-models create <slug> --detach --yes --json   # prints deployment_id
jl deploy status <id> --json                                   # check until "running" or "failed"
jl deploy chat <id> "Say hello" --max-tokens 50
```

Pass `--detach` so that `create` returns at once. Without it, `create` waits until the deployment is running, which can take a long time for a large model.

`jl deploy chat` never waits for a worker. If a worker is ready, it sends the prompt and prints the reply. If not, it starts a worker when none is starting, and then exits with code 1 and a "Try again shortly" message. Wait until `workers.healthy` is at least 1 in `jl deploy get <id> --json`, then try again.

### Use it in your own code

The endpoint is OpenAI-compatible, so any OpenAI client works:

- Base URL: `openai_base_url` from `jl deploy get <id> --json`. It appears only while the deployment is running.
- API key: the JarvisLabs API key.
- `model`: the served model name. For a managed model, `jl deploy get <id>` shows it in the "API model" row. For a custom deployment, it is the `--arg served-model-name` value if you set one, and otherwise the `--model` value.

If no worker is ready, a request may wait while a worker starts, or fail with HTTP 504 "A worker is still starting, please retry shortly". Retry on 408, 429, 500, 502, 503 and 504, with a pause between tries.

### Change, debug and delete

```bash
jl deploy get <id> --json                    # workers, cost so far, error, base URL
jl deploy logs <id> --tail 100 --no-follow   # recent logs from running workers
jl deploy update <id> --min-workers 0 --json
jl deploy delete <id> --yes --json
```

- `jl deploy logs` follows by default, so pass `--no-follow`. Logs come only from running workers and are not stored.
- `jl deploy update` can change the name, the idle timeout, min workers, max workers and concurrent requests. It can change the wait time for custom deployments only. It works only while the deployment is `running`. To change the model, the GPU or the storage, create a new deployment.

### What you pay for

- Each running worker is billed by the hour, from the moment it starts. Min workers bill all the time.
- Storage is billed for as long as the deployment exists, even with 0 workers.
- In `jl deploy managed-models list --json`, `price_per_hour` is the price of one running worker's GPUs, without storage.
- Setting min workers to 0 does not stop the GPU cost at once, because the current workers stop only after the idle timeout. Only `jl deploy delete` stops all billing, and it cannot be undone.

## Before you finish

Before you end a task, find everything that you created and that still costs money:

- `jl list --json` shows instances (containers and VMs), with their status and cost.
- `jl deploy list --json` shows deployments.
- `jl filesystem list --json` shows filesystems.

Then clean up what you created, or tell the user exactly what is still running. If the account belongs to a team, `jl list` can show instances of other team members. Touch only what you created. `jl destroy`, `jl filesystem remove` and `jl deploy delete` cannot be undone, so ask the user first if the data might matter.
