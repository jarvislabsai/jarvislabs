"""Serverless model deployment commands — `jl deploy`.

Thin wrapper over ``client.deployments``: parse flags, call the SDK, render output.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Annotated

import typer

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.cli.models import ImageOption, interactive_chat, read_prompt, stream_reply, user_content
from jarvislabs.regions import normalize_serverless_region

if TYPE_CHECKING:
    from jarvislabs.models import ManagedModel, ManagedModelListResult

BETA_HELP_NOTICE = (
    "[bold bright_yellow]Serverless is in beta. We're actively improving it and would love early feedback. "
    "Limits and behavior may change; if you hit issues or have suggestions, email support@jarvislabs.ai.[/bold bright_yellow]"
)

deploy_app = typer.Typer(
    name="deploy",
    help=(
        "Create and manage serverless model deployments [bold bright_yellow](beta)[/bold bright_yellow].\n\n"
        "[dim]Quick examples:[/dim]\n"
        "  [bold]jl deploy managed-models list[/bold]           [rgb(100,100,100)]models you can deploy[/rgb(100,100,100)]\n"
        "  [bold]jl deploy managed-models create[/bold]         [rgb(100,100,100)]pick a model and deploy it[/rgb(100,100,100)]\n"
        "  [bold]jl deploy list[/bold]                          [rgb(100,100,100)]your deployments[/rgb(100,100,100)]\n"
        "  [bold]jl deploy get <id>[/bold]                      [rgb(100,100,100)]status, base URL and cost[/rgb(100,100,100)]\n"
        '  [bold]jl deploy chat <id> "Hello"[/bold]             [rgb(100,100,100)]send a test message[/rgb(100,100,100)]\n'
        "  [bold]jl deploy logs <id>[/bold]                     [rgb(100,100,100)]live logs[/rgb(100,100,100)]\n"
        "  [bold]jl deploy update <id> --min-workers 0[/bold]   [rgb(100,100,100)]no GPU cost while idle[/rgb(100,100,100)]\n"
        "  [bold]jl deploy delete <id>[/bold]                   [rgb(100,100,100)]delete it and stop all billing[/rgb(100,100,100)]\n"
        "\n" + BETA_HELP_NOTICE
    ),
)
app.add_typer(deploy_app, rich_help_panel="[bold bright_yellow]Serverless (Beta)[/bold bright_yellow]")

managed_app = typer.Typer(
    name="managed-models",
    help="List and deploy JarvisLabs-managed models [bold bright_yellow](beta)[/bold bright_yellow].",
)
deploy_app.add_typer(managed_app)

RegionHintOption = Annotated[
    str | None, typer.Option("--region", help="Region of the deployment (skips the cross-region search).")
]


def _parse_kv(pairs: list[str] | None, *, flag: str, redact: bool = False) -> dict[str, str]:
    """Parse repeatable ``key=value`` flags. Split on the first `=`; last key wins.

    ``redact`` keeps the offending value out of the error message (for secrets).
    """
    result: dict[str, str] = {}
    for raw in pairs or []:
        if "=" not in raw:
            shown = "" if redact else f" {raw!r}"
            render.die(f"Invalid {flag}{shown}: expected key=value")
        key, value = raw.split("=", 1)
        result[key] = value
    return result


def _finish_create(client, deployment_id: str, region: str, *, detach: bool) -> None:
    """After a create: wait until running (unless detached), then print how to call it."""
    render.success(f"Deployment {deployment_id} created.")
    next_steps = (
        f"  Watch the logs:  jl deploy logs {deployment_id}\n"
        f"  Check status:    jl deploy get {deployment_id}\n"
        f"  Delete it:       jl deploy delete {deployment_id}"
    )

    if detach:
        if state.json_output:
            render.print_json({"deployment_id": deployment_id, "region": render.region_input_label(region)})
            return
        render.info(f"It is starting on our servers.\n{next_steps}")
        return

    render.info(
        "Your model is starting on our servers. Large models can take a while.\n"
        "You can press Ctrl-C or close this terminal at any time; the deployment keeps starting.\n"
        f"{next_steps}"
    )
    try:
        with render.spinner("Waiting for the deployment to be ready..."):
            deployment = client.deployments.wait_until_running(deployment_id, region=region)
    except KeyboardInterrupt:
        # The deployment keeps building — hand back the id in both modes so it's never lost.
        if state.json_output:
            render.print_json({"deployment_id": deployment_id, "region": render.region_input_label(region)})
        else:
            render.info(f"Stopped waiting. The deployment is still starting.\n{next_steps}")
        raise typer.Exit() from None

    base_url = client.deployments.openai_base_url(deployment_id, region=deployment.region)
    if state.json_output:
        render.print_json({**deployment.model_dump(), "openai_base_url": base_url})
        return
    render.deployment_running_handoff(base_url, deployment.served_model)


def _fetch_managed_models(client) -> ManagedModelListResult:
    """The managed-model catalog with each model's hourly price per running worker.

    The catalog has no prices, so they come from the GPU on-demand price (as in `jl gpus`).
    """
    with render.spinner("Fetching managed models..."):
        result = client.deployments.list_managed_models()
        gpu_prices = {(gpu.gpu_type, gpu.region): gpu.price_per_hour for gpu in client.account.gpu_availability()}
        result.currency = client.account.currency()
    for model in result.models:
        price = gpu_prices.get((model.gpu_type, model.region))
        if price is not None and model.gpus_per_worker:
            model.price_per_hour = round(price * model.gpus_per_worker, 2)
    return result


def _resolve_managed_model(client, model_slug: str | None, region: str | None) -> tuple[ManagedModel, str | None]:
    """The model to deploy and the account currency."""
    result = _fetch_managed_models(client)
    render.warn_region_errors(result.region_errors)

    region = region and normalize_serverless_region(region)
    options = [m for m in result.models if model_slug in (None, m.model_slug) and region in (None, m.region)]
    if not options:
        render.die("No matching managed model. See: jl deploy managed-models list")
    if model_slug and len(options) == 1:
        return options[0], result.currency
    if state.json_output or not render.can_prompt():
        if model_slug:
            regions = ", ".join(render.region_label(m.region) for m in options)
            render.die(f"{model_slug} is available in {regions}. Pass --region.")
        render.die("Pass a model slug. See: jl deploy managed-models list")
    return render.select_managed_model(options, result.currency), result.currency


@deploy_app.command("create")
def deploy_create(
    name: Annotated[str, typer.Option("--name", help="Deployment name.")],
    region: Annotated[str, typer.Option("--region", help="Region (IN1 or IN2).")],
    framework: Annotated[str, typer.Option("--framework", help="vllm | sglang | ollama.")],
    gpu: Annotated[str, typer.Option("--gpu", help="GPU type.")],
    gpus_per_worker: Annotated[int, typer.Option("--gpus-per-worker", help="GPUs per worker.")],
    min_workers: Annotated[int, typer.Option("--min-workers", help="Minimum workers (0-100).")],
    max_workers: Annotated[int, typer.Option("--max-workers", help="Maximum workers (0-100).")],
    idle_timeout: Annotated[int, typer.Option("--idle-timeout", help="Idle timeout in seconds (0-86400).")],
    wait_time: Annotated[
        int, typer.Option("--wait-time", help="Seconds a request waits for a free worker before it fails (0-600).")
    ],
    storage: Annotated[int, typer.Option("--storage", help="Deployment filesystem size in GB (min 50).")],
    model: Annotated[str, typer.Option("--model", help="Model id (e.g. a Hugging Face repo id).")],
    concurrent: Annotated[
        int | None, typer.Option("--concurrent", help="Concurrent requests per worker (default 1).", min=1)
    ] = None,
    arg: Annotated[
        list[str] | None,
        typer.Option("--arg", help="Repeatable framework flag as key=value (e.g. --arg max-model-len=8192)."),
    ] = None,
    env: Annotated[
        list[str] | None, typer.Option("--env", help="Repeatable env key=value. HF token via --env HF_TOKEN=...")
    ] = None,
    detach: Annotated[bool, typer.Option("--detach", help="Skip the wait: print the id and exit.")] = False,
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Create a serverless deployment and poll until running (unless --detach).

    [bold bright_yellow]Serverless is in beta. We're actively improving it and would love early feedback. Limits and behavior may change; if you hit issues or have suggestions, email support@jarvislabs.ai.[/bold bright_yellow]
    """
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    args = _parse_kv(arg, flag="--arg")
    env_dict = _parse_kv(env, flag="--env", redact=True)

    details = (
        f"name={name!r}, region={render.region_input_label(region)}, framework={framework}, "
        f"gpu={gpus_per_worker}x {gpu}, workers={min_workers}-{max_workers}, storage={storage}GB"
    )
    if not render.confirm(f"Create deployment ({details})?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Creating deployment..."):
        deployment_id = client.deployments.create(
            name=name,
            region=region,
            framework=framework,
            gpu=gpu,
            model=model,
            gpus_per_worker=gpus_per_worker,
            min_workers=min_workers,
            max_workers=max_workers,
            idle_timeout=idle_timeout,
            wait_time=wait_time,
            storage=storage,
            concurrent=concurrent,
            args=args or None,
            env=env_dict or None,
        )
    _finish_create(client, deployment_id, region, detach=detach)


@deploy_app.command("list")
def deploy_list(
    wide: Annotated[bool, typer.Option("--wide", help="Show all columns (framework, workers, concurrency).")] = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """List your deployments across all serverless regions, newest first."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching deployments..."):
        result = client.deployments.list()

    if state.json_output:
        render.print_json(result)
        return
    render.deployments_table(result, wide=wide)


@deploy_app.command("get")
def deploy_get(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    region: RegionHintOption = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show full detail of one deployment, including worker info."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching deployment..."):
        deployment = client.deployments.get(deployment_id, region=region)

    base_url = None
    if deployment.status == "running":
        base_url = client.deployments.openai_base_url(deployment_id, region=deployment.region)

    if state.json_output:
        payload = deployment.model_dump()
        if base_url:
            payload["openai_base_url"] = base_url
        render.print_json(payload)
        return
    render.deployment_detail(deployment, base_url=base_url)


@deploy_app.command("status")
def deploy_status(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    region: RegionHintOption = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Status check for one deployment."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Checking status..."):
        status = client.deployments.status(deployment_id, region=region)

    if state.json_output:
        render.print_json(status)
        return
    render.deployment_status_line(status)


@deploy_app.command("logs")
def deploy_logs(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    region: RegionHintOption = None,
    tail: Annotated[int, typer.Option("--tail", help="Last N lines from each worker (0 = live only).")] = 100,
    follow: Annotated[
        bool, typer.Option("--follow/--no-follow", "-f", help="Keep streaming new lines (Ctrl-C to stop).")
    ] = True,
    worker: Annotated[
        int | None, typer.Option("--worker", help="Only this worker id (ids shown in `jl deploy get`).")
    ] = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Stream a deployment's logs, live from its running workers (logs are not stored)."""
    cli_options.apply_command_options(json_output=json_output)
    if state.json_output:
        render.die("jl deploy logs doesn't support --json. Use --no-follow for a one-time read.")

    client = get_client()
    # Read the record before streaming: it resolves the region once (so logs()
    # skips its own search) and carries the status used to explain an empty
    # stream — no parsing it out of notice text.
    with render.spinner("Fetching deployment..."):
        deployment = client.deployments.get(deployment_id, region=region)

    events = client.deployments.logs(deployment_id, region=deployment.region, tail=tail, follow=follow, worker=worker)
    try:
        for kind, text in events:
            if kind == "log":
                print(text, flush=True)  # noqa: T201 - deployment logs intentionally stream raw lines to stdout.
                continue
            if text == "no active workers":
                text = _no_workers_message(deployment.status, deployment_id)
            render.info(text)
    except KeyboardInterrupt:
        raise typer.Exit() from None
    except BrokenPipeError:
        # The reader went away (e.g. `| head`); exit quietly. Point stdout at
        # devnull so the interpreter's final flush doesn't complain.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise typer.Exit() from None


def _no_workers_message(status: str, deployment_id: str) -> str:
    """Why the log stream is empty, in terms of the deployment's status."""
    if status in ("failed", "cleaning"):
        return (
            "No running workers — the deployment failed; logs stream live and are not stored. "
            f"For the failure reason: jl deploy get {deployment_id}"
        )
    if status == "deleting":
        return "No running workers — the deployment is being deleted."
    return "No workers up yet — logs start once a worker is running. Try again shortly."


@deploy_app.command("update")
def deploy_update(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    name: Annotated[str | None, typer.Option("--name", help="New name.")] = None,
    idle_timeout: Annotated[int | None, typer.Option("--idle-timeout", help="New idle timeout in seconds.")] = None,
    wait_time: Annotated[
        int | None,
        typer.Option(
            "--wait-time", help="New wait time: seconds a request waits for a free worker (custom deployments only)."
        ),
    ] = None,
    min_workers: Annotated[int | None, typer.Option("--min-workers", help="New minimum workers.")] = None,
    max_workers: Annotated[int | None, typer.Option("--max-workers", help="New maximum workers.")] = None,
    concurrent: Annotated[int | None, typer.Option("--concurrent", help="New concurrent requests per worker.")] = None,
    region: RegionHintOption = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Change a running deployment: name, idle timeout, scaling, or wait time.

    To change the GPU, model, or storage, delete it and create a new one.

    [bold]Examples:[/bold]

      [dim]# Scale to zero: no GPU cost while idle[/dim]
      [cyan]jl deploy update <id> --min-workers 0[/cyan]

      [dim]# Keep one worker warm and allow up to three[/dim]
      [cyan]jl deploy update <id> --min-workers 1 --max-workers 3[/cyan]
    """
    cli_options.apply_command_options(json_output=json_output)
    if all(value is None for value in (name, idle_timeout, wait_time, min_workers, max_workers, concurrent)):
        render.die(
            "Pass at least one of --name, --idle-timeout, --wait-time, --min-workers, --max-workers, --concurrent."
        )

    client = get_client()
    with render.spinner("Updating deployment..."):
        deployment = client.deployments.update(
            deployment_id,
            name=name,
            idle_timeout=idle_timeout,
            wait_time=wait_time,
            min_workers=min_workers,
            max_workers=max_workers,
            concurrent=concurrent,
            region=region,
        )

    if state.json_output:
        render.print_json(deployment)
        return
    render.deployment_detail(deployment)


@deploy_app.command("delete")
def deploy_delete(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    yes: cli_options.YesOption = False,
    region: RegionHintOption = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Delete a deployment and stop its billing."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)

    if not render.confirm(f"Delete deployment {deployment_id}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Deleting deployment..."):
        client.deployments.delete(deployment_id, region=region)

    if state.json_output:
        render.print_json({"success": True, "deployment_id": deployment_id})
        return
    render.success(f"Deleting deployment {deployment_id}.")


@deploy_app.command("chat")
def deploy_chat(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    prompt: Annotated[str | None, typer.Argument(help="Message to send. Omit it for an interactive chat.")] = None,
    system: Annotated[str | None, typer.Option("--system", help="System prompt.")] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Maximum output tokens (default: the model server's default).", min=1),
    ] = None,
    image: ImageOption = None,
    region: RegionHintOption = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Test a running deployment: send it a message, or chat with it.

    The reply streams to stdout; thinking and token usage go to stderr.
    If no worker is ready, it doesn't wait or send your message: it starts a worker (unless one is already
    starting) and exits with an error. Try again in a few minutes.

    [bold]Examples:[/bold]

      [dim]# One reply[/dim]
      [cyan]jl deploy chat <id> "Say hello"[/cyan]

      [dim]# Interactive chat[/dim]
      [cyan]jl deploy chat <id>[/cyan]

      [dim]# Full OpenAI-style response as JSON[/dim]
      [cyan]jl deploy chat <id> "Hi" --json[/cyan]

      [dim]# With an image[/dim]
      [cyan]jl deploy chat <id> "What is in this picture?" --image photo.jpg[/cyan]

    In interactive chat: Ctrl-C stops a reply · Ctrl-D or /exit quits.
    """
    cli_options.apply_command_options(json_output=json_output)
    text = read_prompt(prompt)
    if image and not text:
        render.die("Pass a prompt with --image. See: jl deploy chat --help")
    content = user_content(text, image)
    if not text and (state.json_output or not render.can_prompt()):
        render.die("Pass a prompt. See: jl deploy chat --help")
    client = get_client()
    with render.spinner("Fetching deployment..."):
        deployment = client.deployments.get(deployment_id, region=region)
    if deployment.status != "running":
        render.die(
            f"Deployment {deployment_id} is {deployment.status}; it must be running. Check: jl deploy get {deployment_id}"
        )

    model = deployment.served_model
    base_url = client.deployments.openai_base_url(deployment_id, region=deployment.region)
    workers = deployment.workers
    if not workers.healthy:
        check = f"Try again shortly. Check progress: jl deploy get {deployment_id}"
        if workers.provisioning or workers.total:
            render.die(f"This deployment is starting a worker. This can take a few minutes. {check}")
        # A worker starts only when a request arrives, so send a tiny one (never the user's prompt) and stop.
        with render.spinner("Starting a worker..."):
            client.models.chat_submit(model, [{"role": "user", "content": "ping"}], base_url=base_url, max_tokens=1)
        render.die(
            f"No worker was running, so this deployment is starting one now. This can take a few minutes. {check}"
        )

    params = {"base_url": base_url, **({"max_tokens": max_tokens} if max_tokens else {})}
    messages = [{"role": "system", "content": system}] if system else []

    def send(history: list[dict]) -> dict | None:
        return stream_reply(client.models.chat_stream(model, history, **params))

    if not text:
        interactive_chat(f"{deployment.name or deployment_id} ({model})", messages, send)
        return

    messages.append({"role": "user", "content": content})
    if state.json_output:
        render.print_json(client.models.chat(model, messages, **params))
        return
    if send(messages) is None:
        raise typer.Exit(1)


@managed_app.command("list")
def managed_models_list(json_output: cli_options.JsonOption = False) -> None:
    """List the managed models you can deploy, across all serverless regions."""
    cli_options.apply_command_options(json_output=json_output)
    result = _fetch_managed_models(get_client())

    if state.json_output:
        render.print_json(result)
        return
    render.managed_models_table(result)


@managed_app.command("create")
def managed_models_create(
    model_slug: Annotated[
        str | None,
        typer.Argument(
            help="Managed model slug. Omit it to pick from a list; you are then asked for each setting you didn't pass."
        ),
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="Deployment name (default: the model slug).")] = None,
    region: Annotated[
        str | None, typer.Option("--region", help="Region (IN1 or IN2). Optional: found from the model when omitted.")
    ] = None,
    min_workers: Annotated[int, typer.Option("--min-workers", help="Minimum workers.")] = 0,
    max_workers: Annotated[int, typer.Option("--max-workers", help="Maximum workers.")] = 1,
    idle_timeout: Annotated[int, typer.Option("--idle-timeout", help="Idle timeout in seconds.")] = 300,
    concurrent: Annotated[
        int | None,
        typer.Option("--concurrent", help="Concurrent requests per worker (default: the model's tested value).", min=1),
    ] = None,
    detach: Annotated[bool, typer.Option("--detach", help="Skip the wait: print the id and exit.")] = False,
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Deploy a managed model and poll until running (unless --detach).

    The model's recipe sets the GPU, framework, and engine settings; you choose scaling.

    [bold bright_yellow]Serverless is in beta. We're actively improving it and would love early feedback. Limits and behavior may change; if you hit issues or have suggestions, email support@jarvislabs.ai.[/bold bright_yellow]
    """
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    client = get_client()
    model, currency = _resolve_managed_model(client, model_slug, region)
    name = name or model.model_slug

    if model_slug is None:  # Picked from the list: ask for each setting not given as a flag.
        render.managed_settings_help()
        name = render.ask_value("name", "Deployment name:", name)
        # The server checks the limits; the prompts only ask for whole numbers.
        min_workers = render.ask_value("min_workers", "Min workers:", min_workers, low=0)
        # Starting from max(min, 1) prefills a usable value when min was raised above the default.
        max_workers = render.ask_value("max_workers", "Max workers:", max_workers, low=max(min_workers, 1))
        idle_timeout = render.ask_value("idle_timeout", "Idle timeout (seconds):", idle_timeout, low=0)
        concurrent = render.ask_value(
            "concurrent",
            "Concurrent requests per worker:",
            concurrent or model.concurrent_requests or 1,
            low=1,
        )

    details = (
        f"model={model.model_slug}, gpu={model.gpus_per_worker}x {model.gpu_type}/worker, name={name!r}, "
        f"region={render.region_label(model.region)}, workers={min_workers}-{max_workers}, "
        f"idle_timeout={idle_timeout}s, concurrent={concurrent or model.concurrent_requests}, "
        f"cost={render.managed_cost_label(model, currency, max_workers)}"
    )
    if not render.confirm(f"Create deployment ({details})?", skip=state.yes):
        raise typer.Exit()

    with render.spinner("Creating deployment..."):
        deployment_id = client.deployments.create_managed(
            model.model_slug,
            name=name,
            region=model.region,
            min_workers=min_workers,
            max_workers=max_workers,
            idle_timeout=idle_timeout,
            concurrent=concurrent,
        )
    _finish_create(client, deployment_id, model.region, detach=detach)
