"""Serverless model deployment commands — `jl deploy`.

Thin wrapper over ``client.deployments``: parse flags, call the SDK, render output.
"""

from __future__ import annotations

import typer

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.cli.options import value_or_default

deploy_app = typer.Typer(name="deploy", help="Create and manage serverless model deployments.")
app.add_typer(deploy_app, rich_help_panel="Workloads")


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


@deploy_app.command("create")
def deploy_create(
    name: str = typer.Option(..., "--name", help="Deployment name."),
    region: str = typer.Option(..., "--region", help="Region (IN1 or IN2)."),
    framework: str = typer.Option(..., "--framework", help="vllm | sglang | ollama."),
    gpu: str = typer.Option(..., "--gpu", help="GPU type (v1: H100 | L4)."),
    gpus_per_worker: int = typer.Option(..., "--gpus-per-worker", help="GPUs per worker."),
    min_workers: int = typer.Option(..., "--min-workers", help="Minimum workers (0-100)."),
    max_workers: int = typer.Option(..., "--max-workers", help="Maximum workers (0-100)."),
    idle_timeout: int = typer.Option(..., "--idle-timeout", help="Idle timeout in seconds (0-86400)."),
    wait_time: int = typer.Option(..., "--wait-time", help="Wait time in seconds (0-600)."),
    storage: int = typer.Option(
        ...,
        "--storage",
        help="Deployment filesystem size in GB.",
    ),
    model: str = typer.Option(..., "--model", help="Model id (e.g. a Hugging Face repo id)."),
    concurrent: int | None = typer.Option(None, "--concurrent", help="Concurrent requests per worker (default 1)."),
    arg: list[str] | None = typer.Option(
        None,
        "--arg",
        help="Repeatable framework flag as key=value (e.g. --arg max-model-len=8192).",
    ),
    env: list[str] | None = typer.Option(
        None, "--env", help="Repeatable env key=value. HF token via --env HF_TOKEN=..."
    ),
    detach: bool = typer.Option(False, "--detach", help="Skip the wait: print the id and exit."),
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Create a serverless deployment and poll until running (unless --detach).

    Use --arg to pass extra framework flags as key=value.
    """
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    concurrent = value_or_default(concurrent, None)
    args = _parse_kv(value_or_default(arg, None), flag="--arg")
    env_dict = _parse_kv(value_or_default(env, None), flag="--env", redact=True)

    details = (
        f"name={name!r}, region={render.region_input_label(region)}, framework={framework}, "
        f"gpu={gpus_per_worker}x {gpu}, workers={min_workers}-{max_workers}, storage={storage}GB"
    )
    if not render.confirm(f"Create deployment ({details})?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
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
        wait=False,
    )
    # Print the id immediately so Ctrl-C can never orphan it silently (human mode only).
    if not state.json_output:
        render.info(f"Deployment {deployment_id} created.")

    if detach:
        if state.json_output:
            render.print_json({"deployment_id": deployment_id, "region": render.region_input_label(region)})
            return
        render.info(f"Detached. Check: jl deploy get {deployment_id} · Delete: jl deploy delete {deployment_id}")
        return

    try:
        with render.spinner("Deploying..."):
            deployment = client.deployments.wait_until_running(deployment_id, region=region)
    except KeyboardInterrupt:
        render.info(
            f"Detached. The deployment keeps running. "
            f"Check: jl deploy get {deployment_id} · Delete: jl deploy delete {deployment_id}"
        )
        raise typer.Exit() from None

    base_url = client.deployments.openai_base_url(deployment_id, region=deployment.region)
    if state.json_output:
        render.print_json({**deployment.model_dump(), "openai_base_url": base_url})
        return
    served_model = args.get("served-model-name", model)
    render.deployment_running_handoff(base_url, served_model)


@deploy_app.command("list")
def deploy_list(
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
    render.deployments_table(result)


@deploy_app.command("get")
def deploy_get(
    deployment_id: str = typer.Argument(..., help="Deployment id."),
    region: str | None = typer.Option(None, "--region", help="Region hint (fast-path)."),
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show full detail of one deployment, including worker info."""
    cli_options.apply_command_options(json_output=json_output)
    region = value_or_default(region, None)
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
    deployment_id: str = typer.Argument(..., help="Deployment id."),
    region: str | None = typer.Option(None, "--region", help="Region hint (fast-path)."),
    json_output: cli_options.JsonOption = False,
) -> None:
    """Cheap status check for one deployment."""
    cli_options.apply_command_options(json_output=json_output)
    region = value_or_default(region, None)
    client = get_client()
    with render.spinner("Checking status..."):
        status = client.deployments.status(deployment_id, region=region)

    if state.json_output:
        render.print_json(status)
        return
    render.deployment_status_line(status)


@deploy_app.command("update")
def deploy_update(
    deployment_id: str = typer.Argument(..., help="Deployment id."),
    name: str | None = typer.Option(None, "--name", help="New name."),
    idle_timeout: int | None = typer.Option(None, "--idle-timeout", help="New idle timeout in seconds."),
    wait_time: int | None = typer.Option(None, "--wait-time", help="New wait time in seconds."),
    region: str | None = typer.Option(None, "--region", help="Region hint (fast-path)."),
    json_output: cli_options.JsonOption = False,
) -> None:
    """Patch a running deployment. Only name/idle_timeout/wait_time are mutable; rescale = recreate."""
    cli_options.apply_command_options(json_output=json_output)
    name = value_or_default(name, None)
    idle_timeout = value_or_default(idle_timeout, None)
    wait_time = value_or_default(wait_time, None)
    region = value_or_default(region, None)

    client = get_client()
    with render.spinner("Updating deployment..."):
        deployment = client.deployments.update(
            deployment_id, name=name, idle_timeout=idle_timeout, wait_time=wait_time, region=region
        )

    if state.json_output:
        render.print_json(deployment)
        return
    render.deployment_detail(deployment)


@deploy_app.command("delete")
def deploy_delete(
    deployment_id: str = typer.Argument(..., help="Deployment id."),
    yes: cli_options.YesOption = False,
    region: str | None = typer.Option(None, "--region", help="Region hint (fast-path)."),
    json_output: cli_options.JsonOption = False,
) -> None:
    """Tear down a deployment."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    region = value_or_default(region, None)

    if state.json_output and not state.yes:
        render.die("--json requires --yes for delete")
    if not render.confirm(f"Delete deployment {deployment_id}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Deleting deployment..."):
        client.deployments.delete(deployment_id, region=region)

    if state.json_output:
        render.print_json({"success": True, "deployment_id": deployment_id})
        return
    render.success(f"Deleting deployment {deployment_id}.")
