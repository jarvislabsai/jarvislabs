"""Machine commands — lifecycle, SSH, exec, and file transfer (registered on root app)."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import typer

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.exceptions import SSHError, ValidationError
from jarvislabs.ssh import (
    build_remote_shell_command,
    build_scp_command,
    harden_ssh_parts,
    parse_ssh_command,
    split_ssh_command,
)

if TYPE_CHECKING:
    from jarvislabs.models import Instance

_MACHINE_PANEL = "Machine Management"
_ACCESS_PANEL = "Remote Access"


def _resolve_ssh(machine_id: int) -> tuple[Instance, list[str]]:
    client = get_client()
    with render.spinner("Fetching instance..."):
        inst = client.instances.get(machine_id)

    if inst.status != "Running":
        if inst.status == "Paused":
            render.die(f"Instance {machine_id} is paused. Resume it first: jl resume {machine_id}")
        if inst.status in {"Creating", "Resuming"}:
            render.die(f"Instance {machine_id} is not ready yet (status: {inst.status}). Wait for it to reach Running.")
        render.die(f"Instance {machine_id} is not available for SSH (status: {inst.status}).")

    if not inst.ssh_command:
        render.die(f"Instance {machine_id} has no SSH command (status: {inst.status}).")

    try:
        return inst, harden_ssh_parts(split_ssh_command(inst.ssh_command))
    except SSHError:
        render.die(f"Cannot parse SSH command: {inst.ssh_command}")


def _remote_home(ssh_command: str | None) -> str:
    """Derive the remote home/workspace directory from the SSH command.

    Containers run as root and use /home as a workspace.
    VMs run as a regular user (ubuntu, cloud, …) whose home is /home/<user>.
    """
    if ssh_command:
        try:
            info = parse_ssh_command(ssh_command)
            if info.user and info.user != "root":
                return f"/home/{info.user}"
        except SSHError:
            pass
    return "/home"


def _default_upload_dest(source: Path, ssh_command: str | None = None) -> str:
    name = source.name or source.resolve().name
    return f"{_remote_home(ssh_command)}/{name}"


def _default_download_dest(source: str) -> str:
    cleaned = source.rstrip("/")
    name = PurePosixPath(cleaned).name
    if not name:
        raise ValueError(f"Cannot infer a local destination from remote path: {source}")
    return name


@app.command("list", rich_help_panel=_MACHINE_PANEL)
def instance_list(
    json_output: cli_options.JsonOption = False,
) -> None:
    """List all instances."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching instances..."):
        instances = client.instances.list()
        currency = client.account.currency()

    if state.json_output:
        render.print_json(instances)
        return

    render.instances_table(instances, currency)


@app.command("get", rich_help_panel=_MACHINE_PANEL)
def instance_get(
    machine_id: int = typer.Argument(..., help="Instance ID."),
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show details of a specific instance."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching instance..."):
        inst = client.instances.get(machine_id)
        currency = client.account.currency()

    if state.json_output:
        render.print_json(inst)
        return

    render.instance_detail(inst, currency)


@app.command("create", rich_help_panel=_MACHINE_PANEL)
def instance_create(
    gpu: str = typer.Option(..., "--gpu", "-g", help="GPU type (e.g. H100, A100, L4)."),
    vm: bool = typer.Option(False, "--vm", help="Create a VM instance (SSH-only, no container)."),
    template: str = typer.Option("pytorch", "--template", "-t", help="Framework template for container instances."),
    storage: int = typer.Option(40, "--storage", "-s", help="Storage in GB."),
    name: str = typer.Option("Name me", "--name", "-n", help="Instance name."),
    num_gpus: int = typer.Option(1, "--num-gpus", help="Number of GPUs."),
    region: str | None = typer.Option(None, "--region", help="Optional region pin (e.g. IN1, IN2, EU1)."),
    http_ports: str = typer.Option("", "--http-ports", help="Comma-separated HTTP ports to expose (e.g. 7860,8080)."),
    script_id: str | None = typer.Option(None, "--script-id", help="Startup script ID to run on launch."),
    script_args: str = typer.Option("", "--script-args", help="Arguments passed to startup script."),
    fs_id: int | None = typer.Option(None, "--fs-id", help="Filesystem ID to attach."),
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Create a new GPU instance (container or VM)."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)

    # Handle --vm flag
    if vm:
        if template != "pytorch":
            render.die("--vm and --template cannot be used together.")
        template = "vm"
        if storage == 40:
            storage = 100
        if http_ports:
            render.die("--http-ports is not supported with --vm. VMs are SSH-only.")
    if template.strip().lower() == "vm" and not vm:
        render.die("Use --vm instead of --template vm.")

    details = [f"gpu={num_gpus}x {gpu}", f"template={template}", f"storage={storage}GB", f"name={name!r}"]
    if region:
        details.append(f"region={region.upper()}")
    if http_ports:
        details.append(f"http_ports={http_ports!r}")
    if script_id:
        details.append(f"script_id={script_id}")
    if script_args:
        details.append(f"script_args={script_args!r}")
    if fs_id is not None:
        details.append(f"fs_id={fs_id}")
    prompt = f"Create instance ({', '.join(details)})?"
    if not render.confirm(prompt, skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Creating instance — this may take a few seconds..."):
        inst = client.instances.create(
            gpu_type=gpu,
            num_gpus=num_gpus,
            template=template,
            storage=storage,
            name=name,
            region=region,
            http_ports=http_ports,
            script_id=script_id,
            script_args=script_args,
            fs_id=fs_id,
        )

    if state.json_output:
        render.print_json(inst)
        return

    render.success(f"Instance {inst.machine_id} is Running.")
    render.instance_detail(inst, client.account.currency())


@app.command("rename", rich_help_panel=_MACHINE_PANEL)
def instance_rename(
    machine_id: int = typer.Argument(..., help="Instance ID to rename."),
    name: str = typer.Option(..., "--name", "-n", help="New instance name."),
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Rename an instance."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    if not render.confirm(f"Rename instance {machine_id} to {name!r}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Renaming instance..."):
        client.instances.rename(machine_id, name)

    if state.json_output:
        render.print_json({"success": True, "machine_id": machine_id, "name": name})
        return

    render.success(f"Instance {machine_id} renamed to {name!r}.")


@app.command("pause", rich_help_panel=_MACHINE_PANEL)
def instance_pause(
    machine_id: int = typer.Argument(..., help="Instance ID to pause."),
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Pause a running instance."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    client = get_client()
    with render.spinner("Checking instance..."):
        client.instances.get(machine_id)

    if not render.confirm(f"Pause instance {machine_id}?", skip=state.yes):
        raise typer.Exit()

    with render.spinner("Pausing instance..."):
        client.instances.pause(machine_id)

    if state.json_output:
        render.print_json({"success": True, "machine_id": machine_id})
        return

    render.success(f"Instance {machine_id} paused.")


@app.command("resume", rich_help_panel=_MACHINE_PANEL)
def instance_resume(
    machine_id: int = typer.Argument(..., help="Instance ID to resume."),
    gpu: str | None = typer.Option(None, "--gpu", "-g", help="Resume with a different GPU type."),
    num_gpus: int | None = typer.Option(None, "--num-gpus", help="Change number of GPUs."),
    storage: int | None = typer.Option(None, "--storage", "-s", help="Expand storage (GB). Can only increase."),
    name: str | None = typer.Option(None, "--name", "-n", help="Rename instance."),
    http_ports: str = typer.Option("", "--http-ports", help="Comma-separated HTTP ports to expose (e.g. 7860,8080)."),
    script_id: str | None = typer.Option(None, "--script-id", help="Startup script ID to use on resume."),
    script_args: str | None = typer.Option(None, "--script-args", help="Arguments passed to startup script."),
    fs_id: int | None = typer.Option(None, "--fs-id", help="Filesystem ID to attach."),
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Resume a paused instance. Optionally swap GPU, expand storage, or rename."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    changes: list[str] = []
    if gpu:
        changes.append(f"gpu={gpu}")
    if num_gpus is not None:
        changes.append(f"num_gpus={num_gpus}")
    if storage is not None:
        changes.append(f"storage={storage}GB")
    if name is not None:
        changes.append(f"name={name!r}")
    if http_ports:
        changes.append(f"http_ports={http_ports!r}")
    if script_id is not None:
        changes.append(f"script_id={script_id}")
    if script_args is not None:
        changes.append(f"script_args={script_args!r}")
    if fs_id is not None:
        changes.append(f"fs_id={fs_id}")

    details = ", ".join(changes) if changes else "current configuration"
    if not render.confirm(f"Resume instance {machine_id} with {details}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Resuming instance..."):
        inst = client.instances.resume(
            machine_id,
            gpu_type=gpu,
            num_gpus=num_gpus,
            storage=storage,
            name=name,
            http_ports=http_ports,
            script_id=script_id,
            script_args=script_args,
            fs_id=fs_id,
        )

    if inst.machine_id != machine_id:
        render.warning(f"Instance ID changed: {machine_id} → {inst.machine_id}")

    if state.json_output:
        render.print_json(inst)
        return

    render.success(f"Instance {inst.machine_id} is Running.")
    render.instance_detail(inst, client.account.currency())


@app.command("destroy", rich_help_panel=_MACHINE_PANEL)
def instance_destroy(
    machine_id: int = typer.Argument(..., help="Instance ID to destroy."),
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Permanently destroy an instance."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    client = get_client()
    with render.spinner("Checking instance..."):
        client.instances.get(machine_id)

    if not render.confirm(
        f"Destroy instance {machine_id}? This cannot be undone.",
        skip=state.yes,
    ):
        raise typer.Exit()

    with render.spinner("Destroying instance..."):
        client.instances.destroy(machine_id)

    if state.json_output:
        render.print_json({"success": True, "machine_id": machine_id})
        return

    render.success(f"Instance {machine_id} destroyed.")


@app.command("ssh", rich_help_panel=_ACCESS_PANEL)
def instance_ssh(
    machine_id: int = typer.Argument(..., help="Instance ID."),
    print_command: bool = typer.Option(False, "--print-command", "-p", help="Print SSH command instead of connecting."),
    json_output: cli_options.JsonOption = False,
) -> None:
    """SSH into a running instance."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching instance..."):
        inst = client.instances.get(machine_id)

    if not inst.ssh_command:
        render.die(f"Instance {machine_id} has no SSH command (status: {inst.status}).")

    if print_command:
        render.stdout_console.print(inst.ssh_command)
        return

    if state.json_output:
        render.print_json({"ssh_command": inst.ssh_command})
        return

    if inst.status != "Running":
        if inst.status == "Paused":
            render.die(f"Instance {machine_id} is paused. Resume it first: jl resume {machine_id}")
        if inst.status in {"Creating", "Resuming"}:
            render.die(f"Instance {machine_id} is not ready yet (status: {inst.status}). Wait for it to reach Running.")
        render.die(f"Instance {machine_id} is not available for SSH (status: {inst.status}).")

    try:
        parts = harden_ssh_parts(split_ssh_command(inst.ssh_command))
    except SSHError:
        render.die(f"Cannot parse SSH command: {inst.ssh_command}")

    render.info(f"Connecting to {machine_id}...")
    raise SystemExit(subprocess.call(parts))


@app.command(
    "exec",
    rich_help_panel=_ACCESS_PANEL,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def instance_exec(
    ctx: typer.Context,
    machine_id: int = typer.Argument(..., help="Instance ID."),
    json_output: cli_options.JsonOption = False,
) -> None:
    """Execute a command on a running instance."""
    cli_options.apply_command_options(json_output=json_output)
    if not ctx.args:
        render.die(f"No command specified. Use -- to separate: jl exec {machine_id} -- <command>")

    _, parts = _resolve_ssh(machine_id)
    try:
        remote_command = build_remote_shell_command(ctx.args)
    except (SSHError, ValidationError):
        render.die(f"Cannot prepare SSH command for instance {machine_id}.")

    command_label = shlex.join(ctx.args)
    parts.append(remote_command)

    if state.json_output:
        completed = subprocess.run(parts, capture_output=True, text=True, check=False)
        render.print_json(
            {
                "machine_id": machine_id,
                "command": command_label,
                "exit_code": completed.returncode,
                "stdout": getattr(completed, "stdout", ""),
                "stderr": getattr(completed, "stderr", ""),
            }
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return

    render.info(f"Executing on {machine_id}: {command_label}")
    exit_code = subprocess.call(parts)
    if exit_code != 0:
        if exit_code == 255:
            render.warning(f"SSH or remote command failed on {machine_id} (exit code 255).")
        else:
            render.warning(f"Command on {machine_id} exited with code {exit_code}.")
    raise SystemExit(exit_code)


@app.command("upload", rich_help_panel=_ACCESS_PANEL)
def instance_upload(
    machine_id: int = typer.Argument(..., help="Instance ID."),
    source: Path = typer.Argument(
        ..., exists=True, readable=True, resolve_path=True, help="Local file or directory to upload."
    ),
    dest: str | None = typer.Argument(None, help="Remote destination path. Defaults to remote home directory."),
    json_output: cli_options.JsonOption = False,
) -> None:
    """Upload a local file or directory to a running instance."""
    cli_options.apply_command_options(json_output=json_output)
    inst, ssh_parts = _resolve_ssh(machine_id)
    remote_dest = dest or _default_upload_dest(source, inst.ssh_command)
    recursive = source.is_dir()

    if dest is not None:
        if recursive:
            remote_prep = remote_dest.rstrip("/") or remote_dest
        else:
            remote_prep = PurePosixPath(remote_dest).parent.as_posix()
        prep_command = build_remote_shell_command(["mkdir", "-p", remote_prep])
        if subprocess.call([*ssh_parts, prep_command]) != 0:
            render.die(f"Failed to prepare remote destination {remote_prep}.")

    try:
        parts = build_scp_command(
            inst.ssh_command,
            source=str(source),
            dest=remote_dest,
            upload=True,
            recursive=recursive,
        )
    except SSHError:
        render.die(f"Cannot prepare upload command for instance {machine_id}.")

    if state.json_output:
        completed = subprocess.run(parts, capture_output=True, text=True, check=False)
        render.print_json(
            {
                "machine_id": machine_id,
                "direction": "upload",
                "source": str(source),
                "dest": remote_dest,
                "recursive": recursive,
                "exit_code": completed.returncode,
            }
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return

    render.info(f"Uploading to {machine_id}: {source} -> {remote_dest}")
    raise SystemExit(subprocess.call(parts))


@app.command("download", rich_help_panel=_ACCESS_PANEL)
def instance_download(
    machine_id: int = typer.Argument(..., help="Instance ID."),
    source: str = typer.Argument(..., help="Remote file or directory to download."),
    dest: Path | None = typer.Argument(None, resolve_path=True, help="Local destination path. Defaults to ./<name>."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Download directories recursively."),
    json_output: cli_options.JsonOption = False,
) -> None:
    """Download a remote file or directory from a running instance."""
    cli_options.apply_command_options(json_output=json_output)
    inst, _ = _resolve_ssh(machine_id)

    try:
        local_dest = dest or Path(_default_download_dest(source))
    except ValueError as exc:
        render.die(str(exc))

    try:
        parts = build_scp_command(
            inst.ssh_command,
            source=source,
            dest=str(local_dest),
            upload=False,
            recursive=recursive,
        )
    except SSHError:
        render.die(f"Cannot prepare download command for instance {machine_id}.")

    if state.json_output:
        completed = subprocess.run(parts, capture_output=True, text=True, check=False)
        render.print_json(
            {
                "machine_id": machine_id,
                "direction": "download",
                "source": source,
                "dest": str(local_dest),
                "recursive": recursive,
                "exit_code": completed.returncode,
            }
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return

    render.info(f"Downloading from {machine_id}: {source} -> {local_dest}")
    raise SystemExit(subprocess.call(parts))
