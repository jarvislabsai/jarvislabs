"""Output helpers for the CLI — tables, JSON, errors, confirmations."""

from __future__ import annotations

import json
import sys
from typing import Any

from pydantic import BaseModel
from rich import box
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from jarvislabs.constants import REGION_CODE_TO_REGION, REGION_DISPLAY_CODES

TABLE_BOX = box.ROUNDED
HEADER_STYLE = "bold"
TITLE_STYLE = "bold"
BORDER_STYLE = "dim"

theme = Theme(
    {
        "info": "dim",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "title": "bold",
    }
)

console = Console(stderr=True, theme=theme)
stdout_console = Console(theme=theme)


# ── JSON output ──────────────────────────────────────────────────────────────


def jsonable(data: Any) -> Any:
    """Convert Pydantic models inside lists or dicts before JSON printing."""
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, list):
        return [jsonable(item) for item in data]
    if isinstance(data, dict):
        return {key: jsonable(value) for key, value in data.items()}
    return data


def print_json(data: list[BaseModel] | BaseModel | dict) -> None:
    raw = jsonable(data)
    stdout_console.print_json(json.dumps(raw, default=str))


# ── Account Status ────────────────────────────────────────────────────────────


def account_status(info, bal, metrics, sym: str) -> None:
    from rich.panel import Panel
    from rich.text import Text

    content = Text()
    content.append(f"{info.name}", style="bold")
    content.append(f"  {info.user_id}\n", style="cyan")
    content.append("\n")
    content.append("Balance  ", style="dim")
    content.append(f"{sym}{bal.balance:.2f}", style="bold green")
    content.append("    Grants  ", style="dim")
    content.append(f"{sym}{bal.grants:.2f}", style="bold yellow")
    content.append("\n")
    content.append("Running  ", style="dim")
    content.append(f"{metrics.running_instances}", style="bold green")
    content.append("         Paused  ", style="dim")
    content.append(f"{metrics.paused_instances}", style="bold yellow")

    panel = Panel(
        content,
        title="[bold cyan]⚡ Account[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=False,
    )
    stdout_console.print(panel)


# ── Tables ───────────────────────────────────────────────────────────────────


def _table(title: str | None = None, **kwargs) -> Table:
    """Create a table with standard styling."""
    return Table(
        title=title,
        box=TABLE_BOX,
        title_style=TITLE_STYLE,
        header_style=HEADER_STYLE,
        border_style=BORDER_STYLE,
        **kwargs,
    )


def instances_table(instances: list, currency: str = "USD") -> None:
    if not instances:
        info("No instances found.")
        return

    sym = "₹" if currency == "INR" else "$"

    table = _table("Instances", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Resource", style="bold", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Storage", justify="right")
    table.add_column("Region", no_wrap=True)
    table.add_column("Cost", justify="right")
    table.add_column("Template")

    for inst in instances:
        status_style = _status_style(inst.status)
        table.add_row(
            str(inst.machine_id),
            inst.name or "—",
            f"[{status_style}]{inst.status}[/{status_style}]",
            resource_label(inst),
            instance_type(inst),
            f"{inst.storage_gb}GB" if inst.storage_gb else "—",
            region_label(inst.region),
            _cost_cell(inst, sym),
            inst.template,
        )

    stdout_console.print(table)


def _link_value(url: str | None) -> str:
    return f"[link={url}][magenta]{url}[/magenta][/link]" if url else "—"


def _service_url_rows(inst) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if inst.url:
        rows.append(("URL", _link_value(inst.url)))

    http_ports = [port.strip() for port in (inst.http_ports or "").split(",") if port.strip()]

    if inst.vs_url:
        rows.append(("Port 7007", _link_value(inst.vs_url)))

    endpoint_urls = list(inst.endpoints or [])
    if endpoint_urls:
        rows.append(("Port 6006", _link_value(endpoint_urls[0])))

    for port, endpoint in zip(http_ports, endpoint_urls[1:], strict=False):
        rows.append((f"Port {port}", _link_value(endpoint)))

    for index, endpoint in enumerate(endpoint_urls[1 + len(http_ports) :], start=1 + len(http_ports)):
        rows.append((f"Endpoint {index}", _link_value(endpoint)))

    return rows


def instance_detail(inst, currency: str = "USD") -> None:
    sym = "₹" if currency == "INR" else "$"
    table = Table(show_header=False, box=None, padding=(0, 2), border_style=BORDER_STYLE)
    table.add_column("Field", style="dim")
    # Avoid cutting off long values like notebook URLs with auth tokens.
    table.add_column("Value", overflow="fold")

    status_style = _status_style(inst.status)

    cost_label = "Storage cost" if inst.status == "Paused" else "Session cost"

    rows = [
        ("ID", f"[cyan]{inst.machine_id}[/cyan]"),
        ("Name", f"[bold]{inst.name or '—'}[/bold]"),
        ("Status", f"[{status_style}]{inst.status}[/{status_style}]"),
        ("Resource", f"[bold]{resource_label(inst)}[/bold]"),
        ("Type", instance_type(inst)),
        ("Template", inst.template),
        ("Storage", f"{inst.storage_gb}GB" if inst.storage_gb else "—"),
        ("Region", region_label(inst.region)),
        (cost_label, f"[green]{sym}{inst.cost:.2f}[/green]"),
        ("SSH", f"[cyan]{inst.ssh_command}[/cyan]" if inst.ssh_command else "—"),
    ]

    # Public IP for running VMs only (paused VMs have stale, released IPs)
    _ip = (inst.public_ip or "").strip()
    if inst.template == "vm" and inst.status == "Running" and _ip:
        rows.append(("Public IP", f"[cyan]{_ip}[/cyan]"))

    # VMs are SSH-only, skip the always-empty HTTP Ports row for them
    if inst.template != "vm":
        rows.append(("HTTP Ports", inst.http_ports or "—"))

    rows.extend(_service_url_rows(inst))

    for field, value in rows:
        table.add_row(field, value)

    stdout_console.print(table)


def ssh_keys_table(keys: list) -> None:
    if not keys:
        info("No SSH keys found.")
        return

    table = _table("SSH Keys")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Key", style="dim", max_width=50)

    for key in keys:
        display_key = key.ssh_key[:40] + "..." if len(key.ssh_key) > 40 else key.ssh_key
        table.add_row(key.key_id, key.key_name, display_key)

    stdout_console.print(table)


def scripts_table(scripts: list) -> None:
    if not scripts:
        info("No startup scripts found.")
        return

    table = _table("Startup Scripts")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")

    for script in scripts:
        table.add_row(str(script.script_id), script.script_name or "—")

    stdout_console.print(table)


def templates_table(templates: list) -> None:
    if not templates:
        info("No templates found.")
        return

    table = _table("Templates")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="bold")
    table.add_column("Category", style="dim")

    for template in templates:
        table.add_row(template.id, template.title, template.category or "—")

    stdout_console.print(table)


def filesystems_table(filesystems: list) -> None:
    if not filesystems:
        info("No filesystems found.")
        return

    table = _table("Filesystems")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Storage", justify="right")
    table.add_column("Region", no_wrap=True)

    for filesystem in filesystems:
        storage = f"{filesystem.storage}GB" if filesystem.storage is not None else "—"
        table.add_row(str(filesystem.fs_id), filesystem.fs_name or "—", storage, region_label(filesystem.region))

    stdout_console.print(table)


def resource_label(inst) -> str:
    """Return the compact resource text shown in list/detail views."""
    if is_cpu_vm(inst):
        cores = inst.cores or "—"
        ram = f"{inst.ram}GB" if inst.ram is not None else "—"
        return f"{cores} vCPU / {ram} RAM"
    return f"{inst.num_gpus or 1}x {inst.gpu_type or '—'}"


def is_cpu_vm(inst) -> bool:
    """Return True when an instance should be displayed as a CPU VM."""
    return getattr(inst, "template", None) == "vm" and getattr(inst, "gpu_type", None) == "CPU"


def instance_type(inst) -> str:
    """Display-only billing type. Reserved wins because backend can return both signals."""
    if getattr(inst, "committed_resource_id", None) or getattr(inst, "reservation_info", None):
        return "reserved"
    if getattr(inst, "is_spot", False):
        return "spot"
    return "on-demand"


def gpu_table(gpus: list, currency: str = "USD", *, show_legend: bool = True) -> None:
    """Render GPU container and GPU VM availability tables."""
    if not gpus:
        info("No GPU data available.")
        return

    sym = "₹" if currency == "INR" else "$"
    container_gpus = [g for g in gpus if g.workload_type in ("container", None)]
    vm_gpus = [g for g in gpus if g.workload_type in ("vm", None)]

    if container_gpus:
        _gpu_subtable(container_gpus, sym, title="GPU Containers", show_spot=True)
    if vm_gpus:
        _gpu_subtable(vm_gpus, sym, title="GPU VMs", show_spot=False)
    if show_legend:
        availability_legend(show_spot_note=bool(container_gpus))


def _gpu_subtable(gpus: list, sym: str, title: str, *, show_spot: bool) -> None:
    available = [g for g in gpus if _on_demand_available(g)]
    unavailable = [g for g in gpus if not _on_demand_available(g)]

    table = _table(title=title)
    table.add_column("", no_wrap=True)
    table.add_column("GPU", no_wrap=True)
    table.add_column("Region", no_wrap=True)
    table.add_column("VRAM", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("CPUs", justify="right")
    table.add_column(f"{sym}/hr", justify="right")
    if show_spot:
        table.add_column("Spot", justify="right")

    for gpu in available:
        spot_price = display_spot_price(gpu)
        spot_available = getattr(gpu, "num_free_devices", 0) > 0
        row = [
            "[green]●[/green]",
            f"[bold]{gpu.gpu_type}[/bold]",
            region_label(gpu.region),
            f"{gpu.vram}GB" if gpu.vram else "—",
            f"{gpu.ram_per_gpu}GB" if gpu.ram_per_gpu else "—",
            str(gpu.cpus_per_gpu) if gpu.cpus_per_gpu else "—",
            f"[green]{sym}{gpu.price_per_hour:.2f}[/green]" if gpu.price_per_hour else "—",
        ]
        if show_spot:
            if spot_price is not None and spot_available:
                row.append(f"[green]{sym}{spot_price:.2f}[/green]")
            elif spot_price is not None:
                row.append(f"[dim]{sym}{spot_price:.2f}[/dim]")
            else:
                row.append("—")
        table.add_row(*row)

    for gpu in unavailable:
        spot_price = display_spot_price(gpu)
        row = [
            "[dim]○[/dim]",
            f"[dim]{gpu.gpu_type}[/dim]",
            f"[dim]{region_label(gpu.region)}[/dim]",
            f"[dim]{gpu.vram}GB[/dim]" if gpu.vram else "[dim]—[/dim]",
            f"[dim]{gpu.ram_per_gpu}GB[/dim]" if gpu.ram_per_gpu else "[dim]—[/dim]",
            f"[dim]{gpu.cpus_per_gpu}[/dim]" if gpu.cpus_per_gpu else "[dim]—[/dim]",
            f"[dim]{sym}{gpu.price_per_hour:.2f}[/dim]" if gpu.price_per_hour else "[dim]—[/dim]",
        ]
        if show_spot:
            row.append(f"[dim]{sym}{spot_price:.2f}[/dim]" if spot_price is not None else "[dim]—[/dim]")
        table.add_row(*row)

    stdout_console.print(table)


def _on_demand_available(gpu) -> bool:
    free = getattr(gpu, "effective_num_free_devices", None)
    if free is None:
        free = getattr(gpu, "num_free_devices", 0)
    return free > 0


def display_spot_price(gpu) -> float | None:
    """Return a spot price only for GPU container rows."""
    if getattr(gpu, "workload_type", None) == "vm":
        return None
    return getattr(gpu, "spot_price", None)


def availability_legend(*, show_spot_note: bool = False) -> None:
    """Print the shared available/unavailable marker legend."""
    stdout_console.print("[green]●[/green] available  [dim]○ unavailable[/dim]")
    if show_spot_note:
        stdout_console.print("[dim]dim Spot price = spot price exists, but no spot capacity is free right now[/dim]")


def cpu_vm_table(cpu_meta: dict, currency: str = "USD", *, show_legend: bool = True) -> None:
    """Render CPU VM sizes from backend cpu_meta combinations."""
    combinations = cpu_meta.get("combinations") or []
    if not combinations:
        info("No CPU VM data available.")
        return

    sym = "₹" if currency == "INR" else "$"
    rows = []
    for combo in combinations:
        regions = combo.get("regions") or {}
        for region, available in regions.items():
            rows.append(
                (
                    int(combo.get("vcpus") or 0),
                    int(combo.get("ram_gb") or 0),
                    region,
                    bool(available),
                    combo.get("price_per_hour", combo.get("price")),
                )
            )

    if not rows:
        info("No CPU VM data available.")
        return

    rows.sort(key=lambda row: (row[0], row[1], region_label(row[2])))
    table = _table(title="CPU VMs")
    table.add_column("", no_wrap=True)
    table.add_column("vCPUs", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("Region", no_wrap=True)
    table.add_column(f"{sym}/hr", justify="right")

    for vcpus, ram_gb, region, available, price in rows:
        marker = "[green]●[/green]" if available else "[dim]○[/dim]"
        price_cell = f"{sym}{float(price):.2f}" if price is not None else "—"
        if available:
            table.add_row(marker, str(vcpus), f"{ram_gb}GB", region_label(region), f"[green]{price_cell}[/green]")
        else:
            table.add_row(
                marker,
                f"[dim]{vcpus}[/dim]",
                f"[dim]{ram_gb}GB[/dim]",
                f"[dim]{region_label(region)}[/dim]",
                f"[dim]{price_cell}[/dim]",
            )

    stdout_console.print(table)
    if show_legend:
        availability_legend()


# ── Messages ─────────────────────────────────────────────────────────────────


def success(msg: str) -> None:
    from jarvislabs.cli import state

    if state.json_output:
        return
    console.print(f"[green]✓[/green] {msg}")


def error(msg: str) -> None:
    from jarvislabs.cli import state

    if state.json_output:
        return
    console.print(f"[red]✗[/red] {msg}", style="red")


def info(msg: str) -> None:
    from jarvislabs.cli import state

    if state.json_output:
        return
    console.print(f"[dim]{msg}[/dim]")


def warning(msg: str) -> None:
    from jarvislabs.cli import state

    if state.json_output:
        return
    console.print(f"[yellow]![/yellow] {msg}")


# ── Confirmation ─────────────────────────────────────────────────────────────


def confirm(msg: str, *, skip: bool = False) -> bool:
    """Ask for confirmation. Returns True if confirmed or skip=True (--yes flag)."""
    if skip:
        return True
    try:
        response = console.input(f"[yellow]?[/yellow] {msg} [dim]\\[y/N][/dim] ")
        return response.strip().lower() in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        console.print()
        return False


# ── Spinner ──────────────────────────────────────────────────────────────────


def spinner(msg: str):
    """Rich spinner context manager for wrapping API calls. Suppressed in --json mode."""
    from contextlib import nullcontext

    from jarvislabs.cli import state

    if state.json_output:
        return nullcontext()
    return console.status(f"[bold]{msg}[/bold]", spinner="dots")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _cost_cell(inst, sym: str) -> str:
    """Format cost for table display with contextual color."""
    if inst.cost <= 0:
        return "[dim]—[/dim]"
    return f"[green]{sym}{inst.cost:.2f}[/green]"


def region_label(region: str | None) -> str:
    """Convert backend region IDs to short CLI labels like IN2 and EU1."""
    if not region:
        return "—"
    return REGION_DISPLAY_CODES.get(region, region)


def region_input_label(region: str | None, *, default: str | None = None) -> str:
    """Convert user region input or backend region IDs to short CLI labels."""
    if not region:
        return region_label(default)
    normalized = region.strip()
    internal = REGION_CODE_TO_REGION.get(normalized.lower(), normalized)
    return region_label(internal)


def _status_style(status: str) -> str:
    """Map instance status to a Rich color."""
    return {
        "Running": "green",
        "Paused": "yellow",
        "Failed": "red",
        "Creating": "blue",
        "Resuming": "blue",
        "Pausing": "yellow",
        "Destroying": "red",
    }.get(status, "white")


def die(msg: str, code: int = 1) -> None:
    """Print error and exit. Emits JSON to stdout when --json is active."""
    from jarvislabs.cli import state

    if state.json_output:
        print_json({"error": msg})
    else:
        error(msg)
    sys.exit(code)
