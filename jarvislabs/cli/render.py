"""Output helpers for the CLI — tables, JSON, errors, confirmations."""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from typing import Any, NoReturn

import questionary
import typer
from pydantic import BaseModel
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme

from jarvislabs.cli import options as cli_options, state
from jarvislabs.constants import EUROPE_REGION, REGION_CODE_TO_REGION, REGION_DISPLAY_CODES
from jarvislabs.instances import is_cpu_vm
from jarvislabs.regions import region_code

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


def _print_detail_table(rows: list[tuple[str, str]]) -> None:
    """Print a borderless field/value detail view (shared by instance and deployment detail)."""
    table = Table(show_header=False, box=None, padding=(0, 2), border_style=BORDER_STYLE)
    table.add_column("Field", style="dim")
    # Avoid cutting off long values like notebook URLs with auth tokens.
    table.add_column("Value", overflow="fold")
    for field, value in rows:
        table.add_row(field, value)
    stdout_console.print(table)


def instances_table(instances: list, currency: str = "USD") -> None:
    if not instances:
        info("No instances found.")
        return

    sym = currency_symbol(currency)

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
    sym = currency_symbol(currency)
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
    ]

    if inst.ssh_command:
        rows.append(("SSH", f"[cyan]{inst.ssh_command}[/cyan]"))

    if inst.public_ip:
        rows.append(("Public IP", f"[cyan]{inst.public_ip}[/cyan]"))

    if inst.private_ip:
        rows.append(("Private IP", f"[cyan]{inst.private_ip}[/cyan]"))

    if inst.vpc_id:
        rows.append(("VPC", f"[cyan]{inst.vpc_id}[/cyan]"))

    # VMs are SSH-only, skip the always-empty HTTP Ports row for them
    if inst.template != "vm":
        rows.append(("HTTP Ports", inst.http_ports or "—"))

    rows.extend(_service_url_rows(inst))
    _print_detail_table(rows)


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


def vpcs_table(vpcs: list) -> None:
    if not vpcs:
        info("No VPCs found.")
        return

    # Status is omitted — listed VPCs are always active; --json includes it.
    table = _table("VPCs")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Region", no_wrap=True)
    table.add_column("CIDR", no_wrap=True)
    table.add_column("Gateway", no_wrap=True)
    table.add_column("Default", justify="center")

    for vpc in vpcs:
        table.add_row(
            vpc.vpc_id,
            escape(vpc.name) if vpc.name else "—",
            region_label(vpc.region),
            vpc.cidr or "—",
            vpc.gateway_ip or "—",
            "✓" if vpc.is_default else "—",
        )

    stdout_console.print(table)


def vpc_detail(vpc) -> None:
    rows = [
        ("ID", f"[cyan]{vpc.vpc_id}[/cyan]"),
        ("Name", f"[bold]{escape(vpc.name) if vpc.name else '—'}[/bold]"),
        ("Region", region_label(vpc.region)),
        ("CIDR", vpc.cidr or "—"),
        ("Gateway", vpc.gateway_ip or "—"),
        ("Default", "yes" if vpc.is_default else "no"),
        ("Status", vpc.status or "—"),
    ]
    _print_detail_table(rows)


def vpc_ips_table(ips: list) -> None:
    if not ips:
        info("No IPs allocated in this VPC.")
        return

    table = _table("VPC IPs")
    table.add_column("Private IP", style="cyan", no_wrap=True)
    table.add_column("Machine ID", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("MAC", style="dim", no_wrap=True)

    for ip in ips:
        style = "green" if ip.status == "attached" else "yellow"
        table.add_row(
            ip.private_ip,
            str(ip.machine_id) if ip.machine_id is not None else "—",
            f"[{style}]{ip.status}[/{style}]" if ip.status else "—",
            ip.mac_address or "—",
        )

    stdout_console.print(table)


def runs_table(rows: list) -> None:
    """Render managed runs. Rows are (run_id, machine, kind, state, lifecycle, started) strings."""
    table = _table("Managed Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Machine", style="bold")
    table.add_column("Kind")
    table.add_column("State")
    table.add_column("Lifecycle")
    table.add_column("Started")
    for row in rows:
        table.add_row(*row)
    stdout_console.print(table)


# ── Deployments ────────────────────────────────────────────────────────────────


def _deployment_status_style(status: str) -> str:
    """Map a (lowercase) deployment status to a Rich color."""
    return {
        "running": "green",
        "starting": "blue",
        "filesystem_created": "blue",
        "downloading_model": "blue",
        "model_downloaded": "blue",
        "failed": "red",
        "cleaning": "red",
        "deleting": "yellow",
        "deleted": "dim",
    }.get(status, "white")


def _gpu_label(gpus_to_use: dict | None) -> str:
    """Render the GPU type from a {'gpus': [type]} payload."""
    gpus = (gpus_to_use or {}).get("gpus") or []
    return ", ".join(str(g) for g in gpus) if gpus else "—"


def _workers_label(summary) -> str:
    lo, hi = summary.min_workers, summary.max_workers
    if lo is None and hi is None:
        return "—"
    return f"{lo if lo is not None else '—'}-{hi if hi is not None else '—'}"


def _started_label(start_time) -> str:
    return start_time.strftime("%Y-%m-%d %H:%M") if start_time else "—"


def _deployment_cost_cell(dep) -> str:
    """Total accrued cost for a deployment; dim when nothing has accrued yet."""
    sym = currency_symbol(dep.currency)
    style = "green" if dep.total_cost > 0 else "dim"
    return f"[{style}]{sym}{dep.total_cost:.2f}[/{style}]"


def deployments_table(result, *, wide: bool = False) -> None:
    """Render the deployments list, newest first.

    The default view shows the columns most useful at a glance; ``wide`` adds
    the configuration columns (framework, workers, concurrency). The full
    record is always one ``jl deploy get <id>`` (or ``--json``) away.
    """
    warn_region_errors(result.region_errors)

    if not result.deployments:
        info("No deployments.")
        return

    table = _table("Deployments", show_lines=True)
    table.add_column("ID", no_wrap=True)  # full id stays intact so it's copy-pasteable
    table.add_column("Name", overflow="fold")  # the one column that wraps, so the others stay readable
    table.add_column("Region", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    if wide:
        table.add_column("Framework", no_wrap=True)
    table.add_column("GPU", no_wrap=True)
    if wide:
        table.add_column("Workers", justify="right")
        table.add_column("Concurrent", justify="right")
    table.add_column("Started", no_wrap=True)
    table.add_column("Cost", justify="right", no_wrap=True)

    for dep in result.deployments:
        style = _deployment_status_style(dep.status)
        row = [
            escape(dep.deployment_id),
            escape(dep.name or "—"),
            region_label(dep.region),
            f"[{style}]{escape(dep.status)}[/{style}]",
        ]
        if wide:
            row.append(escape(dep.framework or "—"))
        row.append(escape(_gpu_label(dep.gpus_to_use)))
        if wide:
            row.append(_workers_label(dep))
            row.append(str(dep.concurrent_requests) if dep.concurrent_requests is not None else "—")
        row.append(_started_label(dep.start_time))
        row.append(_deployment_cost_cell(dep))
        table.add_row(*row)

    stdout_console.print(table)
    if not wide:
        info("Tip: run 'jl deploy get <ID>' for full details · --wide for all columns")


def deployment_detail(dep, *, base_url: str | None = None) -> None:
    """Full detail view including worker info (folds the would-be `health` command)."""
    style = _deployment_status_style(dep.status)
    rows = [
        ("ID", f"[cyan]{escape(dep.deployment_id)}[/cyan]"),
        ("Name", f"[bold]{escape(dep.name or '—')}[/bold]"),
        ("Status", f"[{style}]{escape(dep.status)}[/{style}]"),
        ("Region", region_label(dep.region)),
        ("Framework", escape(dep.framework or "—")),
        ("GPU", f"{escape(_gpu_label(dep.gpus_to_use))} ({dep.gpus_per_worker or '—'}/worker)"),
        ("Workers", f"{_workers_label(dep)} (min-max)"),
        ("Concurrent", str(dep.concurrent_requests) if dep.concurrent_requests is not None else "—"),
        ("Idle timeout", f"{dep.idle_timeout}s" if dep.idle_timeout is not None else "—"),
        ("Wait time", f"{dep.wait_time}s" if dep.wait_time is not None else "—"),
        ("Storage", f"{dep.storage}GB" if dep.storage is not None else "—"),
        ("Model", _model_label(dep)),
    ]
    if dep.managed_model:
        rows.append(("API model", escape(dep.managed_model.api_model_name or "—")))

    other_args = {k: v for k, v in (dep.args or {}).items() if k != "model"}
    if other_args:
        rows.append(("Args", _kv_label(other_args)))

    if dep.env:
        rows.append(("Env", _kv_label(dep.env)))

    workers = dep.workers
    rows.append(("Queue depth", str(dep.queue_depth) if dep.queue_depth is not None else "—"))
    rows.append(
        ("Workers (live)", f"{workers.total} total · {workers.healthy} healthy · {workers.provisioning} provisioning")
    )
    rows.extend(
        (
            f"  worker {w.worker_id if w.worker_id is not None else i}",
            f"{escape(str(w.status))} (last used: {escape(str(w.last_used or '—'))})",
        )
        for i, w in enumerate(workers.list, start=1)
    )

    rows.append(("Total cost", _deployment_cost_cell(dep)))
    rows.append(("Created", _started_label(dep.created_at)))
    rows.append(("Updated", _started_label(dep.updated_at)))
    if dep.error_message:
        rows.append(("Error", f"[red]{escape(dep.error_message)}[/red]"))
    if base_url:
        rows.append(("Base URL", _link_value(base_url)))

    _print_detail_table(rows)


def _kv_label(data: dict) -> str:
    return ", ".join(f"{escape(str(key))}={escape(str(value))}" for key, value in data.items())


def deployment_status_line(status) -> None:
    """One-line status: `IN2 · <id> · running` (status colored)."""
    style = _deployment_status_style(status.status)
    region = region_label(status.region)
    stdout_console.print(f"{region} · {escape(status.deployment_id)} · [{style}]{escape(status.status)}[/{style}]")


def deployment_running_handoff(base_url: str, model: str | None) -> None:
    """Print the base URL, a paste-ready OpenAI snippet, and a first-request note.
    Printed markup-safe so brackets/braces are not mangled."""
    stdout_console.print(f"Base URL: {base_url}", markup=False, soft_wrap=True)
    stdout_console.print(_openai_snippet(base_url, model or "<model>"), markup=False, soft_wrap=True)
    stdout_console.print(
        "Note: the first request may be slow while a worker starts up.",
        markup=False,
    )


def _openai_snippet(base_url: str, model: str) -> str:
    return (
        "from openai import OpenAI\n"
        f'client = OpenAI(base_url="{base_url}", api_key="<YOUR_JL_API_KEY>")\n'
        "resp = client.chat.completions.create(\n"
        f'    model="{model}",\n'
        '    messages=[{"role": "user", "content": "Hello"}],\n'
        ")\n"
        "print(resp.choices[0].message.content)"
    )


# ── Token factory ────────────────────────────────────────────────────────────


def _price(value: float, currency: str) -> str:
    return f"{currency_symbol(currency)}{value:.2f}"


def _context_label(tokens: int | None) -> str:
    if not tokens:
        return "—"
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
    return f"{round(tokens / 1000)}K"


def _prices(pricing) -> dict[str, str]:
    return {
        "Input": _price(pricing.input_per_million_tokens, pricing.currency),
        "Cached": _price(pricing.cached_input_per_million_tokens, pricing.currency),
        "Output": _price(pricing.output_per_million_tokens, pricing.currency),
    }


def models_table(models: list) -> None:
    table = _table("Models · price per 1M tokens")
    table.add_column("ID", style="cyan", no_wrap=True)
    for heading in ("Context", "Input", "Cached", "Output"):
        table.add_column(heading, justify="right", no_wrap=True)
    for model in models:
        table.add_row(escape(model.id), _context_label(model.context_length), *_prices(model.pricing).values())
    stdout_console.print(table)
    info("Tip: run 'jl models get <ID>' for details and ready-to-use code")


def select_model(models: list):
    rows = []
    for model in models:
        prices = _prices(model.pricing)
        rows.append((model.id, _context_label(model.context_length), prices["Input"], prices["Output"]))
    headings = ("MODEL", "CONTEXT", "INPUT/1M", "OUTPUT/1M")
    colors = ("fg:ansicyan", "fg:ansibrightblack", "fg:ansigreen", "fg:ansigreen")
    return select_row("Select a model:", headings, colors, rows, models)


def _thinking_lines(thinking: dict | None) -> str:
    if not thinking or not thinking.get("supported"):
        return "[dim]Not supported[/dim]"
    keys = thinking.get("template_parameters", {})
    options = []
    if "toggle" in keys:
        options.append((keys["toggle"], "true | false"))
    if "effort" in keys:
        options.append((keys["effort"], " | ".join(thinking["supported_efforts"])))
    if not options:
        return "[green]Supported[/green]"
    width = max(len(key) for key, _ in options)
    lines = [f"[yellow]{key:<{width}}[/yellow]   {values}" for key, values in options]
    return "\n".join(["[green]Supported[/green]   [dim]set in chat_template_kwargs[/dim]", *lines])


def model_detail(model, base_url: str) -> None:
    header = f"[bold]{escape(model.name or model.id)}[/bold]   [cyan]{escape(model.id)}[/cyan]"
    if model.description:
        header += f"\n[dim]{escape(model.description)}[/dim]"
    prices = _prices(model.pricing)
    width = max(len(price) for price in prices.values())
    pricing = "\n".join(f"{name:<8}[green]{price:>{width}}[/green]" for name, price in prices.items())

    stdout_console.print()
    stdout_console.print(Padding(header, (0, 0, 1, 2)))
    _print_detail_table(
        [
            ("Context", f"{model.context_length:,} tokens" if model.context_length else "—"),
            ("Price per 1M", pricing),
            ("Thinking", _thinking_lines(model.thinking)),
            ("Model card", _link_value(model.model_card_url)),
            ("Base URL", _link_value(base_url)),
        ]
    )


def model_snippets(model_id: str, base_url: str) -> None:
    body = json.dumps({"model": model_id, "messages": [{"role": "user", "content": "Hello"}]})
    curl = (
        f"curl {base_url}/chat/completions \\\n"
        '  -H "Authorization: Bearer $JL_API_KEY" \\\n'
        '  -H "Content-Type: application/json" \\\n'
        f"  -d '{body}'"
    )
    for title, code, lexer in (("Python", _openai_snippet(base_url, model_id), "python"), ("curl", curl, "bash")):
        stdout_console.print()
        stdout_console.print(Rule(f"[bold]{title}[/bold]", align="left", style="dim"))
        stdout_console.print()
        stdout_console.print(Syntax(code, lexer, theme="ansi_dark", background_color="default", word_wrap=True))
    stdout_console.print()


def chat_text(text: str) -> None:
    stdout_console.print(text, end="", markup=False, highlight=False, soft_wrap=True)


def chat_thinking(text: str) -> None:
    console.print(text, end="", style="dim italic", markup=False, highlight=False, soft_wrap=True)


def chat_prompt() -> str:
    console.print()
    return console.input("[bold cyan]>[/bold cyan] ")


def chat_usage(usage: dict | None, pricing) -> None:
    if not usage:
        return
    tokens_in, tokens_out = usage["prompt_tokens"], usage["completion_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cost = (
        (tokens_in - cached) * pricing.input_per_million_tokens
        + cached * pricing.cached_input_per_million_tokens
        + tokens_out * pricing.output_per_million_tokens
    ) / 1_000_000
    info(f"\n{tokens_in:,} in · {tokens_out:,} out · {currency_symbol(pricing.currency)}{cost:.6f}")


def resource_label(inst) -> str:
    """Return the compact resource text shown in list/detail views."""
    if is_cpu_vm(inst):
        cores = inst.cores or "—"
        ram = f"{inst.ram}GB" if inst.ram is not None else "—"
        return f"{cores} vCPU / {ram} RAM"
    return f"{inst.num_gpus or 1}x {inst.gpu_type or '—'}"


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

    sym = currency_symbol(currency)
    container_gpus = [g for g in gpus if g.workload_type in ("container", None)]
    vm_gpus = [g for g in gpus if g.workload_type in ("vm", None)]

    if container_gpus:
        _gpu_subtable(container_gpus, sym, title="GPU Containers", show_spot=True)
    if vm_gpus:
        _gpu_subtable(vm_gpus, sym, title="GPU VMs", show_spot=False)
    if show_legend:
        availability_legend()


def _gpu_subtable(gpus: list, sym: str, title: str, *, show_spot: bool) -> None:
    available = [g for g in gpus if _on_demand_available(g)]
    unavailable = [g for g in gpus if not _on_demand_available(g)]
    # EU is being phased out — sink its rows to the bottom of each group (stable otherwise).
    available.sort(key=lambda g: g.region == EUROPE_REGION)
    unavailable.sort(key=lambda g: g.region == EUROPE_REGION)

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


def availability_legend() -> None:
    """Print the shared available/unavailable marker legend."""
    stdout_console.print("[green]●[/green] available  [dim]○ unavailable[/dim]")


def cpu_vm_table(cpu_meta: dict, currency: str = "USD", *, show_legend: bool = True) -> None:
    """Render CPU VM sizes from backend cpu_meta combinations."""
    combinations = cpu_meta.get("combinations") or []
    if not combinations:
        info("No CPU VM data available.")
        return

    sym = currency_symbol(currency)
    rows = []
    for combo in combinations:
        region_availability = combo.get("regions") or {}
        for region, available in region_availability.items():
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


def jsonable_cpu_meta(cpu_meta: dict) -> dict:
    """Return CPU metadata with CLI display region codes and no legacy top-level region."""
    sanitized = dict(cpu_meta)
    sanitized.pop("region", None)
    combinations = []
    for combo in sanitized.get("combinations", []):
        item = dict(combo)
        regions = item.get("regions")
        if isinstance(regions, dict):
            item["regions"] = {
                REGION_DISPLAY_CODES[region]: available
                for region, available in regions.items()
                if region in REGION_DISPLAY_CODES
            }
        combinations.append(item)
    if combinations:
        sanitized["combinations"] = combinations
    return sanitized


# ── Messages ─────────────────────────────────────────────────────────────────


def success(msg: str) -> None:
    if state.json_output:
        return
    console.print(f"[green]✓[/green] {msg}")


def error(msg: str) -> None:
    if state.json_output:
        return
    console.print(f"[red]✗[/red] {msg}", style="red")


def info(msg: str) -> None:
    if state.json_output:
        return
    console.print(f"[dim]{msg}[/dim]")


def warning(msg: str) -> None:
    if state.json_output:
        return
    console.print(f"[yellow]![/yellow] {msg}")


# ── Confirmation ─────────────────────────────────────────────────────────────


def confirm(msg: str, *, skip: bool = False) -> bool:
    """Ask for confirmation. Returns True if confirmed or skip=True (--yes flag).

    In --json mode confirmation cannot be prompted, so a missing --yes is a
    hard error — never a silent decline a script would read as success.
    """
    if skip:
        return True
    if state.json_output:
        die("--json requires --yes")
    try:
        response = console.input(f"[yellow]?[/yellow] {msg} [dim]\\[y/N][/dim] ")
        return response.strip().lower() in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        console.print()
        return False


# ── Spinner ──────────────────────────────────────────────────────────────────


def spinner(msg: str):
    """Rich spinner context manager for wrapping API calls. Suppressed in --json mode."""
    if state.json_output:
        return nullcontext()
    return console.status(f"[bold]{msg}[/bold]", spinner="dots")


# ── Helpers ──────────────────────────────────────────────────────────────────


def currency_symbol(currency: str) -> str:
    return "₹" if currency == "INR" else "$"


def _cost_cell(inst, sym: str) -> str:
    """Format cost for table display with contextual color."""
    if inst.cost <= 0:
        return "[dim]—[/dim]"
    return f"[green]{sym}{inst.cost:.2f}[/green]"


def region_label(region: str | None) -> str:
    """Convert backend region IDs to short CLI labels like IN2 and EU1."""
    if not region:
        return "—"
    return region_code(region)


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


def die(msg: str, code: int = 1) -> NoReturn:
    """Print error and exit. Emits JSON to stdout when --json is active."""
    if state.json_output:
        print_json({"error": msg})
    else:
        error(msg)
    sys.exit(code)


def managed_models_table(result) -> None:
    warn_region_errors(result.region_errors)
    table = _table("Managed Models · GPUs per worker")
    table.add_column("Slug", style="cyan", no_wrap=True)
    table.add_column("Region", no_wrap=True)
    table.add_column("GPU", no_wrap=True)
    table.add_column("Context", justify="right")
    table.add_column("Concurrent", justify="right")
    for model in result.models:
        table.add_row(
            escape(model.model_slug),
            region_label(model.region),
            _managed_gpu_label(model),
            _context_label(model.context_length),
            str(model.concurrent_requests or "—"),
        )
    stdout_console.print(table)
    info("Deploy one: jl deploy managed-models create <SLUG>, or leave out <SLUG> to pick from a list")


def select_managed_model(models: list):
    rows = [
        (
            model.display_name or model.model_slug,
            model.model_slug,
            region_label(model.region),
            _managed_gpu_label(model),
            _context_label(model.context_length),
        )
        for model in models
    ]
    headings = ("MODEL", "SLUG", "REGION", "GPU", "CONTEXT")
    colors = ("bold", "fg:ansicyan", "fg:ansiyellow", "fg:ansigreen", "fg:ansibrightblack")
    return select_row("Select a managed model:", headings, colors, rows, models)


def _managed_gpu_label(model) -> str:
    return f"{model.gpus_per_worker or '—'}x {model.gpu_type or '—'}"


def _model_label(dep) -> str:
    managed = dep.managed_model
    if managed:
        return (
            f"{escape(managed.display_name or managed.model_slug)} [dim]({escape(managed.model_slug)}, managed)[/dim]"
        )
    return escape(dep.model or "—")


def warn_region_errors(region_errors: list) -> None:
    for region_error in region_errors:
        warning(f"Could not reach {region_label(region_error.region)}: {region_error.error}")


# ── Interactive prompts ──────────────────────────────────────────────────────

PROMPT_STYLE = questionary.Style([("pointer", "fg:#73daca bold"), ("separator", "fg:ansibrightblack bold")])


def can_prompt() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def select_row(message: str, headings: tuple[str, ...], colors: tuple[str, ...], rows: list, values: list) -> Any:
    """Arrow-key picker shown as an aligned table; returns the chosen row's value. Ctrl+C exits."""
    widths = [max(len(cell) for cell in column) for column in zip(headings, *rows, strict=True)]

    def pad(cells) -> list[str]:
        return [cell.ljust(width + 2) for cell, width in zip(cells, widths, strict=True)]

    choices = [questionary.Separator("".join(pad(headings)))]
    choices += [
        questionary.Choice(title=list(zip(colors, pad(row), strict=True)), value=value)
        for row, value in zip(rows, values, strict=True)
    ]
    choice = questionary.select(message, choices=choices, style=PROMPT_STYLE).ask()
    if choice is None:
        raise typer.Exit()
    return choice


def ask_value(param: str, message: str, value, *, low: int | None = None, high: int | None = None):
    """Ask for a setting, prefilled with its value; whole numbers when low/high are given.

    A flag given on the command line is kept without asking. Ctrl+C exits.
    """
    if cli_options.option_was_explicit(param):
        return value

    def validate(text: str) -> bool | str:
        if low is None:
            return bool(text.strip()) or "This can't be empty."
        return (text.strip().isdigit() and low <= int(text) <= high) or f"Enter a whole number from {low} to {high}."

    if low is not None:
        value = min(max(value, low), high)
    answer = questionary.text(message, default=str(value), validate=validate, style=PROMPT_STYLE).ask()
    if answer is None:
        raise typer.Exit()
    return answer.strip() if low is None else int(answer)
