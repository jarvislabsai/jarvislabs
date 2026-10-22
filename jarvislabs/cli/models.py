from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.constants import MODELS_API_URL
from jarvislabs.exceptions import NotFoundError

if TYPE_CHECKING:
    from jarvislabs.models import ModelInfo

ModelIdArgument = Annotated[str | None, typer.Argument(help="Model ID. Omit it to pick from a list.")]

# Chat sessions keep history across separate `jl models chat` calls (for agents) until ended or unused this long.
CHAT_SESSIONS_DIR = Path(tempfile.gettempdir()) / "jl-chats"
CHAT_SESSION_TTL_S = 900

models_app = typer.Typer(name="models", help="Pay-per-token model APIs: call a model directly, no deployment needed.")
app.add_typer(models_app, rich_help_panel="Token Factory")


@models_app.command("list")
def models_list(json_output: cli_options.JsonOption = False) -> None:
    """List the models you can call, with pricing per 1M tokens."""
    cli_options.apply_command_options(json_output=json_output)
    with render.spinner("Fetching models..."):
        models = get_client().models.list()

    if state.json_output:
        render.print_json(models)
        return
    render.models_table(models)


@models_app.command("get")
def models_get(model_id: ModelIdArgument = None, json_output: cli_options.JsonOption = False) -> None:
    """Show a model's details and ready-to-use code to call it."""
    cli_options.apply_command_options(json_output=json_output)
    model = _resolve_model(get_client(), model_id)

    if state.json_output:
        render.print_json({**model.model_dump(), "base_url": MODELS_API_URL})
        return
    render.model_detail(model, MODELS_API_URL)
    render.model_snippets(model.id, MODELS_API_URL)


@models_app.command("chat")
def models_chat(
    model_id: ModelIdArgument = None,
    prompt: Annotated[str | None, typer.Argument(help="Message to send. Omit it for an interactive chat.")] = None,
    system: Annotated[str | None, typer.Option("--system", help="System prompt.")] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Maximum output tokens.", min=1)] = 32768,
    thinking: Annotated[bool | None, typer.Option("--thinking/--no-thinking", help="Turn thinking on or off.")] = None,
    effort: Annotated[str | None, typer.Option("--effort", help="Thinking effort, e.g. low or high.")] = None,
    session: Annotated[
        str | None,
        typer.Option(
            "--session",
            help="Continue a session by its ID (every reply prints one). Deleted after 15 minutes unused.",
        ),
    ] = None,
    continue_session: Annotated[
        bool, typer.Option("--continue", "-c", help="Continue the most recent session (no ID needed).")
    ] = False,
    end: Annotated[str | None, typer.Option("--end", help="End a session and delete its history.")] = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Chat with a model: pass a prompt for one reply, or omit it for an interactive chat.

    The reply streams to stdout; thinking and token usage go to stderr.

    [bold]Examples:[/bold]

      [dim]# Interactive chat (pick a model from the list)[/dim]
      [cyan]jl models chat[/cyan]

      [dim]# One reply[/dim]
      [cyan]jl models chat <model_id> "Explain KV cache in two lines"[/cyan]

      [dim]# Add piped text to the prompt[/dim]
      [cyan]cat notes.txt | jl models chat <model_id> "Summarize this"[/cyan]

      [dim]# System prompt, output limit, thinking controls[/dim]
      [cyan]jl models chat <model_id> "Hi" --system "Be brief." --max-tokens 500 --no-thinking[/cyan]
      [cyan]jl models chat <model_id> "Prove it" --effort high[/cyan]

      [dim]# Full OpenAI-style response as JSON[/dim]
      [cyan]jl models chat <model_id> "Hi" --json[/cyan]

      [dim]# Multi-turn chat across separate calls (for agents and scripts)[/dim]
      [cyan]jl models chat <model_id> "My name is Jay"[/cyan]                     [dim]# prints a session ID[/dim]
      [cyan]jl models chat <model_id> "What is my name?" --session <ID>[/cyan]
      [cyan]jl models chat <model_id> "And my age?" -c[/cyan]                     [dim]# continues the most recent session[/dim]
      [cyan]jl models chat --end <ID>[/cyan]                                     [dim]# deletes the history[/dim]

    In interactive chat: Ctrl-C stops a reply · Ctrl-D or /exit quits.
    See supported thinking options with: jl models get <model_id>
    """
    cli_options.apply_command_options(json_output=json_output)
    _prune_sessions()
    if end:
        _session_path(end).unlink(missing_ok=True)
        if state.json_output:
            render.print_json({"success": True, "session": end})
            return
        render.success(f"Session {end} ended.")
        return

    text = "\n\n".join(part for part in (prompt, _read_stdin()) if part)
    if (session or continue_session) and not text:
        render.die("Pass a prompt with --session or -c. See: jl models chat --help")
    session = session or (_latest_session() if continue_session else None)
    messages = _load_session(session) if session else []
    session = session or uuid.uuid4().hex[:8]
    if system and not messages:
        messages.append({"role": "system", "content": system})

    client = get_client()
    model = _resolve_model(client, model_id)
    params = {**_thinking_params(model, thinking, effort), "max_tokens": max_tokens}

    if not text:
        if state.json_output or not render.can_prompt():
            render.die("Pass a prompt. See: jl models chat --help")
        _interactive_chat(client, model, messages, params)
        return

    messages.append({"role": "user", "content": text})
    if state.json_output:
        response = client.models.chat(model.id, messages, **params)
        reply = response["choices"][0]["message"].get("content") or ""
        _save_session(session, [*messages, {"role": "assistant", "content": reply}])
        render.print_json({**response, "session": session})
        return
    reply = _stream_reply(client, model, messages, params)
    if reply is not None:
        _save_session(session, [*messages, {"role": "assistant", "content": reply}])
        render.info(f"Session {session} · end: jl models chat --end {session}")


def _resolve_model(client, model_id: str | None) -> ModelInfo:
    if model_id:
        try:
            with render.spinner("Fetching model..."):
                return client.models.get(model_id)
        except NotFoundError:
            render.die(f"Unknown model: {model_id}. See: jl models list")
    if state.json_output or not render.can_prompt():
        render.die("Pass a model ID. See: jl models list")
    with render.spinner("Fetching models..."):
        models = client.models.list()
    return render.select_model(models)


def _session_path(session_id: str) -> Path:
    if not session_id.isalnum():
        render.die(f"Invalid session ID: {session_id}")
    return CHAT_SESSIONS_DIR / f"{session_id}.json"


def _load_session(session_id: str) -> list[dict]:
    path = _session_path(session_id)
    if not path.exists():
        render.die(f"Session {session_id} not found or expired. Send a prompt without --session to start a new one.")
    return json.loads(path.read_text())["messages"]


def _latest_session() -> str:
    paths = sorted(CHAT_SESSIONS_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not paths:
        render.die("No session to continue. Send a prompt without -c to start one.")
    return paths[-1].stem


def _save_session(session_id: str, messages: list[dict]) -> None:
    CHAT_SESSIONS_DIR.mkdir(mode=0o700, exist_ok=True)
    _session_path(session_id).write_text(json.dumps({"messages": messages}))


def _prune_sessions() -> None:
    """Delete sessions unused for CHAT_SESSION_TTL_S, so forgotten ones don't pile up."""
    cutoff = time.time() - CHAT_SESSION_TTL_S
    for path in CHAT_SESSIONS_DIR.glob("*.json"):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)


def _read_stdin() -> str:
    # Only pipes and files: other non-terminal stdins (e.g. agents) may never close, so reading would hang.
    mode = os.fstat(sys.stdin.fileno()).st_mode
    return sys.stdin.read().strip() if stat.S_ISFIFO(mode) or stat.S_ISREG(mode) else ""


def _thinking_params(model: ModelInfo, thinking: bool | None, effort: str | None) -> dict:
    support = model.thinking
    keys = support.template_parameters
    kwargs = {}
    if thinking is not None:
        if "toggle" not in keys:
            render.die(f"{model.id} can't turn thinking on or off.")
        kwargs[keys["toggle"]] = thinking
    if effort:
        if "effort" not in keys:
            render.die(f"{model.id} doesn't support --effort.")
        if effort not in support.supported_efforts:
            render.die(f"--effort must be one of: {', '.join(support.supported_efforts)}")
        kwargs[keys["effort"]] = effort
    return {"chat_template_kwargs": kwargs} if kwargs else {}


def _stream_reply(client, model: ModelInfo, messages: list[dict], params: dict) -> str | None:
    """Print a reply as it streams; returns its text, or None when Ctrl-C stopped it."""
    reply, thought, usage = "", False, None
    try:
        for chunk in client.models.chat_stream(model.id, messages, **params):
            usage = chunk.get("usage") or usage
            delta = chunk["choices"][0]["delta"] if chunk.get("choices") else {}
            if delta.get("reasoning_content"):
                render.chat_thinking(delta["reasoning_content"])
                thought = True
            if delta.get("content"):
                if thought and not reply:
                    render.chat_thinking("\n\n")
                render.chat_text(delta["content"])
                reply += delta["content"]
    except KeyboardInterrupt:
        render.chat_text("\n")
        render.info("Stopped.")
        return None
    render.chat_text("\n")
    render.chat_usage(usage, model.pricing)
    return reply


def _interactive_chat(client, model: ModelInfo, messages: list[dict], params: dict) -> None:
    render.info(f"Chatting with {model.id}. Ctrl-C stops a reply · Ctrl-D or /exit quits.")
    while True:
        try:
            text = render.chat_prompt().strip()
        except (EOFError, KeyboardInterrupt):
            render.chat_text("\n")
            return
        if text == "/exit":
            return
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        reply = _stream_reply(client, model, messages, params)
        if reply is None:
            messages.pop()
        else:
            messages.append({"role": "assistant", "content": reply})
