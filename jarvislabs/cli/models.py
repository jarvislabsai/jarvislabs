from __future__ import annotations

import os
import stat
import sys
from typing import TYPE_CHECKING, Annotated

import typer

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.constants import MODELS_API_URL
from jarvislabs.exceptions import NotFoundError

if TYPE_CHECKING:
    from jarvislabs.models import ModelInfo

ModelIdArgument = Annotated[str | None, typer.Argument(help="Model ID. Omit it to pick from a list.")]

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
    max_tokens: Annotated[int | None, typer.Option("--max-tokens", help="Maximum output tokens.", min=1)] = None,
    thinking: Annotated[bool | None, typer.Option("--thinking/--no-thinking", help="Turn thinking on or off.")] = None,
    effort: Annotated[str | None, typer.Option("--effort", help="Thinking effort, e.g. low or high.")] = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Chat with a model: pass a prompt for one reply, or omit it for an interactive chat."""
    cli_options.apply_command_options(json_output=json_output)
    text = "\n\n".join(part for part in (prompt, _read_stdin()) if part)
    client = get_client()
    model = _resolve_model(client, model_id)
    params = _thinking_params(model, thinking, effort)
    if max_tokens:
        params["max_tokens"] = max_tokens
    messages = [{"role": "system", "content": system}] if system else []

    if not text:
        if state.json_output or not render.can_prompt():
            render.die("Pass a prompt. See: jl models chat --help")
        _interactive_chat(client, model, messages, params)
        return

    messages.append({"role": "user", "content": text})
    if state.json_output:
        render.print_json(client.models.chat(model.id, messages, **params))
        return
    _stream_reply(client, model, messages, params)


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


def _read_stdin() -> str:
    # Only pipes and files: other non-terminal stdins (e.g. agents) may never close, so reading would hang.
    mode = os.fstat(sys.stdin.fileno()).st_mode
    return sys.stdin.read().strip() if stat.S_ISFIFO(mode) or stat.S_ISREG(mode) else ""


def _thinking_params(model: ModelInfo, thinking: bool | None, effort: str | None) -> dict:
    support = model.thinking or {}
    keys = support.get("template_parameters", {})
    kwargs = {}
    if thinking is not None:
        if "toggle" not in keys:
            render.die(f"{model.id} can't turn thinking on or off.")
        kwargs[keys["toggle"]] = thinking
    if effort:
        if "effort" not in keys:
            render.die(f"{model.id} doesn't support --effort.")
        if effort not in support["supported_efforts"]:
            render.die(f"--effort must be one of: {', '.join(support['supported_efforts'])}")
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
