from __future__ import annotations

import base64
import json
import mimetypes
import os
import select
import stat
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.constants import MODELS_API_URL, MODELS_DEFAULT_MAX_TOKENS
from jarvislabs.exceptions import NotFoundError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from jarvislabs.models import ModelInfo, ModelPricing

ModelIdArgument = Annotated[str | None, typer.Argument(help="Model ID. Omit it to pick from a list.")]
ImageOption = Annotated[
    list[str] | None,
    typer.Option("--image", help="Image to send with the prompt: a local file or an http(s) URL. Repeatable."),
]
ParamOption = Annotated[
    list[str] | None,
    typer.Option(
        "--param",
        help="Extra request field as key=value, e.g. temperature=0.2. The value is read as JSON when it is "
        "valid JSON; @file reads it from a file. Repeatable.",
    ),
]
# Request fields jl sets itself, and base_url (where the request and API key go); --param can't replace them.
RESERVED_PARAMS = frozenset({"model", "messages", "stream", "stream_options", "base_url"})

# Chat sessions keep history across separate `jl models chat` calls (for agents) until ended or unused this long.
CHAT_SESSIONS_DIR = Path(tempfile.gettempdir()) / "jl-chats"
CHAT_SESSION_TTL_S = 900
# With a prompt given, piped input is added only if it starts arriving within this time (see _read_stdin).
STDIN_WAIT_S = 1.0

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
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Maximum output tokens.", min=1)
    ] = MODELS_DEFAULT_MAX_TOKENS,
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
    image: ImageOption = None,
    param: ParamOption = None,
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

      [dim]# Images (local file or URL), for models that accept them[/dim]
      [cyan]jl models chat <model_id> "What is in this picture?" --image photo.jpg[/cyan]

      [dim]# Any other OpenAI request field[/dim]
      [cyan]jl models chat <model_id> "List 3 colors" --param temperature=0.2 --param response_format='{"type": "json_object"}'[/cyan]
      [cyan]jl models chat <model_id> "Weather in Pune?" --param tools=@tools.json --json[/cyan]

      [dim]# Multi-turn chat across separate calls (for agents and scripts)[/dim]
      [cyan]jl models chat <model_id> "My name is Jay"[/cyan]                     [dim]# prints a session ID[/dim]
      [cyan]jl models chat <model_id> "What is my name?" --session <ID>[/cyan]
      [cyan]jl models chat <model_id> "And my age?" -c[/cyan]                     [dim]# continues the most recent session[/dim]

    In interactive chat: Ctrl-C stops a reply · Ctrl-D or /exit quits.
    See supported thinking options with: jl models get <model_id>
    """
    cli_options.apply_command_options(json_output=json_output)
    _prune_sessions()
    text = read_prompt(prompt)
    if (session or continue_session) and not text:
        render.die("Pass a prompt with --session or -c. See: jl models chat --help")
    if image and not text:
        render.die("Pass a prompt with --image. See: jl models chat --help")
    extra = request_params(param)
    session = session or (_latest_session() if continue_session else None)
    messages = _load_session(session) if session else []
    session = session or uuid.uuid4().hex[:8]
    if system and not messages:
        messages.append({"role": "system", "content": system})

    client = get_client()
    model = _resolve_model(client, model_id)
    if image and model.input_modalities and "image" not in model.input_modalities:
        render.die(f"{model.id} doesn't accept images. See the Modalities in: jl models get {model.id}")
    params = merge_params({**_thinking_params(model, thinking, effort), "max_tokens": max_tokens}, extra)

    def send(history: list[dict]) -> dict | None:
        return stream_reply(client.models.chat_stream(model.id, history, **params), model.pricing)

    if not text:
        if state.json_output or not render.can_prompt():
            render.die("Pass a prompt. See: jl models chat --help")
        interactive_chat(model.id, messages, send)
        return

    messages.append({"role": "user", "content": user_content(text, image)})
    if state.json_output:
        response = client.models.chat(model.id, messages, **params)
        message = response["choices"][0]["message"]
        thinking = message.get("reasoning_content") or message.get("reasoning")
        _save_session(session, [*messages, assistant_message(message.get("content") or "", thinking)])
        render.print_json({**response, "session": session})
        return
    reply = send(messages)
    if reply is None:
        raise typer.Exit(1)
    _save_session(session, [*messages, reply])
    render.info(f"Session {session}")


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


def read_prompt(prompt: str | None) -> str:
    """The prompt argument plus any piped or redirected input, joined by a blank line."""
    return "\n\n".join(part for part in (prompt, _read_stdin(wait=prompt is None)) if part)


def _read_stdin(*, wait: bool) -> str:
    """Piped or redirected input; "" for a terminal or anything else.

    Some callers (e.g. agents) leave a pipe open without ever writing to it, so reading it
    would hang. When a prompt was given (``wait=False``), a pipe is read only if input or
    end-of-file arrives within STDIN_WAIT_S. Without a prompt the input is all there is, so wait.
    """
    mode = os.fstat(sys.stdin.fileno()).st_mode
    if stat.S_ISREG(mode):
        return sys.stdin.read().strip()
    if not stat.S_ISFIFO(mode):
        return ""
    # select() can't watch pipes on Windows, so there a pipe is always read.
    if not wait and os.name != "nt" and not select.select([sys.stdin], [], [], STDIN_WAIT_S)[0]:
        return ""
    return sys.stdin.read().strip()


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


def assistant_message(content: str, thinking: str | None = None) -> dict:
    """An assistant turn for the chat history. The thinking goes back to the model on later turns
    as ``reasoning_content``: some chat templates (e.g. GLM's) use earlier thinking, others drop it."""
    message = {"role": "assistant", "content": content}
    if thinking:
        message["reasoning_content"] = thinking
    return message


def user_content(text: str, images: list[str] | None) -> str | list[dict]:
    """A user message's content: the text alone, or the text followed by the images."""
    if not images:
        return text
    return [
        {"type": "text", "text": text},
        *({"type": "image_url", "image_url": {"url": _image_url(i)}} for i in images),
    ]


def _image_url(image: str) -> str:
    """URLs pass through (the model server downloads them); a local file is sent inline as base64."""
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = Path(image).expanduser()
    if not path.is_file():
        render.die(f"Image not found: {image}")
    mime = mimetypes.guess_type(path.name)[0]
    if not mime or not mime.startswith("image/"):
        render.die(f"{image} doesn't look like an image (by its file extension).")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def request_params(pairs: list[str] | None) -> dict[str, Any]:
    """Extra request fields from --param key=value. A value is JSON when it parses, else a string;
    ``@path`` reads the value from a file."""
    params: dict[str, Any] = {}
    for key, raw in cli_options.parse_kv(pairs, flag="--param").items():
        if key in RESERVED_PARAMS:
            render.die(f"--param {key} is set by jl and can't be changed.")
        if raw.startswith("@"):
            path = Path(raw[1:]).expanduser()
            if not path.is_file():
                render.die(f"--param {key}: file not found: {raw[1:]}")
            raw = path.read_text()
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    return params


def merge_params(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Request fields from jl's own options, overridden by --param. ``chat_template_kwargs`` from both are
    combined, and a --param max_completion_tokens replaces jl's max_tokens (the API takes only one)."""
    merged = {**base, **extra}
    if isinstance(base.get("chat_template_kwargs"), dict) and isinstance(extra.get("chat_template_kwargs"), dict):
        merged["chat_template_kwargs"] = {**base["chat_template_kwargs"], **extra["chat_template_kwargs"]}
    if "max_completion_tokens" in extra and "max_tokens" not in extra:
        merged.pop("max_tokens", None)
    return merged


def stream_reply(chunks: Iterable[dict[str, Any]], pricing: ModelPricing | None = None) -> dict | None:
    """Print a reply as it streams; returns it as an assistant message, or None when Ctrl-C stopped it
    or no answer came back."""
    reply, thinking, usage, finish_reason = "", "", None, None
    try:
        for chunk in chunks:
            usage = chunk.get("usage") or usage
            choice = chunk["choices"][0] if chunk.get("choices") else {}
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            # The Token Factory sends both names; a deployment's engine (vLLM) sends only "reasoning".
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if thought:
                render.chat_thinking(thought)
                thinking += thought
            if delta.get("content"):
                if thinking and not reply:
                    render.chat_thinking("\n\n")
                render.chat_text(delta["content"])
                reply += delta["content"]
    except KeyboardInterrupt:
        render.chat_text("\n")
        render.info("Stopped.")
        return None
    render.chat_text("\n")
    render.chat_usage(usage, pricing)
    if finish_reason == "length":
        render.warning("The reply hit the --max-tokens limit and was cut off. Raise --max-tokens to get all of it.")
    elif finish_reason == "tool_calls":
        render.warning("The model wants to call a tool. Add --json to get the tool calls.")
    elif not reply:
        render.warning("The model returned no answer.")
    return assistant_message(reply, thinking) if reply else None


def interactive_chat(name: str, messages: list[dict], send: Callable[[list[dict]], dict | None]) -> None:
    """Chat turn by turn; ``send`` streams one reply to the history and returns it (None if there's none)."""
    render.info(f"Chatting with {name}. Ctrl-C stops a reply · Ctrl-D or /exit quits.")
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
        reply = send(messages)
        if reply is None:
            messages.pop()
        else:
            messages.append(reply)
