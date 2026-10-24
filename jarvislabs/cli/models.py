from __future__ import annotations

import base64
import mimetypes
import os
import select
import stat
import sys
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
    image: ImageOption = None,
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

    In interactive chat: Ctrl-C stops a reply · Ctrl-D or /exit quits.
    See supported thinking options with: jl models get <model_id>
    """
    cli_options.apply_command_options(json_output=json_output)
    text = read_prompt(prompt)
    if image and not text:
        render.die("Pass a prompt with --image. See: jl models chat --help")
    if not text and (state.json_output or not render.can_prompt()):
        render.die("Pass a prompt. See: jl models chat --help")
    messages = [{"role": "system", "content": system}] if system else []

    client = get_client()
    model = _resolve_model(client, model_id)
    if image and model.input_modalities and "image" not in model.input_modalities:
        render.die(f"{model.id} doesn't accept images. See the Modalities in: jl models get {model.id}")
    params = {**_thinking_params(model, thinking, effort), "max_tokens": max_tokens}

    def send(history: list[dict]) -> dict | None:
        return stream_reply(client.models.chat_stream(model.id, history, **params), model.pricing)

    if not text:
        interactive_chat(model.id, messages, send)
        return

    messages.append({"role": "user", "content": user_content(text, image)})
    if state.json_output:
        render.print_json(client.models.chat(model.id, messages, **params))
        return
    if send(messages) is None:
        raise typer.Exit(1)


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
    # select() can't watch pipes on Windows, so there a pipe is always read to the end.
    if wait or os.name == "nt":
        return sys.stdin.read().strip()
    # Read what arrives, and stop once the pipe closes or stays quiet for STDIN_WAIT_S.
    fd, chunks = sys.stdin.fileno(), []
    while select.select([fd], [], [], STDIN_WAIT_S)[0]:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks).decode(sys.stdin.encoding or "utf-8", errors="replace").strip()


def _thinking_params(model: ModelInfo, thinking: bool | None, effort: str | None) -> dict:
    support = model.thinking
    keys = support.template_parameters
    kwargs = {}
    if thinking is not None:
        if "toggle" not in keys:
            render.die(f"{model.id} can't turn thinking on or off. See: jl models get {model.id}")
        kwargs[keys["toggle"]] = thinking
    if effort:
        if "effort" not in keys:
            render.die(f"{model.id} doesn't support --effort. See: jl models get {model.id}")
        if effort not in support.supported_efforts:
            render.die(f"--effort must be one of: {', '.join(support.supported_efforts)}")
        kwargs[keys["effort"]] = effort
    return {"chat_template_kwargs": kwargs} if kwargs else {}


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
    try:
        data = path.read_bytes()
    except OSError as exc:
        render.die(f"Can't read {image}: {exc.strerror}")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


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
            # Servers name the thinking field either way.
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
        render.warning("The reply hit the output token limit and was cut off. Raise --max-tokens to get all of it.")
    elif finish_reason is None:
        render.warning("The reply stopped before the model finished (the connection ended early).")
        return None
    elif not reply:
        render.warning("The model returned no answer.")
    if not reply:
        return None
    # Keep the thinking in the history as reasoning_content; some models use it on later turns.
    return {"role": "assistant", "content": reply, **({"reasoning_content": thinking} if thinking else {})}


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
