from __future__ import annotations

import base64
import os
from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from jarvislabs.cli import models as models_cli, render, state
from jarvislabs.constants import MODELS_API_URL
from jarvislabs.exceptions import InsufficientBalanceError, JarvislabsError, NotFoundError, error_from_response
from jarvislabs.models import ModelInfo, ModelThinking
from jarvislabs.token_factory import Models

READ_STDIN = models_cli._read_stdin  # The real one; tests stub it out by default.

QWEN = {
    "id": "qwen",
    "name": "Qwen 3.5 27B",
    "context_length": 128000,
    "pricing": {
        "currency": "USD",
        "input_per_million_tokens": 0.2,
        "cached_input_per_million_tokens": 0.05,
        "output_per_million_tokens": 0.8,
    },
}
DEEPSEEK = {
    **QWEN,
    "id": "deepseek",
    "thinking": {
        "supported": True,
        "supported_efforts": ["low", "high", "max"],
        "template_parameters": {"toggle": "thinking", "effort": "reasoning_effort"},
    },
}
HELLO = [{"role": "user", "content": "Hello"}]


@pytest.fixture(autouse=True)
def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "spinner", lambda *a, **k: nullcontext())
    monkeypatch.setattr(models_cli, "_read_stdin", lambda **_: "")
    state.json_output = False
    yield
    state.json_output = False


def _client(monkeypatch, **methods):
    monkeypatch.setattr(models_cli, "get_client", lambda: SimpleNamespace(models=SimpleNamespace(**methods)))


def _capture(fn, *args, **kwargs):
    buf = StringIO()
    old_console = render.stdout_console
    render.stdout_console = Console(file=buf, color_system=None, width=150)
    try:
        fn(*args, **kwargs)
    finally:
        render.stdout_console = old_console
    return buf.getvalue()


def _errors(monkeypatch):
    errors = []
    monkeypatch.setattr(render, "error", errors.append)
    return errors


# ── SDK ─────────────────────────────────────────────────────────────────────────


def test_sdk_list_and_get(mock_transport):
    mock_transport.request.side_effect = [{"data": [QWEN]}, QWEN]
    models = Models(mock_transport)
    assert models.list()[0].id == "qwen"
    assert models.get("qwen").name == "Qwen 3.5 27B"
    assert mock_transport.request.call_args.args == ("GET", "models/qwen")


def test_sdk_chat(mock_transport):
    Models(mock_transport).chat("qwen", HELLO, max_tokens=10)
    mock_transport.request.assert_called_once_with(
        "POST",
        "chat/completions",
        json={"model": "qwen", "messages": HELLO, "max_tokens": 10},
        base_url=MODELS_API_URL,
    )


def test_sdk_chat_stream(mock_transport):
    mock_transport.stream_lines.return_value = iter(
        ['data: {"choices": [{"delta": {"content": "Hi"}}]}', "", ": ping", "data: [DONE]"]
    )
    chunks = list(Models(mock_transport).chat_stream("qwen", HELLO))
    assert chunks == [{"choices": [{"delta": {"content": "Hi"}}]}]
    body = mock_transport.stream_lines.call_args.kwargs["json"]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_sdk_chat_stream_error(mock_transport):
    mock_transport.stream_lines.return_value = iter(['data: {"error": {"message": "Upstream failed"}}'])
    with pytest.raises(JarvislabsError, match="Upstream failed"):
        list(Models(mock_transport).chat_stream("qwen", HELLO))


def test_sdk_chat_submit(mock_transport):
    Models(mock_transport).chat_submit("qwen", HELLO, base_url="https://x/v1", max_tokens=1)
    mock_transport.submit_stream.assert_called_once_with(
        "chat/completions",
        json={"model": "qwen", "messages": HELLO, "max_tokens": 1, "stream": True},
        base_url="https://x/v1",
    )


def test_402_is_insufficient_balance():
    assert isinstance(error_from_response(402, {"error": {"message": "Add credits."}}), InsufficientBalanceError)


# ── list / get ──────────────────────────────────────────────────────────────────


def test_list_table(monkeypatch):
    _client(monkeypatch, list=lambda: [ModelInfo(**QWEN)])
    out = _capture(models_cli.models_list)
    assert "qwen" in out
    assert "128K" in out
    assert "$0.20" in out
    assert "$0.05" in out
    assert "$0.80" in out


def test_list_json(monkeypatch):
    models = [ModelInfo(**QWEN)]
    _client(monkeypatch, list=lambda: models)
    printed = []
    monkeypatch.setattr(render, "print_json", printed.append)
    models_cli.models_list(json_output=True)
    assert printed == [models]


@pytest.mark.parametrize(("tokens", "label"), [(None, "—"), (131072, "131K"), (1048576, "1.05M"), (2_000_000, "2M")])
def test_context_label(tokens, label):
    assert render._context_label(tokens) == label


@pytest.mark.parametrize(("value", "label"), [(0.5, "$0.50"), (0.15, "$0.15"), (0.003, "$0.003"), (0.0, "$0.00")])
def test_price_keeps_small_values(value, label):
    assert render._price(value, "USD") == label


def test_get_detail_and_snippets(monkeypatch):
    _client(monkeypatch, get=lambda model_id: ModelInfo(**DEEPSEEK))
    out = _capture(models_cli.models_get, "deepseek")
    assert "128,000 tokens" in out
    assert "Input   $0.20" in out
    assert "thinking           true | false" in out
    assert "reasoning_effort   low | high | max" in out
    assert f'OpenAI(base_url="{MODELS_API_URL}"' in out
    assert f"curl {MODELS_API_URL}/chat/completions" in out


def test_thinking_not_supported():
    assert render._thinking_lines(ModelThinking()) == "[dim]Not supported[/dim]"


def test_get_json(monkeypatch):
    _client(monkeypatch, get=lambda model_id: ModelInfo(**QWEN))
    printed = []
    monkeypatch.setattr(render, "print_json", printed.append)
    models_cli.models_get("qwen", json_output=True)
    assert printed[0]["base_url"] == MODELS_API_URL


def test_get_unknown_model(monkeypatch):
    def missing(model_id):
        raise NotFoundError("Unknown model.")

    _client(monkeypatch, get=missing)
    errors = _errors(monkeypatch)
    with pytest.raises(SystemExit):
        models_cli.models_get("nope")
    assert errors == ["Unknown model: nope. See: jl models list"]


def test_get_without_id_needs_a_terminal(monkeypatch):
    _client(monkeypatch)
    monkeypatch.setattr(render, "can_prompt", lambda: False)
    errors = _errors(monkeypatch)
    with pytest.raises(SystemExit):
        models_cli.models_get(None)
    assert errors == ["Pass a model ID. See: jl models list"]


def test_get_without_id_picks_from_list(monkeypatch):
    qwen = ModelInfo(**QWEN)
    _client(monkeypatch, list=lambda: [qwen])
    monkeypatch.setattr(render, "can_prompt", lambda: True)
    monkeypatch.setattr(render, "select_model", lambda models: models[0])
    shown = []
    monkeypatch.setattr(render, "model_detail", lambda model, base_url: shown.append(model))
    monkeypatch.setattr(render, "model_snippets", lambda *a: None)
    models_cli.models_get(None)
    assert shown == [qwen]


def test_select_model_rows(monkeypatch):
    seen = []
    monkeypatch.setattr(render, "select_row", lambda message, headings, rows, values: seen.extend(rows))
    render.select_model([ModelInfo(**QWEN)])
    assert seen == [("qwen", "128K", "$0.20", "$0.80")]


# ── chat ────────────────────────────────────────────────────────────────────────


OK_CHUNKS = ({"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]},)


def _chat_client(monkeypatch, chunks=OK_CHUNKS, reply=None):
    calls = []

    def chat_stream(model_id, messages, **params):
        calls.append((messages, params))
        yield from chunks

    def chat(model_id, messages, **params):
        calls.append((messages, params))
        return reply

    _client(monkeypatch, get=lambda model_id: ModelInfo(**DEEPSEEK), chat_stream=chat_stream, chat=chat)
    return calls


def test_chat_streams_reply_and_usage(monkeypatch):
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "Hmm."}}]},
        {"choices": [{"delta": {"content": "Hi "}}]},
        {"choices": [{"delta": {"content": "there"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}},
    ]
    _chat_client(monkeypatch, chunks)
    thinking, notes = [], []
    monkeypatch.setattr(render, "chat_thinking", thinking.append)
    monkeypatch.setattr(render, "info", notes.append)
    assert _capture(models_cli.models_chat, "deepseek", "Hello") == "Hi there\n"
    assert thinking == ["Hmm.", "\n\n"]
    assert notes == ["\n1,000,000 in · 1,000,000 out · $1.000000"]


def test_chat_piped_input_and_options(monkeypatch):
    calls = _chat_client(monkeypatch)
    monkeypatch.setattr(models_cli, "_read_stdin", lambda **_: "file text")
    monkeypatch.setattr(render, "chat_usage", lambda *a: None)
    models_cli.models_chat("deepseek", "Summarize", system="Be brief.", max_tokens=50, thinking=False, effort="high")
    assert calls == [
        (
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Summarize\n\nfile text"}],
            {"chat_template_kwargs": {"thinking": False, "reasoning_effort": "high"}, "max_tokens": 50},
        )
    ]


def test_chat_json(monkeypatch):
    reply = {"id": "r1", "choices": [{"message": {"role": "assistant", "content": "Hi"}}]}
    _chat_client(monkeypatch, reply=reply)
    printed = []
    monkeypatch.setattr(render, "print_json", printed.append)
    models_cli.models_chat("deepseek", "Hello", json_output=True)
    assert printed[0]["id"] == "r1"


@pytest.mark.parametrize(
    ("model", "options", "message"),
    [
        (DEEPSEEK, {}, "Pass a prompt. See: jl models chat --help"),
        (DEEPSEEK, {"prompt": "Hi", "effort": "extreme"}, "--effort must be one of: low, high, max"),
        (QWEN, {"prompt": "Hi", "thinking": True}, "qwen can't turn thinking on or off."),
        (QWEN, {"prompt": "Hi", "effort": "low"}, "qwen doesn't support --effort."),
    ],
)
def test_chat_errors(monkeypatch, model, options, message):
    _client(monkeypatch, get=lambda model_id: ModelInfo(**model))
    monkeypatch.setattr(render, "can_prompt", lambda: False)
    errors = _errors(monkeypatch)
    with pytest.raises(SystemExit):
        models_cli.models_chat(model["id"], **options)
    assert errors == [message]


def test_read_stdin_reads_pipes_only(monkeypatch):
    read, write = os.pipe()
    os.write(write, b" piped text \n")
    os.close(write)
    with os.fdopen(read) as pipe:
        monkeypatch.setattr(models_cli.sys, "stdin", pipe)
        assert READ_STDIN(wait=True) == "piped text"
    with open(os.devnull) as devnull:
        monkeypatch.setattr(models_cli.sys, "stdin", devnull)
        assert READ_STDIN(wait=True) == ""


def test_read_stdin_skips_a_silent_open_pipe_when_a_prompt_is_given(monkeypatch):
    monkeypatch.setattr(models_cli, "STDIN_WAIT_S", 0.01)
    read, write = os.pipe()  # the write end stays open and nothing is written: reading would hang
    try:
        with os.fdopen(read) as pipe:
            monkeypatch.setattr(models_cli.sys, "stdin", pipe)
            assert READ_STDIN(wait=False) == ""
    finally:
        os.close(write)


def test_model_info_drops_thinking_defaults_and_keeps_exact_prices():
    model = ModelInfo(
        **{
            **DEEPSEEK,
            "thinking": {**DEEPSEEK["thinking"], "default_enabled": True, "default_effort": "high"},
            "pricing": {**QWEN["pricing"], "input_per_million_tokens": 12.3201},
        }
    )
    dumped = model.model_dump()
    assert "default_enabled" not in dumped["thinking"]
    assert "default_effort" not in dumped["thinking"]
    assert dumped["pricing"]["input_per_million_tokens"] == 12.3201


def test_get_detail_shows_modalities(monkeypatch):
    model = ModelInfo(**{**QWEN, "input_modalities": ["text", "image"], "output_modalities": ["text"]})
    _client(monkeypatch, get=lambda model_id: model)
    out = _capture(models_cli.models_get, "qwen")
    assert "Input    text, image" in out
    assert "Output   text" in out


def test_stream_reply_warns_when_cut_off(monkeypatch):
    warnings = []
    monkeypatch.setattr(render, "warning", warnings.append)
    chunks = [{"choices": [{"delta": {"content": "Half"}, "finish_reason": "length"}]}]
    assert _capture(models_cli.stream_reply, chunks) == "Half\n"
    assert "--max-tokens" in warnings[0]


def test_stream_reply_without_answer_returns_none(monkeypatch):
    warnings = []
    monkeypatch.setattr(render, "warning", warnings.append)
    monkeypatch.setattr(render, "chat_thinking", lambda text: None)
    chunks = [{"choices": [{"delta": {"reasoning_content": "Hmm"}, "finish_reason": "length"}]}]
    assert models_cli.stream_reply(chunks) is None
    assert warnings


def test_stream_reply_reads_vllm_reasoning(monkeypatch):
    thoughts = []
    monkeypatch.setattr(render, "chat_thinking", thoughts.append)
    chunks = [{"choices": [{"delta": {"reasoning": "Hmm"}}]}, {"choices": [{"delta": {"content": "Hi"}}]}]
    _capture(models_cli.stream_reply, chunks)
    assert thoughts[0] == "Hmm"


def test_model_info_accepts_nulls():
    model = ModelInfo(id="m", pricing=None, thinking=None, input_modalities=None, output_modalities=None)
    assert model.pricing is None
    assert model.thinking.supported is False
    assert model.input_modalities == []
    assert render._prices(None) == {"Input": "—", "Cached": "—", "Output": "—"}


def test_chat_usage_without_prices_shows_tokens(monkeypatch):
    notes = []
    monkeypatch.setattr(render, "info", notes.append)
    render.chat_usage({"prompt_tokens": 10, "completion_tokens": 5})
    assert notes == ["\n10 in · 5 out"]


def test_snippets_set_max_tokens():
    out = _capture(render.model_snippets, "glm-5.3", MODELS_API_URL)
    assert "max_tokens=32768" in out
    assert '"max_tokens": 32768' in out


def test_stream_reply_keeps_thinking_for_the_history(monkeypatch):
    monkeypatch.setattr(render, "chat_thinking", lambda text: None)
    chunks = [{"choices": [{"delta": {"reasoning": "Hm"}}]}, {"choices": [{"delta": {"reasoning": "m"}}]}]
    chunks.append({"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]})
    message = models_cli.stream_reply(chunks)
    assert message == {"role": "assistant", "content": "Hi", "reasoning_content": "Hmm"}


def test_chat_usage_shows_thinking_tokens(monkeypatch):
    notes = []
    monkeypatch.setattr(render, "info", notes.append)
    usage = {"prompt_tokens": 10, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 38}}
    render.chat_usage(usage)
    assert notes == ["\n10 in · 50 out (38 thinking)"]


# ── --image and --param ────────────────────────────────────────────────────────

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


def test_user_content_without_images_is_text():
    assert models_cli.user_content("Hi", None) == "Hi"


def test_user_content_adds_local_file_and_url(tmp_path):
    photo = tmp_path / "photo.png"
    photo.write_bytes(PNG)
    content = models_cli.user_content("What is this?", [str(photo), "https://x.test/cat.jpg"])
    assert content[0] == {"type": "text", "text": "What is this?"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert content[2] == {"type": "image_url", "image_url": {"url": "https://x.test/cat.jpg"}}


@pytest.mark.parametrize(
    ("name", "message"),
    [("missing.png", "Image not found: "), ("notes.txt", "doesn't look like an image")],
)
def test_bad_image_paths(monkeypatch, tmp_path, name, message):
    (tmp_path / "notes.txt").write_text("hi")
    errors = _errors(monkeypatch)
    with pytest.raises(SystemExit):
        models_cli.user_content("Hi", [str(tmp_path / name)])
    assert message in errors[0]


def test_chat_sends_image(monkeypatch, tmp_path):
    photo = tmp_path / "photo.png"
    photo.write_bytes(PNG)
    calls = _chat_client(monkeypatch)
    monkeypatch.setattr(
        models_cli, "_resolve_model", lambda c, m: ModelInfo(**{**DEEPSEEK, "input_modalities": ["text", "image"]})
    )
    monkeypatch.setattr(render, "chat_usage", lambda *a: None)
    models_cli.models_chat("deepseek", "Describe", image=[str(photo)], max_tokens=50)
    messages, params = calls[0]
    assert messages[-1]["content"][1]["type"] == "image_url"
    assert params == {"max_tokens": 50}


def test_chat_refuses_images_for_text_only_models(monkeypatch, tmp_path):
    _chat_client(monkeypatch)
    monkeypatch.setattr(
        models_cli, "_resolve_model", lambda c, m: ModelInfo(**{**DEEPSEEK, "input_modalities": ["text"]})
    )
    errors = _errors(monkeypatch)
    with pytest.raises(SystemExit):
        models_cli.models_chat("deepseek", "Describe", image=["photo.png"])
    assert errors == ["deepseek doesn't accept images. See the Modalities in: jl models get deepseek"]


def test_chat_image_needs_a_prompt(monkeypatch):
    errors = _errors(monkeypatch)
    with pytest.raises(SystemExit):
        models_cli.models_chat("deepseek", image=["photo.png"])
    assert errors == ["Pass a prompt with --image. See: jl models chat --help"]


def test_sdk_chat_accepts_named_arguments(mock_transport):
    Models(mock_transport).chat(model_id="qwen", messages=HELLO)
    assert mock_transport.request.call_args.kwargs["json"] == {"model": "qwen", "messages": HELLO}


def test_stream_that_ends_early_is_not_a_reply(monkeypatch):
    warnings = []
    monkeypatch.setattr(render, "warning", warnings.append)
    assert _capture(models_cli.stream_reply, [{"choices": [{"delta": {"content": "Partial"}}]}]) == "Partial\n"
    assert models_cli.stream_reply([{"choices": [{"delta": {"content": "Partial"}}]}]) is None
    assert "stopped before the model finished" in warnings[0]


def test_read_stdin_stops_on_a_quiet_open_pipe_after_data(monkeypatch):
    monkeypatch.setattr(models_cli, "STDIN_WAIT_S", 0.05)
    read, write = os.pipe()
    os.write(write, b"context")  # data, but the writer stays open
    try:
        with os.fdopen(read) as pipe:
            monkeypatch.setattr(models_cli.sys, "stdin", pipe)
            assert READ_STDIN(wait=False) == "context"
    finally:
        os.close(write)


@pytest.mark.parametrize("show", [render.error, render.warning, render.info, render.success])
def test_message_text_is_not_read_as_markup(monkeypatch, show):
    buf = StringIO()
    monkeypatch.setattr(render, "console", Console(file=buf, color_system=None))
    show("echo 'hello [world]' [/]")
    assert "echo 'hello [world]' [/]" in buf.getvalue()


def test_confirm_text_is_not_read_as_markup(monkeypatch):
    buf = StringIO()
    console = Console(file=buf, color_system=None)
    monkeypatch.setattr(console, "input", lambda prompt, **k: console.print(prompt, end="") or "n")
    monkeypatch.setattr(render, "console", console)
    assert render.confirm("Delete deployment [/]?") is False
    assert "Delete deployment [/]?" in buf.getvalue()
