from __future__ import annotations

import os
from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from jarvislabs.cli import models as models_cli, render, state
from jarvislabs.constants import MODELS_API_URL
from jarvislabs.exceptions import InsufficientBalanceError, JarvislabsError, NotFoundError, error_from_response
from jarvislabs.models import ModelInfo
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
def _setup(monkeypatch):
    monkeypatch.setattr(render, "spinner", lambda *a, **k: nullcontext())
    monkeypatch.setattr(models_cli, "_read_stdin", lambda: "")
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
        "POST", "chat/completions", json={"model": "qwen", "messages": HELLO, "max_tokens": 10}, base_url=MODELS_API_URL
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
    assert render._thinking_lines(None) == "[dim]Not supported[/dim]"


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
    monkeypatch.setattr(render, "select_row", lambda message, headings, colors, rows, values: seen.extend(rows))
    render.select_model([ModelInfo(**QWEN)])
    assert seen == [("qwen", "128K", "$0.20", "$0.80")]


# ── chat ────────────────────────────────────────────────────────────────────────


def _chat_client(monkeypatch, chunks=(), reply=None):
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
        {"choices": [{"delta": {"content": "there"}}]},
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
    monkeypatch.setattr(models_cli, "_read_stdin", lambda: "file text")
    monkeypatch.setattr(render, "chat_usage", lambda *a: None)
    models_cli.models_chat("deepseek", "Summarize", system="Be brief.", max_tokens=50, thinking=False, effort="high")
    assert calls == [
        (
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Summarize\n\nfile text"}],
            {"chat_template_kwargs": {"thinking": False, "reasoning_effort": "high"}, "max_tokens": 50},
        )
    ]


def test_chat_json(monkeypatch):
    _chat_client(monkeypatch, reply={"id": "r1"})
    printed = []
    monkeypatch.setattr(render, "print_json", printed.append)
    models_cli.models_chat("deepseek", "Hello", json_output=True)
    assert printed == [{"id": "r1"}]


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
        assert READ_STDIN() == "piped text"
    with open(os.devnull) as devnull:
        monkeypatch.setattr(models_cli.sys, "stdin", devnull)
        assert READ_STDIN() == ""
