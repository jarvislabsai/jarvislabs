from __future__ import annotations

import os
from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace

import pytest
import typer
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
    monkeypatch.setattr(models_cli, "CHAT_SESSIONS_DIR", tmp_path)
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
        {"choices": [{"delta": {"content": "there"}}]},
        {"choices": [], "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}},
    ]
    _chat_client(monkeypatch, chunks)
    thinking, notes = [], []
    monkeypatch.setattr(render, "chat_thinking", thinking.append)
    monkeypatch.setattr(render, "info", notes.append)
    assert _capture(models_cli.models_chat, "deepseek", "Hello") == "Hi there\n"
    assert thinking == ["Hmm.", "\n\n"]
    assert notes[0] == "\n1,000,000 in · 1,000,000 out · $1.000000"
    assert notes[1].startswith("Session ")


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
    assert printed[0]["session"]


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


# ── chat sessions ───────────────────────────────────────────────────────────────


@pytest.fixture
def sessions_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(models_cli.uuid, "uuid4", lambda: SimpleNamespace(hex="abc12345ffff"))
    monkeypatch.setattr(render, "chat_usage", lambda *a: None)
    return tmp_path


def test_session_keeps_history_across_calls(monkeypatch, sessions_dir):
    calls = _chat_client(monkeypatch, [{"choices": [{"delta": {"content": "Hi Jay"}}]}])
    notes = []
    monkeypatch.setattr(render, "info", notes.append)
    _capture(models_cli.models_chat, "deepseek", "I am Jay", system="Be brief.")
    assert notes[-1].startswith("Session abc12345 ")

    _capture(models_cli.models_chat, "deepseek", "Who am I?", session="abc12345")
    assert calls[1][0] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "I am Jay"},
        {"role": "assistant", "content": "Hi Jay"},
        {"role": "user", "content": "Who am I?"},
    ]


def test_session_json_saves_reply_and_returns_id(monkeypatch, sessions_dir):
    reply = {"choices": [{"message": {"role": "assistant", "content": "Hi", "reasoning_content": "Hmm"}}]}
    _chat_client(monkeypatch, reply=reply)
    printed = []
    monkeypatch.setattr(render, "print_json", printed.append)
    models_cli.models_chat("deepseek", "Hello", json_output=True)
    assert printed[0]["session"] == "abc12345"
    saved = models_cli._load_session("abc12345")
    assert saved[-1] == {"role": "assistant", "content": "Hi"}


def test_session_not_saved_when_reply_stopped(monkeypatch, sessions_dir):
    def chat_stream(model_id, messages, **params):
        raise KeyboardInterrupt
        yield

    _client(monkeypatch, get=lambda model_id: ModelInfo(**DEEPSEEK), chat_stream=chat_stream)
    monkeypatch.setattr(render, "info", lambda m: None)
    with pytest.raises(typer.Exit):
        _capture(models_cli.models_chat, "deepseek", "Hello")
    assert not list(sessions_dir.glob("*.json"))


def test_end_deletes_session(monkeypatch, sessions_dir):
    models_cli._save_session("abc123", [{"role": "user", "content": "Hi"}])
    monkeypatch.setattr(render, "success", lambda m: None)
    models_cli.models_chat(end="abc123")
    assert not (sessions_dir / "abc123.json").exists()


def test_unused_sessions_expire(sessions_dir):
    models_cli._save_session("old", [])
    models_cli._save_session("fresh", [])
    stale = models_cli.time.time() - models_cli.CHAT_SESSION_TTL_S - 1
    os.utime(sessions_dir / "old.json", (stale, stale))
    models_cli._prune_sessions()
    assert [p.name for p in sessions_dir.glob("*.json")] == ["fresh.json"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"session": "missing"},
            "Session missing not found or expired. Send a prompt without --session to start a new one.",
        ),
        ({"session": "../etc"}, "Invalid session ID: ../etc"),
    ],
)
def test_session_errors(monkeypatch, sessions_dir, options, message):
    _chat_client(monkeypatch)
    errors = _errors(monkeypatch)
    with pytest.raises(SystemExit):
        models_cli.models_chat("deepseek", "Hi", **options)
    assert errors == [message]


def test_session_needs_a_prompt(monkeypatch, sessions_dir):
    _chat_client(monkeypatch)
    errors = _errors(monkeypatch)
    with pytest.raises(SystemExit):
        models_cli.models_chat("deepseek", session="abc123")
    assert errors == ["Pass a prompt with --session or -c. See: jl models chat --help"]


def test_continue_uses_most_recent_session(monkeypatch, sessions_dir):
    models_cli._save_session("older", [{"role": "user", "content": "old"}])
    models_cli._save_session("newer", [{"role": "user", "content": "new"}])
    stale = models_cli.time.time() - 60
    os.utime(sessions_dir / "older.json", (stale, stale))
    calls = _chat_client(monkeypatch)
    monkeypatch.setattr(render, "info", lambda m: None)
    _capture(models_cli.models_chat, "deepseek", "Next", continue_session=True)
    assert calls[0][0][0] == {"role": "user", "content": "new"}


def test_continue_without_sessions(monkeypatch, sessions_dir):
    _chat_client(monkeypatch)
    errors = _errors(monkeypatch)
    with pytest.raises(SystemExit):
        models_cli.models_chat("deepseek", "Hi", continue_session=True)
    assert errors == ["No session to continue. Send a prompt without -c to start one."]


def test_end_unknown_session(monkeypatch, sessions_dir):
    errors = _errors(monkeypatch)
    with pytest.raises(SystemExit):
        models_cli.models_chat(end="missing")
    assert errors == ["Session missing not found or already ended."]


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
