"""A local stand-in for the Models API's Anthropic Messages and Responses endpoints.

The real endpoints (`/v1/messages`, `/v1/responses`) aren't public yet, so the preview harnesses
are verified against this instead: it speaks enough of both protocols for Claude Code and Codex
to complete one turn, records every request, and returns a fixed reply.

Run it by hand while trying a client:

    uv run python -m tests.cli.mock_endpoints --port 8765 --log /tmp/requests.jsonl

or import ``MockModelsAPI`` from a test (it picks a free port).
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPLY = "Hi from the mock."
MODEL = "deepseek-v4.1-flash-1017"


def _message(text: str, model: str) -> dict[str, Any]:
    return {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 8, "output_tokens": 5},
    }


def _response(text: str, model: str) -> dict[str, Any]:
    return {
        "id": "resp_mock",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_mock",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
    }


def _anthropic_events(text: str, model: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 8, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


def _responses_events(text: str, model: str) -> list[tuple[str, dict[str, Any]]]:
    item = {
        "type": "message",
        "id": "msg_mock",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    return [
        (
            "response.created",
            {"type": "response.created", "response": {**_response("", model), "status": "in_progress", "output": []}},
        ),
        ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0, "item": item}),
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": "msg_mock",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": "msg_mock",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            },
        ),
        (
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": "msg_mock",
                "output_index": 0,
                "content_index": 0,
                "text": text,
            },
        ),
        (
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": "msg_mock",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        ),
        (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_mock",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                },
            },
        ),
        ("response.completed", {"type": "response.completed", "response": _response(text, model)}),
    ]


class _Handler(BaseHTTPRequestHandler):
    server_version = "jl-mock/1"

    def log_message(self, *args: object) -> None:
        """Keep the test output for the client, not the mock's access log."""

    def _send(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _event_stream(self, events: list[tuple[str, dict[str, Any]]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for name, data in events:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL,
                            "object": "model",
                            "name": "DeepSeek V4.1 Flash",
                            "context_length": 1048576,
                        }
                    ],
                },
            )
            return
        self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            self._send(400, {"error": "body is not JSON"})
            return
        record = {
            "path": self.path,
            "headers": {name.lower(): value for name, value in self.headers.items()},
            "body": body,
        }
        self.server.requests.append(record)  # type: ignore[attr-defined]
        if self.server.log is not None:  # type: ignore[attr-defined]
            with self.server.log.open("a", encoding="utf-8") as fh:  # type: ignore[attr-defined]
                fh.write(json.dumps(record) + "\n")
        model = str(body.get("model") or MODEL)
        route = urlparse(self.path).path.rstrip("/")
        if route == "/v1/messages":
            if body.get("stream"):
                self._event_stream(_anthropic_events(REPLY, model))
            else:
                self._send(200, _message(REPLY, model))
            return
        if route == "/v1/responses":
            if body.get("stream"):
                self._event_stream(_responses_events(REPLY, model))
            else:
                self._send(200, _response(REPLY, model))
            return
        self._send(404, {"error": f"no route {self.path}"})


class MockModelsAPI:
    """A mock Anthropic/Responses server on a free local port, recording every request."""

    def __init__(self, log: Path | None = None) -> None:
        """Bind the server; call start() to serve in a background thread.

        Args:
            log: Write one JSON line per request here, or None to only keep them in memory.
        """
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.requests = []  # type: ignore[attr-defined]
        self._httpd.log = log  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The port the server is bound to."""
        return int(self._httpd.server_address[1])

    @property
    def requests(self) -> list[dict[str, Any]]:
        """Every request received so far, in order."""
        return self._httpd.requests  # type: ignore[attr-defined]

    def start(self) -> MockModelsAPI:
        """Serve in a daemon thread and return self."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and close the socket."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> MockModelsAPI:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def main() -> None:
    """Run the mock until Ctrl-C, printing the port to point a client at."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default 8765).")
    parser.add_argument("--log", type=Path, help="Write each request as a JSON line here.")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    server.requests = []  # type: ignore[attr-defined]
    server.log = args.log  # type: ignore[attr-defined]
    print(f"mock Models API on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
