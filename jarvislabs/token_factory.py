from __future__ import annotations

import json
from typing import TYPE_CHECKING

from jarvislabs.constants import MODELS_API_URL
from jarvislabs.exceptions import JarvislabsError
from jarvislabs.models import ModelInfo

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jarvislabs.transport import Transport


class Models:
    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def list(self) -> list[ModelInfo]:
        resp = self._t.request("GET", "models", base_url=MODELS_API_URL)
        return [ModelInfo(**item) for item in resp["data"]]

    def get(self, model_id: str) -> ModelInfo:
        return ModelInfo(**self._t.request("GET", f"models/{model_id}", base_url=MODELS_API_URL))

    # Chat calls default to the Token Factory; pass ``base_url`` to call another
    # OpenAI-compatible endpoint, such as a deployment's ``openai_base_url``, and
    # ``read_timeout`` (seconds) when the first reply can be slow, e.g. while a worker starts.
    def chat(
        self,
        model_id: str,
        messages: list[dict],
        *,
        base_url: str = MODELS_API_URL,
        read_timeout: float | None = None,
        **params,
    ) -> dict:
        body = {"model": model_id, "messages": messages, **params}
        return self._t.request("POST", "chat/completions", json=body, base_url=base_url, read_timeout=read_timeout)

    def chat_stream(
        self,
        model_id: str,
        messages: list[dict],
        *,
        base_url: str = MODELS_API_URL,
        read_timeout: float | None = None,
        **params,
    ) -> Iterator[dict]:
        body = {
            "model": model_id,
            "messages": messages,
            **params,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        lines = self._t.stream_lines(
            "chat/completions", method="POST", json=body, base_url=base_url, read_timeout=read_timeout
        )
        for line in lines:
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line.removeprefix("data: "))
            if "error" in chunk:
                raise JarvislabsError(chunk["error"]["message"])
            yield chunk
