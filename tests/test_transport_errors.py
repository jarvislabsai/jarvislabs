"""Tests for backend error-response parsing and the streaming transport."""

from __future__ import annotations

import httpx
import pytest

from jarvislabs.exceptions import (
    APIError,
    AuthError,
    InsufficientBalanceError,
    NotFoundError,
    error_from_response,
    extract_message,
    validation_field,
)
from jarvislabs.transport import Transport


@pytest.mark.parametrize(
    "loc, expected",
    [
        (["body", "storage"], "storage"),
        (["body"], None),
        (["body", "gpus", 0], "gpus"),
        ([], None),
        (None, None),
    ],
)
def test_validation_field(loc, expected):
    assert validation_field(loc) == expected


def _detail(**item):
    return {"detail": [item]}


def test_less_than_equal_prefixes_field():
    msg = extract_message(
        _detail(type="less_than_equal", msg="Input should be less than or equal to 1000", loc=["body", "storage"])
    )
    assert msg.startswith("storage: ")


def test_string_pattern_mismatch_hides_regex():
    msg = extract_message(
        _detail(type="string_pattern_mismatch", msg="String should match pattern '^(vllm)$'", loc=["body", "framework"])
    )
    assert msg == "Invalid framework"
    assert "^(vllm)$" not in msg


def test_legacy_regex_substring_still_rewritten():
    msg = extract_message(_detail(msg="value does not match regex", loc=["body", "framework"]))
    assert msg == "Invalid framework"


def test_string_too_short_prefixes_field():
    msg = extract_message(
        _detail(type="string_too_short", msg="String should have at least 1 character", loc=["body", "name"])
    )
    assert msg.startswith("name: ")


def test_body_only_loc_unchanged():
    msg = extract_message(_detail(type="value_error", msg="some error", loc=["body"]))
    assert msg == "some error"  # no "Invalid None", no prefix


def test_integer_loc_uses_last_string():
    msg = extract_message(_detail(type="value_error", msg="bad", loc=["body", "gpus", 0]))
    assert msg.startswith("gpus: ")


def test_missing_loc_unchanged():
    msg = extract_message(_detail(type="value_error", msg="standalone"))
    assert msg == "standalone"


def test_non_dict_item_degrades():
    msg = extract_message({"detail": ["just a string"]})
    assert msg == "just a string"


def test_non_string_msg_does_not_crash():
    msg = extract_message(_detail(type="value_error", msg=12345, loc=["body", "storage"]))
    assert "storage" in msg


def test_message_and_error_shapes():
    assert extract_message({"message": "boom"}) == "boom"
    assert extract_message({"error": "kaboom"}) == "kaboom"
    assert extract_message({"detail": "plain detail"}) == "plain detail"


def test_error_from_response_maps_status_codes():
    assert isinstance(error_from_response(401, {"message": "nope"}), AuthError)
    assert isinstance(error_from_response(404, {"message": "gone"}), NotFoundError)
    assert isinstance(error_from_response(403, {"message": "Insufficient balance"}), InsufficientBalanceError)


def test_error_from_response_403_without_balance_is_generic():
    err = error_from_response(403, {"message": "Machine does not belong to this user"})
    assert isinstance(err, APIError) and not isinstance(err, InsufficientBalanceError)
    assert err.status_code == 403


def test_error_from_response_unmapped_status_carries_code():
    err = error_from_response(500, {"message": "kaboom"})
    assert isinstance(err, APIError)
    assert err.status_code == 500
    assert err.message == "kaboom"


# ── stream_lines ──


def _stream_transport(handler) -> Transport:
    """A Transport whose httpx client talks to an in-memory handler."""
    transport = Transport(token="t")
    transport._client = httpx.Client(transport=httpx.MockTransport(handler))
    return transport


def test_stream_lines_yields_lines():
    def handler(request):
        return httpx.Response(200, content=b"data: [worker-1] hello\n\n: ping\n\ndata: [worker-1] bye\n\n")

    t = _stream_transport(handler)
    lines = list(t.stream_lines("management/dep1/logs", base_url="https://example.test"))
    assert "data: [worker-1] hello" in lines
    assert ": ping" in lines
    assert "data: [worker-1] bye" in lines


def test_stream_lines_passes_params():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, content=b"")

    t = _stream_transport(handler)
    list(t.stream_lines("management/dep1/logs", params={"tail": 100, "follow": True}, base_url="https://example.test"))
    assert seen["params"] == {"tail": "100", "follow": "true"}


def test_stream_lines_maps_http_errors():
    def handler(request):
        return httpx.Response(404, json={"detail": "Deployment not found"})

    t = _stream_transport(handler)
    with pytest.raises(NotFoundError, match="Deployment not found"):
        list(t.stream_lines("management/dep1/logs", base_url="https://example.test"))


def test_stream_lines_maps_429_with_detail():
    def handler(request):
        return httpx.Response(429, json={"detail": "Too many concurrent log streams"})

    t = _stream_transport(handler)
    with pytest.raises(APIError, match="Too many concurrent log streams"):
        list(t.stream_lines("management/dep1/logs", base_url="https://example.test"))


def test_stream_lines_wraps_connection_errors():
    def handler(request):
        raise httpx.ConnectError("boom")

    t = _stream_transport(handler)
    with pytest.raises(APIError, match="Stream failed"):
        list(t.stream_lines("management/dep1/logs", base_url="https://example.test"))


def test_stream_lines_wraps_timeouts():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    t = _stream_transport(handler)
    with pytest.raises(APIError, match="Stream timed out"):
        list(t.stream_lines("management/dep1/logs", base_url="https://example.test"))
