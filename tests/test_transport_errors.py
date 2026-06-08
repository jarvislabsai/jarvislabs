"""Tests for the HTTP error-message extractor."""

from __future__ import annotations

import pytest

from jarvislabs.transport import _extract_error_message, _validation_field


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
    assert _validation_field(loc) == expected


def _detail(**item):
    return {"detail": [item]}


def test_less_than_equal_prefixes_field():
    msg = _extract_error_message(
        _detail(type="less_than_equal", msg="Input should be less than or equal to 1000", loc=["body", "storage"])
    )
    assert msg.startswith("storage: ")


def test_string_pattern_mismatch_hides_regex():
    msg = _extract_error_message(
        _detail(type="string_pattern_mismatch", msg="String should match pattern '^(vllm)$'", loc=["body", "framework"])
    )
    assert msg == "Invalid framework"
    assert "^(vllm)$" not in msg


def test_legacy_regex_substring_still_rewritten():
    msg = _extract_error_message(_detail(msg="value does not match regex", loc=["body", "framework"]))
    assert msg == "Invalid framework"


def test_string_too_short_prefixes_field():
    msg = _extract_error_message(
        _detail(type="string_too_short", msg="String should have at least 1 character", loc=["body", "name"])
    )
    assert msg.startswith("name: ")


def test_body_only_loc_unchanged():
    msg = _extract_error_message(_detail(type="value_error", msg="some error", loc=["body"]))
    assert msg == "some error"  # no "Invalid None", no prefix


def test_integer_loc_uses_last_string():
    msg = _extract_error_message(_detail(type="value_error", msg="bad", loc=["body", "gpus", 0]))
    assert msg.startswith("gpus: ")


def test_missing_loc_unchanged():
    msg = _extract_error_message(_detail(type="value_error", msg="standalone"))
    assert msg == "standalone"


def test_non_dict_item_degrades():
    msg = _extract_error_message({"detail": ["just a string"]})
    assert msg == "just a string"


def test_non_string_msg_does_not_crash():
    msg = _extract_error_message(_detail(type="value_error", msg=12345, loc=["body", "storage"]))
    assert "storage" in msg


def test_message_and_error_shapes():
    assert _extract_error_message({"message": "boom"}) == "boom"
    assert _extract_error_message({"error": "kaboom"}) == "kaboom"
    assert _extract_error_message({"detail": "plain detail"}) == "plain detail"
