from __future__ import annotations

import pytest

from jarvislabs import regions
from jarvislabs.constants import CHENNAI_REGION, EUROPE_REGION, INDIA_NOIDA_REGION
from jarvislabs.exceptions import ValidationError
from jarvislabs.regions import normalize_serverless_region, serverless_region_url

# ── normalize_serverless_region ────────────────────────────────────────────────


def test_normalize_accepts_display_code():
    assert normalize_serverless_region("IN2") == INDIA_NOIDA_REGION
    assert normalize_serverless_region("in2") == INDIA_NOIDA_REGION


def test_normalize_accepts_internal_id():
    assert normalize_serverless_region(INDIA_NOIDA_REGION) == INDIA_NOIDA_REGION


def test_normalize_accepts_in1():
    # Serverless is now live in Chennai (IN1), so it normalizes instead of rejecting.
    assert normalize_serverless_region("IN1") == CHENNAI_REGION


def test_normalize_rejects_eu():
    with pytest.raises(ValidationError):
        normalize_serverless_region("EU1")


def test_normalize_rejects_bad_string():
    with pytest.raises(ValidationError):
        normalize_serverless_region("nope")


# ── serverless_region_url / region label ───────────────────────────────────────


def test_serverless_region_url_known():
    assert serverless_region_url(INDIA_NOIDA_REGION) == "https://serverlessn.jarvislabs.net/"


def test_serverless_region_url_unknown_raises():
    with pytest.raises(ValidationError):
        serverless_region_url(EUROPE_REGION)


def test_region_label_maps_and_passes_none():
    assert regions.region_code(INDIA_NOIDA_REGION) == "IN2"
    assert regions.region_code(CHENNAI_REGION) == "IN1"
    assert regions.region_code(None) is None
