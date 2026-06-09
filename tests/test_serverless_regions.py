from __future__ import annotations

import pytest

from jarvislabs import regions
from jarvislabs.constants import CHENNAI_REGION, EUROPE_REGION, INDIA_NOIDA_REGION
from jarvislabs.exceptions import APIError, AuthError, NotFoundError, RegionResolutionError, ValidationError
from jarvislabs.regions import (
    normalize_serverless_region,
    resolve_deployment_region,
    search_serverless_regions,
    serverless_region_url,
)

TWO_REGIONS = frozenset({INDIA_NOIDA_REGION, CHENNAI_REGION})


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
    assert regions.label(INDIA_NOIDA_REGION) == "IN2"
    assert regions.label(CHENNAI_REGION) == "IN1"
    assert regions.label(None) is None


# ── search_serverless_regions ───────────────────────────────────────────────────────────────


def test_fan_out_found():
    found, unreachable = search_serverless_regions(lambda region: f"r:{region}", allowed_regions=TWO_REGIONS)
    assert sorted(found) == [(CHENNAI_REGION, f"r:{CHENNAI_REGION}"), (INDIA_NOIDA_REGION, f"r:{INDIA_NOIDA_REGION}")]
    assert unreachable == []


def test_fan_out_404_skipped():
    def op(region):
        if region == CHENNAI_REGION:
            raise NotFoundError("nope")
        return "ok"

    found, unreachable = search_serverless_regions(op, allowed_regions=TWO_REGIONS)
    assert found == [(INDIA_NOIDA_REGION, "ok")]
    assert unreachable == []


def test_fan_out_transient_recorded_as_unreachable():
    def op(region):
        if region == CHENNAI_REGION:
            raise APIError(0, "connection failed")
        return "ok"

    found, unreachable = search_serverless_regions(op, allowed_regions=TWO_REGIONS)
    assert found == [(INDIA_NOIDA_REGION, "ok")]
    assert unreachable == [(CHENNAI_REGION, "connection failed")]


def test_fan_out_auth_reraises():
    def op(region):
        raise AuthError("bad token")

    with pytest.raises(AuthError):
        search_serverless_regions(op, allowed_regions=TWO_REGIONS)


def test_fan_out_non_transient_apierror_reraises():
    def op(region):
        raise APIError(400, "bad request")

    with pytest.raises(APIError):
        search_serverless_regions(op, allowed_regions=TWO_REGIONS)


# ── resolve_deployment_region ─────────────────────────────────────────────────────────────


def test_resolve_fast_path_hint_hit():
    calls: list[str] = []

    def op(region):
        calls.append(region)
        return "ok"

    cache: dict[str, str] = {}
    region, result = resolve_deployment_region(
        "dep1", op, hint=INDIA_NOIDA_REGION, cache=cache, allowed_regions=TWO_REGIONS
    )
    assert (region, result) == (INDIA_NOIDA_REGION, "ok")
    assert calls == [INDIA_NOIDA_REGION]  # no fan-out
    assert cache == {"dep1": INDIA_NOIDA_REGION}


def test_resolve_fast_path_cache_hit():
    calls: list[str] = []

    def op(region):
        calls.append(region)
        return "ok"

    cache = {"dep1": CHENNAI_REGION}
    region, _ = resolve_deployment_region("dep1", op, cache=cache, allowed_regions=TWO_REGIONS)
    assert region == CHENNAI_REGION
    assert calls == [CHENNAI_REGION]


def test_resolve_fast_path_404_evicts_and_falls_back():
    cache = {"dep1": CHENNAI_REGION}

    def op(region):
        if region == CHENNAI_REGION:
            raise NotFoundError("gone here")
        return "ok"

    region, result = resolve_deployment_region("dep1", op, cache=cache, allowed_regions=TWO_REGIONS)
    assert (region, result) == (INDIA_NOIDA_REGION, "ok")
    assert cache == {"dep1": INDIA_NOIDA_REGION}  # evicted then repopulated by fan-out


def test_resolve_exactly_one():
    def op(region):
        if region == INDIA_NOIDA_REGION:
            return "ok"
        raise NotFoundError("nope")

    cache: dict[str, str] = {}
    region, result = resolve_deployment_region("dep1", op, cache=cache, allowed_regions=TWO_REGIONS)
    assert (region, result) == (INDIA_NOIDA_REGION, "ok")
    assert cache == {"dep1": INDIA_NOIDA_REGION}


def test_resolve_collision_raises():
    with pytest.raises(RegionResolutionError) as exc:
        resolve_deployment_region("dep1", lambda region: "ok", allowed_regions=TWO_REGIONS)
    msg = str(exc.value)
    assert "IN1" in msg and "IN2" in msg
    assert "disambiguate" in msg


def test_resolve_zero_clean_not_found():
    def op(region):
        raise NotFoundError("nope")

    with pytest.raises(NotFoundError) as exc:
        resolve_deployment_region("dep1", op, allowed_regions=TWO_REGIONS)
    assert "not found in any region" in str(exc.value)


def test_resolve_zero_with_unreachable_raises_resolution():
    def op(region):
        if region == CHENNAI_REGION:
            raise APIError(503, "down")
        raise NotFoundError("nope")

    with pytest.raises(RegionResolutionError) as exc:
        resolve_deployment_region("dep1", op, allowed_regions=TWO_REGIONS)
    msg = str(exc.value)
    assert "could not check" in msg
    assert "IN1" in msg


def test_resolve_fast_path_transient_falls_back():
    cache = {"dep1": CHENNAI_REGION}

    def op(region):
        if region == CHENNAI_REGION:
            raise APIError(0, "timeout")
        return "ok"

    region, result = resolve_deployment_region("dep1", op, cache=cache, allowed_regions=TWO_REGIONS)
    assert (region, result) == (INDIA_NOIDA_REGION, "ok")
    assert cache == {"dep1": INDIA_NOIDA_REGION}
