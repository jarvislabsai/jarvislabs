"""Canonical region helpers.

The single home for everything region-related: translating between internal region ids
(e.g. "india-noida-01"), user-facing display codes (e.g. "IN2"), and backend base URLs —
plus serverless deployment routing (which serverless host owns a given deployment id).
Region *data* lives in ``constants.py``; this module is the logic that operates on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from jarvislabs.constants import (
    DEFAULT_REGION,
    DEPLOYMENT_TRANSIENT_STATUS,
    REGION_CODE_TO_REGION,
    REGION_DISPLAY_CODES,
    REGION_URLS,
    SERVERLESS_REGION_URLS,
    SERVERLESS_REGIONS,
)
from jarvislabs.exceptions import (
    APIError,
    AuthError,
    NotFoundError,
    RegionResolutionError,
    ValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


def label(region: str | None) -> str | None:
    """Internal region id -> user-facing display code (e.g. "india-noida-01" -> "IN2"). None passes through."""
    return REGION_DISPLAY_CODES.get(region, region)


def format_codes() -> str:
    """Valid region codes in product order: IN1, IN2, EU1."""
    return ", ".join(label(region) for region in REGION_DISPLAY_CODES)


def base_url(region: str | None) -> str:
    """Backend base URL for an internal region id (defaults to the default region)."""
    if region is None:
        return REGION_URLS[DEFAULT_REGION]
    try:
        return REGION_URLS[region]
    except KeyError as exc:
        raise ValidationError(f"Unknown region {region!r}. Use one of: {format_codes()}") from exc


def normalize_input(region: str | None) -> str | None:
    """Validate a user-supplied region (display code or internal id) and return the internal id.

    Returns None for a None/blank input. Raises ValidationError for an unknown region.
    """
    if region is None:
        return None

    normalized = region.strip()
    if not normalized:
        return None

    if normalized in REGION_DISPLAY_CODES:
        return normalized

    upper = normalized.upper()
    if upper in REGION_DISPLAY_CODES.values():
        return REGION_CODE_TO_REGION[upper.lower()]

    raise ValidationError(f"Unknown region {region!r}. Use one of: {format_codes()}")


# ── Serverless ───────────────────────────────────────────────────────────────
# Serverless runs as independent hosts per region (its own URL map and allowed set)
# with no global registry, so finding the region that owns a deployment means asking
# each host in turn. Everything below is serverless-specific.


def serverless_region_url(region: str) -> str:
    """Serverless host URL for an internal region id."""
    try:
        return SERVERLESS_REGION_URLS[region]
    except KeyError:
        raise ValidationError(f"No serverless host for region {label(region)}.") from None


def _serverless_valid_codes(allowed_regions: frozenset[str]) -> str:
    """Serverless region display codes, sorted, for error messages (e.g. 'IN1, IN2')."""
    return ", ".join(sorted(label(region) for region in allowed_regions))


def normalize_serverless_region(region: str) -> str:
    """Accept a display code (IN2) or internal id (case-insensitive) and return
    the internal region id, validated against the serverless allowed set."""
    key = region.strip().lower()
    resolved = REGION_CODE_TO_REGION.get(key, key)
    if resolved not in SERVERLESS_REGIONS:
        raise ValidationError(
            f"Region {label(resolved)} is not a valid serverless region. Use one of: {_serverless_valid_codes(SERVERLESS_REGIONS)}."
        )
    return resolved


def search_serverless_regions(
    op: Callable[[str], T],
    *,
    allowed_regions: frozenset[str] = SERVERLESS_REGIONS,
) -> tuple[list[tuple[str, T]], list[tuple[str, str]]]:
    """Call ``op(region)`` against every serverless region and collect the results.

    Returns (found, unreachable): found is [(region, result)] where the id was
    present; unreachable is [(region, message)] for regions that could not be
    reached. Not-found regions are skipped; auth and other errors re-raise.
    """
    found: list[tuple[str, T]] = []
    unreachable: list[tuple[str, str]] = []
    for region in sorted(allowed_regions):
        try:
            found.append((region, op(region)))
        except NotFoundError:
            continue
        except AuthError:
            raise
        except APIError as exc:
            if exc.status_code in DEPLOYMENT_TRANSIENT_STATUS:
                unreachable.append((region, str(exc)))
            else:
                raise
    return found, unreachable


def resolve_deployment_region(
    deployment_id: str,
    op: Callable[[str], T],
    *,
    hint: str | None = None,
    cache: dict[str, str] | None = None,
    allowed_regions: frozenset[str] = SERVERLESS_REGIONS,
) -> tuple[str, T]:
    """Resolve a deployment id to its region and the result of ``op(region)``.

    Tries the hint or cached region first and returns immediately on a hit (no fan-out).
    Otherwise checks each region and raises if the id matches more than one reachable
    region or if a region can't be reached.
    """
    fast_region = hint or (cache.get(deployment_id) if cache is not None else None)
    if fast_region is not None and fast_region in allowed_regions:
        try:
            result = op(fast_region)
        except NotFoundError:
            if cache is not None:
                cache.pop(deployment_id, None)
        except AuthError:
            raise
        except APIError as exc:
            if exc.status_code not in DEPLOYMENT_TRANSIENT_STATUS:
                raise
        else:
            if cache is not None:
                cache[deployment_id] = fast_region
            return fast_region, result

    found, unreachable = search_serverless_regions(op, allowed_regions=allowed_regions)

    if len(found) == 1:
        region, result = found[0]
        if cache is not None:
            cache[deployment_id] = region
        return region, result

    if len(found) > 1:
        a, b = (label(region) for region, _ in found[:2])
        raise RegionResolutionError(f"Deployment {deployment_id} found in {a} and {b} — pass --region to disambiguate.")

    if unreachable:
        codes = ", ".join(label(region) for region, _ in unreachable)
        raise RegionResolutionError(
            f"Deployment {deployment_id} not found; could not check {codes}. Retry, or pass --region."
        )

    raise NotFoundError(f"Deployment {deployment_id} not found in any region.")
