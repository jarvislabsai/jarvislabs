"""Serverless region routing: normalize region codes, build host URLs, fan out
reads across regions, and resolve a deployment id to the region that owns it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from jarvislabs.constants import (
    DEPLOYMENT_TRANSIENT_STATUS,
    REGION_CODE_TO_REGION,
    REGION_DISPLAY_CODES,
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


def region_display(region: str | None) -> str | None:
    """Internal region id -> display code (IN1/IN2); None passes through."""
    return None if region is None else REGION_DISPLAY_CODES.get(region, region)


def serverless_region_url(region: str) -> str:
    """Serverless host URL for an internal region id."""
    try:
        return SERVERLESS_REGION_URLS[region]
    except KeyError:
        raise ValidationError(f"No serverless host for region {region_display(region)}.") from None


def _display_order(regions: frozenset[str]) -> list[str]:
    """Internal region ids sorted by display code (IN1, IN2, ...)."""
    return sorted(regions, key=lambda region: region_display(region) or region)


def _valid_codes(regions: frozenset[str]) -> str:
    return ", ".join(region_display(region) for region in _display_order(regions))


def normalize_serverless_region(region: str) -> str:
    """Accept a display code (IN2) or internal id (case-insensitive) and return
    the internal region id, validated against the serverless allowed set."""
    key = region.strip().lower()
    resolved = REGION_CODE_TO_REGION.get(key, key)
    if resolved not in SERVERLESS_REGIONS:
        shown = region_display(resolved)
        raise ValidationError(
            f"Region {shown} is not a valid serverless region. Use one of: {_valid_codes(SERVERLESS_REGIONS)}."
        )
    return resolved


def fan_out_read(
    op: Callable[[str], T],
    *,
    regions: frozenset[str] = SERVERLESS_REGIONS,
) -> tuple[list[tuple[str, T]], list[tuple[str, str]]]:
    """Call ``op(region)`` across regions in display order.

    Returns (found, unreachable): found is [(region, result)] where the id was
    present; unreachable is [(region, message)] for regions that could not be
    reached. Not-found regions are skipped; auth and other errors re-raise.
    """
    found: list[tuple[str, T]] = []
    unreachable: list[tuple[str, str]] = []
    for region in _display_order(regions):
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


def resolve_region(
    deployment_id: str,
    op: Callable[[str], T],
    *,
    hint: str | None = None,
    cache: dict[str, str] | None = None,
    regions: frozenset[str] = SERVERLESS_REGIONS,
) -> tuple[str, T]:
    """Resolve a deployment id to its region and the result of ``op(region)``.

    Tries the hint or cached region first, otherwise checks each region. Raises if
    the id matches more than one region or if a region can't be reached.
    """
    fast_region = hint or (cache.get(deployment_id) if cache is not None else None)
    if fast_region is not None and fast_region in regions:
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

    found, unreachable = fan_out_read(op, regions=regions)

    if len(found) == 1:
        region, result = found[0]
        if cache is not None:
            cache[deployment_id] = region
        return region, result

    if len(found) > 1:
        a, b = (region_display(region) for region, _ in found[:2])
        raise RegionResolutionError(f"Deployment {deployment_id} found in {a} and {b} — pass --region to disambiguate.")

    if unreachable:
        codes = ", ".join(region_display(region) for region, _ in unreachable)
        raise RegionResolutionError(
            f"Deployment {deployment_id} not found; could not check {codes}. Retry, or pass --region."
        )

    raise NotFoundError(f"Deployment {deployment_id} not found in any region.")
