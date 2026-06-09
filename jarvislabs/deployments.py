"""Serverless model deployments resource."""

from __future__ import annotations

import contextlib
import time
from datetime import datetime
from typing import TYPE_CHECKING

from jarvislabs.constants import (
    DEPLOYMENT_POLL_INTERVAL_S,
    DEPLOYMENT_POLL_MAX_TRANSIENT_ERRORS,
    DEPLOYMENT_TERMINAL_FAILURE,
    DEPLOYMENT_TERMINAL_OTHER,
    DEPLOYMENT_TERMINAL_SUCCESS,
    DEPLOYMENT_TRANSIENT_STATUS,
    DEPLOYMENT_WAIT_TIMEOUT_S,
)
from jarvislabs.exceptions import APIError, JarvislabsError, NotFoundError, ValidationError
from jarvislabs.models import (
    Deployment,
    DeploymentListResult,
    DeploymentStatus,
    DeploymentSummary,
    RegionError,
)
from jarvislabs.regions import (
    normalize_serverless_region,
    region_code,
    resolve_deployment_region,
    search_serverless_regions,
    serverless_region_url,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvislabs.transport import Transport


class Deployments:
    """Serverless model deployments: create, list, get, status, update, delete.

    Resolves a deployment's region (with an in-memory id->region cache) and routes
    each call to the right serverless host.
    """

    def __init__(self, transport: Transport) -> None:
        self._t = transport
        self._region_cache: dict[str, str] = {}

    def create(
        self,
        *,
        name: str,
        region: str,
        framework: str,
        gpu: str,
        model: str,
        gpus_per_worker: int,
        min_workers: int,
        max_workers: int,
        idle_timeout: int,
        wait_time: int,
        storage: int,
        concurrent: int | None = None,
        args: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        wait: bool = True,
    ) -> Deployment | str:
        """Create a deployment in the chosen region.

        ``wait=True`` polls until the deployment is running and returns the final
        ``Deployment``; ``wait=False`` returns the deployment id.
        """
        internal_region = normalize_serverless_region(region)
        payload: dict = {
            "name": name,
            "region": internal_region,
            "framework": framework,
            "gpus_to_use": {"gpus": [gpu]},
            "gpus_per_worker": gpus_per_worker,
            "min_workers": min_workers,
            "max_workers": max_workers,
            "idle_timeout": idle_timeout,
            "wait_time": wait_time,
            "storage": storage,
            "args": {**(args or {}), "model": model},
        }
        if env is not None:
            payload["env"] = env
        if concurrent is not None:
            payload["concurrent_requests"] = concurrent

        base_url = serverless_region_url(internal_region)
        resp = self._t.request("POST", "management/create", json=payload, base_url=base_url)
        deployment_id = resp.get("deployment_id") if isinstance(resp, dict) else None
        if not deployment_id:
            raise JarvislabsError("Failed to create deployment: no deployment_id returned.")

        self._region_cache[deployment_id] = internal_region
        if not wait:
            return deployment_id
        return self.wait_until_running(deployment_id, region=internal_region)

    def list(self) -> DeploymentListResult:
        """Fan out across all serverless regions, merge, and sort newest-first."""

        def fetch(region: str) -> list[DeploymentSummary]:
            base_url = serverless_region_url(region)
            resp = self._t.request("GET", "management/list", base_url=base_url)
            items = resp.get("deployments", []) if isinstance(resp, dict) else []
            return [DeploymentSummary(**{**item, "region": region}) for item in items]

        found, unreachable = search_serverless_regions(fetch)
        if not found and unreachable:
            raise JarvislabsError("; ".join(f"{region_code(r)}: {msg}" for r, msg in unreachable))

        deployments: list[DeploymentSummary] = []
        for region, rows in found:
            for row in rows:
                self._region_cache[row.deployment_id] = region
                deployments.append(row)
        deployments.sort(key=_summary_sort_key, reverse=True)

        region_errors = [RegionError(region=region, error=msg) for region, msg in unreachable]
        return DeploymentListResult(deployments=deployments, region_errors=region_errors)

    def get(self, deployment_id: str, *, region: str | None = None) -> Deployment:
        """Full record. Fast-path a region hint/cache; otherwise fan out."""
        _, deployment = self._resolve(deployment_id, region, self._fetch_get)
        return deployment

    def status(self, deployment_id: str, *, region: str | None = None) -> DeploymentStatus:
        """Lightweight status with the resolved region attached."""
        resolved, status = self._resolve(deployment_id, region, self._fetch_status)
        return DeploymentStatus(deployment_id=deployment_id, region=resolved, status=status)

    def update(
        self,
        deployment_id: str,
        *,
        name: str | None = None,
        idle_timeout: int | None = None,
        wait_time: int | None = None,
        region: str | None = None,
    ) -> Deployment:
        """Patch name/idle_timeout/wait_time on a running deployment, then re-fetch."""
        patch: dict = {}
        if name is not None:
            patch["name"] = name
        if idle_timeout is not None:
            patch["idle_timeout"] = idle_timeout
        if wait_time is not None:
            patch["wait_time"] = wait_time
        if not patch:
            raise ValidationError("Pass at least one of name / idle_timeout / wait_time.")

        resolved, deployment = self._resolve(deployment_id, region, self._fetch_get)
        base_url = serverless_region_url(resolved)
        try:
            self._t.request("PATCH", f"management/{deployment_id}", json=patch, base_url=base_url)
        except APIError as exc:
            if exc.status_code == 409:
                raise ValidationError(
                    f"Deployment is in '{deployment.status}'; only running deployments can be updated. "
                    "Min/max workers are not patchable — recreate to rescale."
                ) from exc
            raise
        return self.get(deployment_id, region=resolved)

    def delete(self, deployment_id: str, *, region: str | None = None) -> bool:
        """Find the deployment's region, then delete it."""
        resolved, _ = self._resolve(deployment_id, region, self._fetch_get)
        base_url = serverless_region_url(resolved)
        with contextlib.suppress(NotFoundError):
            self._t.request("DELETE", f"management/{deployment_id}", base_url=base_url)
        self._region_cache.pop(deployment_id, None)
        return True

    # ── helpers ──

    def wait_until_running(
        self, deployment_id: str, *, region: str | None = None, timeout: float = DEPLOYMENT_WAIT_TIMEOUT_S
    ) -> Deployment:
        """Poll status until a terminal state, or until ``timeout`` seconds elapse.

        The poller recognizes terminal statuses (anything else means in-progress);
        the timeout is a backstop so an unrecognized terminal status can't poll forever.
        """
        resolved = self._resolve_for_read(deployment_id, region)
        base_url = serverless_region_url(resolved)
        transient = 0
        deadline = time.monotonic() + timeout
        while True:
            try:
                resp = self._t.request("GET", f"management/{deployment_id}/status", base_url=base_url)
                status = resp.get("status") if isinstance(resp, dict) else None
                transient = 0
            except NotFoundError:
                transient += 1
                if transient >= DEPLOYMENT_POLL_MAX_TRANSIENT_ERRORS:
                    raise
                time.sleep(DEPLOYMENT_POLL_INTERVAL_S)
                continue
            except APIError as exc:
                if exc.status_code not in DEPLOYMENT_TRANSIENT_STATUS:
                    raise
                transient += 1
                if transient >= DEPLOYMENT_POLL_MAX_TRANSIENT_ERRORS:
                    raise
                time.sleep(DEPLOYMENT_POLL_INTERVAL_S)
                continue

            if status in DEPLOYMENT_TERMINAL_SUCCESS:
                self._region_cache[deployment_id] = resolved
                return self._fetch_get(deployment_id, resolved)
            if status in DEPLOYMENT_TERMINAL_FAILURE:
                raise JarvislabsError(
                    f"Deployment {deployment_id} {status}: {self._failure_reason(deployment_id, resolved)}"
                )
            if status in DEPLOYMENT_TERMINAL_OTHER:
                raise JarvislabsError(f"Deployment {deployment_id} is being deleted (status: {status}).")

            if time.monotonic() >= deadline:
                raise JarvislabsError(
                    f"Stopped waiting on deployment {deployment_id} after {int(timeout)}s "
                    f"(last status: {status!r}). It may still be coming up — large model "
                    f"downloads can take a while. Check: jl deploy get {deployment_id}"
                )
            time.sleep(DEPLOYMENT_POLL_INTERVAL_S)

    def openai_base_url(self, deployment_id: str, *, region: str | None = None) -> str:
        """OpenAI-compatible base URL: ``<host>/openai/<id>`` (no /v1, no trailing slash)."""
        resolved = self._resolve_for_read(deployment_id, region)
        host = serverless_region_url(resolved).rstrip("/")
        return f"{host}/openai/{deployment_id}"

    def _resolve_hint(self, region: str | None) -> str | None:
        return normalize_serverless_region(region) if region else None

    def _resolve(
        self, deployment_id: str, region: str | None, fetch: Callable[[str, str], Deployment | str]
    ) -> tuple[str, Deployment | str]:
        """Resolve the deployment's region (hint/cache fast-path, else fan out), running ``fetch``."""
        return resolve_deployment_region(
            deployment_id,
            lambda r: fetch(deployment_id, r),
            hint=self._resolve_hint(region),
            cache=self._region_cache,
        )

    def _resolve_for_read(self, deployment_id: str, region: str | None) -> str:
        """Resolve a region for a read-only op: hint fast-path, else the cheap status endpoint."""
        hint = self._resolve_hint(region)
        if hint is not None:
            return hint
        resolved, _ = self._resolve(deployment_id, region, self._fetch_status)
        return resolved

    def _fetch_get(self, deployment_id: str, region: str) -> Deployment:
        base_url = serverless_region_url(region)
        resp = self._t.request("GET", f"management/{deployment_id}", base_url=base_url)
        return Deployment(**{**resp, "region": region})

    def _fetch_status(self, deployment_id: str, region: str) -> str:
        base_url = serverless_region_url(region)
        resp = self._t.request("GET", f"management/{deployment_id}/status", base_url=base_url)
        return resp.get("status", "") if isinstance(resp, dict) else ""

    def _failure_reason(self, deployment_id: str, region: str) -> str:
        """Return the deployment's error_message, or 'reason unavailable' if it can't be read."""
        try:
            return self._fetch_get(deployment_id, region).error_message or "reason unavailable"
        except NotFoundError:
            return "reason unavailable"


def _summary_sort_key(summary: DeploymentSummary) -> tuple[bool, datetime]:
    """Newest-first by start_time; rows with no start_time sort last."""
    start = summary.start_time
    if start is not None and start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    return (start is not None, start or datetime.min)
