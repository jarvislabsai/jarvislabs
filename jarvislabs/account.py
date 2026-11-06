"""Account resource: balance, user info, resource metrics, catalog views."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from jarvislabs.constants import EUROPE_GPU_TYPES, EUROPE_REGION, REGION_URLS
from jarvislabs.models import (
    ApiKey,
    Balance,
    CreatedApiKey,
    ResourceMetrics,
    ServerMetaGPU,
    ServerMetaResponse,
    Template,
    UserInfo,
)
from jarvislabs.responses import _normalize_success

if TYPE_CHECKING:
    from jarvislabs.transport import Transport


class Account:
    """Balance, user info, and resource metrics."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def balance(self) -> Balance:
        resp = self._t.request("GET", "users/balance")
        return Balance(**resp)

    def user_info(self) -> UserInfo:
        resp = self._t.request("GET", "users/user_info")
        return UserInfo(**resp)

    def create_api_key(self, name: str, kind: Literal["payg", "subscription"]) -> CreatedApiKey:
        """Create a named API key on the account.

        The key counts towards the account's cap on active keys.

        Args:
            name: The key's name, shown on the console's API keys page.
            kind: "subscription" for a key charged to the plan, "payg" for one charged to
                the wallet.

        Returns:
            The new key, with its secret.

        Raises:
            APIError: With the server's message, when the name is already in use (409) or
                the account is at its cap on active keys (429).
        """
        resp = self._t.request("POST", "users/api_keys", json={"name": name, "kind": kind})
        return CreatedApiKey.model_validate(resp)

    def list_api_keys(self, kind: Literal["payg", "subscription"] | None = None) -> list[ApiKey]:
        """Return the account's active API keys, newest first, without their secrets.

        Args:
            kind: Only keys of this kind, or None for every key.
        """
        resp = self._t.request("GET", "users/api_keys", params={"kind": kind} if kind else None)
        return [ApiKey.model_validate(key) for key in resp]

    def revoke_api_key(self, key_id: str) -> None:
        """Revoke one of the account's API keys, so its requests are refused from now on.

        Revoking a key that is already revoked does nothing.

        Args:
            key_id: The key's ID.

        Raises:
            NotFoundError: When the account has no key with this ID.
        """
        self._t.request("DELETE", f"users/api_keys/{key_id}")

    def resource_metrics(self) -> ResourceMetrics:
        resp = self._t.request("GET", "misc/resource_metrics")
        return ResourceMetrics(**resp)

    def templates(self) -> list[Template]:
        resp = self._t.request("GET", "misc/frameworks")
        return [Template(**t) for t in resp.get("frameworks", [])]

    def gpu_availability(self) -> list[ServerMetaGPU]:
        """GPU types, pricing, and current availability for new creates."""
        return self.gpu_availability_from(self.resources())

    def gpu_availability_from(self, meta: ServerMetaResponse) -> list[ServerMetaGPU]:
        """Build GPU availability rows from a server_meta response already in memory."""
        return [_listing_gpu_view(gpu) for gpu in meta.server_meta if gpu.region in REGION_URLS]

    def resources(self) -> ServerMetaResponse:
        """Fetch the backend resource catalog used by jl resources."""
        resp = self._t.request("GET", "misc/server_meta")
        return ServerMetaResponse(**resp)

    def currency(self) -> str:
        """Return 'INR' or 'USD' based on user's payment location."""
        resp = self._t.request("GET", "misc/")
        return "INR" if _normalize_success(resp) else "USD"


def _listing_gpu_view(gpu: ServerMetaGPU) -> ServerMetaGPU:
    if gpu.region != EUROPE_REGION or gpu.gpu_type not in EUROPE_GPU_TYPES or gpu.num_free_devices <= 0:
        return gpu
    return gpu.model_copy(update={"num_free_devices": 1})
