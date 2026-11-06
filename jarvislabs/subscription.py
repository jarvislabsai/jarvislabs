"""Subscription resource: the plan's usage, the wallet setting, and the billing page."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from jarvislabs.constants import MODELS_API_URL
from jarvislabs.exceptions import NotFoundError
from jarvislabs.models import SubscriptionUsage

if TYPE_CHECKING:
    from jarvislabs.transport import Transport


class Subscription:
    """The account's Models API subscription."""

    def __init__(self, transport: Transport) -> None:
        """Wrap the transport shared with the other resources.

        Args:
            transport: The authenticated transport, holding the Jarvis key.
        """
        self._t = transport

    def usage(self) -> SubscriptionUsage | None:
        """Return the plan, paid period, windows and slots for the whole account.

        Returns:
            The account's usage, or None when it has no active subscription.
        """
        try:
            resp = self._t.request("GET", "subscription/usage", base_url=MODELS_API_URL)
        except NotFoundError:
            return None
        return SubscriptionUsage.model_validate(resp)

    def set_use_balance(self, on: bool) -> bool:
        """Set whether the wallet pays for requests once the plan's limits are reached.

        Saves the setting on the account's subscription.

        Args:
            on: True to pay from the wallet, False to refuse those requests.

        Returns:
            The setting the server saved.

        Raises:
            NotFoundError: The account has never subscribed.
        """
        resp = self._t.request("PUT", "subscription/use_balance", json={"enabled": on})
        return cast("dict[str, bool]", resp)["use_balance"]

    def portal_url(self) -> str:
        """Return a one-visit link to the account's Stripe billing page.

        Returns:
            The billing page URL, which expires after a few minutes.

        Raises:
            NotFoundError: The account has never subscribed.
        """
        return cast("dict[str, str]", self._t.request("POST", "subscription/portal"))["url"]
