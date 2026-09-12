"""Small, fail-closed HTTP adapter for Listonic's unofficial API."""

from __future__ import annotations

import collections.abc as c
import dataclasses
import datetime as dt
import logging
import threading
from urllib.parse import quote

import httpx

from ..auth.credentials import (
    ConnectionStatus,
    CredentialError,
    CredentialReference,
    CredentialStore,
    PermanentRefreshError,
    ProviderAccount,
    RefreshCallback,
)

logger = logging.getLogger(__name__)

LISTONIC_BASE_URL = "https://listonic.com"
LISTONIC_PROVIDER = "listonic"


class ListonicError(RuntimeError):
    """Base class for safe Listonic adapter errors."""


class ListonicAuthenticationError(ListonicError):
    """Raised when a Listonic account cannot authenticate."""


class ListonicForbiddenError(ListonicError):
    """Raised when Listonic refuses an authenticated request."""


class ListonicNotFoundError(ListonicError):
    """Raised when a Listonic object is not present."""


class ListonicConflictError(ListonicError):
    """Raised when Listonic reports a conflicting mutation."""


class ListonicInvalidRequestError(ListonicError):
    """Raised when Listonic rejects request validation."""


class ListonicRateLimitError(ListonicError):
    """Raised when Listonic asks the caller to slow down."""


class ListonicContractDriftError(ListonicError):
    """Raised when the pinned Listonic contract no longer matches."""


@dataclasses.dataclass(frozen=True, slots=True)
class ListonicCircuitBreaker:
    """Thread-safe-enough immutable-looking circuit breaker state holder.

    The lock is deliberately internal.  A single contract mismatch opens this
    provider-only breaker; it is never shared with another integration.
    """

    _state: list[bool] = dataclasses.field(
        default_factory=lambda: [False], repr=False, compare=False
    )
    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def is_open(self) -> bool:
        """Return whether Listonic calls have been disabled."""
        with self._lock:
            return self._state[0]

    @property
    def opened(self) -> bool:
        """Alias for :attr:`is_open` used by integration status displays."""
        return self.is_open

    @property
    def open(self) -> bool:
        """Alias for :attr:`is_open`."""
        return self.is_open

    def trip(self) -> None:
        """Open the breaker permanently until explicit operator reset."""
        with self._lock:
            self._state[0] = True

    def reset(self) -> None:
        """Close the breaker after the operator has verified the contract."""
        with self._lock:
            self._state[0] = False

    record_contract_drift = trip


# A shorter public name is useful to embedding applications.
ContractDriftCircuitBreaker = ListonicCircuitBreaker


@dataclasses.dataclass(frozen=True, slots=True)
class ShoppingList:
    """A Listonic list using provider IDs only inside the adapter."""

    id: str
    name: str
    item_count: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ShoppingItem:
    """A Listonic item using provider IDs only inside the adapter."""

    id: str
    list_id: str
    name: str
    amount: float | None = None
    unit: str | None = None
    checked: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class ListonicSessionToken:
    """Tokens imported from an isolated browser session."""

    access_token: str
    refresh_token: str
    expires_at: dt.datetime


class ListonicProvider:
    """Call only the verified Listonic paths with exact request field casing."""

    BASE_URL = LISTONIC_BASE_URL

    def __init__(
        self,
        credential_store: CredentialStore,
        *,
        accounts: c.Mapping[str, ProviderAccount] | None = None,
        enabled: bool = False,
        allow_unofficial: bool = False,
        http_client: httpx.Client | None = None,
        breaker: ListonicCircuitBreaker | None = None,
        refresh: RefreshCallback | None = None,
    ) -> None:
        """Create an adapter.

        Args:
            credential_store:
                The shared credential store.  Only its opaque reference is used.
            accounts (optional):
                Accounts keyed by local profile name.
            enabled (optional):
                Independent Listonic feature flag.
            allow_unofficial (optional):
                Explicit acknowledgement of the unofficial API.
            http_client (optional):
                A raw ``httpx.Client``; injection is intended for contract tests.
            breaker (optional):
                Provider-specific circuit breaker.
            refresh (optional):
                A previously verified refresh implementation.  None fails closed.
        """
        if accounts is None:
            accounts = {}
        self.credential_store = credential_store
        self.accounts = dict(accounts)
        self.enabled = enabled
        self.allow_unofficial = allow_unofficial
        self.client = http_client or httpx.Client(base_url=LISTONIC_BASE_URL)
        self._owns_client = http_client is None
        self.breaker = breaker or ListonicCircuitBreaker()
        self._refresh_callback = refresh
        self._session_tokens: dict[CredentialReference, ListonicSessionToken] = {}

    @property
    def available(self) -> bool:
        """Return whether calls are allowed by both gates and the breaker."""
        return self.enabled and self.allow_unofficial and not self.breaker.is_open

    @property
    def circuit_open(self) -> bool:
        """Return whether contract drift has disabled this provider."""
        return self.breaker.is_open

    @property
    def circuit_breaker(self) -> ListonicCircuitBreaker:
        """Return the Listonic-only circuit breaker."""
        return self.breaker

    @property
    def is_available(self) -> bool:
        """Alias for :attr:`available` used by final assembly."""
        return self.available

    def close(self) -> None:
        """Close an internally-owned HTTP client."""
        if self._owns_client:
            self.client.close()

    def register_account(self, account: ProviderAccount) -> None:
        """Route a newly onboarded account by its local profile."""
        self.accounts[account.profile] = account

    def register_session_tokens(
        self, account: ProviderAccount, tokens: ListonicSessionToken
    ) -> None:
        """Cache imported access state without persisting the access token."""
        self.register_account(account=account)
        self._session_tokens[account.credential_ref] = tokens

    def list_lists(self, account: ProviderAccount) -> list[ShoppingList]:
        """Return the account's shopping lists."""
        payload = self._request(account=account, method="GET", path="/api/lists")
        try:
            values = _collection(payload=payload, key_names=("lists", "Lists"))
            return [_shopping_list(value) for value in values]
        except ListonicContractDriftError:
            self.breaker.trip()
            raise

    def get_list(self, account: ProviderAccount, list_id: str) -> ShoppingList:
        """Return one shopping list by its exact provider ID."""
        payload = self._request(
            account=account, method="GET", path=f"/api/lists/{_path_id(list_id)}"
        )
        try:
            return _shopping_list(_object(payload=payload))
        except ListonicContractDriftError:
            self.breaker.trip()
            raise

    def list_items(self, account: ProviderAccount, list_id: str) -> list[ShoppingItem]:
        """Return all items for one shopping list."""
        payload = self._request(
            account=account, method="GET", path=f"/api/lists/{_path_id(list_id)}/items"
        )
        try:
            values = _collection(payload=payload, key_names=("items", "Items"))
            return [_shopping_item(value, list_id=list_id) for value in values]
        except ListonicContractDriftError:
            self.breaker.trip()
            raise

    def add_item(
        self,
        account: ProviderAccount,
        list_id: str,
        *,
        name: str,
        amount: float | None = None,
        unit: str | None = None,
    ) -> ShoppingItem:
        """Add one item using the verified mixed-case request contract."""
        payload = self._request(
            account=account,
            method="POST",
            path=f"/api/lists/{_path_id(list_id)}/items",
            json={"Amount": amount, "Unit": unit, "name": name},
        )
        try:
            return _shopping_item(_object(payload=payload), list_id=list_id)
        except ListonicContractDriftError:
            self.breaker.trip()
            raise

    def add_items(
        self,
        account: ProviderAccount,
        list_id: str,
        items: c.Iterable[dict[str, object]],
    ) -> list[ShoppingItem]:
        """Add items serially without retrying an uncertain mutation."""
        return [
            self.add_item(
                account=account,
                list_id=list_id,
                name=_required_string(item, "name"),
                amount=_number_or_none(item.get("amount")),
                unit=_string_or_none(item.get("unit")),
            )
            for item in items
        ]

    def set_item_checked(
        self, account: ProviderAccount, list_id: str, item_id: str, *, checked: bool
    ) -> ShoppingItem:
        """Check or uncheck one item using the verified PATCH casing."""
        payload = self._request(
            account=account,
            method="PATCH",
            path=f"/api/lists/{_path_id(list_id)}/items/{_path_id(item_id)}",
            json={"checked": checked, "itemId": item_id},
        )
        try:
            return _shopping_item(_object(payload=payload), list_id=list_id)
        except ListonicContractDriftError:
            self.breaker.trip()
            raise

    def remove_item(self, account: ProviderAccount, list_id: str, item_id: str) -> None:
        """Remove one exact item through the pinned item endpoint."""
        self._request(
            account=account,
            method="DELETE",
            path=f"/api/lists/{_path_id(list_id)}/items/{_path_id(item_id)}",
            expected_status=204,
        )

    def _request(
        self,
        *,
        account: ProviderAccount,
        method: str,
        path: str,
        json: dict[str, object] | None = None,
        expected_status: int | None = None,
    ) -> object:
        if not self.available:
            raise ListonicError("Listonic integration is unavailable")
        if account.provider != LISTONIC_PROVIDER:
            raise ListonicAuthenticationError("account is not a Listonic account")
        try:
            token = self._access_token(account=account)
            response = self.client.request(
                method,
                f"{LISTONIC_BASE_URL}{path}",
                headers={"Authorization": f"Bearer {token}"},
                json=json,
            )
        except ListonicError:
            raise
        except httpx.HTTPError, CredentialError, PermanentRefreshError:
            raise ListonicError("Listonic request failed") from None

        expected = expected_status or _EXPECTED_STATUS[(method, _path_kind(path))]
        if response.status_code != expected:
            self._raise_response_status(response=response, expected=expected)
        if expected == 204:
            return None
        try:
            return response.json()
        except ValueError, TypeError:
            self.breaker.trip()
            raise ListonicContractDriftError(
                "Listonic response format changed"
            ) from None

    def _access_token(self, account: ProviderAccount) -> str:
        if (
            self.credential_store.account_status(account.credential_ref)
            is not ConnectionStatus.CONNECTED
        ):
            raise ListonicAuthenticationError("Listonic authentication failed")
        cached = self._session_tokens.get(account.credential_ref)
        now = dt.datetime.now(dt.UTC)
        if cached is not None and cached.expires_at > now + dt.timedelta(seconds=60):
            return cached.access_token
        if self._refresh_callback is None:
            self.breaker.trip()
            raise ListonicAuthenticationError("Listonic refresh is unsupported")
        try:
            token = self.credential_store.get_access_token(
                account.credential_ref, self._refresh_callback
            )
        except CredentialError, PermanentRefreshError:
            self.breaker.trip()
            raise ListonicAuthenticationError(
                "Listonic authentication failed"
            ) from None
        return token

    def _raise_response_status(self, response: httpx.Response, expected: int) -> None:
        del expected
        if response.status_code == 401:
            raise ListonicAuthenticationError("Listonic authentication failed")
        if response.status_code == 403:
            raise ListonicForbiddenError("Listonic request forbidden")
        if response.status_code == 404:
            raise ListonicNotFoundError("Listonic object not found")
        if response.status_code == 409:
            raise ListonicConflictError("Listonic request conflicts")
        if response.status_code in (400, 422):
            raise ListonicInvalidRequestError("Listonic request was rejected")
        if response.status_code == 429:
            raise ListonicRateLimitError("Listonic rate limit reached")
        if response.status_code >= 500:
            raise ListonicError("Listonic service unavailable")
        self.breaker.trip()
        raise ListonicContractDriftError("Listonic response contract changed")


_EXPECTED_STATUS: dict[tuple[str, str], int] = {
    ("GET", "lists"): 200,
    ("GET", "list"): 200,
    ("GET", "items"): 200,
    ("POST", "items"): 201,
    ("PATCH", "item"): 200,
    ("DELETE", "item"): 204,
}


def _path_kind(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    if parts[-1:] == ["lists"]:
        return "lists"
    if parts[-1:] == ["items"]:
        return "items"
    if len(parts) == 3 and parts[0:2] == ["api", "lists"]:
        return "list"
    return "item"


def _path_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("provider ID must not be empty")
    return quote(value, safe="")


def _object(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ListonicContractDriftError("Listonic response format changed")
    return payload


def _collection(payload: object, key_names: tuple[str, ...]) -> list[dict[str, object]]:
    values: object = payload
    if isinstance(payload, dict):
        for key in key_names:
            if key in payload:
                values = payload[key]
                break
        else:
            raise ListonicContractDriftError("Listonic response format changed")
    if not isinstance(values, list):
        raise ListonicContractDriftError("Listonic response format changed")
    if not all(isinstance(value, dict) for value in values):
        raise ListonicContractDriftError("Listonic response format changed")
    return [value for value in values if isinstance(value, dict)]


def _shopping_list(value: dict[str, object]) -> ShoppingList:
    identifier = _field(value, "id", "Id", "ID")
    name = _field(value, "name", "Name")
    count = _field(value, "itemCount", "ItemCount", "item_count", default=None)
    if not isinstance(identifier, str) or not isinstance(name, str):
        raise ListonicContractDriftError("Listonic response format changed")
    if count is not None and not isinstance(count, int):
        raise ListonicContractDriftError("Listonic response format changed")
    return ShoppingList(id=identifier, name=name, item_count=count)


def _shopping_item(value: dict[str, object], list_id: str) -> ShoppingItem:
    identifier = _field(value, "id", "Id", "ID", "itemId")
    name = _field(value, "name", "Name")
    amount = _field(value, "amount", "Amount", default=None)
    unit = _field(value, "unit", "Unit", default=None)
    checked = _field(value, "checked", "Checked", default=False)
    if not isinstance(identifier, str) or not isinstance(name, str):
        raise ListonicContractDriftError("Listonic response format changed")
    if amount is not None and not isinstance(amount, (int, float)):
        raise ListonicContractDriftError("Listonic response format changed")
    if unit is not None and not isinstance(unit, str):
        raise ListonicContractDriftError("Listonic response format changed")
    if not isinstance(checked, bool):
        raise ListonicContractDriftError("Listonic response format changed")
    return ShoppingItem(
        id=identifier,
        list_id=list_id,
        name=name,
        amount=float(amount) if isinstance(amount, int) else amount,
        unit=unit,
        checked=checked,
    )


def _field(value: dict[str, object], *names: str, default: object = ...) -> object:
    for name in names:
        if name in value:
            return value[name]
    if default is ...:
        raise ListonicContractDriftError("Listonic response format changed")
    return default


def _required_string(value: dict[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return result.strip()


def _string_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("amount must be a number or null")
    return float(value)


__all__ = [
    "ContractDriftCircuitBreaker",
    "LISTONIC_BASE_URL",
    "LISTONIC_PROVIDER",
    "ListonicAuthenticationError",
    "ListonicCircuitBreaker",
    "ListonicConflictError",
    "ListonicContractDriftError",
    "ListonicError",
    "ListonicForbiddenError",
    "ListonicInvalidRequestError",
    "ListonicNotFoundError",
    "ListonicProvider",
    "ListonicRateLimitError",
    "ListonicSessionToken",
    "ShoppingItem",
    "ShoppingList",
]
