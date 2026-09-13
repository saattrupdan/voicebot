"""Small, fail-closed HTTP adapter for Listonic's unofficial API."""

from __future__ import annotations

import base64
import binascii
import collections.abc as c
import dataclasses
import datetime as dt
import json
import logging
import math
import threading
import time
import uuid
from urllib.parse import quote, urlencode

import httpx

from ..auth.credentials import (
    ConnectionStatus,
    CredentialError,
    CredentialReference,
    CredentialStore,
    PermanentRefreshError,
    ProviderAccount,
    RefreshCallback,
    RefreshContractError,
    TokenSet,
)

logger = logging.getLogger(__name__)

LISTONIC_BASE_URL = "https://api.listonic.com"
LISTONIC_PROVIDER = "listonic"
LISTONIC_WEB_VERSION = "web:4.0.0"
# Public protocol metadata shipped in Listonic's browser application. It is not a
# user credential; pinning it avoids executing or trusting remote JavaScript at runtime.
_LISTONIC_WEB_CLIENT = "listonicv2:fjdfsoj9874jdfhjkh34jkhffdfff"
_LISTONIC_WEB_AUTHORISATION = base64.b64encode(
    _LISTONIC_WEB_CLIENT.encode("ascii")
).decode("ascii")


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
    """Call pinned Listonic paths with exact casing and per-operation gates."""

    BASE_URL = LISTONIC_BASE_URL

    def __init__(
        self,
        credential_store: CredentialStore,
        *,
        accounts: c.Mapping[str, ProviderAccount] | None = None,
        enabled: bool = False,
        allow_unofficial: bool = False,
        allow_unverified_item_removal: bool = False,
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
            allow_unverified_item_removal (optional):
                Explicit operator gate for the separately controlled DELETE operation.
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
        self.allow_unverified_item_removal = allow_unverified_item_removal
        self.client = http_client or httpx.Client(base_url=LISTONIC_BASE_URL)
        self._owns_client = http_client is None
        self.breaker = breaker or ListonicCircuitBreaker()
        self._lcode = str(int(time.time() * 1000))
        self._refresh_callback = refresh or self._refresh_token
        self._session_tokens: dict[CredentialReference, ListonicSessionToken] = {}
        for account in self.accounts.values():
            self._rehydrate_session_tokens(account)

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
        """Cache imported access state in the credential backend and this process."""
        self.register_account(account=account)
        self.credential_store.store_provider_secret(
            account.credential_ref,
            "listonic_access",
            json.dumps(
                {
                    "access_token": tokens.access_token,
                    "expires_at": tokens.expires_at.isoformat(),
                },
                separators=(",", ":"),
            ),
        )
        self._session_tokens[account.credential_ref] = tokens

    def list_lists(self, account: ProviderAccount) -> list[ShoppingList]:
        """Return the account's shopping lists."""
        payload = self._request(
            account=account,
            method="GET",
            path="/api/lists",
            params={
                "includeShares": "true",
                "archive": "false",
                "includeItems": "true",
            },
        )
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
            json={"checked": int(checked), "itemId": item_id},
        )
        try:
            return _shopping_item(_object(payload=payload), list_id=list_id)
        except ListonicContractDriftError:
            self.breaker.trip()
            raise

    def remove_item(self, account: ProviderAccount, list_id: str, item_id: str) -> None:
        """Remove one item only when the separate endpoint gate is explicit."""
        if not self.allow_unverified_item_removal:
            raise ListonicContractDriftError("Listonic item removal is not enabled")
        self._request(
            account=account,
            method="DELETE",
            path=f"/api/lists/{_path_id(list_id)}/items/{_path_id(item_id)}",
            expected_status=200,
        )

    def _request(
        self,
        *,
        account: ProviderAccount,
        method: str,
        path: str,
        json: dict[str, object] | None = None,
        params: dict[str, str] | None = None,
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
                headers=self._request_headers(account, access_token=token),
                params=params,
                json=json,
            )
        except ListonicError:
            raise
        except httpx.HTTPError, CredentialError, PermanentRefreshError:
            raise ListonicError("Listonic request failed") from None

        expected = expected_status or _EXPECTED_STATUS[(method, _path_kind(path))]
        if response.status_code != expected:
            self._raise_response_status(response=response, expected=expected)
        if method == "DELETE" or expected == 204:
            return None
        try:
            return response.json()
        except ValueError, TypeError:
            self.breaker.trip()
            raise ListonicContractDriftError(
                "Listonic response format changed"
            ) from None

    def _request_headers(
        self, account: ProviderAccount, *, access_token: str | None = None
    ) -> dict[str, str]:
        headers = {
            "Version": LISTONIC_WEB_VERSION,
            "LCode": self._lcode,
            "DeviceId": str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL, f"voicebot:listonic:{account.credential_ref}"
                )
            ),
            "Culture": "en",
        }
        if access_token is not None:
            headers["Authorization"] = f"Bearer {access_token}"
        return headers

    def _refresh_token(self, refresh_token: str) -> TokenSet:
        headers = {
            "Version": LISTONIC_WEB_VERSION,
            "ClientAuthorization": f"Bearer {_LISTONIC_WEB_AUTHORISATION}",
            "DeviceId": str(uuid.uuid5(uuid.NAMESPACE_URL, "voicebot:listonic")),
            "Culture": "en",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            response = self.client.post(
                f"{LISTONIC_BASE_URL}/api/loginextended",
                params={
                    "provider": "refresh_token",
                    "autoMerge": "1",
                    "autoDestruct": "1",
                },
                content=urlencode({"refresh_token": refresh_token}),
                headers=headers,
            )
        except httpx.HTTPError:
            raise CredentialError("Listonic token refresh failed") from None
        if response.status_code in {400, 401, 403}:
            raise PermanentRefreshError("Listonic refresh credential was rejected")
        if response.status_code in {408, 429} or response.status_code >= 500:
            raise CredentialError("Listonic token refresh failed")
        if not 200 <= response.status_code < 300:
            raise RefreshContractError("Listonic refresh endpoint changed")
        try:
            payload = response.json()
        except TypeError, ValueError:
            raise RefreshContractError("Listonic refresh response changed") from None
        if not isinstance(payload, dict):
            raise RefreshContractError("Listonic refresh response changed")
        access = payload.get("access_token")
        rotated_refresh = payload.get("refresh_token")
        if not isinstance(access, str) or not isinstance(rotated_refresh, str):
            raise RefreshContractError("Listonic refresh response changed")
        expiry = _jwt_expiry(access)
        if expiry is None or expiry <= dt.datetime.now(dt.UTC):
            raise RefreshContractError("Listonic refresh response changed")
        return TokenSet(
            access_token=access, expires_at=expiry, refresh_token=rotated_refresh
        )

    def _access_token(self, account: ProviderAccount) -> str:
        if (
            self.credential_store.account_status(account.credential_ref)
            is not ConnectionStatus.CONNECTED
        ):
            raise ListonicAuthenticationError("Listonic authentication failed")
        cached = self._session_tokens.get(account.credential_ref)
        now = dt.datetime.now(dt.UTC)
        if cached is not None and cached.expires_at > now:
            return cached.access_token
        self._session_tokens.pop(account.credential_ref, None)
        self.credential_store.delete_provider_secret(
            account.credential_ref, "listonic_access"
        )
        if self._refresh_callback is None:
            self.credential_store.mark_disconnected(account.credential_ref)
            raise ListonicAuthenticationError(
                "Listonic session expired; re-onboarding is required"
            )
        try:
            token = self.credential_store.get_access_token(
                account.credential_ref, self._refresh_callback
            )
            expiry = _jwt_expiry(token)
            if expiry is None:
                raise RefreshContractError(
                    "Listonic token refresh returned invalid data"
                )
            self.credential_store.store_provider_secret(
                account.credential_ref,
                "listonic_access",
                json.dumps(
                    {"access_token": token, "expires_at": expiry.isoformat()},
                    separators=(",", ":"),
                ),
            )
            self._session_tokens[account.credential_ref] = ListonicSessionToken(
                access_token=token, refresh_token="", expires_at=expiry
            )
        except PermanentRefreshError:
            self.credential_store.mark_disconnected(account.credential_ref)
            raise ListonicAuthenticationError(
                "Listonic refresh failed; re-onboarding is required"
            ) from None
        except RefreshContractError:
            self.breaker.trip()
            raise ListonicContractDriftError(
                "Listonic refresh response changed"
            ) from None
        except CredentialError:
            raise ListonicError("Listonic refresh is temporarily unavailable") from None
        return token

    def _rehydrate_session_tokens(self, account: ProviderAccount) -> None:
        value = self.credential_store.load_provider_secret(
            account.credential_ref, "listonic_access"
        )
        if value is None:
            return
        try:
            payload = json.loads(value)
            access_token = payload["access_token"]
            expires_at = dt.datetime.fromisoformat(payload["expires_at"])
            if (
                not isinstance(access_token, str)
                or not access_token
                or expires_at.tzinfo is None
            ):
                raise ValueError
        except KeyError, TypeError, ValueError, json.JSONDecodeError:
            self.credential_store.delete_provider_secret(
                account.credential_ref, "listonic_access"
            )
            return
        self._session_tokens[account.credential_ref] = ListonicSessionToken(
            access_token=access_token,
            refresh_token="",
            expires_at=expires_at.astimezone(dt.UTC),
        )

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
    ("DELETE", "item"): 200,
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
    count = _field(
        value, "itemCount", "ItemCount", "ItemsCount", "item_count", default=None
    )
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
    amount_value = _response_amount(amount)
    if unit is not None and not isinstance(unit, str):
        raise ListonicContractDriftError("Listonic response format changed")
    if isinstance(checked, bool):
        checked_value = checked
    elif isinstance(checked, int) and checked in {0, 1}:
        checked_value = bool(checked)
    else:
        raise ListonicContractDriftError("Listonic response format changed")
    return ShoppingItem(
        id=identifier,
        list_id=list_id,
        name=name,
        amount=amount_value,
        unit=unit,
        checked=checked_value,
    )


def _response_amount(value: object) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ListonicContractDriftError("Listonic response format changed")
    try:
        amount = float(value) if isinstance(value, (int, float, str)) else math.nan
    except ValueError:
        amount = math.nan
    if not math.isfinite(amount):
        raise ListonicContractDriftError("Listonic response format changed")
    return amount


def _jwt_expiry(token: str) -> dt.datetime | None:
    """Read only the expiry claim from a bounded JWT-shaped access token."""
    if len(token) > 8192:
        return None
    parts = token.split(".")
    if len(parts) != 3 or not parts[1]:
        return None
    try:
        payload_bytes = base64.b64decode(
            parts[1] + "=" * (-len(parts[1]) % 4), altchars=b"-_", validate=True
        )
        if len(payload_bytes) > 4096:
            return None
        payload = json.loads(payload_bytes)
    except binascii.Error, UnicodeDecodeError, ValueError:
        return None
    if not isinstance(payload, c.Mapping):
        return None
    value = payload.get("exp")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        return None
    try:
        return dt.datetime.fromtimestamp(value, tz=dt.UTC)
    except OverflowError, OSError, ValueError:
        return None


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
