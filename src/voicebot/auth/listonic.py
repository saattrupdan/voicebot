"""Safe local onboarding for Listonic's unofficial integration."""

from __future__ import annotations

import collections.abc as c
import dataclasses
import datetime as dt
import json
import typing as t

from ..providers.listonic import (
    LISTONIC_BASE_URL,
    LISTONIC_PROVIDER,
    ListonicSessionToken,
)
from .credentials import (
    ConnectionStatus,
    CredentialError,
    CredentialStore,
    ProviderAccount,
    ProviderAuthHandler,
)

LISTONIC_LOGIN_URL = f"{LISTONIC_BASE_URL}/login"


class IsolatedBrowserSession(t.Protocol):
    """Minimal browser surface required by Listonic onboarding.

    Implementations must arrange for the user to enter credentials in the browser.
    There is intentionally no password parameter or password-reading method here.
    """

    def open_login(self, url: str) -> None:
        """Open the provider login page in the isolated browser."""

    def export_tokens(self) -> object:
        """Export access/refresh state, never the login password."""

    def destroy(self) -> None:
        """Destroy the isolated browser profile and its cookies."""


class IsolatedBrowserFactory(t.Protocol):
    """Factory for disposable browser sessions."""

    def __call__(self) -> IsolatedBrowserSession:
        """Create an isolated browser session."""


@dataclasses.dataclass(frozen=True, slots=True)
class ListonicAuthStatus:
    """Non-secret status returned by local integration setup."""

    profile: str
    state: str


class ListonicAuthHandler:
    """Import only browser-exported tokens into the shared credential store."""

    def __init__(
        self,
        credential_store: CredentialStore,
        browser_factory: IsolatedBrowserFactory,
        *,
        token_sink: c.Callable[[ProviderAccount, ListonicSessionToken], None]
        | None = None,
        clock: c.Callable[[], dt.datetime] | None = None,
    ) -> None:
        """Create a Listonic onboarding handler.

        Args:
            credential_store:
                Shared store where only the refresh credential is persisted.
            browser_factory:
                Factory that creates a disposable, isolated browser profile.
            token_sink (optional):
                Callback for the provider's process-local access-token cache.
            clock (optional):
                Clock used when checking imported token expiry.
        """
        self.credential_store = credential_store
        self.browser_factory = browser_factory
        self.token_sink = token_sink
        self.clock = clock or (lambda: dt.datetime.now(dt.UTC))
        self._accounts: dict[str, ProviderAccount] = {}

    def login(self, profile: str) -> ProviderAccount:
        """Onboard a profile without receiving or retaining its password.

        The browser owns the login interaction.  Only the three explicitly selected
        token fields are copied out, and the session is destroyed on every outcome.
        """
        if not isinstance(profile, str) or not profile.strip():
            raise ValueError("profile must not be empty")
        session = self.browser_factory()
        try:
            session.open_login(LISTONIC_LOGIN_URL)
            imported = _import_session_tokens(session.export_tokens(), now=self.clock())
            account = self.credential_store.connect(
                LISTONIC_PROVIDER, profile.strip(), refresh_token=imported.refresh_token
            )
            self.credential_store.store_provider_secret(
                account.credential_ref,
                "listonic_access",
                json.dumps(
                    {
                        "access_token": imported.access_token,
                        "expires_at": imported.expires_at.isoformat(),
                    },
                    separators=(",", ":"),
                ),
            )
            self._accounts[account.profile] = account
            if self.token_sink is not None:
                self.token_sink(account, imported)
            return account
        except CredentialError:
            raise
        except Exception:
            raise CredentialError("Listonic onboarding failed") from None
        finally:
            try:
                session.destroy()
            except Exception:
                pass

    def status(
        self, profile: str | None = None
    ) -> ListonicAuthStatus | list[ListonicAuthStatus]:
        """Return non-sensitive local connection state."""
        if profile is not None:
            account = self._accounts.get(profile)
            state = "connected"
            if account is None:
                state = "disconnected"
            elif not _has_valid_access_state(
                self.credential_store, account, now=self.clock()
            ):
                state = "re-onboarding required"
            elif (
                self.credential_store.account_status(account.credential_ref)
                is not ConnectionStatus.CONNECTED
            ):
                state = "disconnected"
            return ListonicAuthStatus(profile=profile, state=state)
        statuses: list[ListonicAuthStatus] = []
        for name in sorted(self._accounts):
            status = self.status(profile=name)
            assert isinstance(status, ListonicAuthStatus)
            statuses.append(status)
        return statuses

    def disconnect(self, profile: str) -> None:
        """Erase the profile's refresh credential and local account state."""
        account = self._accounts.get(profile)
        if account is None:
            return
        try:
            self.credential_store.disconnect(account)
        except CredentialError:
            # CredentialStore has already attempted local deletion and reports a safe
            # generic error.  Setup callers do not need provider error details.
            raise
        finally:
            self.credential_store.delete_provider_secret(
                account.credential_ref, "listonic_access"
            )
            self._accounts.pop(profile, None)


def listonic_auth_handler(
    credential_store: CredentialStore,
    browser_factory: IsolatedBrowserFactory,
    *,
    token_sink: c.Callable[[ProviderAccount, ListonicSessionToken], None] | None = None,
) -> ProviderAuthHandler:
    """Build the provider-agnostic authentication hook for final assembly."""
    return ListonicAuthHandler(
        credential_store=credential_store,
        browser_factory=browser_factory,
        token_sink=token_sink,
    )


build_listonic_auth_handler = listonic_auth_handler


def _has_valid_access_state(
    credential_store: CredentialStore, account: ProviderAccount, *, now: dt.datetime
) -> bool:
    value = credential_store.load_provider_secret(
        account.credential_ref, "listonic_access"
    )
    if value is None:
        return False
    try:
        payload = json.loads(value)
        access_token = payload["access_token"]
        expires_at = dt.datetime.fromisoformat(payload["expires_at"])
    except KeyError, TypeError, ValueError, json.JSONDecodeError:
        return False
    return (
        isinstance(access_token, str)
        and bool(access_token)
        and expires_at.tzinfo is not None
        and expires_at.astimezone(dt.UTC) > now.astimezone(dt.UTC)
    )


def _import_session_tokens(value: object, *, now: dt.datetime) -> ListonicSessionToken:
    """Select and validate token fields from a browser-only export."""
    if isinstance(value, ListonicSessionToken):
        return value
    if not isinstance(value, c.Mapping):
        raise ValueError("browser token export is invalid")

    token_value = _find_token_mapping(value)
    if token_value is None:
        raise ValueError("browser token export has no token state")
    access = token_value.get("access_token")
    refresh = token_value.get("refresh_token")
    expiry = token_value.get("expires_at")
    if expiry is None:
        expires_in = token_value.get("expires_in")
        if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool):
            expiry = now + dt.timedelta(seconds=expires_in)
    if not isinstance(access, str) or not access:
        raise ValueError("browser token export has no access token")
    if not isinstance(refresh, str) or not refresh:
        raise ValueError("browser token export has no refresh token")
    parsed_expiry = _parse_expiry(expiry, now=now)
    if parsed_expiry <= now.astimezone(dt.UTC):
        raise ValueError("browser access token has expired")
    return ListonicSessionToken(
        access_token=access, refresh_token=refresh, expires_at=parsed_expiry
    )


def _find_token_mapping(value: c.Mapping[str, object]) -> c.Mapping[str, object] | None:
    """Find a token object in browser storage without copying unrelated state."""
    normalised = {
        str(key).casefold().replace("-", "_"): item for key, item in value.items()
    }
    access = normalised.get("access_token", normalised.get("accesstoken"))
    refresh = normalised.get("refresh_token", normalised.get("refreshtoken"))
    if isinstance(access, str) and isinstance(refresh, str):
        return {
            "access_token": access,
            "refresh_token": refresh,
            "expires_at": normalised.get("expires_at", normalised.get("expiresat")),
            "expires_in": normalised.get("expires_in", normalised.get("expiresin")),
        }
    for item in value.values():
        if isinstance(item, c.Mapping):
            found = _find_token_mapping(item)
            if found is not None:
                return found
        if isinstance(item, list):
            for nested in item:
                if isinstance(nested, c.Mapping):
                    found = _find_token_mapping(nested)
                    if found is not None:
                        return found
        if isinstance(item, str):
            try:
                decoded = json.loads(item)
            except TypeError, ValueError:
                continue
            if isinstance(decoded, c.Mapping):
                found = _find_token_mapping(decoded)
                if found is not None:
                    return found
    return None


def _parse_expiry(value: object, *, now: dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("browser token expiry is invalid") from None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = dt.datetime.fromtimestamp(value, tz=dt.UTC)
    else:
        raise ValueError("browser token expiry is invalid")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("browser token expiry must be timezone-aware")
    return parsed.astimezone(dt.UTC)


__all__ = [
    "LISTONIC_LOGIN_URL",
    "IsolatedBrowserFactory",
    "IsolatedBrowserSession",
    "ListonicAuthHandler",
    "ListonicAuthStatus",
    "build_listonic_auth_handler",
    "listonic_auth_handler",
]
