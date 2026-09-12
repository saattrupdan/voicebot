"""Credential storage and token lifecycle primitives.

Only refresh credentials are persisted by this module.  Access tokens and provider
account metadata are deliberately process-local; the storage layer can persist the
opaque reference returned by :meth:`CredentialStore.connect`.
"""

from __future__ import annotations

import collections.abc as c
import dataclasses
import datetime as dt
import enum
import re
import secrets
import threading
import typing as t

import keyring
from keyring.errors import PasswordDeleteError

CredentialReference: t.TypeAlias = str


class CredentialError(Exception):
    """Base class for credential lifecycle errors."""


class CredentialDisconnectedError(CredentialError):
    """Raised when an account has been marked disconnected."""


class PermanentRefreshError(CredentialError):
    """Raised by a provider when a refresh credential cannot be used again."""


class ConnectionStatus(enum.StrEnum):
    """Local state of a provider account."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    NEEDS_REAUTH = "needs_reauth"


@dataclasses.dataclass(frozen=True, slots=True)
class TokenSet:
    """Tokens returned by a provider's token endpoint.

    Args:
        access_token: Short-lived token used for provider requests.
        expires_at: Absolute expiry time for ``access_token``.
        refresh_token: Optional rotated refresh credential.
    """

    access_token: str
    expires_at: dt.datetime
    refresh_token: str | None = None

    @classmethod
    def from_expires_in(
        cls,
        access_token: str,
        expires_in: float,
        refresh_token: str | None = None,
        *,
        now: dt.datetime | None = None,
    ) -> TokenSet:
        """Create a token set from the provider's lifetime in seconds."""
        issued_at = now or _utc_now()
        return cls(
            access_token=access_token,
            expires_at=issued_at + dt.timedelta(seconds=expires_in),
            refresh_token=refresh_token,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderAccount:
    """The non-secret metadata a caller may persist in SQLite."""

    provider: str
    profile: str
    credential_ref: CredentialReference
    scopes: tuple[str, ...] = ()
    status: ConnectionStatus = ConnectionStatus.CONNECTED


class CredentialBackend(t.Protocol):
    """Backend contract for refresh credential persistence."""

    def get(self, credential_ref: CredentialReference) -> str | None:
        """Return a refresh credential, if present."""

    def set(self, credential_ref: CredentialReference, value: str) -> None:
        """Create or overwrite a refresh credential."""

    def replace(self, credential_ref: CredentialReference, value: str) -> None:
        """Atomically replace a refresh credential."""

    def delete(self, credential_ref: CredentialReference) -> None:
        """Erase a refresh credential."""


class RefreshCallback(t.Protocol):
    """Callback implemented by provider adapters for token refreshes."""

    def __call__(self, refresh_token: str) -> TokenSet:
        """Exchange a refresh credential for a token set."""


class RevocationCallback(t.Protocol):
    """Callback implemented by providers that support grant revocation."""

    def __call__(self, refresh_token: str) -> None:
        """Revoke a refresh credential at the provider."""


class ProviderAuthHandler(t.Protocol):
    """Interface used by local integration onboarding commands.

    Provider implementations own OAuth/browser details.  They must return only
    metadata or write through :class:`CredentialStore`; they must never put a
    secret in a CLI result or log message.
    """

    def login(self, profile: str) -> ProviderAccount | None:
        """Onboard the provider account for a local profile."""

    def status(self, profile: str | None = None) -> object:
        """Return provider status for one profile or all profiles."""

    def disconnect(self, profile: str) -> None:
        """Revoke and erase the provider account for a local profile."""


class MemoryCredentialBackend:
    """Thread-safe, process-only backend for tests and explicit headless use.

    This backend never writes to disk.  It is intentionally not selected as a
    fallback when the operating-system keyring is unavailable.
    """

    def __init__(self) -> None:
        """Initialise an empty process-local credential collection."""
        self._values: dict[CredentialReference, str] = {}
        self._lock = threading.RLock()

    def get(self, credential_ref: CredentialReference) -> str | None:
        """Return a credential without exposing the backend's internal mapping."""
        with self._lock:
            return self._values.get(credential_ref)

    def set(self, credential_ref: CredentialReference, value: str) -> None:
        """Store a credential in memory."""
        _validate_credential(value)
        with self._lock:
            self._values[credential_ref] = value

    def replace(self, credential_ref: CredentialReference, value: str) -> None:
        """Replace a credential with one atomic mapping assignment."""
        _validate_credential(value)
        with self._lock:
            self._values[credential_ref] = value

    def delete(self, credential_ref: CredentialReference) -> None:
        """Erase a credential from memory."""
        with self._lock:
            self._values.pop(credential_ref, None)


InMemoryCredentialBackend = MemoryCredentialBackend


class KeyringCredentialBackend:
    """Operating-system keyring backend for refresh credentials."""

    def __init__(
        self, service_name: str = "voicebot", *, keyring_api: object = keyring
    ) -> None:
        """Initialise the keyring service namespace.

        Args:
            service_name (optional): Keyring service name. Defaults to ``voicebot``.
            keyring_api (optional): Keyring-compatible object for tests. Defaults to
                the installed keyring module.
        """
        self._service_name = service_name
        self._keyring = _KeyringAdapter(keyring_api)

    def get(self, credential_ref: CredentialReference) -> str | None:
        """Read a refresh credential from the OS keyring."""
        return self._keyring.get_password(self._service_name, credential_ref)

    def set(self, credential_ref: CredentialReference, value: str) -> None:
        """Store a refresh credential in the OS keyring."""
        _validate_credential(value)
        self._keyring.set_password(self._service_name, credential_ref, value)

    def replace(self, credential_ref: CredentialReference, value: str) -> None:
        """Overwrite a keyring entry without deleting the previous value first."""
        _validate_credential(value)
        self._keyring.set_password(self._service_name, credential_ref, value)

    def delete(self, credential_ref: CredentialReference) -> None:
        """Delete a keyring entry, treating an absent entry as already erased."""
        try:
            self._keyring.delete_password(self._service_name, credential_ref)
        except PasswordDeleteError:
            return


@dataclasses.dataclass(frozen=True, slots=True)
class _CachedToken:
    token: str
    expires_at: dt.datetime


class CredentialStore:
    """Manage refresh credentials and an in-memory access-token cache."""

    def __init__(
        self,
        backend: CredentialBackend | None = None,
        *,
        refresh_skew: dt.timedelta = dt.timedelta(seconds=60),
        clock: c.Callable[[], dt.datetime] | None = None,
        status_callback: c.Callable[[ProviderAccount, ConnectionStatus], None]
        | None = None,
    ) -> None:
        """Initialise credential persistence and the process-local token cache.

        Args:
            backend (optional): Refresh credential backend. Defaults to the OS keyring.
            refresh_skew (optional): Refresh this long before expiry. Defaults to 60
                seconds.
            clock (optional): Injectable UTC clock. Defaults to the system clock.
        """
        if refresh_skew < dt.timedelta(0):
            raise ValueError("refresh_skew must not be negative")
        self._backend = backend if backend is not None else KeyringCredentialBackend()
        self._refresh_skew = refresh_skew
        self._clock = clock or _utc_now
        self._status_callback = status_callback
        self._cache: dict[CredentialReference, _CachedToken] = {}
        self._accounts: dict[CredentialReference, ProviderAccount] = {}
        self._locks: dict[CredentialReference, threading.Lock] = {}
        self._lock = threading.RLock()

    @classmethod
    def memory_only(
        cls,
        *,
        refresh_skew: dt.timedelta = dt.timedelta(seconds=60),
        clock: c.Callable[[], dt.datetime] | None = None,
    ) -> CredentialStore:
        """Create a store which cannot persist credentials beyond this process."""
        return cls(
            backend=MemoryCredentialBackend(), refresh_skew=refresh_skew, clock=clock
        )

    def connect(
        self,
        provider: str,
        profile: str,
        *,
        refresh_token: str | None = None,
        scopes: c.Iterable[str] = (),
        credential_ref: CredentialReference | None = None,
    ) -> ProviderAccount:
        """Register an account and optionally store its refresh credential.

        The returned reference is opaque and contains neither the provider nor the
        profile.  It is suitable for persistence in provider-account metadata.
        """
        if not provider.strip() or not profile.strip():
            raise ValueError("provider and profile must not be empty")
        reference = credential_ref or new_credential_reference()
        account = ProviderAccount(
            provider=provider,
            profile=profile,
            credential_ref=reference,
            scopes=tuple(scopes),
        )
        with self._lock:
            self._accounts[reference] = account
            self._locks.setdefault(reference, threading.Lock())
        if refresh_token is not None:
            self.store_refresh_token(reference, refresh_token)
        return account

    def store_refresh_token(
        self, credential_ref: CredentialReference, refresh_token: str
    ) -> None:
        """Store a refresh credential without putting it in application metadata."""
        self._backend.set(credential_ref, refresh_token)
        self._set_status(credential_ref, ConnectionStatus.CONNECTED)

    save_refresh_token = store_refresh_token

    def load_refresh_token(self, credential_ref: CredentialReference) -> str | None:
        """Load a refresh credential for a provider adapter."""
        return self._backend.get(credential_ref)

    get_refresh_token = load_refresh_token

    def replace_refresh_token(
        self, credential_ref: CredentialReference, refresh_token: str
    ) -> None:
        """Atomically replace a rotated refresh credential."""
        self._backend.replace(credential_ref, refresh_token)

    def delete_refresh_token(self, credential_ref: CredentialReference) -> None:
        """Erase one refresh credential and any cached access token."""
        self._backend.delete(credential_ref)
        self.mark_disconnected(credential_ref)

    def get_access_token(
        self,
        credential_ref: CredentialReference,
        refresh: RefreshCallback,
        *,
        now: dt.datetime | None = None,
    ) -> str:
        """Return a cached access token or serialise one refresh for the account."""
        current_time = _normalise_time(now or self._clock())
        with self._lock:
            account = self._accounts.get(credential_ref)
            lock = self._locks.setdefault(credential_ref, threading.Lock())
            cached = self._cache.get(credential_ref)
            if account and account.status is not ConnectionStatus.CONNECTED:
                raise CredentialDisconnectedError("credential is disconnected")
            if _is_fresh(cached, current_time, self._refresh_skew):
                assert cached is not None
                return cached.token

        with lock:
            with self._lock:
                account = self._accounts.get(credential_ref)
                cached = self._cache.get(credential_ref)
                if account and account.status is not ConnectionStatus.CONNECTED:
                    raise CredentialDisconnectedError("credential is disconnected")
                if _is_fresh(cached, current_time, self._refresh_skew):
                    assert cached is not None
                    return cached.token
            refresh_token = self.load_refresh_token(credential_ref)
            if refresh_token is None:
                raise CredentialError("refresh credential is missing")
            try:
                token_set = refresh(refresh_token)
                _validate_token_set(token_set)
                if token_set.refresh_token is not None:
                    self.replace_refresh_token(credential_ref, token_set.refresh_token)
            except PermanentRefreshError:
                self.mark_disconnected(credential_ref)
                raise PermanentRefreshError("permanent token refresh failure") from None
            except Exception:
                # Provider exceptions may echo request data.  Keep the public error
                # deliberately generic rather than forwarding their text.
                raise CredentialError("token refresh failed") from None
            with self._lock:
                self._cache[credential_ref] = _CachedToken(
                    token=token_set.access_token,
                    expires_at=_normalise_time(token_set.expires_at),
                )
                self._set_status(credential_ref, ConnectionStatus.CONNECTED)
            return token_set.access_token

    access_token = get_access_token

    def clear_access_token(self, credential_ref: CredentialReference) -> None:
        """Discard one cached access token without disconnecting its account."""
        with self._lock:
            self._cache.pop(credential_ref, None)

    def disconnect(
        self,
        account: ProviderAccount | CredentialReference,
        *,
        revoke: RevocationCallback | None = None,
    ) -> None:
        """Revoke where possible, then erase local credential and cache."""
        credential_ref = (
            account.credential_ref if isinstance(account, ProviderAccount) else account
        )
        with self._lock:
            account_lock = self._locks.setdefault(credential_ref, threading.Lock())
        with account_lock:
            revocation_failed = False
            refresh_token = self.load_refresh_token(credential_ref)
            if revoke is not None and refresh_token is not None:
                try:
                    revoke(refresh_token)
                except Exception:
                    # Provider exceptions may echo request data, so do not re-raise
                    # them even though the local deletion below must still happen.
                    revocation_failed = True
            deletion_failed = False
            try:
                self._backend.delete(credential_ref)
            except Exception:
                deletion_failed = True
            finally:
                self.mark_disconnected(credential_ref)
        if deletion_failed:
            raise CredentialError("credential deletion failed")
        if revocation_failed:
            raise CredentialError("provider revocation failed")

    def mark_disconnected(self, credential_ref: CredentialReference) -> None:
        """Mark an account disconnected and discard its process-local token."""
        with self._lock:
            self._cache.pop(credential_ref, None)
            self._set_status(credential_ref, ConnectionStatus.DISCONNECTED)

    def account_status(self, credential_ref: CredentialReference) -> ConnectionStatus:
        """Return local account state, defaulting to disconnected."""
        with self._lock:
            account = self._accounts.get(credential_ref)
            return account.status if account else ConnectionStatus.DISCONNECTED

    def account(self, credential_ref: CredentialReference) -> ProviderAccount | None:
        """Return non-secret account metadata, if registered."""
        with self._lock:
            return self._accounts.get(credential_ref)

    def _set_status(
        self, credential_ref: CredentialReference, status: ConnectionStatus
    ) -> None:
        with self._lock:
            account = self._accounts.get(credential_ref)
            if account is not None:
                updated = dataclasses.replace(account, status=status)
                self._accounts[credential_ref] = updated
            else:
                updated = None
        if updated is not None and self._status_callback is not None:
            try:
                self._status_callback(updated, status)
            except Exception:
                # Persistence must not turn a safe credential failure into a secret
                # bearing exception from the keyring layer.
                pass


class _KeyringAdapter:
    """Small typed adapter around keyring modules and test doubles."""

    def __init__(self, keyring_api: object) -> None:
        self._api = keyring_api

    def get_password(self, service: str, username: str) -> str | None:
        return t.cast(
            c.Callable[[str, str], str | None], getattr(self._api, "get_password")
        )(service, username)

    def set_password(self, service: str, username: str, password: str) -> None:
        t.cast(c.Callable[[str, str, str], None], getattr(self._api, "set_password"))(
            service, username, password
        )

    def delete_password(self, service: str, username: str) -> None:
        t.cast(c.Callable[[str, str], None], getattr(self._api, "delete_password"))(
            service, username
        )


def redact_secret(value: str | None) -> str:
    """Return a fixed marker rather than a secret or a partial secret."""
    return "[REDACTED]" if value is not None else "[MISSING]"


def redact_text(text: str, secrets_to_redact: c.Iterable[str] = ()) -> str:
    """Replace known secrets and email addresses in text."""
    result = text
    for secret in sorted(set(secrets_to_redact), key=len, reverse=True):
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return re.sub(
        r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w-])", "[REDACTED]", result
    )


def redact_headers(headers: c.Mapping[str, object]) -> dict[str, object]:
    """Redact authorisation and cookie header values case-insensitively."""
    sensitive = {"authorization", "cookie", "set-cookie", "proxy-authorization"}
    return {
        name: redact_secret(str(value)) if name.casefold() in sensitive else value
        for name, value in headers.items()
    }


def redact_mapping(mapping: c.Mapping[str, object]) -> dict[str, object]:
    """Redact sensitive fields recursively in a provider payload or error."""
    sensitive_words = (
        "authorization",
        "cookie",
        "password",
        "secret",
        "token",
        "oauth_code",
        "code",
        "email",
        "account_id",
        "accountid",
    )

    def clean(value: object, key: str = "") -> object:
        if any(word in key.casefold() for word in sensitive_words):
            return redact_secret(str(value))
        if isinstance(value, dict):
            return {
                str(child_key): clean(child, str(child_key))
                for child_key, child in value.items()
            }
        if isinstance(value, list):
            return [clean(child) for child in value]
        if isinstance(value, tuple):
            return tuple(clean(child) for child in value)
        return value

    return t.cast(dict[str, object], clean(dict(mapping)))


def new_credential_reference() -> CredentialReference:
    """Generate an opaque, non-sensitive credential reference."""
    return f"cred_{secrets.token_urlsafe(24)}"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _normalise_time(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


def _is_fresh(
    cached: _CachedToken | None, now: dt.datetime, skew: dt.timedelta
) -> bool:
    return cached is not None and now < cached.expires_at - skew


def _validate_credential(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("credential must be a non-empty string")


def _validate_token_set(token_set: TokenSet) -> None:
    if not isinstance(token_set, TokenSet) or not token_set.access_token:
        raise CredentialError("provider returned an invalid access token")
    if not isinstance(token_set.expires_at, dt.datetime):
        raise CredentialError("provider returned an invalid token expiry")


__all__ = [
    "ConnectionStatus",
    "CredentialAccount",
    "CredentialBackend",
    "CredentialDisconnectedError",
    "CredentialError",
    "CredentialReference",
    "CredentialRef",
    "CredentialStore",
    "InMemoryCredentialBackend",
    "KeyringCredentialBackend",
    "MemoryCredentialBackend",
    "PermanentRefreshError",
    "ProviderAccount",
    "ProviderAuthHandler",
    "RefreshCallback",
    "RevocationCallback",
    "TokenSet",
    "new_credential_reference",
    "redact_headers",
    "redact_mapping",
    "redact_secret",
    "redact_text",
]

CredentialAccount = ProviderAccount
CredentialRef = CredentialReference
