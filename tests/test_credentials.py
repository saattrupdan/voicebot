"""Tests for credential lifecycle primitives."""

import datetime as dt
import threading
import time

import pytest

from voicebot.auth import (
    ConnectionStatus,
    CredentialDisconnectedError,
    CredentialError,
    CredentialStore,
    MemoryCredentialBackend,
    PermanentRefreshError,
    TokenSet,
    redact_headers,
    redact_mapping,
    redact_text,
)

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def test_reference_and_refresh_credential_are_separate() -> None:
    """Keep opaque references free of provider and profile values."""
    backend = MemoryCredentialBackend()
    store = CredentialStore(backend=backend)
    account = store.connect("spotify", "household", refresh_token="refresh-secret")

    assert account.credential_ref.startswith("cred_")
    assert "spotify" not in account.credential_ref
    assert "household" not in account.credential_ref
    assert backend.get(account.credential_ref) == "refresh-secret"


def test_access_tokens_are_cached_and_rotated_refresh_is_replaced() -> None:
    """Cache access tokens and atomically retain a rotated refresh token."""
    store = CredentialStore.memory_only(clock=lambda: NOW)
    account = store.connect("provider", "profile", refresh_token="old-refresh")
    calls: list[str] = []

    def refresh(refresh_token: str) -> TokenSet:
        calls.append(refresh_token)
        return TokenSet(
            access_token="access-token",
            expires_at=NOW + dt.timedelta(minutes=10),
            refresh_token="new-refresh",
        )

    assert store.get_access_token(account.credential_ref, refresh) == "access-token"
    assert store.get_access_token(account.credential_ref, refresh) == "access-token"
    assert calls == ["old-refresh"]
    assert store.load_refresh_token(account.credential_ref) == "new-refresh"


def test_refresh_is_serialised_per_account() -> None:
    """Ensure concurrent callers perform only one refresh."""
    store = CredentialStore.memory_only(clock=lambda: NOW)
    account = store.connect("provider", "profile", refresh_token="refresh")
    calls = 0
    calls_lock = threading.Lock()

    def refresh(refresh_token: str) -> TokenSet:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.03)
        return TokenSet("access", NOW + dt.timedelta(minutes=5))

    threads = [
        threading.Thread(
            target=store.get_access_token, args=(account.credential_ref, refresh)
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1


def test_permanent_refresh_failure_disconnects_account() -> None:
    """Disconnect accounts after a permanent provider refresh failure."""
    store = CredentialStore.memory_only(clock=lambda: NOW)
    account = store.connect("provider", "profile", refresh_token="refresh")

    def refresh(refresh_token: str) -> TokenSet:
        raise PermanentRefreshError("invalid grant")

    with pytest.raises(PermanentRefreshError):
        store.get_access_token(account.credential_ref, refresh)
    assert store.account_status(account.credential_ref) is ConnectionStatus.DISCONNECTED
    with pytest.raises(CredentialDisconnectedError):
        store.get_access_token(account.credential_ref, refresh)


def test_disconnect_revokes_then_erases_even_if_revocation_fails() -> None:
    """Erase local credentials even when provider revocation fails."""
    store = CredentialStore.memory_only()
    account = store.connect("provider", "profile", refresh_token="refresh-secret")
    revoked: list[str] = []

    def revoke(refresh_token: str) -> None:
        revoked.append(refresh_token)
        raise RuntimeError("provider failure")

    with pytest.raises(CredentialError) as error:
        store.disconnect(account, revoke=revoke)
    assert "refresh-secret" not in str(error.value)
    assert revoked == ["refresh-secret"]
    assert store.load_refresh_token(account.credential_ref) is None
    assert store.account_status(account.credential_ref) is ConnectionStatus.DISCONNECTED


def test_redaction_helpers_remove_sensitive_values() -> None:
    """Redact headers, mappings, email addresses, and known secrets."""
    assert redact_text("access-secret", ["access-secret"]) == "[REDACTED]"
    assert redact_headers({"Authorization": "Bearer secret", "X-Test": "ok"}) == {
        "Authorization": "[REDACTED]",
        "X-Test": "ok",
    }
    result = redact_mapping({"access_token": "secret", "nested": {"email": "a@b"}})
    assert result == {"access_token": "[REDACTED]", "nested": {"email": "[REDACTED]"}}
