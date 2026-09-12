"""Spotify Authorization Code + PKCE onboarding."""

from __future__ import annotations

import base64
import collections.abc as c
import hashlib
import http.server
import secrets
import time
import typing as t
import urllib.parse
import webbrowser
from dataclasses import dataclass

import httpx

from .credentials import (
    ConnectionStatus,
    CredentialError,
    CredentialReference,
    CredentialStore,
    PermanentRefreshError,
    ProviderAccount,
    TokenSet,
)

SPOTIFY_SCOPES: tuple[str, str] = (
    "user-read-playback-state",
    "user-modify-playback-state",
)
SPOTIFY_SCOPE = " ".join(SPOTIFY_SCOPES)
AUTHORISATION_ENDPOINT = "https://accounts.spotify.com/authorize"
TOKEN_ENDPOINT = "https://accounts.spotify.com/api/token"


class SpotifyOAuthError(CredentialError):
    """Raised when Spotify onboarding cannot be completed safely."""


def make_pkce_verifier() -> str:
    """Create a high-entropy RFC 7636 code verifier."""
    return secrets.token_urlsafe(64)


def make_pkce_challenge(verifier: str) -> str:
    """Return the S256 PKCE challenge for ``verifier``."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def make_oauth_state() -> str:
    """Create an unpredictable CSRF state value."""
    return secrets.token_urlsafe(32)


@dataclass(frozen=True, slots=True)
class SpotifyCallback:
    """The values received by the loopback OAuth callback."""

    code: str
    state: str


CallbackResult: t.TypeAlias = SpotifyCallback | tuple[str, str]


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Capture one OAuth callback without writing request values to logs."""

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urllib.parse.urlparse(self.path)
        values = urllib.parse.parse_qs(parsed.query)
        server = t.cast(_CallbackServer, self.server)
        server.callback = SpotifyCallback(
            code=values.get("code", [""])[0], state=values.get("state", [""])[0]
        )
        self.send_response(200 if server.callback.code else 400)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Spotify forbindelsen kan lukkes nu.\n")

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class _CallbackServer(http.server.ThreadingHTTPServer):
    """Small loopback server used for one login attempt."""

    callback: SpotifyCallback | None = None


class SpotifyAuthHandler:
    """Onboard and disconnect Spotify accounts for local profiles.

    Args:
        client_id:
            The Spotify application client ID. It is not a secret.
        credential_store:
            Shared store for refresh credentials and access-token caching.
        http_client:
            Optional raw ``httpx.Client`` used for token requests.
        open_browser:
            Optional browser opener, useful for headless deployments and tests.
        callback_waiter:
            Optional callback implementation. It receives the redirect URI and is useful
            for an application that already owns its loopback listener.
    """

    def __init__(
        self,
        client_id: str,
        credential_store: CredentialStore | None = None,
        *,
        credentials: CredentialStore | None = None,
        store: CredentialStore | None = None,
        http_client: httpx.Client | None = None,
        open_browser: c.Callable[[str], object] | None = None,
        callback_waiter: c.Callable[[str], CallbackResult] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        callback_timeout: float = 300.0,
        authorisation_endpoint: str = AUTHORISATION_ENDPOINT,
        token_endpoint: str = TOKEN_ENDPOINT,
    ) -> None:
        """Initialise Spotify OAuth configuration."""
        if not client_id.strip():
            raise ValueError("client_id must not be empty")
        if (
            sum(value is not None for value in (credential_store, credentials, store))
            > 1
        ):
            raise ValueError("provide only one credential store")
        self.client_id = client_id
        self.credential_store = (
            credential_store or credentials or store or CredentialStore()
        )
        self.http_client = http_client or httpx.Client(timeout=20.0)
        self.open_browser = open_browser or webbrowser.open
        self.callback_waiter = callback_waiter
        self.host = host
        self.port = port
        self.callback_timeout = callback_timeout
        self.authorisation_endpoint = authorisation_endpoint
        self.token_endpoint = token_endpoint
        self._accounts: dict[str, ProviderAccount] = {}

    def login(self, profile: str) -> ProviderAccount | None:
        """Run browser consent and save the resulting refresh credential."""
        if not profile.strip():
            raise ValueError("profile must not be empty")
        verifier = make_pkce_verifier()
        state = make_oauth_state()
        if self.callback_waiter is not None:
            # A supplied waiter still gets the exact redirect URI generated below.
            redirect_uri = f"http://{self.host}:{self.port}/callback"
            callback = self._start_external_callback(
                redirect_uri=redirect_uri, verifier=verifier, state=state
            )
        else:
            callback, redirect_uri = self._start_loopback(
                verifier=verifier, state=state
            )
        if callback.state != state or not callback.code:
            raise SpotifyOAuthError("Spotify OAuth state validation failed")
        token_set = self.exchange_code(
            code=callback.code, verifier=verifier, redirect_uri=redirect_uri
        )
        if token_set.refresh_token is None:
            raise SpotifyOAuthError("Spotify did not return offline access")
        old = self._accounts.get(profile)
        if old is not None:
            self.credential_store.disconnect(old)
        account = self.credential_store.connect(
            "spotify",
            profile,
            refresh_token=token_set.refresh_token,
            scopes=SPOTIFY_SCOPES,
        )
        self._accounts[profile] = account
        return account

    def status(self, profile: str | None = None) -> object:
        """Return non-secret connection metadata."""
        if profile is not None:
            account = self._accounts.get(profile)
            return self._status(account=account, profile=profile)
        return [
            self._status(account=account, profile=name)
            for name, account in self._accounts.items()
        ]

    def disconnect(self, profile: str) -> None:
        """Erase local Spotify credentials.

        Spotify does not provide a dependable server-side revocation endpoint. Users can
        remove this app from Spotify account permissions after this local operation.
        """
        account = self._accounts.pop(profile, None)
        if account is not None:
            self.credential_store.disconnect(account)

    def authorization_url(self, redirect_uri: str, state: str, challenge: str) -> str:
        """Build a consent URL requesting exactly the playback scopes."""
        query = urllib.parse.urlencode(
            {
                "client_id": self.client_id,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "state": state,
                "scope": SPOTIFY_SCOPE,
                "code_challenge_method": "S256",
                "code_challenge": challenge,
            }
        )
        return f"{self.authorisation_endpoint}?{query}"

    def exchange_code(self, code: str, verifier: str, redirect_uri: str) -> TokenSet:
        """Exchange an authorisation code for tokens without exposing them."""
        try:
            response = self.http_client.post(
                self.token_endpoint,
                data={
                    "client_id": self.client_id,
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "code_verifier": verifier,
                },
            )
            payload = response.json()
        except Exception:
            raise SpotifyOAuthError("Spotify token exchange failed") from None
        if response.status_code >= 400 or not isinstance(payload, dict):
            raise SpotifyOAuthError("Spotify token exchange failed")
        access = payload.get("access_token")
        refresh = payload.get("refresh_token")
        expires = payload.get("expires_in")
        if not isinstance(access, str) or not isinstance(refresh, str):
            raise SpotifyOAuthError("Spotify token exchange returned invalid data")
        if not isinstance(expires, (int, float)) or isinstance(expires, bool):
            raise SpotifyOAuthError("Spotify token exchange returned invalid data")
        return TokenSet.from_expires_in(
            access_token=access, expires_in=float(expires), refresh_token=refresh
        )

    def refresh(self, refresh_token: str) -> TokenSet:
        """Refresh an access token, raising a safe permanent error when needed."""
        try:
            response = self.http_client.post(
                self.token_endpoint,
                data={
                    "client_id": self.client_id,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
            payload = response.json()
        except Exception:
            raise SpotifyOAuthError("Spotify token refresh failed") from None
        if response.status_code == 400:
            raise PermanentRefreshError("Spotify refresh credential is invalid")
        if response.status_code >= 400 or not isinstance(payload, dict):
            raise SpotifyOAuthError("Spotify token refresh failed")
        access = payload.get("access_token")
        expires = payload.get("expires_in")
        rotated = payload.get("refresh_token")
        if not isinstance(access, str) or not isinstance(expires, (int, float)):
            raise SpotifyOAuthError("Spotify token refresh returned invalid data")
        return TokenSet.from_expires_in(
            access_token=access,
            expires_in=float(expires),
            refresh_token=rotated if isinstance(rotated, str) else None,
        )

    def _start_external_callback(
        self, redirect_uri: str, verifier: str, state: str
    ) -> SpotifyCallback:
        url = self.authorization_url(
            redirect_uri=redirect_uri,
            state=state,
            challenge=make_pkce_challenge(verifier=verifier),
        )
        self.open_browser(url)
        assert self.callback_waiter is not None
        callback = self.callback_waiter(redirect_uri)
        if isinstance(callback, SpotifyCallback):
            return callback
        code, callback_state = callback
        return SpotifyCallback(code=code, state=callback_state)

    def _start_loopback(self, verifier: str, state: str) -> tuple[SpotifyCallback, str]:
        server = _CallbackServer((self.host, self.port), _CallbackHandler)
        redirect_uri = f"http://{self.host}:{server.server_port}/callback"
        url = self.authorization_url(
            redirect_uri=redirect_uri,
            state=state,
            challenge=make_pkce_challenge(verifier=verifier),
        )
        self.open_browser(url)
        deadline = time.monotonic() + self.callback_timeout
        server.timeout = max(0.0, self.callback_timeout)
        try:
            while server.callback is None and time.monotonic() < deadline:
                server.timeout = max(0.0, deadline - time.monotonic())
                server.handle_request()
        finally:
            server.server_close()
        if server.callback is None:
            raise SpotifyOAuthError("Spotify OAuth callback timed out")
        return server.callback, redirect_uri

    def _status(
        self, account: ProviderAccount | None, profile: str
    ) -> dict[str, object]:
        connected = (
            account is not None
            and self.credential_store.account_status(account.credential_ref)
            is ConnectionStatus.CONNECTED
        )
        return {
            "profile": profile,
            "connected": connected,
            "scopes": list(account.scopes) if account is not None else [],
        }


# British spelling is public while the common API spelling remains available.
SpotifyAuth = SpotifyAuthHandler
SpotifyOAuth = SpotifyAuthHandler
SPOTIFY_AUTH_SCOPES = SPOTIFY_SCOPES
make_pkce_code_challenge = make_pkce_challenge
make_state = make_oauth_state
CredentialRef = CredentialReference

__all__ = [
    "AUTHORISATION_ENDPOINT",
    "SPOTIFY_SCOPE",
    "SPOTIFY_SCOPES",
    "SPOTIFY_AUTH_SCOPES",
    "SpotifyAuth",
    "SpotifyAuthHandler",
    "SpotifyCallback",
    "SpotifyOAuth",
    "SpotifyOAuthError",
    "make_oauth_state",
    "make_pkce_challenge",
    "make_pkce_code_challenge",
    "make_pkce_verifier",
    "make_state",
]
