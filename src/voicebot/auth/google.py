"""Google OAuth 2.0 Authorization Code + PKCE for Calendar.

This module deliberately owns only the OAuth flow.  Refresh credentials are handed to
:class:`~voicebot.auth.credentials.CredentialStore`; they are never returned in a
status object or included in an exception.
"""

from __future__ import annotations

import base64
import collections.abc as c
import dataclasses
import hashlib
import http.server
import secrets
import threading
import urllib.parse
import webbrowser

import httpx

from .credentials import (
    ConnectionStatus,
    CredentialError,
    CredentialStore,
    PermanentRefreshError,
    ProviderAccount,
    ProviderAuthHandler,
    TokenSet,
)

GOOGLE_PROVIDER = "google_calendar"
GOOGLE_SCOPES: tuple[str, ...] = (
    "openid",
    "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
    "https://www.googleapis.com/auth/calendar.events.readonly",
)
# Aliases make the constants convenient for callers without creating configurable
# endpoints.  Provider hosts must remain constants so model/configuration input cannot
# redirect an OAuth credential.
CALENDAR_SCOPES = GOOGLE_SCOPES
GOOGLE_CALENDAR_SCOPES = GOOGLE_SCOPES
READ_ONLY_SCOPES = GOOGLE_SCOPES
GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_REVOCATION_ENDPOINT = "https://oauth2.googleapis.com/revoke"
AUTHORIZATION_ENDPOINT = GOOGLE_AUTHORIZATION_ENDPOINT
TOKEN_ENDPOINT = GOOGLE_TOKEN_ENDPOINT
REVOCATION_ENDPOINT = GOOGLE_REVOCATION_ENDPOINT


class GoogleAuthError(CredentialError):
    """A safe, provider-independent OAuth failure."""


class GoogleAuthCallbackError(GoogleAuthError):
    """The loopback callback was invalid or did not arrive in time."""


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorisationRequest:
    """Values kept private while one browser authorisation is in progress."""

    url: str
    redirect_uri: str
    state: str
    code_verifier: str


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Capture one OAuth callback without writing request data to logs."""

    def do_GET(self) -> None:  # noqa: N802 - dictated by BaseHTTPRequestHandler
        server = self.server
        if not isinstance(server, _CallbackServer):
            return
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/oauth2callback":
            self.send_response(404)
            self.end_headers()
            return
        values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        server.callback = {key: items[0] for key, items in values.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Authorisation complete. You may close this window.")
        server.ready.set()

    def log_message(self, format: str, *args: object) -> None:
        """Avoid logging callback query strings, which can contain an auth code."""
        return None


class _CallbackServer(http.server.ThreadingHTTPServer):
    """Small loopback-only callback server used by the default onboarding flow."""

    callback: dict[str, str] | None = None

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _CallbackHandler)
        self.ready = threading.Event()


class GoogleCalendarAuthHandler(ProviderAuthHandler):
    """Onboard Google Calendar accounts for local profiles.

    Args:
        credential_store:
            Store used for the refresh credential and access-token cache.
        client_id:
            OAuth client ID from local application configuration.
        client_secret (optional):
            OAuth client secret, when using a confidential client.
        http_client (optional):
            Injectable ``httpx.Client``; useful for tests and local proxies.
        browser_opener (optional):
            Function opening the consent URL. Defaults to ``webbrowser.open``.
        callback_waiter (optional):
            Test/headless callback implementation. It receives the redirect URI and
            returns callback query parameters.
    """

    def __init__(
        self,
        credential_store: CredentialStore,
        client_id: str,
        client_secret: str | None = None,
        *,
        http_client: httpx.Client | None = None,
        browser_opener: c.Callable[[str], object] | None = None,
        callback_waiter: c.Callable[[str], c.Mapping[str, str]] | None = None,
        loopback_host: str = "127.0.0.1",
        loopback_port: int = 0,
        callback_timeout: float = 300.0,
        on_account_connected: c.Callable[[ProviderAccount], None] | None = None,
        on_account_disconnected: c.Callable[[str], None] | None = None,
    ) -> None:
        """Initialise an OAuth handler with injectable browser and HTTP seams."""
        if not client_id.strip():
            raise ValueError("client_id must not be empty")
        if loopback_host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("loopback_host must identify the local machine")
        if not 0 <= loopback_port <= 65535:
            raise ValueError("loopback_port must be between 0 and 65535")
        if callback_timeout <= 0:
            raise ValueError("callback_timeout must be positive")
        self.credential_store = credential_store
        self.client_id = client_id
        self.client_secret = client_secret
        self.http_client = http_client or httpx.Client(timeout=20.0)
        self._owns_http_client = http_client is None
        self.browser_opener = browser_opener or webbrowser.open
        self.callback_waiter = callback_waiter
        self.loopback_host = loopback_host
        self.loopback_port = loopback_port
        self.callback_timeout = callback_timeout
        self.on_account_connected = on_account_connected
        self.on_account_disconnected = on_account_disconnected
        self._accounts: dict[str, ProviderAccount] = {}

    def login(self, profile: str) -> ProviderAccount | None:
        """Run local-loopback OAuth and connect the selected profile."""
        if not profile.strip():
            raise ValueError("profile must not be empty")
        request: AuthorisationRequest
        server: _CallbackServer | None = None
        if self.callback_waiter is None:
            server = _CallbackServer((self.loopback_host, self.loopback_port))
            port = server.server_address[1]
            redirect_uri = f"http://{self.loopback_host}:{port}/oauth2callback"
            request = self.create_authorisation_request(redirect_uri=redirect_uri)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                self.browser_opener(request.url)
                if not server.ready.wait(timeout=self.callback_timeout):
                    raise GoogleAuthCallbackError("authorisation callback timed out")
                callback = server.callback or {}
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
        else:
            # A deterministic redirect URI lets command-line/headless callers provide
            # their own browser callback while retaining state and PKCE validation.
            redirect_uri = f"http://{self.loopback_host}:0/oauth2callback"
            request = self.create_authorisation_request(redirect_uri=redirect_uri)
            self.browser_opener(request.url)
            callback = dict(self.callback_waiter(redirect_uri))

        self._validate_callback(callback=callback, request=request)
        token_set = self.exchange_code(
            code=callback["code"],
            redirect_uri=request.redirect_uri,
            code_verifier=request.code_verifier,
        )
        if token_set.refresh_token is None:
            raise GoogleAuthError("offline authorisation was not granted")

        previous = self._accounts.get(profile)
        account = self.credential_store.connect(
            GOOGLE_PROVIDER,
            profile,
            refresh_token=token_set.refresh_token,
            scopes=GOOGLE_SCOPES,
        )
        self._accounts[profile] = account
        if previous is not None:
            self.credential_store.disconnect(previous, revoke=self.revoke_token)
        if self.on_account_connected is not None:
            self.on_account_connected(account)
        return account

    def status(self, profile: str | None = None) -> object:
        """Return safe local connection status for one or all profiles."""
        if profile is not None:
            account = self._accounts.get(profile)
            return {
                "provider": GOOGLE_PROVIDER,
                "profile": profile,
                "status": (
                    self.credential_store.account_status(account.credential_ref).value
                    if account is not None
                    else ConnectionStatus.DISCONNECTED.value
                ),
            }
        return {name: self.status(name) for name in sorted(self._accounts)}

    def disconnect(self, profile: str) -> None:
        """Revoke and erase a profile's local Google credentials."""
        account = self._accounts.pop(profile, None)
        if account is None:
            return
        try:
            self.credential_store.disconnect(account, revoke=self.revoke_token)
        finally:
            if self.on_account_disconnected is not None:
                self.on_account_disconnected(profile)

    def create_authorisation_request(
        self, *, redirect_uri: str
    ) -> AuthorisationRequest:
        """Create a stateful PKCE consent request with the fixed least scopes."""
        parsed = urllib.parse.urlsplit(redirect_uri)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("redirect_uri must be a loopback HTTP URL")
        state = _random_urlsafe(32)
        verifier = _random_urlsafe(64)
        challenge = _pkce_challenge(verifier)
        query = urllib.parse.urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": " ".join(GOOGLE_SCOPES),
                "access_type": "offline",
                "prompt": "consent",
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        return AuthorisationRequest(
            url=f"{GOOGLE_AUTHORIZATION_ENDPOINT}?{query}",
            redirect_uri=redirect_uri,
            state=state,
            code_verifier=verifier,
        )

    def exchange_code(
        self, *, code: str, redirect_uri: str, code_verifier: str
    ) -> TokenSet:
        """Exchange an authorisation code without exposing the provider response."""
        data: dict[str, str] = {
            "code": code,
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": code_verifier,
        }
        if self.client_secret is not None:
            data["client_secret"] = self.client_secret
        try:
            response = self.http_client.post(GOOGLE_TOKEN_ENDPOINT, data=data)
        except httpx.HTTPError:
            raise GoogleAuthError("Google token service unavailable") from None
        return _token_set_from_response(response=response, permanent=False)

    def refresh_token(self, refresh_token: str) -> TokenSet:
        """Refresh an access token for ``CredentialStore``."""
        data = {
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "grant_type": "refresh_token",
        }
        if self.client_secret is not None:
            data["client_secret"] = self.client_secret
        try:
            response = self.http_client.post(GOOGLE_TOKEN_ENDPOINT, data=data)
        except httpx.HTTPError:
            raise GoogleAuthError("Google token service unavailable") from None
        return _token_set_from_response(response=response, permanent=True)

    def revoke_token(self, refresh_token: str) -> None:
        """Revoke a refresh credential at Google."""
        try:
            response = self.http_client.post(
                GOOGLE_REVOCATION_ENDPOINT, data={"token": refresh_token}
            )
        except httpx.HTTPError:
            raise GoogleAuthError("Google revocation unavailable") from None
        if response.status_code not in {200, 400}:
            raise GoogleAuthError("credential revocation failed")

    def close(self) -> None:
        """Close an internally-created HTTP client."""
        if self._owns_http_client:
            self.http_client.close()

    # American spellings are retained for callers using Google's terminology.
    create_authorization_request = create_authorisation_request
    refresh = refresh_token
    revoke = revoke_token

    def _validate_callback(
        self, *, callback: c.Mapping[str, str], request: AuthorisationRequest
    ) -> None:
        if callback.get("state") != request.state:
            raise GoogleAuthCallbackError("invalid authorisation callback")
        if callback.get("error") or not callback.get("code"):
            raise GoogleAuthCallbackError("authorisation was not completed")


def _token_set_from_response(response: httpx.Response, *, permanent: bool) -> TokenSet:
    if response.status_code >= 400:
        if permanent and response.status_code in {400, 401, 403}:
            raise PermanentRefreshError("Google authorisation failed")
        raise GoogleAuthError("Google token request failed")
    try:
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError
        access_token = payload["access_token"]
        expires_in = float(payload.get("expires_in", 3600))
    except AttributeError, ValueError, TypeError, KeyError:
        raise GoogleAuthError("invalid Google token response") from None
    if not isinstance(access_token, str) or not access_token or expires_in <= 0:
        raise GoogleAuthError("invalid Google token response")
    refresh_token = payload.get("refresh_token")
    if refresh_token is not None and (
        not isinstance(refresh_token, str) or not refresh_token
    ):
        raise GoogleAuthError("invalid Google token response")
    granted_scope = payload.get("scope")
    if granted_scope is not None and (
        not isinstance(granted_scope, str)
        or set(granted_scope.split()) != set(GOOGLE_SCOPES)
    ):
        raise GoogleAuthError("Google granted unexpected scopes")
    return TokenSet.from_expires_in(
        access_token=access_token, expires_in=expires_in, refresh_token=refresh_token
    )


def _random_urlsafe(byte_count: int) -> str:
    return secrets.token_urlsafe(byte_count)


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


AuthorizationRequest = AuthorisationRequest
GoogleAuthHandler = GoogleCalendarAuthHandler
GoogleCalendarAuth = GoogleCalendarAuthHandler
GoogleOAuthHandler = GoogleCalendarAuthHandler

__all__ = [
    "AUTHORIZATION_ENDPOINT",
    "AuthorisationRequest",
    "AuthorizationRequest",
    "CALENDAR_SCOPES",
    "GOOGLE_AUTHORIZATION_ENDPOINT",
    "GOOGLE_CALENDAR_SCOPES",
    "GOOGLE_PROVIDER",
    "GOOGLE_REVOCATION_ENDPOINT",
    "GOOGLE_SCOPES",
    "GOOGLE_TOKEN_ENDPOINT",
    "GoogleAuthCallbackError",
    "GoogleAuthError",
    "GoogleCalendarAuth",
    "GoogleCalendarAuthHandler",
    "GoogleAuthHandler",
    "GoogleOAuthHandler",
    "READ_ONLY_SCOPES",
    "REVOCATION_ENDPOINT",
    "TOKEN_ENDPOINT",
]
