"""Contract tests for the read-only Google Calendar integration."""

import base64
import hashlib
import json
import urllib.parse

import httpx
import pytest

from voicebot.auth.credentials import CredentialStore
from voicebot.auth.google import (
    GOOGLE_SCOPES,
    GoogleAuthCallbackError,
    GoogleCalendarAuthHandler,
)
from voicebot.providers.google_calendar import GoogleCalendarProvider
from voicebot.tool_runtime import ToolContext, ToolStatus
from voicebot.tools.calendar import CalendarToolAdapter, make_calendar_tool_specs

STARTS = "2025-01-01T00:00:00+00:00"
ENDS = "2025-01-02T00:00:00+00:00"


def test_authorisation_request_uses_exact_scopes_and_pkce() -> None:
    """Consent requests use offline access, state, and S256 PKCE."""
    handler = GoogleCalendarAuthHandler(
        CredentialStore.memory_only(), "client-id", browser_opener=lambda _: None
    )
    request = handler.create_authorisation_request(
        redirect_uri="http://127.0.0.1:1234/oauth2callback"
    )
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(request.url).query)
    assert query["scope"] == [" ".join(GOOGLE_SCOPES)]
    assert query["access_type"] == ["offline"]
    assert query["code_challenge_method"] == ["S256"]
    challenge = (
        base64.urlsafe_b64encode(
            hashlib.sha256(request.code_verifier.encode()).digest()
        )
        .rstrip(b"=")
        .decode()
    )
    assert query["code_challenge"] == [challenge]
    assert len(request.state) >= 32


def test_login_rejects_csrf_state_and_does_not_store_tokens() -> None:
    """A callback from another authorisation attempt cannot connect an account."""
    store = CredentialStore.memory_only()
    handler = GoogleCalendarAuthHandler(
        store,
        "client-id",
        browser_opener=lambda _: None,
        callback_waiter=lambda _: {"state": "attacker", "code": "code"},
    )
    with pytest.raises(GoogleAuthCallbackError):
        handler.login("dan")
    status = handler.status("dan")
    assert isinstance(status, dict)
    assert status["status"] == "disconnected"


def test_refresh_and_revocation_are_form_posts_without_secret_leakage() -> None:
    """Refresh rotation and revocation use raw HTTP form requests."""
    seen: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/revoke"):
            return httpx.Response(200, request=request)
        return httpx.Response(
            200,
            request=request,
            json={"access_token": "access", "expires_in": 3600, "refresh_token": "new"},
        )

    client = httpx.Client(transport=httpx.MockTransport(transport))
    handler = GoogleCalendarAuthHandler(
        CredentialStore.memory_only(), "client-id", http_client=client
    )
    token = handler.refresh_token("refresh-secret")
    handler.revoke_token("refresh-secret")
    assert token.access_token == "access"
    assert b"refresh-secret" in seen[0].content
    assert b"refresh-secret" in seen[1].content
    assert "refresh-secret" not in str(handler.status())


def _adapter(transport: httpx.MockTransport) -> CalendarToolAdapter:
    store = CredentialStore.memory_only()
    account = store.connect("google_calendar", "dan", refresh_token="refresh")
    client = httpx.Client(transport=transport)
    provider = GoogleCalendarProvider(store, client_id="client-id", http_client=client)
    return CalendarToolAdapter(
        provider,
        store,
        profile_aliases=[("Daniel", "dan"), ("Anna", "anna")],
        profile_accounts={"dan": account.credential_ref},
        calendar_bindings={"dan": {"arbejde": "calendar-id"}},
    )


def test_events_are_field_filtered_and_private_events_become_busy() -> None:
    """Model data excludes IDs/descriptions and masks private event titles."""

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(
                200,
                request=request,
                json={"access_token": "access", "expires_in": 3600},
            )
        return httpx.Response(
            200,
            request=request,
            json={
                "items": [
                    {
                        "id": "provider-id",
                        "summary": "Private secret",
                        "description": "do not expose",
                        "visibility": "private",
                        "start": {"dateTime": STARTS},
                        "end": {"dateTime": ENDS},
                    }
                ]
            },
        )

    adapter = _adapter(httpx.MockTransport(transport))
    result = adapter.list_calendar_events(
        ToolContext(state={}),
        {
            "profile_name": "Daniel",
            "calendar_name": "arbejde",
            "starts_at": STARTS,
            "ends_at": ENDS,
            "query": None,
            "max_results": 50,
        },
    )
    assert result.status is ToolStatus.OK
    encoded = json.dumps(result.as_dict())
    assert "Optaget" in encoded
    assert "provider-id" not in encoded
    assert "do not expose" not in encoded
    assert "Private secret" not in encoded


def test_profile_ambiguity_and_cross_profile_binding_fail_closed() -> None:
    """Aliases clarify locally and a binding cannot borrow another profile's account."""
    store = CredentialStore.memory_only()
    first = store.connect("google_calendar", "dan", refresh_token="one")
    second = store.connect("google_calendar", "anna", refresh_token="two")
    provider = GoogleCalendarProvider(
        store,
        client_id="client-id",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, request=request, json={"items": []})
            )
        ),
    )
    adapter = CalendarToolAdapter(
        provider,
        store,
        profile_aliases=[("same", "dan"), ("same", "anna")],
        profile_accounts={"dan": first.credential_ref, "anna": second.credential_ref},
        calendar_bindings={"dan": {"home": "home-id"}, "anna": {"home": "work-id"}},
    )
    ambiguous = adapter.list_calendar_events(
        ToolContext(state={}),
        {
            "profile_name": "same",
            "calendar_name": "home",
            "starts_at": STARTS,
            "ends_at": ENDS,
            "query": None,
            "max_results": 1,
        },
    )
    assert ambiguous.status is ToolStatus.NEEDS_CLARIFICATION
    assert set(ambiguous.candidates) == {"dan", "anna"}
    assert {spec.name for spec in make_calendar_tool_specs(adapter)} == {
        "list_calendar_events",
        "get_calendar_availability",
    }
