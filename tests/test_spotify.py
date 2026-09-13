"""Contract tests for the Spotify integration."""

import urllib.parse

import httpx

from voicebot.auth.credentials import CredentialStore, ProviderAccount, TokenSet
from voicebot.auth.spotify import (
    SPOTIFY_CALLBACK_PORT,
    SpotifyAuthHandler,
    SpotifyCallback,
)
from voicebot.providers.spotify import SpotifyProvider
from voicebot.tool_runtime import ToolContext, ToolStatus, validate_arguments
from voicebot.tools.spotify import create_spotify_tool_specs

_SCOPES = ("user-read-playback-state", "user-modify-playback-state")


def test_callback_uses_fixed_registerable_port() -> None:
    """Spotify's exact redirect URI can be registered before onboarding."""
    auth = SpotifyAuthHandler(client_id="client", store=CredentialStore.memory_only())

    assert auth.port == SPOTIFY_CALLBACK_PORT == 8766


def _account(store: CredentialStore) -> ProviderAccount:
    return store.connect(
        "spotify", "household", refresh_token="refresh-secret", scopes=_SCOPES
    )


def test_oauth_pkce_state_and_exact_scopes() -> None:
    """Login uses a matching state, S256 challenge, and playback scopes only."""
    opened: list[str] = []

    def token(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/token")
        return httpx.Response(
            200,
            json={
                "access_token": "access-secret",
                "refresh_token": "refresh",
                "expires_in": 3600,
            },
        )

    store = CredentialStore.memory_only()
    client = httpx.Client(transport=httpx.MockTransport(token))
    auth = SpotifyAuthHandler(
        client_id="client",
        store=store,
        http_client=client,
        open_browser=opened.append,
        port=4567,
        callback_waiter=lambda uri: SpotifyCallback(
            code="one-time-code",
            state=urllib.parse.parse_qs(urllib.parse.urlparse(opened[0]).query)[
                "state"
            ][0],
        ),
    )

    account = auth.login("household")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(opened[0]).query)
    assert account is not None
    assert account.scopes == _SCOPES
    assert query["scope"] == [" ".join(_SCOPES)]
    assert query["code_challenge_method"] == ["S256"]
    assert "client_secret" not in opened[0]


def test_play_binds_search_result_and_hides_ids() -> None:
    """Playback uses only a URI returned by search and never returns device IDs."""
    requests: list[httpx.Request] = []

    def spotify(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/search"):
            return httpx.Response(
                200,
                json={
                    "tracks": {
                        "items": [
                            {
                                "name": "Kind of Blue",
                                "type": "track",
                                "uri": "spotify:track:private",
                            }
                        ]
                    }
                },
            )
        if request.url.path.endswith("/devices"):
            return httpx.Response(
                200, json={"devices": [{"id": "device-private", "name": "Kitchen"}]}
            )
        return httpx.Response(204)

    store = CredentialStore.memory_only()
    account = _account(store)
    provider = SpotifyProvider(
        store=store,
        accounts={"household": account},
        device_aliases={"kitchen": "Kitchen"},
        refresh_callback=lambda token: TokenSet.from_expires_in("access-token", 3600),
        http_client=httpx.Client(transport=httpx.MockTransport(spotify)),
    )
    result = provider.play(
        context=ToolContext(state={}),
        query="Kind of Blue",
        profile_name="household",
        device_name="kitchen",
    )

    assert result.status is ToolStatus.OK
    assert "device-private" not in result.to_json()
    assert requests[-1].url.query == b"device_id=device-private"
    assert requests[-1].content == b'{"uris":["spotify:track:private"]}'


def test_ambiguous_devices_and_rate_limit_are_safe() -> None:
    """Ambiguity and throttling return structured, non-provider-error results."""
    calls = 0

    def spotify(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.path.endswith("/devices") and calls == 1:
            return httpx.Response(
                200,
                json={
                    "devices": [
                        {"id": "one", "name": "Kitchen"},
                        {"id": "two", "name": "Office"},
                    ]
                },
            )
        return httpx.Response(429, headers={"Retry-After": "12"})

    store = CredentialStore.memory_only()
    account = _account(store)
    provider = SpotifyProvider(
        store=store,
        accounts={"household": account},
        refresh_callback=lambda token: TokenSet.from_expires_in("access-token", 3600),
        http_client=httpx.Client(transport=httpx.MockTransport(spotify)),
    )
    result = provider.control(
        context=ToolContext(state={}), action="pause", profile_name="household"
    )
    assert result.status is ToolStatus.NEEDS_CLARIFICATION
    assert result.candidates == ["Kitchen", "Office"]

    rate_limited = provider.list_devices(
        context=ToolContext(state={}), profile_name="household"
    )
    assert rate_limited.status is ToolStatus.RATE_LIMITED
    assert rate_limited.data == {"retry_after_seconds": 12}


def test_tool_specs_have_no_uri_argument() -> None:
    """Model-visible Spotify schemas expose names and aliases, not provider IDs."""
    specs = create_spotify_tool_specs(
        SpotifyProvider(store=CredentialStore.memory_only())
    )
    assert {spec.name for spec in specs} == {
        "spotify_play",
        "spotify_control",
        "spotify_set_volume",
        "spotify_now_playing",
        "spotify_list_devices",
    }
    for spec in specs:
        properties = spec.parameters["properties"]
        assert isinstance(properties, dict)
        assert "uri" not in properties


def test_spotify_schemas_reject_malformed_nullable_values() -> None:
    """Nullable Spotify fields remain strict under the merged validator."""
    specs = {
        spec.name: spec
        for spec in create_spotify_tool_specs(
            SpotifyProvider(store=CredentialStore.memory_only())
        )
    }
    play = specs["spotify_play"]
    valid = {
        "profile_name": None,
        "query": "Kind of Blue",
        "media_type": None,
        "device_name": None,
    }

    assert validate_arguments(schema=play.parameters, arguments=valid) is None
    for field, value in (
        ("profile_name", 42),
        ("media_type", True),
        ("media_type", "podcast"),
        ("device_name", []),
    ):
        malformed = {**valid, field: value}
        assert (
            validate_arguments(schema=play.parameters, arguments=malformed) is not None
        )


def test_device_aliases_are_scoped_to_spotify_profile() -> None:
    """Identical aliases cannot route a device across Spotify accounts."""
    store = CredentialStore.memory_only()
    home = store.connect(
        "spotify", "home", refresh_token="refresh-home", scopes=_SCOPES
    )
    work = store.connect(
        "spotify", "work", refresh_token="refresh-work", scopes=_SCOPES
    )
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        token = request.headers["Authorization"].removeprefix("Bearer ")
        requests.append((token, request.url.params.get("device_id", "")))
        device = "home-id" if token == "access-home" else "work-id"
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "devices": [{"id": device, "name": "Speaker", "is_active": True}]
                },
            )
        return httpx.Response(204)

    provider = SpotifyProvider(
        store,
        accounts={"home": home, "work": work},
        device_aliases={"home": {"speaker": "home-id"}, "work": {"speaker": "work-id"}},
        refresh_callback=lambda refresh: TokenSet.from_expires_in(
            f"access-{refresh.removeprefix('refresh-')}", 3600
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    for profile in ("home", "work"):
        result = provider.control(
            ToolContext({}), "pause", device_name="speaker", profile_name=profile
        )
        assert result.status is ToolStatus.OK
    assert requests == [
        ("access-home", ""),
        ("access-home", "home-id"),
        ("access-work", ""),
        ("access-work", "work-id"),
    ]


def test_refresh_rotation_and_disconnect_keep_secrets_local() -> None:
    """Refresh rotation is stored in the credential backend and disconnect erases it."""
    calls: list[httpx.Request] = []

    def token(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "access-secret",
                "refresh_token": "rotated-refresh-secret",
                "expires_in": 3600,
            },
        )

    store = CredentialStore.memory_only()
    auth = SpotifyAuthHandler(
        client_id="client",
        store=store,
        http_client=httpx.Client(transport=httpx.MockTransport(token)),
    )
    account = store.connect(
        "spotify", "household", refresh_token="refresh-secret", scopes=_SCOPES
    )
    result = auth.refresh("refresh-secret")
    store.replace_refresh_token(account.credential_ref, result.refresh_token or "")
    assert result.access_token == "access-secret"
    assert store.load_refresh_token(account.credential_ref) == "rotated-refresh-secret"

    auth._accounts["household"] = account
    auth.disconnect("household")
    assert store.load_refresh_token(account.credential_ref) is None
    status = auth.status("household")
    assert isinstance(status, dict)
    assert "refresh-secret" not in status["profile"]
    assert calls[0].content.find(b"access-secret") == -1
