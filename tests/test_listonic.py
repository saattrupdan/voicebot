"""Sanitised contract tests for the unofficial Listonic adapter."""

import base64
import collections.abc as c
import datetime as dt
import json
import pathlib
import threading

import httpx
import pytest

from scripts.integrations import _PersistedAuthHandler
from voicebot.auth import (
    ConnectionStatus,
    CredentialError,
    CredentialStore,
    MemoryCredentialBackend,
)
from voicebot.auth.listonic import (
    LISTONIC_LOGIN_URL,
    IsolatedBrowserSession,
    ListonicAuthHandler,
)
from voicebot.providers.listonic import (
    LISTONIC_BASE_URL,
    ListonicAuthenticationError,
    ListonicCircuitBreaker,
    ListonicContractDriftError,
    ListonicError,
    ListonicProvider,
    ListonicSessionToken,
)
from voicebot.storage import Storage
from voicebot.tool_runtime import ToolContext, ToolStatus
from voicebot.tools.shopping import ShoppingTools

NOW = dt.datetime(2030, 1, 1, tzinfo=dt.UTC)


def _jwt(expiry: dt.datetime) -> str:
    def encode(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'exp': expiry.timestamp()})}.signature"


def _provider(handler: c.Callable[[httpx.Request], httpx.Response]) -> ListonicProvider:
    store = CredentialStore.memory_only()
    account = store.connect("listonic", "household", refresh_token="refresh-canary")
    provider = ListonicProvider(
        store,
        accounts={"household": account},
        enabled=True,
        allow_unofficial=True,
        allow_unverified_item_removal=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider.register_session_tokens(
        account,
        ListonicSessionToken(
            "access-canary", "refresh-canary", NOW + dt.timedelta(hours=1)
        ),
    )
    return provider


def test_provider_pins_contract_and_mixed_request_casing() -> None:
    """Pin Listonic's production paths, verbs, and request casing."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "GET" and request.url.path == "/api/lists":
            return httpx.Response(
                200, json=[{"Id": "list-1", "Name": "Groceries", "ItemsCount": 2}]
            )
        if request.method == "POST":
            return httpx.Response(201, json={"id": "item-2", "name": "Bread"})
        raise AssertionError(request.url)

    provider = _provider(handler)
    account = provider.accounts["household"]
    provider.list_lists(account)
    provider.add_item(account, "list-1", name="Bread", amount=2, unit="loaves")

    assert calls[0].url == httpx.URL(
        f"{LISTONIC_BASE_URL}/api/lists"
        "?includeShares=true&archive=false&includeItems=true"
    )
    assert calls[0].headers["Version"] == "web:4.0.0"
    assert calls[0].headers["Culture"] == "en"
    assert calls[0].headers["DeviceId"]
    assert calls[1].method == "POST"
    assert calls[1].url.path == "/api/lists/list-1/items"
    assert json.loads(calls[1].content) == {
        "Amount": 2,
        "Unit": "loaves",
        "name": "Bread",
    }


def test_contract_drift_opens_only_listonic_breaker() -> None:
    """Open the Listonic breaker when a response shape changes."""
    breaker = ListonicCircuitBreaker()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": []})

    provider = _provider(handler)
    provider.breaker = breaker
    with pytest.raises(ListonicContractDriftError):
        provider.list_lists(provider.accounts["household"])
    assert breaker.is_open
    assert not provider.available


def test_tools_use_defaults_and_require_structured_removal_confirmation() -> None:
    """Use defaults safely and never remove before confirmation."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/lists":
            return httpx.Response(200, json=[{"Id": "list-1", "Name": "Groceries"}])
        if request.method == "GET":
            return httpx.Response(
                200, json=[{"Id": "item-1", "Name": "Milk", "Amount": "", "Checked": 0}]
            )
        if request.method == "DELETE":
            return httpx.Response(200)
        raise AssertionError(request.url)

    provider = _provider(handler)
    tools = ShoppingTools(
        provider,
        profile_aliases={"home": "household"},
        list_aliases={"household": {"groceries": "list-1"}},
        default_profile="household",
        default_lists={"household": "list-1"},
    )
    context = ToolContext(state={})
    arguments = {"profile_name": None, "list_name": None, "item_name": "milk"}
    result = tools.remove_shopping_item(context, arguments)
    assert result.status is ToolStatus.CONFIRMATION_REQUIRED
    assert result.data == {
        "profile_name": "home",
        "list_name": "groceries",
        "item_name": "Milk",
        "action": "remove_shopping_item",
    }

    context.state["confirmed"] = True
    result = tools.remove_shopping_item(context, arguments)
    assert result.status is ToolStatus.OK


def test_ambiguous_items_and_cancelled_mutation_do_not_call_mutation_endpoint() -> None:
    """Clarify duplicate names and honour cancellation before PATCH."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "GET" and request.url.path == "/api/lists":
            return httpx.Response(
                200, json={"lists": [{"id": "list-1", "name": "Groceries"}]}
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {"id": "item-1", "name": "Milk", "checked": False},
                    {"id": "item-2", "name": "MILK", "checked": False},
                ],
            )
        raise AssertionError(request.url)

    provider = _provider(handler)
    tools = ShoppingTools(
        provider,
        profile_aliases={"home": "household"},
        default_profile="household",
        default_lists={"household": "list-1"},
    )
    context = ToolContext(state={})
    result = tools.set_shopping_item_checked(
        context,
        {"profile_name": None, "list_name": None, "item_name": "milk", "checked": True},
    )
    assert result.status is ToolStatus.NEEDS_CLARIFICATION
    assert result.candidates == ["Milk", "MILK"]

    context.cancel_event = threading.Event()
    context.cancel_event.set()
    result = tools.set_shopping_item_checked(
        context,
        {"profile_name": None, "list_name": None, "item_name": "Milk", "checked": True},
    )
    assert result.status is ToolStatus.CANCELLED
    assert not any(request.method == "PATCH" for request in calls)


def test_list_requires_explicit_default_and_removal_gate() -> None:
    """Never select the first list or expose unverified removal by default."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(
            200, json={"lists": [{"id": "list-1", "name": "Groceries"}]}
        )

    store = CredentialStore.memory_only()
    account = store.connect("listonic", "household", refresh_token="refresh-canary")
    provider = ListonicProvider(
        store,
        accounts={"household": account},
        enabled=True,
        allow_unofficial=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider.register_session_tokens(
        account,
        ListonicSessionToken(
            "access-canary", "refresh-canary", NOW + dt.timedelta(hours=1)
        ),
    )
    tools = ShoppingTools(provider, default_profile="household")
    result = tools.get_shopping_list(
        ToolContext({}),
        {"profile_name": None, "list_name": None, "include_checked": False},
    )
    assert result.status is ToolStatus.NOT_FOUND
    with pytest.raises(ListonicContractDriftError, match="not enabled"):
        provider.remove_item(account, "list-1", "item-1")
    assert calls == ["GET"]


def test_access_state_rehydrates_and_expires_without_opening_circuit() -> None:
    """Persist access only in credentials and fail closed at its actual expiry."""
    backend = MemoryCredentialBackend()
    first = CredentialStore(backend=backend)
    account = first.connect("listonic", "household", refresh_token="refresh-canary")
    first.store_provider_secret(
        account.credential_ref,
        "listonic_access",
        json.dumps(
            {
                "access_token": "access-canary",
                "expires_at": (
                    dt.datetime.now(dt.UTC) + dt.timedelta(minutes=5)
                ).isoformat(),
            }
        ),
    )
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json={"lists": []})

    second = CredentialStore(backend=backend)
    reconstructed = second.connect(
        "listonic", "household", credential_ref=account.credential_ref
    )
    provider = ListonicProvider(
        second,
        accounts={"household": reconstructed},
        enabled=True,
        allow_unofficial=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.list_lists(reconstructed) == []
    assert seen == ["Bearer access-canary"]
    assert not provider.circuit_open

    second.store_provider_secret(
        account.credential_ref,
        "listonic_access",
        json.dumps(
            {
                "access_token": "expired-canary",
                "expires_at": (
                    dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1)
                ).isoformat(),
            }
        ),
    )
    expired_store = CredentialStore(backend=backend)
    expired_account = expired_store.connect(
        "listonic", "household", credential_ref=account.credential_ref
    )
    expired = ListonicProvider(
        expired_store,
        accounts={"household": expired_account},
        enabled=True,
        allow_unofficial=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ListonicError, match="temporarily unavailable"):
        expired.list_lists(expired_account)
    assert not expired.circuit_open
    assert (
        expired_store.account_status(account.credential_ref)
        is ConnectionStatus.CONNECTED
    )
    assert expired_store.load_refresh_token(account.credential_ref) == "refresh-canary"
    assert (
        expired_store.load_provider_secret(account.credential_ref, "listonic_access")
        is None
    )


def test_expired_access_refreshes_headlessly_and_rotates_keychain_token() -> None:
    """Refresh Listonic without a browser and retain only the rotated credential."""
    backend = MemoryCredentialBackend()
    store = CredentialStore(backend=backend)
    now = dt.datetime.now(dt.UTC)
    account = store.connect("listonic", "household", refresh_token="old-refresh")
    store.store_provider_secret(
        account.credential_ref,
        "listonic_access",
        json.dumps(
            {
                "access_token": "expired",
                "expires_at": (now - dt.timedelta(seconds=1)).isoformat(),
            }
        ),
    )
    refreshed_access = _jwt(now + dt.timedelta(hours=1))
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/loginextended":
            assert request.method == "POST"
            assert dict(request.url.params) == {
                "provider": "refresh_token",
                "autoMerge": "1",
                "autoDestruct": "1",
            }
            assert request.headers["Content-Type"].startswith(
                "application/x-www-form-urlencoded"
            )
            assert request.headers["ClientAuthorization"].startswith("Bearer ")
            assert "Authorization" not in request.headers
            assert request.content == b"refresh_token=old-refresh"
            return httpx.Response(
                200,
                json={
                    "access_token": refreshed_access,
                    "refresh_token": "rotated-refresh",
                },
            )
        assert request.url.path == "/api/lists"
        assert request.headers["Authorization"] == f"Bearer {refreshed_access}"
        return httpx.Response(200, json=[])

    provider = ListonicProvider(
        store,
        accounts={"household": account},
        enabled=True,
        allow_unofficial=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.list_lists(account) == []
    assert [request.url.path for request in calls] == [
        "/api/loginextended",
        "/api/lists",
    ]
    assert store.load_refresh_token(account.credential_ref) == "rotated-refresh"
    assert store.load_provider_secret(account.credential_ref, "listonic_access")


def test_rejected_refresh_fails_permanently_before_body_parsing() -> None:
    """Treat a rejected non-JSON refresh as re-onboarding, not transient drift."""
    store = CredentialStore.memory_only()
    account = store.connect("listonic", "household", refresh_token="rejected-refresh")

    def rejected(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/loginextended"
        return httpx.Response(401, text="not json")

    provider = ListonicProvider(
        store,
        accounts={"household": account},
        enabled=True,
        allow_unofficial=True,
        http_client=httpx.Client(transport=httpx.MockTransport(rejected)),
    )
    with pytest.raises(ListonicAuthenticationError, match="re-onboarding"):
        provider.list_lists(account)
    assert store.account_status(account.credential_ref) is ConnectionStatus.DISCONNECTED


def test_transient_and_malformed_refreshes_preserve_connection_state() -> None:
    """Disconnect only rejected credentials and trip drift only for bad contracts."""
    for response, expected_error, breaker_open in (
        (httpx.Response(503, text="unavailable"), ListonicError, False),
        (
            httpx.Response(200, json={"unexpected": "shape"}),
            ListonicContractDriftError,
            True,
        ),
    ):
        store = CredentialStore.memory_only()
        account = store.connect(
            "listonic", "household", refresh_token="still-valid-refresh"
        )
        provider = ListonicProvider(
            store,
            accounts={"household": account},
            enabled=True,
            allow_unofficial=True,
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda request, value=response: value)
            ),
        )
        with pytest.raises(expected_error):
            provider.list_lists(account)
        assert (
            store.account_status(account.credential_ref) is ConnectionStatus.CONNECTED
        )
        assert provider.circuit_open is breaker_open


def test_listonic_login_status_and_disconnect_survive_reconstruction(
    tmp_path: pathlib.Path,
) -> None:
    """CLI-equivalent services share metadata and credential-only access state."""

    class Browser:
        def open_login(self, url: str) -> None:
            assert url.endswith("/login")

        def export_tokens(self) -> object:
            return {
                "access_token": "access-canary",
                "refresh_token": "refresh-canary",
                "expires_at": (NOW + dt.timedelta(hours=1)).isoformat(),
            }

        def destroy(self) -> None:
            return None

    def browser_factory() -> IsolatedBrowserSession:
        return Browser()

    backend = MemoryCredentialBackend()
    path = tmp_path / "state.sqlite"
    storage = Storage(path)
    first_store = CredentialStore(backend=backend)
    first = _PersistedAuthHandler(
        "listonic",
        ListonicAuthHandler(first_store, browser_factory, clock=lambda: NOW),
        storage=storage,
        credentials=first_store,
    )
    account = first.login("household")
    assert account is not None
    storage.close()

    reconstructed_storage = Storage(path)
    reconstructed_store = CredentialStore(backend=backend)
    reconstructed = _PersistedAuthHandler(
        "listonic",
        ListonicAuthHandler(reconstructed_store, browser_factory, clock=lambda: NOW),
        storage=reconstructed_storage,
        credentials=reconstructed_store,
    )
    assert getattr(reconstructed.status("household"), "state") == "connected"
    reconstructed_store.store_provider_secret(
        account.credential_ref,
        "listonic_access",
        json.dumps(
            {
                "access_token": "expired-canary",
                "expires_at": (NOW - dt.timedelta(seconds=1)).isoformat(),
            }
        ),
    )
    assert getattr(reconstructed.status("household"), "state") == "connected"
    reconstructed.disconnect("household")
    assert getattr(reconstructed.status("household"), "state") == "disconnected"
    assert reconstructed_store.load_refresh_token(account.credential_ref) is None
    assert (
        reconstructed_store.load_provider_secret(
            account.credential_ref, "listonic_access"
        )
        is None
    )
    reconstructed_storage.close()


def test_browser_onboarding_destroys_session_and_redacts_failure() -> None:
    """Destroy browser profiles and keep browser failures secret-free."""

    class Browser:
        def __init__(self) -> None:
            self.destroyed = False

        def open_login(self, url: str) -> None:
            assert url == LISTONIC_LOGIN_URL

        def export_tokens(self) -> dict[str, str]:
            return {
                "access_token": "access-canary",
                "refresh_token": "refresh-canary",
                "expires_at": "2030-01-01T01:00:00Z",
                "password": "password-canary",
            }

        def destroy(self) -> None:
            self.destroyed = True

    browser = Browser()
    handler = ListonicAuthHandler(
        CredentialStore.memory_only(), lambda: browser, clock=lambda: NOW
    )
    account = handler.login("household")
    assert account.profile == "household"
    assert browser.destroyed

    current_browser = Browser()
    current_browser.export_tokens = lambda: {
        "auth-persist": json.dumps(
            {
                "state": {
                    "token": _jwt(NOW + dt.timedelta(hours=1)),
                    "refreshToken": "current-refresh",
                }
            }
        )
    }
    current = ListonicAuthHandler(
        CredentialStore.memory_only(), lambda: current_browser, clock=lambda: NOW
    ).login("current")
    assert current.profile == "current"
    assert current_browser.destroyed

    for malformed_token in ("not-a-jwt", "a.e30.b", "x" * 8193):
        malformed_browser = Browser()
        malformed_browser.export_tokens = lambda token=malformed_token: {
            "auth-persist": json.dumps(
                {"state": {"token": token, "refreshToken": "refresh"}}
            )
        }
        with pytest.raises(CredentialError):
            ListonicAuthHandler(
                CredentialStore.memory_only(),
                lambda browser=malformed_browser: browser,
                clock=lambda: NOW,
            ).login("malformed")
        assert malformed_browser.destroyed

    class BrokenBrowser(Browser):
        def export_tokens(self) -> dict[str, str]:
            raise RuntimeError("password-canary")

    broken = BrokenBrowser()
    with pytest.raises(CredentialError) as error:
        ListonicAuthHandler(CredentialStore.memory_only(), lambda: broken).login("home")
    assert "password-canary" not in str(error.value)
    assert broken.destroyed
    assert "password-canary" not in repr(handler.status("household"))
