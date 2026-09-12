"""Sanitised contract tests for the unofficial Listonic adapter."""

import collections.abc as c
import datetime as dt
import json
import threading

import httpx
import pytest

from voicebot.auth import CredentialError, CredentialStore
from voicebot.auth.listonic import ListonicAuthHandler
from voicebot.providers.listonic import (
    LISTONIC_BASE_URL,
    ListonicCircuitBreaker,
    ListonicContractDriftError,
    ListonicProvider,
    ListonicSessionToken,
)
from voicebot.tool_runtime import ToolContext, ToolStatus
from voicebot.tools.shopping import ShoppingTools

NOW = dt.datetime(2030, 1, 1, tzinfo=dt.UTC)


def _provider(handler: c.Callable[[httpx.Request], httpx.Response]) -> ListonicProvider:
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
    return provider


def test_provider_pins_contract_and_mixed_request_casing() -> None:
    """Pin Listonic's production paths, verbs, and request casing."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "GET" and request.url.path == "/api/lists":
            return httpx.Response(
                200, json={"lists": [{"id": "list-1", "Name": "Groceries"}]}
            )
        if request.method == "POST":
            return httpx.Response(201, json={"id": "item-2", "name": "Bread"})
        raise AssertionError(request.url)

    provider = _provider(handler)
    account = provider.accounts["household"]
    provider.list_lists(account)
    provider.add_item(account, "list-1", name="Bread", amount=2, unit="loaves")

    assert calls[0].url == httpx.URL(f"{LISTONIC_BASE_URL}/api/lists")
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
            return httpx.Response(
                200, json={"lists": [{"id": "list-1", "name": "Groceries"}]}
            )
        if request.method == "GET":
            return httpx.Response(
                200, json=[{"id": "item-1", "name": "Milk", "checked": False}]
            )
        if request.method == "DELETE":
            return httpx.Response(204)
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


def test_browser_onboarding_destroys_session_and_redacts_failure() -> None:
    """Destroy browser profiles and keep browser failures secret-free."""

    class Browser:
        def __init__(self) -> None:
            self.destroyed = False

        def open_login(self, url: str) -> None:
            assert url == f"{LISTONIC_BASE_URL}/login"

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

    class BrokenBrowser(Browser):
        def export_tokens(self) -> dict[str, str]:
            raise RuntimeError("password-canary")

    broken = BrokenBrowser()
    with pytest.raises(CredentialError) as error:
        ListonicAuthHandler(CredentialStore.memory_only(), lambda: broken).login("home")
    assert "password-canary" not in str(error.value)
    assert broken.destroyed
    assert "password-canary" not in repr(handler.status("household"))
