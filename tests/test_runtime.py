"""Tests for end-to-end integration assembly."""

from __future__ import annotations

import datetime as dt
import math
import pathlib
import typing as t

import httpx
import pytest
from omegaconf import DictConfig, OmegaConf

from voicebot.auth.credentials import TokenSet
from voicebot.providers.listonic import ListonicProvider, ListonicSessionToken
from voicebot.providers.spotify import SpotifyProvider
from voicebot.runtime import ConfirmationManager, build_integration_runtime
from voicebot.tool_runtime import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolStatus,
    validate_arguments,
    validate_schema,
)
from voicebot.tools import MODEL_TOOL_NAMES, MUTATION_TOOL_NAMES


def _config(path: pathlib.Path) -> DictConfig:
    config = OmegaConf.load("config/config.yaml")
    config.storage.database_path = str(path)
    config.storage.credential_backend = "memory"
    return t.cast(DictConfig, config)


def test_runtime_replaces_all_model_tools_and_honours_kill_switches(
    tmp_path: pathlib.Path,
) -> None:
    """Disabled providers remain real bound tools which fail closed."""
    runtime = build_integration_runtime(_config(tmp_path / "state.sqlite"))
    try:
        assert MODEL_TOOL_NAMES <= set(runtime.registry.names)
        for name in (
            "list_calendar_events",
            "get_calendar_availability",
            "spotify_now_playing",
            "spotify_list_devices",
            "list_shopping_lists",
            "remove_shopping_item",
        ):
            spec = runtime.registry.registry.get(name)
            assert spec is not None
            assert getattr(spec.handler, "__name__", "") != "_unavailable"
        result = runtime.registry.invoke("spotify_now_playing", {"profile_name": None})
        assert result.status is ToolStatus.UNAVAILABLE
    finally:
        runtime.stop()


def test_mutation_classification_and_global_kill_switch(tmp_path: pathlib.Path) -> None:
    """Every mutation is classified and globally blocked without blocking reads."""
    config = _config(tmp_path / "state.sqlite")
    config.mutations_enabled = False
    runtime = build_integration_runtime(config)
    try:
        specs = {
            name: runtime.registry.registry.get(name) for name in runtime.registry.names
        }
        assert {
            name for name, spec in specs.items() if spec is not None and spec.mutates
        } == MUTATION_TOOL_NAMES
        blocked = runtime.registry.invoke(
            "set_timer", {"name": "tea", "duration_seconds": 10}
        )
        readable = runtime.registry.invoke("list_timers", {"name": None})
        assert blocked.status is ToolStatus.UNAVAILABLE
        assert readable.status is ToolStatus.OK
        assert runtime.state.get("running_timers", []) == []
    finally:
        runtime.stop()


def test_listonic_removal_has_independent_provider_gate(tmp_path: pathlib.Path) -> None:
    """Enable Listonic generally without enabling its unverified DELETE path."""
    config = _config(tmp_path / "state.sqlite")
    config.integrations.listonic.enabled = True
    config.integrations.listonic.allow_unofficial = True
    runtime = build_integration_runtime(config)
    try:
        provider = t.cast(ListonicProvider, runtime.providers[2])
        assert provider.available
        assert not provider.allow_unverified_item_removal
        assert not provider.circuit_open
        result = runtime.registry.invoke(
            "remove_shopping_item",
            {"profile_name": "dan", "list_name": None, "item_name": "Milk"},
            ToolContext(runtime.state, device_id="local-device"),
        )
        assert result.status is ToolStatus.UNAVAILABLE
        assert "live-verificering" in result.message_da
    finally:
        runtime.stop()


def test_all_contract_schemas_are_strict_and_numbers_are_finite(
    tmp_path: pathlib.Path,
) -> None:
    """Reject malformed contracts, invalid nullable enums, and non-finite numbers."""
    runtime = build_integration_runtime(_config(tmp_path / "state.sqlite"))
    try:
        for name in runtime.registry.names:
            spec = runtime.registry.registry.get(name)
            assert spec is not None
            assert validate_schema(spec.parameters) is None
        spotify = runtime.registry.registry.get("spotify_play")
        assert spotify is not None
        media = t.cast(
            dict[str, object],
            t.cast(dict[str, object], spotify.parameters["properties"])["media_type"],
        )
        assert validate_arguments(media, None) is None
        assert validate_arguments(media, "podcast") is not None
        number_schema: dict[str, object] = {
            "type": "object",
            "properties": {"values": {"type": "array", "items": {"type": "number"}}},
            "required": ["values"],
            "additionalProperties": False,
        }
        for value in (math.nan, math.inf, -math.inf):
            assert validate_arguments(number_schema, {"values": [value]}) is not None
    finally:
        runtime.stop()

    malformed: dict[str, object] = {
        "type": "object",
        "properties": {"value": {"type": "wat"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    with pytest.raises(ValueError, match="Invalid schema"):
        ToolRegistry(
            [
                ToolSpec(
                    "broken",
                    "broken",
                    malformed,
                    lambda context, arguments: ToolResult(ToolStatus.OK),
                )
            ]
        )


def test_real_confirmations_are_device_bound_expiring_and_exactly_once(
    tmp_path: pathlib.Path,
) -> None:
    """Spotify high volume and Listonic removal execute once after local yes."""
    config = _config(tmp_path / "state.sqlite")
    config.integrations.spotify.enabled = True
    config.integrations.spotify.device_aliases = {"dan": {"speaker": "device-1"}}
    config.integrations.listonic.enabled = True
    config.integrations.listonic.allow_unofficial = True
    config.integrations.listonic.allow_unverified_item_removal = True
    config.integrations.listonic.list_aliases = {"dan": {"groceries": "list-1"}}
    config.integrations.listonic.default_lists = {"dan": "groceries"}
    runtime = build_integration_runtime(config)
    spotify = t.cast(SpotifyProvider, runtime.providers[1])
    listonic = t.cast(ListonicProvider, runtime.providers[2])
    spotify_mutations = 0
    listonic_mutations = 0

    def spotify_handler(request: httpx.Request) -> httpx.Response:
        nonlocal spotify_mutations
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "devices": [
                        {"id": "device-1", "name": "Speaker", "is_active": True}
                    ]
                },
            )
        spotify_mutations += 1
        return httpx.Response(204)

    def listonic_handler(request: httpx.Request) -> httpx.Response:
        nonlocal listonic_mutations
        if request.method == "GET" and request.url.path == "/api/lists":
            return httpx.Response(
                200, json={"lists": [{"id": "list-1", "name": "Groceries"}]}
            )
        if request.method == "GET":
            return httpx.Response(
                200, json=[{"id": "item-1", "name": "Milk", "checked": False}]
            )
        listonic_mutations += 1
        return httpx.Response(204)

    spotify.http_client.close()
    spotify.http_client = httpx.Client(transport=httpx.MockTransport(spotify_handler))
    spotify.refresh_callback = lambda refresh: TokenSet.from_expires_in(
        "spotify-access", 3600
    )
    spotify_account = runtime.credentials.connect(
        "spotify",
        "dan",
        refresh_token="spotify-refresh",
        scopes=("user-read-playback-state", "user-modify-playback-state"),
    )
    spotify.register_account(spotify_account)
    listonic.client.close()
    listonic.client = httpx.Client(transport=httpx.MockTransport(listonic_handler))
    listonic_account = runtime.credentials.connect(
        "listonic", "dan", refresh_token="listonic-refresh"
    )
    listonic.register_session_tokens(
        listonic_account,
        ListonicSessionToken(
            "listonic-access",
            "listonic-refresh",
            dt.datetime.now(dt.UTC) + dt.timedelta(hours=1),
        ),
    )

    try:
        volume_arguments = {
            "profile_name": "dan",
            "volume_percent": 90,
            "device_name": "speaker",
        }
        pending = runtime.registry.invoke(
            "spotify_set_volume",
            volume_arguments,
            ToolContext(runtime.state, operation_id="volume-1", device_id="device-a"),
        )
        assert pending.status is ToolStatus.CONFIRMATION_REQUIRED
        operation = runtime.storage.operations.get("volume-1")
        assert operation is not None and operation.status == "pending_confirmation"
        manager = t.cast(ConfirmationManager, runtime.state["confirmation_manager"])
        assert manager.resolve(True, device_id="device-b").status is ToolStatus.CONFLICT
        still_pending = runtime.storage.operations.get("volume-1")
        assert still_pending is not None
        assert still_pending.status == "pending_confirmation"
        accepted = manager.resolve(True, device_id="device-a")
        assert accepted.status is ToolStatus.OK
        assert spotify_mutations == 1
        assert (
            manager.resolve(True, device_id="device-a").status is ToolStatus.NOT_FOUND
        )
        replay = runtime.registry.invoke(
            "spotify_set_volume",
            volume_arguments,
            ToolContext(runtime.state, operation_id="volume-1", device_id="device-a"),
        )
        assert replay.status is ToolStatus.OK
        assert spotify_mutations == 1

        pending = runtime.registry.invoke(
            "spotify_set_volume",
            volume_arguments,
            ToolContext(runtime.state, operation_id="volume-no", device_id="device-a"),
        )
        assert pending.status is ToolStatus.CONFIRMATION_REQUIRED
        assert manager.resolve(False, device_id="device-a").status is ToolStatus.OK
        assert spotify_mutations == 1

        manager.expiry = dt.timedelta(seconds=-1)
        pending = runtime.registry.invoke(
            "spotify_set_volume",
            volume_arguments,
            ToolContext(
                runtime.state, operation_id="volume-expired", device_id="device-a"
            ),
        )
        assert pending.status is ToolStatus.CONFIRMATION_REQUIRED
        replay = runtime.registry.invoke(
            "spotify_set_volume",
            volume_arguments,
            ToolContext(
                runtime.state, operation_id="volume-expired", device_id="device-a"
            ),
        )
        assert replay.status is ToolStatus.INVALID_REQUEST
        assert manager.resolve(True, device_id="device-a").status is ToolStatus.CONFLICT
        expired_operation = runtime.storage.operations.get("volume-expired")
        assert expired_operation is not None
        assert expired_operation.status == "failed"
        assert expired_operation.result is not None
        assert expired_operation.result["status"] == "invalid_request"
        assert spotify_mutations == 1
        manager.expiry = dt.timedelta(minutes=2)

        removal_arguments = {
            "profile_name": "dan",
            "list_name": None,
            "item_name": "Milk",
        }
        pending = runtime.registry.invoke(
            "remove_shopping_item",
            removal_arguments,
            ToolContext(runtime.state, operation_id="remove-1", device_id="device-a"),
        )
        assert pending.status is ToolStatus.CONFIRMATION_REQUIRED
        removed = manager.resolve(True, device_id="device-a")
        assert removed.status is ToolStatus.OK
        assert listonic_mutations == 1
        assert removed.status is not ToolStatus.CONFIRMATION_REQUIRED
    finally:
        runtime.stop()


def test_documented_safety_configuration_matches_defaults() -> None:
    """Keep operator-facing safety flags and reminder guarantees consistent."""
    config = OmegaConf.load("config/config.yaml")
    readme = pathlib.Path("README.md").read_text()
    plan = pathlib.Path("docs/tool-integrations-plan.md").read_text()
    assert config.mutations_enabled is True
    assert config.integrations.listonic.allow_unverified_item_removal is False
    assert "allow_unverified_item_removal" in readme
    assert "at-least-once" in readme
    assert "can repeat a reminder" in readme
    assert "not automatically\n  mentioned" in plan
    assert "uv run src/scripts/integrations.py status" in readme


def test_shipped_config_builds_and_stops_without_hydra(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shipped config can assemble its runtime outside a Hydra command."""
    config_path = pathlib.Path(__file__).parents[1] / "config" / "config.yaml"
    config = t.cast(DictConfig, OmegaConf.load(config_path))
    config.storage.credential_backend = "memory"
    monkeypatch.chdir(tmp_path)

    runtime = build_integration_runtime(config)
    runtime.start()
    try:
        assert runtime.scheduler.is_running
        assert runtime._worker is not None
        assert runtime._worker.is_alive()
    finally:
        runtime.stop()
    assert not runtime.scheduler.is_running
    assert (tmp_path / ".local/state/voicebot.sqlite").exists()


def test_runtime_starts_and_stops_one_scheduler_worker(tmp_path: pathlib.Path) -> None:
    """The bot-owned runtime wires both durable workers to one lifecycle."""
    runtime = build_integration_runtime(_config(tmp_path / "state.sqlite"))
    runtime.start()
    try:
        assert runtime.scheduler.is_running
        assert runtime._worker is not None
        assert runtime._worker.is_alive()
    finally:
        runtime.stop()
    assert not runtime.scheduler.is_running
