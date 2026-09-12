"""Tests for end-to-end integration assembly."""

from __future__ import annotations

import pathlib
import typing as t

from omegaconf import DictConfig, OmegaConf

from voicebot.runtime import build_integration_runtime
from voicebot.tool_runtime import ToolStatus
from voicebot.tools import MODEL_TOOL_NAMES


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
            assert spec.handler.__name__ != "_unavailable"
        result = runtime.registry.invoke("spotify_now_playing", {"profile_name": None})
        assert result.status is ToolStatus.UNAVAILABLE
    finally:
        runtime.stop()


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
