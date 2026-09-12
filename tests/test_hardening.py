"""Regression tests for the durable runtime hardening seams."""

from __future__ import annotations

import pathlib
import typing as t

import pytest
from omegaconf import DictConfig, OmegaConf

from scripts import integrations
from voicebot.runtime import build_integration_runtime
from voicebot.storage import Storage
from voicebot.tool_runtime import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolRuntime,
    ToolSpec,
    ToolStatus,
)


def test_tool_operations_are_idempotent_and_redacted(tmp_path: pathlib.Path) -> None:
    """A completed call is returned once and never exposes its token payload."""
    storage = Storage(tmp_path / "state.sqlite")
    calls = 0

    def handler(context: ToolContext, arguments: dict[str, object]) -> ToolResult:
        nonlocal calls
        del context, arguments
        calls += 1
        return ToolResult(ToolStatus.OK, data={"access_token": "secret-token"})

    runtime = ToolRuntime(
        ToolRegistry(
            [
                ToolSpec(
                    "mutate",
                    "mutate",
                    {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    handler,
                )
            ]
        )
    )
    context = ToolContext({"storage": storage}, operation_id="call-1")
    first = runtime.invoke("mutate", {}, context)
    second = runtime.invoke(
        "mutate", {}, ToolContext({"storage": storage}, operation_id="call-1")
    )
    try:
        assert calls == 1
        assert first.data == {"access_token": "[REDACTED]"}
        assert second.data == first.data
        operation = storage.operations.get_by_key("mutate:call-1")
        assert operation is not None
        assert operation.status == "completed"
    finally:
        storage.close()


def test_runtime_uses_configured_device_and_restores_legacy_tools(
    tmp_path: pathlib.Path,
) -> None:
    """Fresh runtime assembly exposes the local device and old tool handlers."""
    config = t.cast(DictConfig, OmegaConf.load("config/config.yaml"))
    config.storage.database_path = str(tmp_path / "state.sqlite")
    config.storage.credential_backend = "memory"
    runtime = build_integration_runtime(config)
    try:
        assert runtime.state["device_id"] == "kitchen-speaker"
        assert {"get_weather", "get_news", "search_web", "meow"} <= set(
            runtime.registry.names
        )
    finally:
        runtime.stop()


def test_missing_listonic_helper_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Listonic setup reports a local prerequisite instead of prompting."""
    monkeypatch.delenv("LISTONIC_BROWSER_HELPER", raising=False)
    monkeypatch.setattr(integrations.shutil, "which", lambda name: None)
    with pytest.raises(integrations.BrowserHelperUnavailable, match="helper"):
        integrations._isolated_browser_factory()
