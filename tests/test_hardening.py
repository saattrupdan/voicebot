"""Regression tests for the durable runtime hardening seams."""

from __future__ import annotations

import json
import pathlib
import subprocess
import threading
import time
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
                    mutates=True,
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


def test_concurrent_identical_mutations_share_stored_result(
    tmp_path: pathlib.Path,
) -> None:
    """Only the atomic operation owner executes while peers recover its result."""
    storage = Storage(tmp_path / "state.sqlite")
    calls = 0
    started = threading.Event()

    def handler(context: ToolContext, arguments: dict[str, object]) -> ToolResult:
        nonlocal calls
        del context, arguments
        calls += 1
        started.set()
        time.sleep(0.05)
        return ToolResult(ToolStatus.OK, data={"done": True})

    schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    runtime = ToolRuntime(
        ToolRegistry([ToolSpec("mutate", "mutate", schema, handler, mutates=True)])
    )
    results: list[ToolResult] = []

    def invoke() -> None:
        results.append(
            runtime.invoke(
                "mutate",
                {},
                ToolContext({"storage": storage}, operation_id="same-call"),
            )
        )

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    assert started.wait(timeout=1)
    second.start()
    first.join()
    second.join()
    try:
        assert calls == 1
        assert [result.status for result in results] == [ToolStatus.OK, ToolStatus.OK]
        assert all(result.data == {"done": True} for result in results)
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


def test_listonic_agent_browser_smoke_uses_headed_json_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise onboarding against agent-browser's real JSON response shape."""
    commands: list[list[str]] = []
    eval_response = {
        "success": True,
        "data": {
            "origin": "https://listonic.com/",
            "result": {
                "localStorage": {
                    "auth": json.dumps(
                        {
                            "access_token": "access-canary",
                            "refresh_token": "refresh-canary",
                            "expires_in": 3600,
                        }
                    )
                },
                "sessionStorage": {},
            },
        },
        "error": None,
    }
    responses = {
        "open": {
            "success": True,
            "data": {"title": "Listonic", "url": "https://listonic.com/login"},
            "error": None,
        },
        "eval": eval_response,
        "cookies": {"success": True, "data": {"cookies": []}, "error": None},
        "close": {"success": True, "data": {"closed": True}, "error": None},
    }

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        if "open" in command:
            response = responses["open"]
        elif "eval" in command:
            response = responses["eval"]
        elif "cookies" in command:
            response = responses["cookies"]
        else:
            response = responses["close"]
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(integrations.subprocess, "run", run)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    session = integrations._IsolatedBrowserSession("agent-browser")
    session.open_login("https://listonic.com/login")
    exported = session.export_tokens()
    session.destroy()

    assert isinstance(exported, dict)
    assert "access-canary" in str(exported)
    assert any("--headed" in command for command in commands)
    eval_command = next(command for command in commands if "eval" in command)
    assert "JSON.stringify" not in eval_command
    assert "--json" in eval_command
    sessions = {command[command.index("--session") + 1] for command in commands}
    assert len(sessions) == 1
    assert not any(
        secret in " ".join(command)
        for command in commands
        for secret in ("access-canary", "refresh-canary")
    )


def test_missing_listonic_helper_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Listonic setup reports a local prerequisite instead of prompting."""
    monkeypatch.delenv("LISTONIC_BROWSER_HELPER", raising=False)
    monkeypatch.setattr(integrations.shutil, "which", lambda name: None)
    with pytest.raises(integrations.BrowserHelperUnavailable, match="helper"):
        integrations._isolated_browser_factory()
