"""Contract tests for the bounded Gmail CLI integration."""

from __future__ import annotations

import base64
import datetime as dt
import json
import pathlib
import subprocess

import pytest

from voicebot.providers.gws_gmail import GwsGmailProvider
from voicebot.storage import Storage
from voicebot.tool_runtime import ToolContext, ToolRuntime, ToolStatus
from voicebot.tools.gmail import (
    GmailHandleStore,
    GmailToolAdapter,
    make_gmail_tool_specs,
)


def _encoded(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).rstrip(b"=").decode()


def _message(message_id: str = "private-message-id") -> dict[str, object]:
    return {
        "id": message_id,
        "threadId": "private-thread-id",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Hello"},
                {"name": "From", "value": "sender@example.com"},
            ],
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/html",
                    "body": {
                        "data": _encoded(
                            "<p>Hello <b>world</b></p><script>bad()</script>"
                        )
                    },
                },
                {"mimeType": "text/plain", "body": {"data": _encoded("Plain body")}},
            ],
        },
    }


def _completed(payload: object) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=["gws"], returncode=0, stdout=json.dumps(payload).encode(), stderr=b""
    )


def test_gmail_reads_are_bounded_and_ids_become_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search uses only list/get and never returns provider IDs."""
    responses = [
        {"messages": [{"id": "private-message-id", "threadId": "private-thread-id"}]},
        _message(),
    ]
    commands: list[list[str]] = []

    def run(args: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(args)
        return _completed(responses.pop(0))

    monkeypatch.setattr(subprocess, "run", run)
    adapter = GmailToolAdapter(
        GwsGmailProvider(executable="/usr/local/bin/gws"), profiles=["dan"]
    )
    result = adapter.latest(
        ToolContext(state={}), {"profile_name": "dan", "max_results": 1}
    )

    assert result.status is ToolStatus.OK
    assert "private-message-id" not in json.dumps(result.as_dict())
    assert "private-thread-id" not in json.dumps(result.as_dict())
    assert len(commands) == 2
    assert [*commands[0][1:5]] == ["gmail", "users", "messages", "list"]
    assert [*commands[1][1:5]] == ["gmail", "users", "messages", "get"]
    assert result.data is not None
    messages = result.data.get("messages")
    assert isinstance(messages, list)
    assert isinstance(messages[0], dict)
    assert messages[0].get("message_handle")


def test_nested_mime_prefers_plain_and_handles_expiry() -> None:
    """Plain text wins over HTML and opaque handles expire and remain scoped."""
    now = [dt.datetime.now(dt.UTC)]
    store = GmailHandleStore(clock=lambda: now[0])
    handle = store.put(
        message_id="private-message-id",
        thread_id="private-thread-id",
        profile="dan",
        device="speaker",
    )
    assert store.resolve(handle=handle, profile="dan", device="speaker") == (
        "private-message-id",
        "private-thread-id",
    )
    assert store.resolve(handle=handle, profile="dan", device="other") is None
    now[0] += dt.timedelta(minutes=16)
    assert store.resolve(handle=handle, profile="dan", device="speaker") is None


def test_draft_is_the_only_gmail_mutation_and_private_arguments_are_not_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Draft creation is idempotent while its body stays out of SQLite."""
    seen: list[list[str]] = []

    def run(args: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(args)
        return _completed(
            {"id": "private-draft-id", "message": {"id": "private-message-id"}}
        )

    monkeypatch.setattr(subprocess, "run", run)
    provider = GwsGmailProvider(executable="/usr/local/bin/gws")
    adapter = GmailToolAdapter(provider, profiles=["dan"])
    spec = next(
        spec
        for spec in make_gmail_tool_specs(adapter)
        if spec.name == "create_gmail_draft"
    )
    assert spec.mutates is True
    assert spec.persist_arguments is False
    assert {spec.name for spec in make_gmail_tool_specs(adapter)} == {
        "search_gmail_messages",
        "list_latest_gmail_messages",
        "read_gmail_message",
        "create_gmail_draft",
    }

    with Storage(tmp_path / "state.sqlite") as storage:
        runtime = ToolRuntime()
        runtime.register(spec)
        arguments = {
            "profile_name": "dan",
            "recipients": ["to@example.com"],
            "subject": "Subject",
            "body": "private draft body",
        }
        result = runtime.invoke(
            "create_gmail_draft",
            arguments,
            ToolContext({"storage": storage}, operation_id="draft-call"),
        )
        assert result.data == {"saved": True, "sent": False}
        operation = storage.operations.get_by_key("create_gmail_draft:draft-call")
        assert operation is not None
        assert operation.request == {}
        stored = storage.database.connection.execute(
            "SELECT request_json, result_json FROM operations"
        ).fetchone()
        assert "private draft body" not in " ".join(str(item) for item in stored)
    assert len(seen) == 1
    assert seen[0][1:5] == ["gmail", "users", "drafts", "create"]
