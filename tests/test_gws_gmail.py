"""Contract tests for the bounded Gmail CLI integration."""

from __future__ import annotations

import base64
import datetime as dt
import json
import pathlib
import sys
import threading
import time
import typing as t

import pytest
from omegaconf import DictConfig, OmegaConf

from voicebot.providers.gws_gmail import GmailInvalidResponse, GwsGmailProvider
from voicebot.providers.gws_transport import GwsInvalidResponse, GwsTransport
from voicebot.runtime import build_integration_runtime
from voicebot.storage import Storage
from voicebot.tool_runtime import ToolContext, ToolRuntime, ToolStatus
from voicebot.tools.gmail import (
    GmailHandleStore,
    GmailToolAdapter,
    make_gmail_tool_specs,
)


def _encoded(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).rstrip(b"=").decode()


def _message(
    *, body: dict[str, object] | None = None, snippet: str = "Metadata snippet"
) -> dict[str, object]:
    return {
        "id": "private-message-id",
        "threadId": "private-thread-id",
        "snippet": snippet,
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Hello"},
                {"name": "From", "value": "sender@example.com"},
                {"name": "Date", "value": "today"},
            ],
            **(body or {"mimeType": "text/plain", "body": {"data": _encoded("Body")}}),
        },
    }


def _script(tmp_path: pathlib.Path, source: str, name: str = "gws") -> str:
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(0o700)
    return str(path)


def _json_script(tmp_path: pathlib.Path, payload: object, name: str = "gws") -> str:
    return _script(
        tmp_path, f"import json\nprint(json.dumps({payload!r}))\n", name=name
    )


def test_gmail_reads_are_bounded_and_snippets_come_from_metadata(
    tmp_path: pathlib.Path,
) -> None:
    """List/get are bounded and only metadata get supplies the safe snippet."""
    calls = tmp_path / "calls.jsonl"
    listed = {
        "messages": [
            {
                "id": "private-message-id",
                "threadId": "private-thread-id",
                "snippet": "Wrong list snippet",
            }
        ]
    }
    metadata = _message(snippet="Right metadata snippet")
    executable = _script(
        tmp_path,
        "import json, pathlib, sys\n"
        f"path = pathlib.Path({str(calls)!r})\n"
        "with path.open('a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"payload = {listed!r} if sys.argv[4] == 'list' else {metadata!r}\n"
        "print(json.dumps(payload))\n",
    )
    adapter = GmailToolAdapter(
        GwsGmailProvider(executable=executable), profiles=["dan"]
    )

    result = adapter.latest(
        ToolContext(state={}), {"profile_name": "dan", "max_results": 1}
    )

    assert result.status is ToolStatus.OK
    assert "private-message-id" not in json.dumps(result.as_dict())
    assert "private-thread-id" not in json.dumps(result.as_dict())
    assert result.data is not None
    messages = result.data.get("messages")
    assert isinstance(messages, list)
    assert isinstance(messages[0], dict)
    assert messages[0]["snippet"] == "Right metadata snippet"
    assert messages[0].get("message_handle")
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [command[:4] for command in commands] == [
        ["gmail", "users", "messages", "list"],
        ["gmail", "users", "messages", "get"],
    ]
    get_params = json.loads(commands[1][commands[1].index("--params") + 1])
    assert get_params["format"] == "metadata"
    assert "snippet" in get_params["fields"]


@pytest.mark.parametrize("payload", [{}, {"messages": None}, {"messages": {}}])
def test_message_list_allows_only_a_missing_or_list_messages_field(
    tmp_path: pathlib.Path, payload: dict[str, object]
) -> None:
    """An omitted messages field is empty while present non-lists are malformed."""
    provider = GwsGmailProvider(executable=_json_script(tmp_path, payload))
    if "messages" not in payload:
        assert provider.list_latest_messages(max_results=1) == []
    else:
        with pytest.raises(GmailInvalidResponse):
            provider.list_latest_messages(max_results=1)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            {
                "mimeType": "multipart/mixed",
                "parts": [
                    {
                        "mimeType": "multipart/alternative",
                        "parts": [
                            {
                                "mimeType": "text/html",
                                "body": {"data": _encoded("<p>HTML choice</p>")},
                            },
                            {
                                "mimeType": "text/plain",
                                "body": {"data": _encoded("Nested plain choice")},
                            },
                        ],
                    }
                ],
            },
            "Nested plain choice",
        ),
        (
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    {
                        "mimeType": "text/html",
                        "body": {
                            "data": _encoded(
                                "<p>Hello <b>world</b></p><script>bad()</script>"
                            )
                        },
                    }
                ],
            },
            "Hello world",
        ),
        (
            {"mimeType": "text/plain", "body": {"data": "%%%not-base64%%%"}},
            "(Denne besked indeholder ingen læsbar tekst.)",
        ),
    ],
)
def test_nested_mime_plain_html_and_malformed_base64(
    tmp_path: pathlib.Path, body: dict[str, object], expected: str
) -> None:
    """Real message reads prefer nested plain text, then safe HTML, then fallback."""
    provider = GwsGmailProvider(executable=_json_script(tmp_path, _message(body=body)))
    message = provider.read_message(message_id="private-message-id")
    assert message.body == expected
    assert "bad()" not in message.body


def test_handles_expire_and_remain_profile_and_device_scoped() -> None:
    """Opaque handles cannot cross a profile, device, or their short lifetime."""
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


def test_post_launch_cancellation_does_not_cancel_or_repeat_a_draft(
    tmp_path: pathlib.Path,
) -> None:
    """A launched draft ignores later cancellation and keeps at-most-once execution."""
    launched = tmp_path / "launched"
    executable = _script(
        tmp_path,
        "import json, pathlib, time\n"
        f"path = pathlib.Path({str(launched)!r})\n"
        "with path.open('a') as stream:\n"
        "    stream.write('draft\\n')\n"
        "time.sleep(0.1)\n"
        "print(json.dumps({'id': 'private-draft-id'}))\n",
    )
    cancel_event = threading.Event()

    def cancel_after_launch() -> None:
        while not launched.exists():
            time.sleep(0.001)
        cancel_event.set()

    canceller = threading.Thread(target=cancel_after_launch)
    canceller.start()
    with Storage(tmp_path / "state.sqlite") as storage:
        runtime = ToolRuntime()
        adapter = GmailToolAdapter(
            GwsGmailProvider(executable=executable), profiles=["dan"]
        )
        draft_spec = next(
            spec
            for spec in make_gmail_tool_specs(adapter)
            if spec.name == "create_gmail_draft"
        )
        runtime.register(draft_spec)
        arguments = {
            "profile_name": "dan",
            "recipients": ["to@example.com"],
            "subject": "Subject",
            "body": "private draft body",
        }
        context = ToolContext(
            {"storage": storage}, cancel_event=cancel_event, operation_id="draft-call"
        )
        first = runtime.invoke("create_gmail_draft", arguments, context)
        second = runtime.invoke(
            "create_gmail_draft",
            arguments,
            ToolContext({"storage": storage}, operation_id="draft-call"),
        )
        operation = storage.operations.get_by_key("create_gmail_draft:draft-call")
        assert operation is not None
        assert operation.status == "completed"
        assert operation.request == {}
        stored = storage.database.connection.execute(
            "SELECT request_json, result_json FROM operations"
        ).fetchone()
        assert "private draft body" not in " ".join(str(item) for item in stored)
    canceller.join(timeout=1)
    assert first.status is ToolStatus.OK
    assert second.as_dict() == first.as_dict()
    assert launched.read_text().splitlines() == ["draft"]


def test_draft_timeout_is_persisted_as_non_retryable_outcome_unknown(
    tmp_path: pathlib.Path,
) -> None:
    """A timeout after launch never looks like cancellation or safe retry."""
    launched = tmp_path / "timeout-launched"
    executable = _script(
        tmp_path,
        "import pathlib, time\n"
        f"pathlib.Path({str(launched)!r}).write_text('launched')\n"
        "time.sleep(10)\n",
    )
    with Storage(tmp_path / "state.sqlite") as storage:
        runtime = ToolRuntime()
        adapter = GmailToolAdapter(
            GwsGmailProvider(executable=executable, timeout=0.5), profiles=["dan"]
        )
        runtime.register(
            next(
                spec
                for spec in make_gmail_tool_specs(adapter)
                if spec.name == "create_gmail_draft"
            )
        )
        arguments = {
            "profile_name": "dan",
            "recipients": ["to@example.com"],
            "subject": "Subject",
            "body": "private draft body",
        }
        context = ToolContext({"storage": storage}, operation_id="timeout-draft")
        first = runtime.invoke("create_gmail_draft", arguments, context)
        second = runtime.invoke(
            "create_gmail_draft",
            arguments,
            ToolContext({"storage": storage}, operation_id="timeout-draft"),
        )
        operation = storage.operations.get_by_key("create_gmail_draft:timeout-draft")
        assert operation is not None
        assert operation.status == "failed"
        assert operation.result == first.as_dict()
        assert operation.request == {}
    assert launched.exists()
    assert first.status is ToolStatus.OUTCOME_UNKNOWN
    assert first.retryable is False
    assert second.as_dict() == first.as_dict()


@pytest.mark.parametrize(
    "recipient",
    [
        "first@example.com,second@example.com",
        "first@example.com;second@example.com",
        "Friends:person@example.com;",
        "Person <person@example.com>",
        "person@example.com\r\nBcc:other@example.com",
        "person@localhost",
        "person@-example.com",
        "person@example..com",
        ".person@example.com",
        "person..name@example.com",
    ],
)
def test_draft_rejects_anything_except_one_defect_free_mailbox(recipient: str) -> None:
    """Each recipient array item must be exactly one plain mailbox."""
    adapter = GmailToolAdapter(
        GwsGmailProvider(executable="/does/not/exist"), profiles=["dan"]
    )
    result = adapter.create_draft(
        ToolContext(state={}),
        {
            "profile_name": "dan",
            "recipients": [recipient],
            "subject": "Subject",
            "body": "Body",
        },
    )
    assert result.status is ToolStatus.INVALID_REQUEST


@pytest.mark.parametrize("file_descriptor", [1, 2])
def test_output_overflow_terminates_the_child_without_diagnostics(
    tmp_path: pathlib.Path, file_descriptor: int
) -> None:
    """Either full pipe overflowing the cap immediately terminates the child."""
    terminated = tmp_path / "terminated"
    executable = _script(
        tmp_path,
        "import os, pathlib, signal, time\n"
        f"marker = pathlib.Path({str(terminated)!r})\n"
        "def stop(_signum, _frame):\n"
        "    marker.write_text('terminated')\n"
        "    os._exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"os.write({file_descriptor}, b'secret-diagnostic' * 200)\n"
        "time.sleep(10)\n",
    )
    transport = GwsTransport(executable=executable, timeout=5, max_output_bytes=128)
    started = time.monotonic()
    with pytest.raises(GwsInvalidResponse) as error:
        transport.invoke_read(
            ["gmail", "users", "messages", "list"],
            allowed_commands={("gmail", "users", "messages", "list")},
        )
    assert time.monotonic() - started < 2
    assert terminated.read_text() == "terminated"
    assert "secret-diagnostic" not in str(error.value)


def test_runtime_gmail_uses_only_explicit_shipped_profile_binding(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime assembly neither derives Gmail profiles nor ignores empty config."""
    monkeypatch.setattr(
        "voicebot.providers.gws_transport.shutil.which", lambda _: "/usr/bin/gws"
    )
    config = t.cast(DictConfig, OmegaConf.load("config/config.yaml"))
    config.storage.database_path = str(tmp_path / "bound.sqlite")
    config.storage.credential_backend = "memory"
    config.profiles.eve = {"aliases": ["Eve"]}
    runtime = build_integration_runtime(config)
    try:
        result = runtime.registry.invoke(
            "list_latest_gmail_messages", {"profile_name": "eve", "max_results": 1}
        )
        assert result.status is ToolStatus.NOT_FOUND
        result = runtime.registry.invoke(
            "list_latest_gmail_messages", {"profile_name": "dan", "max_results": 1}
        )
        assert result.status is ToolStatus.UNAVAILABLE
    finally:
        runtime.stop()

    config.storage.database_path = str(tmp_path / "unbound.sqlite")
    config.integrations.gmail.profile_bindings = {}
    runtime = build_integration_runtime(config)
    try:
        result = runtime.registry.invoke(
            "list_latest_gmail_messages", {"profile_name": "dan", "max_results": 1}
        )
        assert result.status is ToolStatus.NOT_FOUND
    finally:
        runtime.stop()


def test_gmail_exposes_only_reads_and_unsent_draft_creation() -> None:
    """No send, modify, or delete operation enters the model allow-list."""
    adapter = GmailToolAdapter(
        GwsGmailProvider(executable="/does/not/exist"), profiles=["dan"]
    )
    specs = make_gmail_tool_specs(adapter)
    assert {spec.name for spec in specs} == {
        "search_gmail_messages",
        "list_latest_gmail_messages",
        "read_gmail_message",
        "create_gmail_draft",
    }
    draft = next(spec for spec in specs if spec.name == "create_gmail_draft")
    assert draft.mutates is True
    assert draft.persist_arguments is False
    assert all(word not in spec.name for spec in specs for word in ("send", "delete"))
