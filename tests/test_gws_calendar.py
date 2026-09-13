"""Contract tests for the local ``gws`` Calendar provider."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import threading
import typing as t

import pytest

from voicebot.auth.credentials import CredentialStore
from voicebot.providers.calendar_domain import (
    GoogleCalendarInvalidResponse,
    GoogleCalendarUnauthenticated,
    GoogleCalendarUnavailable,
)
from voicebot.providers.gws_calendar import GwsCalendarProvider
from voicebot.runtime import build_integration_runtime
from voicebot.tool_runtime import CancelledError, ToolContext, ToolStatus
from voicebot.tools.calendar import CalendarToolAdapter

START = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
END = dt.datetime(2025, 1, 2, tzinfo=dt.UTC)


def _script(tmp_path: pathlib.Path, source: str, name: str = "gws") -> str:
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(0o700)
    return str(path)


def _json_script(tmp_path: pathlib.Path, payload: object, name: str = "gws") -> str:
    return _script(
        tmp_path, f"import json\nprint(json.dumps({payload!r}))\n", name=name
    )


def test_event_arguments_are_bounded_and_read_only(tmp_path: pathlib.Path) -> None:
    """Event reads use only the fixed read-only command and allow-listed fields."""
    captured = tmp_path / "arguments.json"
    executable = _script(
        tmp_path,
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(captured)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "print(json.dumps({'items': []}))\n",
    )
    provider = GwsCalendarProvider(executable=executable)
    assert (
        provider.list_events(
            credential_ref="local",
            calendar_id="primary",
            starts_at=START,
            ends_at=END,
            query="meeting",
            max_results=7,
        )
        == []
    )

    args = t.cast(list[str], json.loads(captured.read_text()))
    assert args[:3] == ["calendar", "events", "list"]
    assert args[-2:] == ["--format", "json"]
    params = json.loads(args[args.index("--params") + 1])
    assert params == {
        "calendarId": "primary",
        "fields": (
            "items(summary,visibility,status,start(date,dateTime),end(date,dateTime))"
        ),
        "maxResults": 7,
        "orderBy": "startTime",
        "q": "meeting",
        "singleEvents": True,
        "timeMax": "2025-01-02T00:00:00Z",
        "timeMin": "2025-01-01T00:00:00Z",
    }


def test_freebusy_arguments_and_output_are_minimised(tmp_path: pathlib.Path) -> None:
    """Free/busy sends calendar IDs only and returns no event details."""
    captured = tmp_path / "arguments.json"
    payload = {"calendars": {"primary": {"busy": [{"start": "a", "end": "b"}]}}}
    executable = _script(
        tmp_path,
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(captured)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        f"print(json.dumps({payload!r}))\n",
    )
    result = GwsCalendarProvider(executable=executable).query_freebusy(
        credential_ref="local", calendar_ids=["primary"], starts_at=START, ends_at=END
    )
    assert [interval.as_dict() for interval in result["primary"]] == [
        {"starts_at": "a", "ends_at": "b"}
    ]
    args = t.cast(list[str], json.loads(captured.read_text()))
    body = json.loads(args[args.index("--json") + 1])
    assert body == {"items": [{"id": "primary"}]}
    assert "--page-all" not in args


@pytest.mark.parametrize(
    ("payload", "returncode", "expected"),
    [
        (b"not json", 0, GoogleCalendarInvalidResponse),
        (b"not json", 1, GoogleCalendarUnavailable),
        (b"login required", 1, GoogleCalendarUnauthenticated),
    ],
)
def test_cli_failures_are_safe_provider_errors(
    tmp_path: pathlib.Path, payload: bytes, returncode: int, expected: type[Exception]
) -> None:
    """Malformed, failed, and unauthenticated CLI calls never expose diagnostics."""
    executable = _script(
        tmp_path,
        "import os\n"
        f"os.write(1, {payload!r})\n"
        f"os.write(2, {payload!r})\n"
        f"raise SystemExit({returncode})\n",
    )
    with pytest.raises(expected) as error:
        GwsCalendarProvider(executable=executable).list_events(
            credential_ref="local", calendar_id="primary", starts_at=START, ends_at=END
        )
    assert "login" not in str(error.value)


def test_timeout_and_oversized_output_fail_closed(tmp_path: pathlib.Path) -> None:
    """Process limits become safe unavailable or invalid-response errors."""
    timeout_executable = _script(
        tmp_path, "import time\ntime.sleep(10)\n", name="timeout-gws"
    )
    with pytest.raises(GoogleCalendarUnavailable):
        GwsCalendarProvider(executable=timeout_executable, timeout=0.2).list_events(
            credential_ref="local", calendar_id="primary", starts_at=START, ends_at=END
        )

    oversized_executable = _json_script(tmp_path, {"items": []}, name="oversized-gws")
    with pytest.raises(GoogleCalendarInvalidResponse):
        GwsCalendarProvider(
            executable=oversized_executable, max_output_bytes=1
        ).list_events(
            credential_ref="local", calendar_id="primary", starts_at=START, ends_at=END
        )


def test_missing_cli_and_cancellation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing executables and cancelled turns perform no external call."""
    monkeypatch.setattr("voicebot.providers.gws_calendar.shutil.which", lambda _: None)
    provider = GwsCalendarProvider()
    with pytest.raises(GoogleCalendarUnavailable):
        provider.list_events(
            credential_ref="local", calendar_id="primary", starts_at=START, ends_at=END
        )
    event = threading.Event()
    event.set()
    with pytest.raises(CancelledError):
        GwsCalendarProvider(executable="/usr/local/bin/gws").list_events(
            credential_ref="local",
            calendar_id="primary",
            starts_at=START,
            ends_at=END,
            context=ToolContext(state={}, cancel_event=event),
        )


def test_tool_uses_local_profile_without_oauth_account(tmp_path: pathlib.Path) -> None:
    """The adapter resolves a gws profile without a fake refresh credential."""
    store = CredentialStore.memory_only()
    adapter = CalendarToolAdapter(
        GwsCalendarProvider(executable=_json_script(tmp_path, {"items": []})),
        store,
        profile_aliases={"mig": "dan"},
        profile_accounts={"dan": "gws-local"},
        calendar_bindings={"dan": {"min kalender": "primary"}},
    )
    result = adapter.list_calendar_events(
        ToolContext(state={}),
        {
            "profile_name": "mig",
            "calendar_name": "min kalender",
            "starts_at": START.isoformat(),
            "ends_at": END.isoformat(),
            "query": None,
            "max_results": 50,
        },
    )
    assert result.status is ToolStatus.OK
    assert result.data == {"events": []}
    assert store.account("gws-local") is None


def test_runtime_rejects_unbound_calendar_profiles_without_gws(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Calendar bindings and aliases do not follow unrelated voice profiles."""
    monkeypatch.setattr(
        "voicebot.providers.gws_calendar.shutil.which", lambda _: "/usr/bin/gws"
    )

    def unexpected_call(*_: object, **__: object) -> list[object]:
        pytest.fail("unbound calendar profile invoked gws")

    monkeypatch.setattr(GwsCalendarProvider, "list_events", unexpected_call)
    from omegaconf import DictConfig, OmegaConf

    config = t.cast(DictConfig, OmegaConf.load("config/config.yaml"))
    config.storage.database_path = str(tmp_path / "state.sqlite")
    config.storage.credential_backend = "memory"
    config.profiles.eve = {"aliases": ["Eve"]}
    config.profile_aliases = {"old eve": "eve"}
    config.integrations.google_calendar.calendar_aliases.eve = {
        "stale calendar": "stale-calendar"
    }
    runtime = build_integration_runtime(config)
    try:
        result = runtime.registry.invoke(
            "list_calendar_events",
            {
                "profile_name": "Eve",
                "calendar_name": "stale calendar",
                "starts_at": START.isoformat(),
                "ends_at": END.isoformat(),
                "query": None,
                "max_results": 1,
            },
        )
        assert result.status is ToolStatus.NOT_FOUND

        alias_result = runtime.registry.invoke(
            "list_calendar_events",
            {
                "profile_name": "old eve",
                "calendar_name": "stale calendar",
                "starts_at": START.isoformat(),
                "ends_at": END.isoformat(),
                "query": None,
                "max_results": 1,
            },
        )
        assert alias_result.status is ToolStatus.NOT_FOUND
    finally:
        runtime.stop()


def test_runtime_uses_gws_and_shipped_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Runtime assembly uses gws and retains the configured Danish binding."""
    monkeypatch.setattr(
        "voicebot.providers.gws_calendar.shutil.which", lambda _: "/usr/bin/gws"
    )
    from omegaconf import DictConfig, OmegaConf

    config = t.cast(DictConfig, OmegaConf.load("config/config.yaml"))
    config.storage.database_path = str(tmp_path / "state.sqlite")
    config.storage.credential_backend = "memory"
    runtime = build_integration_runtime(config)
    try:
        result = runtime.registry.invoke(
            "list_calendar_events",
            {
                "profile_name": "dan",
                "calendar_name": "min kalender",
                "starts_at": START.isoformat(),
                "ends_at": END.isoformat(),
                "query": None,
                "max_results": 1,
            },
        )
        assert result.status is ToolStatus.UNAVAILABLE
    finally:
        runtime.stop()
