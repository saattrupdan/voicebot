"""Contract tests for the local ``gws`` Calendar backend."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess
import threading
import typing as t

import pytest

from voicebot.auth.credentials import CredentialStore
from voicebot.providers.google_calendar import (
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


def _completed(
    payload: object, *, returncode: int = 0
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=["gws"],
        returncode=returncode,
        stdout=json.dumps(payload).encode(),
        stderr=b"",
    )


def test_event_arguments_are_bounded_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event reads use only the fixed read-only command and allow-listed fields."""
    seen: dict[str, object] = {}

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return _completed({"items": []})

    monkeypatch.setattr(subprocess, "run", run)
    provider = GwsCalendarProvider(executable="/usr/local/bin/gws")
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

    args = t.cast(list[str], seen["args"])
    assert args[:3] == ["/usr/local/bin/gws", "calendar", "events"]
    assert args[3] == "list"
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
    kwargs = t.cast(dict[str, object], seen["kwargs"])
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 15.0


def test_freebusy_arguments_and_output_are_minimised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Free/busy sends calendar IDs only and returns no event details."""
    seen: list[str] = []

    def run(args: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        seen.extend(args)
        return _completed(
            {"calendars": {"primary": {"busy": [{"start": "a", "end": "b"}]}}}
        )

    monkeypatch.setattr(subprocess, "run", run)
    result = GwsCalendarProvider(executable="/usr/local/bin/gws").query_freebusy(
        credential_ref="local", calendar_ids=["primary"], starts_at=START, ends_at=END
    )
    assert [interval.as_dict() for interval in result["primary"]] == [
        {"starts_at": "a", "ends_at": "b"}
    ]
    body = json.loads(seen[seen.index("--json") + 1])
    assert body == {"items": [{"id": "primary"}]}
    assert "--page-all" not in seen


@pytest.mark.parametrize(
    ("payload", "returncode", "expected"),
    [
        (b"not json", 0, GoogleCalendarInvalidResponse),
        (b"not json", 1, GoogleCalendarUnavailable),
        (b"login required", 1, GoogleCalendarUnauthenticated),
    ],
)
def test_cli_failures_are_safe_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    returncode: int,
    expected: type[Exception],
) -> None:
    """Malformed, failed, and unauthenticated CLI calls never expose diagnostics."""

    def run(*_: object, **__: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=["gws"], returncode=returncode, stdout=payload, stderr=payload
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(expected) as error:
        GwsCalendarProvider(executable="/usr/local/bin/gws").list_events(
            credential_ref="local", calendar_id="primary", starts_at=START, ends_at=END
        )
    assert "login" not in str(error.value)


def test_timeout_and_oversized_output_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process limits become safe unavailable or invalid-response errors."""

    def timeout(*_: object, **__: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd="gws", timeout=15)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(GoogleCalendarUnavailable):
        GwsCalendarProvider(executable="/usr/local/bin/gws").list_events(
            credential_ref="local", calendar_id="primary", starts_at=START, ends_at=END
        )

    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs: _completed({"items": []})
    )
    with pytest.raises(GoogleCalendarInvalidResponse):
        GwsCalendarProvider(
            executable="/usr/local/bin/gws", max_output_bytes=1
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


def test_tool_uses_local_profile_without_oauth_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter resolves a gws profile without a fake refresh credential."""
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs: _completed({"items": []})
    )
    store = CredentialStore.memory_only()
    adapter = CalendarToolAdapter(
        GwsCalendarProvider(executable="/usr/local/bin/gws"),
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


def test_runtime_selects_gws_backend_and_shipped_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Runtime assembly selects gws and retains the configured Danish binding."""
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
