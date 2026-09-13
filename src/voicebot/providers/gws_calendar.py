"""Read-only Google Calendar provider through the local ``gws`` CLI.

This adapter deliberately does not know about OAuth credentials.  The CLI owns its
already-authenticated local session; this process only supplies bounded Calendar API
parameters and consumes a bounded JSON response.
"""

from __future__ import annotations

import collections.abc as c
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess

from ..tool_runtime import ToolContext
from .calendar_domain import (
    MAX_RESULTS,
    BusyInterval,
    CalendarEvent,
    GoogleCalendarError,
    GoogleCalendarForbidden,
    GoogleCalendarInvalidResponse,
    GoogleCalendarRateLimited,
    GoogleCalendarUnauthenticated,
    GoogleCalendarUnavailable,
    _event_from_payload,
    _rfc3339,
    _validate_interval,
)

GWS_EXECUTABLE = "gws"
GWS_TIMEOUT_SECONDS = 15.0
GWS_MAX_OUTPUT_BYTES = 256 * 1024


class GwsCalendarProvider:
    """Invoke only bounded, read-only Calendar commands in the local CLI."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        timeout: float = GWS_TIMEOUT_SECONDS,
        max_output_bytes: int = GWS_MAX_OUTPUT_BYTES,
    ) -> None:
        """Initialise one fixed executable lookup and bounded process limits."""
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.executable = _resolve_executable(executable)
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes

    def list_events(
        self,
        *,
        credential_ref: str,
        calendar_id: str,
        starts_at: dt.datetime,
        ends_at: dt.datetime,
        query: str | None = None,
        max_results: int = MAX_RESULTS,
        context: ToolContext | None = None,
    ) -> list[CalendarEvent]:
        """List safe event fields from a bounded local CLI request."""
        del credential_ref
        _validate_interval(starts_at=starts_at, ends_at=ends_at)
        if not 1 <= max_results <= MAX_RESULTS:
            raise ValueError("max_results must be between 1 and 50")
        if not calendar_id:
            raise ValueError("calendar_id must not be empty")
        if query is not None and len(query) > 200:
            raise ValueError("query is too long")
        params: dict[str, str | int | bool] = {
            "calendarId": calendar_id,
            "timeMin": _rfc3339(starts_at),
            "timeMax": _rfc3339(ends_at),
            "singleEvents": True,
            "orderBy": "startTime",
            "maxResults": max_results,
            "fields": (
                "items(summary,visibility,status,start(date,dateTime),"
                "end(date,dateTime))"
            ),
        }
        if query:
            params["q"] = query
        payload = self._invoke(
            [
                "calendar",
                "events",
                "list",
                "--params",
                _json_argument(params),
                "--format",
                "json",
            ],
            context=context,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise GoogleCalendarInvalidResponse("invalid Google Calendar response")
        events: list[CalendarEvent] = []
        for item in payload["items"]:
            if not isinstance(item, dict):
                raise GoogleCalendarInvalidResponse("invalid Google Calendar response")
            if item.get("status") == "cancelled":
                continue
            event = _event_from_payload(item=item)
            if event is not None:
                events.append(event)
        events.sort(key=lambda event: event.starts_at)
        return events[:max_results]

    def query_freebusy(
        self,
        *,
        credential_ref: str,
        calendar_ids: c.Sequence[str],
        starts_at: dt.datetime,
        ends_at: dt.datetime,
        context: ToolContext | None = None,
    ) -> dict[str, list[BusyInterval]]:
        """Return busy intervals without exposing event or calendar details."""
        del credential_ref
        _validate_interval(starts_at=starts_at, ends_at=ends_at)
        if not calendar_ids or len(calendar_ids) > MAX_RESULTS:
            raise ValueError("calendar_ids must contain between 1 and 50 calendars")
        if any(not calendar_id for calendar_id in calendar_ids):
            raise ValueError("calendar IDs must not be empty")
        payload = self._invoke(
            [
                "calendar",
                "freebusy",
                "query",
                "--params",
                _json_argument(
                    {"timeMin": _rfc3339(starts_at), "timeMax": _rfc3339(ends_at)}
                ),
                "--json",
                _json_argument({"items": [{"id": value} for value in calendar_ids]}),
                "--format",
                "json",
            ],
            context=context,
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("calendars"), dict
        ):
            raise GoogleCalendarInvalidResponse("invalid Google free/busy response")
        calendars = payload["calendars"]
        result: dict[str, list[BusyInterval]] = {}
        for calendar_id in calendar_ids:
            entry = calendars.get(calendar_id, {})
            if not isinstance(entry, dict):
                raise GoogleCalendarInvalidResponse("invalid Google free/busy response")
            busy = entry.get("busy", [])
            if not isinstance(busy, list):
                raise GoogleCalendarInvalidResponse("invalid Google free/busy response")
            intervals: list[BusyInterval] = []
            for item in busy:
                if not isinstance(item, dict):
                    raise GoogleCalendarInvalidResponse(
                        "invalid Google free/busy response"
                    )
                start, end = item.get("start"), item.get("end")
                if isinstance(start, str) and isinstance(end, str):
                    intervals.append(BusyInterval(starts_at=start, ends_at=end))
            result[calendar_id] = intervals
        return result

    def close(self) -> None:
        """Release no resources; each CLI request is an isolated process."""

    def _invoke(
        self, arguments: list[str], *, context: ToolContext | None
    ) -> dict[str, object]:
        if context is not None:
            context.check_cancelled()
        if self.executable is None:
            raise GoogleCalendarUnavailable("Google Calendar CLI is unavailable")
        try:
            completed = subprocess.run(
                [self.executable, *arguments],
                capture_output=True,
                check=False,
                shell=False,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise GoogleCalendarUnavailable("Google Calendar CLI timed out") from None
        except OSError, subprocess.SubprocessError:
            raise GoogleCalendarUnavailable(
                "Google Calendar CLI is unavailable"
            ) from None
        if context is not None:
            context.check_cancelled()
        stdout = completed.stdout
        stderr = completed.stderr
        if not isinstance(stdout, bytes):
            stdout = str(stdout).encode()
        if not isinstance(stderr, bytes):
            stderr = str(stderr).encode()
        if len(stdout) > self.max_output_bytes or len(stderr) > self.max_output_bytes:
            raise GoogleCalendarInvalidResponse("Google Calendar response is too large")
        if completed.returncode != 0:
            raise _error_from_cli_failure(stderr)
        try:
            payload = json.loads(stdout)
        except UnicodeDecodeError, json.JSONDecodeError:
            raise GoogleCalendarInvalidResponse(
                "invalid Google Calendar CLI response"
            ) from None
        if not isinstance(payload, dict):
            raise GoogleCalendarInvalidResponse("invalid Google Calendar CLI response")
        return payload


def _resolve_executable(executable: str | None) -> str | None:
    if executable is not None:
        path = pathlib.Path(executable)
        if not path.is_absolute():
            raise ValueError("gws executable must be an absolute path")
        return str(path)
    found = shutil.which(GWS_EXECUTABLE)
    if found is None:
        return None
    return os.path.realpath(found)


def _json_argument(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _error_from_cli_failure(stderr: bytes) -> GoogleCalendarError:
    """Map untrusted CLI diagnostics to the existing safe provider statuses."""
    text = stderr[:4096].decode("utf-8", errors="ignore").casefold()
    if any(
        value in text
        for value in (
            "401",
            "unauthorised",
            "unauthorized",
            "authentication",
            "not authenticated",
            "unauthenticated",
            "auth required",
            "no credentials",
            "not logged in",
            "login",
        )
    ):
        return GoogleCalendarUnauthenticated("Google account is not authorised")
    if any(value in text for value in ("403", "forbidden", "permission denied")):
        return GoogleCalendarForbidden("Google Calendar access was denied")
    if "429" in text or "rate limit" in text:
        return GoogleCalendarRateLimited("Google Calendar rate limit reached")
    return GoogleCalendarUnavailable("Google Calendar CLI request failed")


__all__ = [
    "GWS_EXECUTABLE",
    "GWS_MAX_OUTPUT_BYTES",
    "GWS_TIMEOUT_SECONDS",
    "GwsCalendarProvider",
]
