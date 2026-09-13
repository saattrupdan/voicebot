"""Read-only Google Calendar provider through the local ``gws`` CLI.

This adapter deliberately does not know about OAuth credentials.  The CLI owns its
already-authenticated local session; this process only supplies bounded Calendar API
parameters and consumes a bounded JSON response.
"""

from __future__ import annotations

import collections.abc as c
import datetime as dt
import json
import shutil

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
from .gws_transport import (
    GwsForbidden,
    GwsInvalidResponse,
    GwsRateLimited,
    GwsTransport,
    GwsTransportError,
    GwsUnauthenticated,
    GwsUnavailable,
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
        self.transport = GwsTransport(
            executable=executable, timeout=timeout, max_output_bytes=max_output_bytes
        )
        self.executable = self.transport.executable
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
        try:
            return self.transport.invoke(
                arguments,
                allowed_commands={
                    ("calendar", "events", "list", "--params"),
                    ("calendar", "freebusy", "query", "--params"),
                },
                context=context,
            )
        except GwsTransportError as error:
            raise _calendar_error(error) from None


def _json_argument(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _compat_executable_lookup() -> str | None:
    """Retain the module-level lookup seam used by Calendar integrations."""
    return shutil.which(GWS_EXECUTABLE)


def _calendar_error(error: GwsTransportError) -> GoogleCalendarError:
    """Translate shared transport failures to Calendar-safe statuses."""
    mapping: list[tuple[type[GwsTransportError], type[GoogleCalendarError]]] = [
        (GwsUnauthenticated, GoogleCalendarUnauthenticated),
        (GwsForbidden, GoogleCalendarForbidden),
        (GwsRateLimited, GoogleCalendarRateLimited),
        (GwsInvalidResponse, GoogleCalendarInvalidResponse),
        (GwsUnavailable, GoogleCalendarUnavailable),
    ]
    for source, target in mapping:
        if isinstance(error, source):
            return target(str(error))
    return GoogleCalendarUnavailable("Google Calendar is unavailable")


__all__ = [
    "GWS_EXECUTABLE",
    "GWS_MAX_OUTPUT_BYTES",
    "GWS_TIMEOUT_SECONDS",
    "GwsCalendarProvider",
]
