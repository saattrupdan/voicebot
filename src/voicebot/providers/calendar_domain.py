"""Shared domain types for the read-only Google Calendar integration."""

from __future__ import annotations

import collections.abc as c
import dataclasses
import datetime as dt
import typing as t

from ..tool_runtime import ToolContext

MAX_INTERVAL = dt.timedelta(days=31)
MAX_RESULTS = 50


class CalendarProvider(t.Protocol):
    """Structural interface shared by local Calendar providers."""

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
        """Return safe events for a bounded interval."""

    def query_freebusy(
        self,
        *,
        credential_ref: str,
        calendar_ids: c.Sequence[str],
        starts_at: dt.datetime,
        ends_at: dt.datetime,
        context: ToolContext | None = None,
    ) -> dict[str, list[BusyInterval]]:
        """Return busy intervals without event details."""


class GoogleCalendarError(Exception):
    """Base class for safe Google Calendar provider errors."""

    def __init__(self, message: str, *, retry_after: int | None = None) -> None:
        """Create a safe error and retain an optional bounded retry hint."""
        super().__init__(message)
        self.retry_after = retry_after


class GoogleCalendarUnauthenticated(GoogleCalendarError):
    """Google rejected the access token."""


class GoogleCalendarForbidden(GoogleCalendarError):
    """Google denied access to the selected calendar."""


class GoogleCalendarRateLimited(GoogleCalendarError):
    """Google asked the caller to slow down."""


class GoogleCalendarUnavailable(GoogleCalendarError):
    """Google or the network was temporarily unavailable."""


class GoogleCalendarInvalidResponse(GoogleCalendarError):
    """Google returned an unusable response."""


@dataclasses.dataclass(frozen=True, slots=True)
class CalendarEvent:
    """Safe event fields suitable for model-facing output."""

    title: str
    starts_at: str
    ends_at: str
    all_day: bool = False
    calendar_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return only the deliberately allow-listed event fields."""
        result: dict[str, object] = {
            "title": self.title,
            "starts_at": self.starts_at,
            "ends_at": self.ends_at,
            "all_day": self.all_day,
        }
        if self.calendar_name is not None:
            result["calendar_name"] = self.calendar_name
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class BusyInterval:
    """An interval in which a calendar is unavailable, without event details."""

    starts_at: str
    ends_at: str

    def as_dict(self) -> dict[str, str]:
        """Return the model-safe interval representation."""
        return {"starts_at": self.starts_at, "ends_at": self.ends_at}


def _event_from_payload(item: dict[str, object]) -> CalendarEvent | None:
    start = item.get("start")
    end = item.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    all_day = isinstance(start.get("date"), str)
    start_value = start.get("date") if all_day else start.get("dateTime")
    end_value = end.get("date") if all_day else end.get("dateTime")
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        return None
    private = item.get("visibility") == "private"
    title = "Optaget" if private else _safe_title(item.get("summary"))
    return CalendarEvent(
        title=title, starts_at=start_value, ends_at=end_value, all_day=all_day
    )


def _safe_title(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "(uden titel)"
    return value[:500]


def _validate_interval(*, starts_at: dt.datetime, ends_at: dt.datetime) -> None:
    if starts_at.tzinfo is None or ends_at.tzinfo is None:
        raise ValueError("timestamps must include an offset")
    if ends_at <= starts_at:
        raise ValueError("ends_at must be after starts_at")
    if ends_at - starts_at > MAX_INTERVAL:
        raise ValueError("calendar interval is too large")


def _rfc3339(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "BusyInterval",
    "CalendarEvent",
    "CalendarProvider",
    "GoogleCalendarError",
    "GoogleCalendarForbidden",
    "GoogleCalendarInvalidResponse",
    "GoogleCalendarRateLimited",
    "GoogleCalendarUnauthenticated",
    "GoogleCalendarUnavailable",
    "MAX_INTERVAL",
    "MAX_RESULTS",
]
