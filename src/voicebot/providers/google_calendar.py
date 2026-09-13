"""Read-only Google Calendar provider adapter.

Only ``events.list`` and ``freeBusy.query`` are represented here.  There is no generic
request method and no write operation, making it structurally impossible for this
adapter to issue a calendar mutation.
"""

from __future__ import annotations

import collections.abc as c
import dataclasses
import datetime as dt
import typing as t
import urllib.parse

import httpx

from ..auth.credentials import CredentialStore, PermanentRefreshError, TokenSet
from ..auth.google import GOOGLE_TOKEN_ENDPOINT, GoogleCalendarAuthHandler
from ..tool_runtime import ToolContext

# Kept here as a public constant as well as in auth.google for callers assembling a
# provider without importing the auth implementation.
CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"
GOOGLE_CALENDAR_API_BASE = CALENDAR_API_BASE
MAX_INTERVAL = dt.timedelta(days=31)
MAX_RESULTS = 50


class CalendarProvider(t.Protocol):
    """Structural interface shared by local Calendar backends."""

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


class GoogleCalendarProvider:
    """Call the fixed, read-only Google Calendar endpoints."""

    def __init__(
        self,
        credential_store: CredentialStore,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        http_client: httpx.Client | None = None,
        api_base: str = CALENDAR_API_BASE,
        auth_handler: GoogleCalendarAuthHandler | None = None,
    ) -> None:
        """Initialise the provider with a fixed API host and HTTP seam."""
        if api_base != CALENDAR_API_BASE:
            # An endpoint override would make it possible for configuration to redirect
            # bearer tokens.  Tests should use httpx.MockTransport instead.
            raise ValueError("Google Calendar API host cannot be changed")
        self.credential_store = credential_store
        self.auth_handler = auth_handler
        self.client_id = client_id or (
            auth_handler.client_id if auth_handler is not None else ""
        )
        self.client_secret = client_secret or (
            auth_handler.client_secret if auth_handler is not None else None
        )
        self.http_client = http_client or httpx.Client(timeout=20.0)
        self._owns_http_client = http_client is None
        if not self.client_id:
            raise ValueError("client_id or auth_handler must be provided")

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
        """List a bounded, chronologically ordered set of safe event fields."""
        _validate_interval(starts_at=starts_at, ends_at=ends_at)
        if not 1 <= max_results <= MAX_RESULTS:
            raise ValueError("max_results must be between 1 and 50")
        if query is not None and len(query) > 200:
            raise ValueError("query is too long")
        token = self._access_token(credential_ref=credential_ref, context=context)
        params: dict[str, str | int] = {
            "timeMin": _rfc3339(starts_at),
            "timeMax": _rfc3339(ends_at),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": max_results,
            # Do not ask Google for descriptions, attendees, conference data, or IDs.
            "fields": (
                "items(summary,visibility,status,start(date,dateTime),"
                "end(date,dateTime))"
            ),
        }
        if query:
            params["q"] = query
        response = self._request(
            method="GET",
            url=f"{CALENDAR_API_BASE}/calendars/{_quote_calendar_id(calendar_id)}/events",
            token=token,
            credential_ref=credential_ref,
            params=params,
            context=context,
        )
        try:
            payload = response.json()
            items = payload["items"]
            if not isinstance(items, list):
                raise TypeError
        except ValueError, TypeError, KeyError:
            raise GoogleCalendarInvalidResponse(
                "invalid Google Calendar response"
            ) from None
        events: list[CalendarEvent] = []
        for item in items:
            if not isinstance(item, dict) or item.get("status") == "cancelled":
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
        """Query busy intervals without returning event or calendar details."""
        _validate_interval(starts_at=starts_at, ends_at=ends_at)
        if not calendar_ids or len(calendar_ids) > MAX_RESULTS:
            raise ValueError("calendar_ids must contain between 1 and 50 calendars")
        token = self._access_token(credential_ref=credential_ref, context=context)
        response = self._request(
            method="POST",
            url=f"{CALENDAR_API_BASE}/freeBusy",
            token=token,
            credential_ref=credential_ref,
            json_body={
                "timeMin": _rfc3339(starts_at),
                "timeMax": _rfc3339(ends_at),
                "items": [{"id": calendar_id} for calendar_id in calendar_ids],
            },
            context=context,
        )
        try:
            payload = response.json()
            calendars = payload["calendars"]
            if not isinstance(calendars, dict):
                raise TypeError
        except ValueError, TypeError, KeyError:
            raise GoogleCalendarInvalidResponse(
                "invalid Google free/busy response"
            ) from None
        result: dict[str, list[BusyInterval]] = {}
        for calendar_id in calendar_ids:
            entry = calendars.get(calendar_id, {})
            busy = entry.get("busy", []) if isinstance(entry, dict) else []
            if not isinstance(busy, list):
                raise GoogleCalendarInvalidResponse("invalid Google free/busy response")
            intervals: list[BusyInterval] = []
            for item in busy:
                if not isinstance(item, dict):
                    continue
                start, end = item.get("start"), item.get("end")
                if isinstance(start, str) and isinstance(end, str):
                    intervals.append(BusyInterval(starts_at=start, ends_at=end))
            result[calendar_id] = intervals
        return result

    def refresh_access_token(self, refresh_token: str) -> TokenSet:
        """Refresh a token through Google's token endpoint."""
        data = {
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "grant_type": "refresh_token",
        }
        if self.client_secret is not None:
            data["client_secret"] = self.client_secret
        try:
            response = self.http_client.post(GOOGLE_TOKEN_ENDPOINT, data=data)
        except httpx.HTTPError:
            raise GoogleCalendarUnavailable(
                "Google token service unavailable"
            ) from None
        if response.status_code in {400, 401, 403}:
            raise PermanentRefreshError("Google refresh credential rejected")
        if response.status_code >= 500:
            raise GoogleCalendarUnavailable("Google token service unavailable")
        if response.status_code >= 400:
            raise GoogleCalendarError("Google token request failed")
        try:
            payload = response.json()
            access_token = payload["access_token"]
            expires_in = float(payload.get("expires_in", 3600))
            rotated = payload.get("refresh_token")
        except ValueError, TypeError, KeyError:
            raise GoogleCalendarInvalidResponse(
                "invalid Google token response"
            ) from None
        if not isinstance(access_token, str) or not access_token:
            raise GoogleCalendarInvalidResponse("invalid Google token response")
        if rotated is not None and (not isinstance(rotated, str) or not rotated):
            raise GoogleCalendarInvalidResponse("invalid Google token response")
        return TokenSet.from_expires_in(
            access_token=access_token, expires_in=expires_in, refresh_token=rotated
        )

    def revoke_token(self, refresh_token: str) -> None:
        """Revoke a refresh token; an already-invalid token is considered revoked."""
        try:
            response = self.http_client.post(
                "https://oauth2.googleapis.com/revoke", data={"token": refresh_token}
            )
        except httpx.HTTPError:
            raise GoogleCalendarUnavailable("Google revocation unavailable") from None
        if response.status_code not in {200, 400}:
            raise GoogleCalendarError("Google revocation failed")

    def close(self) -> None:
        """Close an internally-created HTTP client."""
        if self._owns_http_client:
            self.http_client.close()

    events_list = list_events
    freebusy_query = query_freebusy
    refresh = refresh_access_token
    revoke = revoke_token

    def _access_token(self, *, credential_ref: str, context: ToolContext | None) -> str:
        if context is not None:
            context.check_cancelled()
        try:
            return self.credential_store.get_access_token(
                credential_ref, self.refresh_access_token
            )
        except PermanentRefreshError:
            raise GoogleCalendarUnauthenticated(
                "Google account needs reauthorisation"
            ) from None
        except Exception as error:
            if isinstance(error, GoogleCalendarUnavailable):
                raise
            raise GoogleCalendarUnauthenticated(
                "Google account is not connected"
            ) from None

    def _request(
        self,
        *,
        method: t.Literal["GET", "POST"],
        url: str,
        token: str,
        credential_ref: str,
        context: ToolContext | None,
        params: c.Mapping[str, str | int] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> httpx.Response:
        if context is not None:
            context.check_cancelled()
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = self.http_client.request(
                method, url, params=params, json=json_body, headers=headers
            )
        except httpx.HTTPError:
            raise GoogleCalendarUnavailable("Google Calendar is unavailable") from None
        if context is not None:
            context.check_cancelled()
        if response.status_code in {401, 403, 429} or response.status_code >= 500:
            if response.status_code == 401:
                self.credential_store.clear_access_token(credential_ref)
                raise GoogleCalendarUnauthenticated("Google account is not authorised")
            if response.status_code == 403:
                raise GoogleCalendarForbidden("Google Calendar access was denied")
            if response.status_code == 429:
                retry_after = _retry_after(response.headers.get("Retry-After"))
                raise GoogleCalendarRateLimited(
                    "Google Calendar rate limit reached", retry_after=retry_after
                )
            raise GoogleCalendarUnavailable("Google Calendar is unavailable")
        if response.status_code >= 400:
            raise GoogleCalendarError("Google Calendar request failed")
        return response


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


def _quote_calendar_id(value: str) -> str:
    if not value:
        raise ValueError("calendar_id must not be empty")
    return urllib.parse.quote(value, safe="")


def _retry_after(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        seconds = int(value)
    except ValueError:
        return None
    return max(0, min(seconds, 3600))


GoogleCalendar = GoogleCalendarProvider

__all__ = [
    "BusyInterval",
    "CalendarProvider",
    "CALENDAR_API_BASE",
    "CalendarEvent",
    "GOOGLE_CALENDAR_API_BASE",
    "GoogleCalendar",
    "GoogleCalendarError",
    "GoogleCalendarForbidden",
    "GoogleCalendarInvalidResponse",
    "GoogleCalendarProvider",
    "GoogleCalendarRateLimited",
    "GoogleCalendarUnauthenticated",
    "GoogleCalendarUnavailable",
    "MAX_INTERVAL",
    "MAX_RESULTS",
]
