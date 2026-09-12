"""Model-visible, read-only Google Calendar tools."""

from __future__ import annotations

import collections.abc as c
import dataclasses
import datetime as dt

from ..auth.credentials import CredentialStore, ProviderAccount
from ..providers.google_calendar import (
    BusyInterval,
    GoogleCalendarError,
    GoogleCalendarForbidden,
    GoogleCalendarProvider,
    GoogleCalendarRateLimited,
    GoogleCalendarUnauthenticated,
    GoogleCalendarUnavailable,
)
from ..resolution import AliasResolver, ResolutionStatus
from ..tool_runtime import ToolContext, ToolResult, ToolSpec, ToolStatus


@dataclasses.dataclass(frozen=True, slots=True)
class CalendarBinding:
    """A local calendar alias and its private provider ID."""

    profile_name: str
    alias: str
    calendar_id: str
    credential_ref: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class CalendarProfileAccount:
    """Local profile routing to one credential-store account."""

    profile_name: str
    credential_ref: str


class CalendarToolAdapter:
    """Resolve local profiles/calendars and invoke the read-only provider.

    ``profile_aliases`` and ``calendar_bindings`` are setup-owned inputs.  They may be
    supplied as dataclasses, mappings, or simple iterables to keep this adapter usable
    with both the SQLite repository and small command-line assemblies.  Provider IDs
    are accepted only through those binding inputs, never from model arguments.
    """

    def __init__(
        self,
        provider: GoogleCalendarProvider,
        credential_store: CredentialStore,
        *,
        profile_aliases: c.Mapping[str, str] | c.Iterable[tuple[str, str]] = (),
        calendar_bindings: c.Iterable[CalendarBinding]
        | c.Mapping[str | tuple[str, str], c.Mapping[str, str] | str] = (),
        profile_accounts: c.Mapping[str, str | ProviderAccount]
        | c.Iterable[CalendarProfileAccount] = (),
        default_profile: str | None = None,
    ) -> None:
        """Initialise local alias routing without retaining provider payloads."""
        self.provider = provider
        self.credential_store = credential_store
        self._profiles = self._normalise_accounts(profile_accounts)
        profile_pairs: list[tuple[str, str]] = []
        if isinstance(profile_aliases, c.Mapping):
            for alias, reference in profile_aliases.items():
                if isinstance(alias, str) and isinstance(reference, str):
                    profile_pairs.append((alias, reference))
        else:
            for pair in profile_aliases:
                if (
                    isinstance(pair, tuple)
                    and len(pair) == 2
                    and isinstance(pair[0], str)
                    and isinstance(pair[1], str)
                ):
                    profile_pairs.append((pair[0], pair[1]))
        profile_pairs.extend(
            (name, name)
            for name in self._profiles
            if not any(reference == name for _, reference in profile_pairs)
        )
        self._profile_resolver = AliasResolver(
            aliases=profile_pairs, default_reference=default_profile
        )
        self._bindings = self._normalise_bindings(calendar_bindings)

    def list_calendar_events(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """List safe event summaries for one exactly resolved local calendar."""
        profile = self._resolve_profile(arguments.get("profile_name"))
        if isinstance(profile, ToolResult):
            return profile
        binding = self._resolve_calendar(
            profile=profile, spoken=arguments.get("calendar_name")
        )
        if isinstance(binding, ToolResult):
            return binding
        try:
            starts_at, ends_at = _parse_interval(arguments)
            query = arguments.get("query")
            max_results = arguments.get("max_results")
            if query is not None and not isinstance(query, str):
                raise ValueError("query must be a string or null")
            if not isinstance(max_results, int) or isinstance(max_results, bool):
                raise ValueError("max_results must be an integer")
            events = self.provider.list_events(
                credential_ref=self._credential_ref(profile=profile, binding=binding),
                calendar_id=binding.calendar_id,
                starts_at=starts_at,
                ends_at=ends_at,
                query=query,
                max_results=max_results,
                context=context,
            )
        except ValueError:
            return _result(
                ToolStatus.INVALID_REQUEST, "Ugyldigt kalenderinterval eller antal."
            )
        except GoogleCalendarError as error:
            return _provider_result(error)
        return ToolResult(
            status=ToolStatus.OK,
            data={
                "events": [
                    dataclasses.replace(event, calendar_name=binding.alias).as_dict()
                    for event in events
                ]
            },
        )

    def get_calendar_availability(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """Return busy intervals for one or more exactly resolved local profiles."""
        names = arguments.get("profile_names")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) for name in names)
        ):
            return _result(ToolStatus.INVALID_REQUEST, "Mindst én person skal angives.")
        profiles: list[str] = []
        for name in names:
            profile = self._resolve_profile(name)
            if isinstance(profile, ToolResult):
                return profile
            if profile not in profiles:
                profiles.append(profile)
        try:
            starts_at, ends_at = _parse_interval(arguments)
            grouped: dict[str, list[tuple[str, list[str]]]] = {}
            for profile in profiles:
                bindings = self._bindings.get(profile, [])
                if not bindings:
                    return _result(ToolStatus.NOT_FOUND, "Kalenderen blev ikke fundet.")
                credential_ref = self._credential_ref(
                    profile=profile, binding=bindings[0]
                )
                grouped.setdefault(credential_ref, []).append(
                    (profile, [binding.calendar_id for binding in bindings])
                )
            by_profile: dict[str, list[BusyInterval]] = {
                profile: [] for profile in profiles
            }
            for credential_ref, profile_items in grouped.items():
                calendar_ids = list(
                    dict.fromkeys(
                        calendar_id for _, ids in profile_items for calendar_id in ids
                    )
                )
                busy = self.provider.query_freebusy(
                    credential_ref=credential_ref,
                    calendar_ids=calendar_ids,
                    starts_at=starts_at,
                    ends_at=ends_at,
                    context=context,
                )
                for profile, ids in profile_items:
                    for calendar_id in ids:
                        by_profile[profile].extend(busy.get(calendar_id, []))
        except ValueError:
            return _result(ToolStatus.INVALID_REQUEST, "Ugyldigt kalenderinterval.")
        except GoogleCalendarError as error:
            return _provider_result(error)
        return ToolResult(
            status=ToolStatus.OK,
            data={
                "profiles": [
                    {
                        "profile_name": profile,
                        "busy_intervals": [
                            interval.as_dict()
                            for interval in _merge_intervals(by_profile[profile])
                        ],
                    }
                    for profile in profiles
                ]
            },
        )

    def _resolve_profile(self, spoken: object) -> str | ToolResult:
        if spoken is not None and not isinstance(spoken, str):
            return _result(ToolStatus.INVALID_REQUEST, "Profilen er ugyldig.")
        result = self._profile_resolver.resolve(spoken)
        if result.status is ResolutionStatus.MATCHED and result.reference is not None:
            if result.reference not in self._profiles:
                return _result(
                    ToolStatus.UNAUTHENTICATED, "Kalenderen er ikke forbundet."
                )
            return result.reference
        if result.status is ResolutionStatus.NEEDS_CLARIFICATION:
            return _result(
                ToolStatus.NEEDS_CLARIFICATION,
                "Hvilken person mener du?",
                candidates=list(result.candidates),
            )
        return _result(ToolStatus.NOT_FOUND, "Personen blev ikke fundet.")

    def _resolve_calendar(
        self, *, profile: str, spoken: object
    ) -> CalendarBinding | ToolResult:
        if spoken is not None and not isinstance(spoken, str):
            return _result(ToolStatus.INVALID_REQUEST, "Kalenderen er ugyldig.")
        bindings = self._bindings.get(profile, [])
        if spoken is None:
            if len(bindings) == 1:
                return bindings[0]
            if not bindings:
                return _result(ToolStatus.NOT_FOUND, "Kalenderen blev ikke fundet.")
            return _result(
                ToolStatus.NEEDS_CLARIFICATION,
                "Hvilken kalender mener du?",
                candidates=[binding.alias for binding in bindings],
            )
        resolver = AliasResolver(
            aliases=(
                (binding.alias, str(index)) for index, binding in enumerate(bindings)
            )
        )
        result = resolver.resolve(spoken)
        if result.status is ResolutionStatus.MATCHED and result.reference is not None:
            return bindings[int(result.reference)]
        if result.status is ResolutionStatus.NEEDS_CLARIFICATION:
            return _result(
                ToolStatus.NEEDS_CLARIFICATION,
                "Hvilken kalender mener du?",
                candidates=[bindings[int(index)].alias for index in result.candidates],
            )
        return _result(ToolStatus.NOT_FOUND, "Kalenderen blev ikke fundet.")

    def _credential_ref(self, *, profile: str, binding: CalendarBinding) -> str:
        reference = binding.credential_ref or self._profiles[profile]
        if not isinstance(reference, str) or not reference:
            raise ValueError("missing credential reference")
        if reference != self._profiles[profile]:
            raise ValueError("calendar binding belongs to another profile")
        return reference

    @staticmethod
    def _normalise_accounts(
        values: c.Mapping[str, str | ProviderAccount]
        | c.Iterable[CalendarProfileAccount],
    ) -> dict[str, str]:
        if isinstance(values, c.Mapping):
            result: dict[str, str] = {}
            for profile, account in values.items():
                if not isinstance(profile, str):
                    continue
                if isinstance(account, ProviderAccount):
                    result[profile] = account.credential_ref
                elif isinstance(account, str) and account:
                    result[profile] = account
            return result
        return {item.profile_name: item.credential_ref for item in values}

    @staticmethod
    def _normalise_bindings(
        values: c.Iterable[CalendarBinding]
        | c.Mapping[str | tuple[str, str], c.Mapping[str, str] | str],
    ) -> dict[str, list[CalendarBinding]]:
        if isinstance(values, c.Mapping):
            result: dict[str, list[CalendarBinding]] = {}
            for key, calendars in values.items():
                if isinstance(key, tuple) and len(key) == 2:
                    profile, alias = key
                    if (
                        isinstance(profile, str)
                        and isinstance(alias, str)
                        and isinstance(calendars, str)
                    ):
                        result.setdefault(profile, []).append(
                            CalendarBinding(profile, alias, calendars)
                        )
                    continue
                if not isinstance(key, str) or not isinstance(calendars, c.Mapping):
                    continue
                result[key] = [
                    CalendarBinding(key, alias, calendar_id)
                    for alias, calendar_id in calendars.items()
                    if isinstance(alias, str) and isinstance(calendar_id, str)
                ]
            return result
        result = {}
        for binding in values:
            result.setdefault(binding.profile_name, []).append(binding)
        return result


def make_calendar_tool_specs(adapter: CalendarToolAdapter) -> tuple[ToolSpec, ToolSpec]:
    """Create strict model-visible specs for the two read-only calendar tools."""
    return (
        ToolSpec(
            name="list_calendar_events",
            description="List chronological calendar events.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "profile_name": {"type": ["string", "null"]},
                    "calendar_name": {"type": ["string", "null"]},
                    "starts_at": {"type": "string", "format": "date-time"},
                    "ends_at": {"type": "string", "format": "date-time"},
                    "query": {"type": ["string", "null"]},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": [
                    "profile_name",
                    "calendar_name",
                    "starts_at",
                    "ends_at",
                    "query",
                    "max_results",
                ],
            },
            handler=adapter.list_calendar_events,
        ),
        ToolSpec(
            name="get_calendar_availability",
            description="Get busy calendar intervals without event details.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "profile_names": {"type": "array", "items": {"type": "string"}},
                    "starts_at": {"type": "string", "format": "date-time"},
                    "ends_at": {"type": "string", "format": "date-time"},
                },
                "required": ["profile_names", "starts_at", "ends_at"],
            },
            handler=adapter.get_calendar_availability,
        ),
    )


# Explicit aliases make final registry assembly discoverable without changing the
# shared registry module.
GoogleCalendarTools = CalendarToolAdapter


def build_calendar_tools(
    provider: GoogleCalendarProvider,
    credential_store: CredentialStore,
    *,
    profile_aliases: c.Mapping[str, str] | c.Iterable[tuple[str, str]] = (),
    calendar_bindings: c.Iterable[CalendarBinding]
    | c.Mapping[str | tuple[str, str], c.Mapping[str, str] | str] = (),
    profile_accounts: c.Mapping[str, str | ProviderAccount]
    | c.Iterable[CalendarProfileAccount] = (),
    default_profile: str | None = None,
) -> CalendarToolAdapter:
    """Build a calendar adapter from setup-owned routing inputs."""
    return CalendarToolAdapter(
        provider,
        credential_store,
        profile_aliases=profile_aliases,
        calendar_bindings=calendar_bindings,
        profile_accounts=profile_accounts,
        default_profile=default_profile,
    )


def list_calendar_events_spec(adapter: CalendarToolAdapter) -> ToolSpec:
    """Create the strict event-list tool specification."""
    return make_calendar_tool_specs(adapter)[0]


def get_calendar_availability_spec(adapter: CalendarToolAdapter) -> ToolSpec:
    """Create the strict availability tool specification."""
    return make_calendar_tool_specs(adapter)[1]


calendar_tool_specs = make_calendar_tool_specs
create_calendar_tool_specs = make_calendar_tool_specs
build_calendar_tool_specs = make_calendar_tool_specs


def _parse_interval(arguments: dict[str, object]) -> tuple[dt.datetime, dt.datetime]:
    starts_raw, ends_raw = arguments.get("starts_at"), arguments.get("ends_at")
    if not isinstance(starts_raw, str) or not isinstance(ends_raw, str):
        raise ValueError("timestamps must be strings")
    try:
        starts_at = dt.datetime.fromisoformat(starts_raw.replace("Z", "+00:00"))
        ends_at = dt.datetime.fromisoformat(ends_raw.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("invalid timestamp") from None
    if starts_at.tzinfo is None or ends_at.tzinfo is None:
        raise ValueError("timestamps must include an offset")
    if ends_at <= starts_at or ends_at - starts_at > dt.timedelta(days=31):
        raise ValueError("interval is outside bounds")
    return starts_at, ends_at


def _provider_result(error: GoogleCalendarError) -> ToolResult:
    if isinstance(error, GoogleCalendarUnauthenticated):
        return _result(ToolStatus.UNAUTHENTICATED, "Kalenderen er ikke forbundet.")
    if isinstance(error, GoogleCalendarForbidden):
        return _result(ToolStatus.FORBIDDEN, "Der er ikke adgang til kalenderen.")
    if isinstance(error, GoogleCalendarRateLimited):
        return _result(
            ToolStatus.RATE_LIMITED,
            "Kalenderen har bedt om færre forespørgsler.",
            retryable=True,
        )
    if isinstance(error, GoogleCalendarUnavailable):
        return _result(
            ToolStatus.UNAVAILABLE,
            "Kalenderen er midlertidigt utilgængelig.",
            retryable=True,
        )
    return _result(
        ToolStatus.UNAVAILABLE, "Kalenderen kunne ikke læses.", retryable=False
    )


def _result(
    status: ToolStatus,
    message: str,
    *,
    candidates: list[object] | None = None,
    retryable: bool = False,
) -> ToolResult:
    return ToolResult(
        status=status,
        message_da=message,
        candidates=candidates or [],
        retryable=retryable,
    )


def _merge_intervals(intervals: list[BusyInterval]) -> list[BusyInterval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item.starts_at)
    merged: list[BusyInterval] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.starts_at <= previous.ends_at:
            merged[-1] = BusyInterval(
                starts_at=previous.starts_at,
                ends_at=max(previous.ends_at, current.ends_at),
            )
        else:
            merged.append(current)
    return merged


__all__ = [
    "CalendarBinding",
    "CalendarProfileAccount",
    "CalendarToolAdapter",
    "GoogleCalendarTools",
    "build_calendar_tool_specs",
    "build_calendar_tools",
    "create_calendar_tool_specs",
    "get_calendar_availability_spec",
    "list_calendar_events_spec",
    "calendar_tool_specs",
    "make_calendar_tool_specs",
]
