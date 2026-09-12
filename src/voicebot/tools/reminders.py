"""Persistent reminder tools backed by the durable storage repositories."""

from __future__ import annotations

import datetime as dt
import logging
import typing as t
from collections.abc import Callable
from zoneinfo import ZoneInfo

from ..storage import Storage
from ..storage.models import Profile, Reminder, timestamp, to_utc, utc_now
from ..tool_runtime import ToolContext, ToolHandler, ToolResult, ToolSpec, ToolStatus

logger = logging.getLogger(__name__)
Clock = Callable[[], dt.datetime]


def create_reminder(context: ToolContext, arguments: dict[str, object]) -> ToolResult:
    """Create a reminder using storage and the clock in ``context.state``."""
    return _create_handler(
        _storage_from_context(context), _clock_from_context(context)
    )(context, arguments)


def list_reminders(context: ToolContext, arguments: dict[str, object]) -> ToolResult:
    """List reminders belonging to the active profile."""
    return _list_handler(_storage_from_context(context), _clock_from_context(context))(
        context, arguments
    )


def cancel_reminder(context: ToolContext, arguments: dict[str, object]) -> ToolResult:
    """Cancel exactly one reminder by its ID or exact name."""
    return _cancel_handler(
        _storage_from_context(context), _clock_from_context(context)
    )(context, arguments)


def create_reminder_spec(storage: Storage, clock: Clock = utc_now) -> ToolSpec:
    """Build the model-visible create-reminder tool specification."""
    return ToolSpec(
        name="create_reminder",
        description="Create one persistent reminder.",
        parameters=_create_schema(),
        handler=_create_handler(storage, clock),
        mutates=True,
    )


def list_reminders_spec(storage: Storage, clock: Clock = utc_now) -> ToolSpec:
    """Build the model-visible list-reminders tool specification."""
    return ToolSpec(
        name="list_reminders",
        description="List persistent reminders in a bounded interval.",
        parameters=_list_schema(),
        handler=_list_handler(storage, clock),
    )


def cancel_reminder_spec(storage: Storage, clock: Clock = utc_now) -> ToolSpec:
    """Build the model-visible cancel-reminder tool specification."""
    return ToolSpec(
        name="cancel_reminder",
        description="Cancel one reminder by exact name or reference.",
        parameters=_cancel_schema(),
        handler=_cancel_handler(storage, clock),
        mutates=True,
    )


# These aliases make the factories easy to discover when the final registry is wired.
create_reminder_tool_spec = create_reminder_spec
list_reminders_tool_spec = list_reminders_spec
cancel_reminder_tool_spec = cancel_reminder_spec


def _create_handler(storage: Storage, clock: Clock) -> ToolHandler:
    def handler(context: ToolContext, arguments: dict[str, object]) -> ToolResult:
        context.raise_if_cancelled()
        profile = _profile(storage, context, arguments.get("profile_name"))
        if isinstance(profile, ToolResult):
            return profile
        now = _now(clock)
        try:
            due_at, original_timezone = _resolve_due_time(arguments, now)
        except ValueError as error:
            return _invalid(context, str(error))
        name = arguments.get("name")
        message = arguments.get("message")
        if not isinstance(message, str) or not message.strip():
            return _invalid(context, "message must not be empty")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            return _invalid(context, "name must not be empty")
        reminder = storage.reminders.create(
            profile.id,
            message.strip(),
            due_at,
            name=name.strip() if isinstance(name, str) else None,
            original_timezone=original_timezone,
            deduplication_key=(
                f"tool:{context.operation_id}" if context.operation_id else None
            ),
            now=now,
        )
        return ToolResult(
            status=ToolStatus.OK,
            operation_id=context.operation_id,
            message_da="Påmindelsen er oprettet.",
            data=_reminder_data(reminder),
        )

    return handler


def _list_handler(storage: Storage, clock: Clock) -> ToolHandler:
    def handler(context: ToolContext, arguments: dict[str, object]) -> ToolResult:
        context.raise_if_cancelled()
        profile = _profile(storage, context, arguments.get("profile_name"))
        if isinstance(profile, ToolResult):
            return profile
        try:
            starts_at = _optional_datetime(arguments.get("starts_at"))
            ends_at = _optional_datetime(arguments.get("ends_at"))
        except ValueError as error:
            return _invalid(context, str(error))
        if starts_at is not None and ends_at is not None and starts_at > ends_at:
            return _invalid(context, "starts_at must not be after ends_at")
        status = arguments.get("status", "all")
        if status not in {"pending", "delivered", "missed", "all"}:
            return _invalid(context, "status is not supported")
        storage_status = (
            t.cast(str, status) if status in {"delivered", "missed"} else None
        )
        reminders = storage.reminders.list(profile_id=profile.id, status=storage_status)
        reminders = [
            reminder
            for reminder in reminders
            if (starts_at is None or reminder.due_at >= starts_at)
            and (ends_at is None or reminder.due_at <= ends_at)
            and (status == "all" or _visible_status(reminder.status) == status)
        ]
        return ToolResult(
            status=ToolStatus.OK,
            operation_id=context.operation_id,
            data={"reminders": [_reminder_data(item) for item in reminders]},
        )

    return handler


def _cancel_handler(storage: Storage, clock: Clock) -> ToolHandler:
    def handler(context: ToolContext, arguments: dict[str, object]) -> ToolResult:
        context.raise_if_cancelled()
        profile = _profile(storage, context, arguments.get("profile_name"))
        if isinstance(profile, ToolResult):
            return profile
        reference = arguments.get("reminder_name")
        if not isinstance(reference, str) or not reference.strip():
            return _invalid(context, "reminder_name must not be empty")
        reference = reference.strip()
        candidates = [
            item
            for item in storage.reminders.list(profile_id=profile.id)
            if item.id == reference or item.name == reference
        ]
        if not candidates:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                operation_id=context.operation_id,
                message_da="Jeg fandt ikke den påmindelse.",
            )
        if len(candidates) != 1:
            return ToolResult(
                status=ToolStatus.CONFLICT,
                operation_id=context.operation_id,
                message_da="Der er flere påmindelser med det navn.",
                candidates=[_reminder_data(item) for item in candidates],
            )
        reminder = candidates[0]
        if reminder.status not in {"scheduled", "queued"}:
            return ToolResult(
                status=ToolStatus.CONFLICT,
                operation_id=context.operation_id,
                message_da="Påmindelsen kan ikke længere annulleres.",
                data=_reminder_data(reminder),
            )
        if not storage.reminders.cancel(reminder.id, now=_now(clock)):
            return ToolResult(
                status=ToolStatus.CONFLICT,
                operation_id=context.operation_id,
                message_da="Påmindelsen blev ændret af en anden proces.",
                retryable=True,
            )
        cancelled = storage.reminders.get(reminder.id) or reminder
        return ToolResult(
            status=ToolStatus.OK,
            operation_id=context.operation_id,
            message_da="Påmindelsen er annulleret.",
            data=_reminder_data(cancelled),
        )

    return handler


def _resolve_due_time(
    arguments: dict[str, object], now: dt.datetime
) -> tuple[dt.datetime, str]:
    delay = arguments.get("delay_seconds")
    due_value = arguments.get("due_at")
    has_delay = delay is not None
    has_due = due_value is not None
    if has_delay == has_due:
        raise ValueError("exactly one of delay_seconds and due_at is required")
    if has_delay:
        if not isinstance(delay, int) or isinstance(delay, bool) or delay <= 0:
            raise ValueError("delay_seconds must be a positive integer")
        due = now + dt.timedelta(seconds=delay)
        return to_utc(due), _timezone_name(now.tzinfo)
    if not isinstance(due_value, (str, dt.datetime)):
        raise ValueError("due_at must be an RFC3339 timestamp")
    due = _parse_datetime(due_value)
    _validate_dst(due)
    if due <= now:
        raise ValueError("due_at must be in the future")
    return to_utc(due), _timezone_name(due.tzinfo)


def _parse_datetime(value: str | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        result = value
    else:
        try:
            result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("due_at must be a valid RFC3339 timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamps must include a timezone offset")
    return result


def _optional_datetime(value: object) -> dt.datetime | None:
    if value is None:
        return None
    if not isinstance(value, (str, dt.datetime)):
        raise ValueError("interval bounds must be RFC3339 timestamps")
    return to_utc(_parse_datetime(value))


def _validate_dst(value: dt.datetime) -> None:
    """Reject nonexistent or ambiguous ZoneInfo local times."""
    timezone = value.tzinfo
    if not isinstance(timezone, ZoneInfo):
        return
    naive = value.replace(tzinfo=None)
    valid_folds = {
        candidate.utcoffset()
        for fold in (0, 1)
        if (candidate := value.replace(fold=fold))
        .astimezone(dt.UTC)
        .astimezone(timezone)
        .replace(tzinfo=None)
        == naive
    }
    if not valid_folds:
        raise ValueError("due_at falls in a daylight-saving transition gap")
    if len(valid_folds) > 1:
        raise ValueError("due_at is ambiguous during a daylight-saving transition")


def _profile(
    storage: Storage, context: ToolContext, profile_name: object
) -> Profile | ToolResult:
    profile_id = context.state.get("profile_id")
    if profile_name is not None:
        if not isinstance(profile_name, str) or not profile_name.strip():
            return _invalid(context, "profile_name must not be empty")
        wanted = profile_name.strip().casefold()
        matches = [
            profile
            for profile in storage.profiles.list()
            if profile.name.casefold() == wanted
        ]
        matches += storage.profiles.resolve_alias(profile_name)
        unique = {profile.id: profile for profile in matches}
        if not unique:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                operation_id=context.operation_id,
                message_da="Jeg fandt ikke profilen.",
            )
        if len(unique) > 1:
            return ToolResult(
                status=ToolStatus.CONFLICT,
                operation_id=context.operation_id,
                message_da="Profilen er ikke entydig.",
                candidates=[
                    {"id": item.id, "name": item.name} for item in unique.values()
                ],
            )
        return next(iter(unique.values()))
    if isinstance(profile_id, str):
        profile = storage.profiles.get(profile_id)
        if profile is not None:
            return profile
    profiles = storage.profiles.list()
    if len(profiles) == 1:
        return profiles[0]
    return ToolResult(
        status=ToolStatus.NEEDS_CLARIFICATION,
        operation_id=context.operation_id,
        message_da="Hvilken profil skal påmindelsen gælde?",
    )


def _storage_from_context(context: ToolContext) -> Storage:
    storage = context.state.get("storage")
    if not isinstance(storage, Storage):
        raise ValueError(
            "reminder tools require Storage in ToolContext.state['storage']"
        )
    return storage


def _clock_from_context(context: ToolContext) -> Clock:
    clock = context.state.get("clock")
    return t.cast(Clock, clock) if callable(clock) else utc_now


def _now(clock: Clock) -> dt.datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("the injected clock must return an aware datetime")
    return value


def _timezone_name(value: dt.tzinfo | None) -> str:
    if value is None:
        return "UTC"
    key = getattr(value, "key", None)
    if isinstance(key, str):
        return key
    offset = value.utcoffset(None)
    if offset is None or offset == dt.timedelta():
        return "UTC"
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def _visible_status(status: str) -> str:
    return "pending" if status in {"scheduled", "queued"} else status


def _reminder_data(reminder: Reminder) -> dict[str, object]:
    return {
        "id": reminder.id,
        "profile_id": reminder.profile_id,
        "name": reminder.name,
        "message": reminder.message,
        "due_at": timestamp(reminder.due_at),
        "original_timezone": reminder.original_timezone,
        "status": _visible_status(reminder.status),
    }


def _invalid(context: ToolContext, message: str) -> ToolResult:
    return ToolResult(
        status=ToolStatus.INVALID_REQUEST,
        operation_id=context.operation_id,
        message_da=message,
    )


def _create_schema() -> dict[str, object]:
    nullable_string = {"type": ["string", "null"]}
    nullable_datetime = {"type": ["string", "null"], "format": "date-time"}
    nullable_delay = {"type": ["integer", "null"], "minimum": 1}
    return {
        "type": "object",
        "properties": {
            "profile_name": nullable_string,
            "name": nullable_string,
            "message": {"type": "string", "minLength": 1},
            "delay_seconds": nullable_delay,
            "due_at": nullable_datetime,
        },
        "required": ["profile_name", "name", "message", "delay_seconds", "due_at"],
        "additionalProperties": False,
    }


def _list_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "profile_name": {"type": ["string", "null"]},
            "starts_at": {"type": ["string", "null"], "format": "date-time"},
            "ends_at": {"type": ["string", "null"], "format": "date-time"},
            "status": {
                "type": "string",
                "enum": ["pending", "delivered", "missed", "all"],
            },
        },
        "required": ["profile_name", "starts_at", "ends_at", "status"],
        "additionalProperties": False,
    }


def _cancel_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "profile_name": {"type": ["string", "null"]},
            "reminder_name": {"type": "string", "minLength": 1},
        },
        "required": ["profile_name", "reminder_name"],
        "additionalProperties": False,
    }


__all__ = [
    "cancel_reminder",
    "cancel_reminder_spec",
    "cancel_reminder_tool_spec",
    "create_reminder",
    "create_reminder_spec",
    "create_reminder_tool_spec",
    "list_reminders",
    "list_reminders_spec",
    "list_reminders_tool_spec",
]
