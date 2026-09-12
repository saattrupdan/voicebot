"""Typed domain models used by the local SQLite storage layer."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import unicodedata
import uuid
from collections.abc import Mapping

UTC = dt.UTC


@dataclasses.dataclass(frozen=True, slots=True)
class Profile:
    """A local person or household account grouping."""

    id: str
    name: str
    created_at: dt.datetime
    updated_at: dt.datetime


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileAlias:
    """A normalised spoken alias belonging to a profile."""

    id: str
    profile_id: str
    alias: str
    normalised_alias: str
    created_at: dt.datetime


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderAccount:
    """Provider routing metadata; credentials themselves are never stored here."""

    id: str
    profile_id: str
    provider: str
    credential_ref: str
    scopes: tuple[str, ...]
    status: str
    expires_at: dt.datetime | None
    metadata: dict[str, object]
    created_at: dt.datetime
    updated_at: dt.datetime


@dataclasses.dataclass(frozen=True, slots=True)
class Binding:
    """A local alias bound to an object at a provider."""

    id: str
    profile_id: str
    provider_account_id: str | None
    alias: str
    normalised_alias: str
    provider_id: str
    created_at: dt.datetime
    expires_at: dt.datetime | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Reminder:
    """A durable reminder scheduled in UTC."""

    id: str
    profile_id: str
    name: str | None
    message: str
    due_at: dt.datetime
    original_timezone: str
    status: str
    deduplication_key: str | None
    created_at: dt.datetime
    updated_at: dt.datetime
    delivery_attempts: int
    claimed_at: dt.datetime | None = None
    delivered_at: dt.datetime | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Notification:
    """An item in the durable notification queue."""

    id: str
    profile_id: str
    kind: str
    payload: dict[str, object]
    available_at: dt.datetime
    status: str
    attempts: int
    claim_token: str | None
    claimed_at: dt.datetime | None
    lease_until: dt.datetime | None
    created_at: dt.datetime
    acknowledged_at: dt.datetime | None
    reminder_id: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Operation:
    """An idempotent operation and its provider-independent outcome."""

    id: str
    idempotency_key: str
    profile_id: str | None
    operation_type: str
    provider: str | None
    status: str
    request: dict[str, object]
    result: dict[str, object] | None
    error: str | None
    created_at: dt.datetime
    updated_at: dt.datetime


@dataclasses.dataclass(frozen=True, slots=True)
class PendingConfirmation:
    """A confirmation which can be resumed after a process restart."""

    id: str
    operation_id: str | None
    profile_id: str
    action: str
    payload: dict[str, object]
    expires_at: dt.datetime
    status: str
    created_at: dt.datetime
    resolved_at: dt.datetime | None


@dataclasses.dataclass(frozen=True, slots=True)
class SchedulerLease:
    """A renewable single-owner scheduler lease."""

    name: str
    owner: str
    acquired_at: dt.datetime
    expires_at: dt.datetime
    updated_at: dt.datetime


@dataclasses.dataclass(frozen=True, slots=True)
class AuditEvent:
    """A redacted, append-only operational event."""

    id: str
    event_type: str
    operation_id: str | None
    profile_id: str | None
    details: dict[str, object]
    created_at: dt.datetime


def new_id() -> str:
    """Return a compact, stable identifier for a local record."""
    return uuid.uuid4().hex


def utc_now() -> dt.datetime:
    """Return the current timezone-aware UTC time."""
    return dt.datetime.now(tz=UTC)


def to_utc(value: dt.datetime) -> dt.datetime:
    """Convert an aware datetime to UTC.

    Raises:
        ValueError:
            If ``value`` does not contain timezone information.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def timestamp(value: dt.datetime) -> str:
    """Serialise an aware datetime as an unambiguous UTC SQLite value."""
    return to_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_timestamp(value: str) -> dt.datetime:
    """Parse a UTC SQLite timestamp."""
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return to_utc(parsed)


def normalise_alias(value: str) -> str:
    """Normalise a spoken alias for exact, deterministic matching."""
    normalised = unicodedata.normalize("NFKC", value).casefold().strip()
    normalised = re.sub(r"^[\W_]+|[\W_]+$", "", normalised, flags=re.UNICODE)
    return " ".join(normalised.split())


def json_object(value: Mapping[str, object]) -> str:
    """Encode a JSON object for SQLite."""
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)


def parse_json_object(value: str) -> dict[str, object]:
    """Decode a JSON object from SQLite."""
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("stored JSON value is not an object")
    return {str(key): item for key, item in decoded.items()}
