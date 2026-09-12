"""Repositories for durable voicebot state.

The repositories deliberately contain no provider or speech concerns. They provide small
transactional operations that runtime workers can safely call after a restart.
"""

from __future__ import annotations

import builtins as b
import datetime as dt
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence

from .database import Database
from .models import (
    AuditEvent,
    Binding,
    Notification,
    Operation,
    OperationAcquisition,
    PendingConfirmation,
    Profile,
    ProfileAlias,
    ProviderAccount,
    Reminder,
    SchedulerLease,
    json_object,
    new_id,
    normalise_alias,
    parse_json_object,
    parse_timestamp,
    timestamp,
    to_utc,
    utc_now,
)


class ProfileRepository:
    """Persist profiles and their exact spoken aliases."""

    def __init__(self, database: Database) -> None:
        """Create a repository backed by ``database``."""
        self.database = database

    def create(
        self,
        name: str,
        *,
        profile_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> Profile:
        """Create a profile."""
        current = to_utc(now or utc_now())
        profile = Profile(
            id=profile_id or new_id(), name=name, created_at=current, updated_at=current
        )
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO profiles(id, name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (profile.id, profile.name, timestamp(current), timestamp(current)),
            )
        return profile

    def get(self, profile_id: str) -> Profile | None:
        """Return a profile by ID, or ``None`` when it does not exist."""
        row = self.database.query_one(
            "SELECT * FROM profiles WHERE id = ?", (profile_id,)
        )
        return _profile(row) if row else None

    def list(self) -> b.list[Profile]:
        """Return profiles in stable creation order."""
        rows = self.database.query("SELECT * FROM profiles ORDER BY created_at, id")
        return [_profile(row) for row in rows]

    def rename(
        self, profile_id: str, name: str, *, now: dt.datetime | None = None
    ) -> Profile:
        """Rename an existing profile."""
        current = to_utc(now or utc_now())
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "UPDATE profiles SET name = ?, updated_at = ? WHERE id = ?",
                (name, timestamp(current), profile_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown profile: {profile_id}")
        result = self.get(profile_id)
        assert result is not None
        return result

    def add_alias(
        self,
        profile_id: str,
        alias: str,
        *,
        alias_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> ProfileAlias:
        """Add a globally unique, exact-match alias to a profile."""
        normalised = normalise_alias(alias)
        if not normalised:
            raise ValueError("alias must contain at least one alphanumeric character")
        current = to_utc(now or utc_now())
        item = ProfileAlias(
            id=alias_id or new_id(),
            profile_id=profile_id,
            alias=alias.strip(),
            normalised_alias=normalised,
            created_at=current,
        )
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO profile_aliases "
                "(id, profile_id, alias, normalised_alias, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    item.id,
                    item.profile_id,
                    item.alias,
                    item.normalised_alias,
                    timestamp(current),
                ),
            )
        return item

    def remove_alias(self, alias: str) -> bool:
        """Remove an alias and report whether a row was removed."""
        normalised = normalise_alias(alias)
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "DELETE FROM profile_aliases WHERE normalised_alias = ?", (normalised,)
            )
        return cursor.rowcount == 1

    def find_aliases(self, alias: str) -> b.list[ProfileAlias]:
        """Return all exact matches for an alias (normally zero or one)."""
        normalised = normalise_alias(alias)
        rows = self.database.query(
            "SELECT * FROM profile_aliases WHERE normalised_alias = ?", (normalised,)
        )
        return [_profile_alias(row) for row in rows]

    def resolve_alias(self, alias: str) -> b.list[Profile]:
        """Resolve an alias without fuzzy matching."""
        rows = self.database.query(
            "SELECT p.* FROM profiles AS p JOIN profile_aliases AS a "
            "ON a.profile_id = p.id WHERE a.normalised_alias = ? "
            "ORDER BY p.created_at, p.id",
            (normalise_alias(alias),),
        )
        return [_profile(row) for row in rows]


class ProviderRepository:
    """Persist provider account metadata and local object bindings."""

    _binding_tables = {
        "calendar": ("calendar_bindings", "calendar_id"),
        "spotify_device": ("spotify_device_aliases", "device_id"),
        "shopping_list": ("shopping_list_bindings", "list_id"),
    }

    def __init__(self, database: Database) -> None:
        """Create a repository backed by ``database``."""
        self.database = database

    def upsert_account(
        self,
        profile_id: str,
        provider: str,
        credential_ref: str,
        *,
        scopes: Sequence[str] = (),
        status: str = "connected",
        expires_at: dt.datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        account_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> ProviderAccount:
        """Create or update the metadata for one profile/provider account."""
        current = to_utc(now or utc_now())
        expires = to_utc(expires_at) if expires_at else None
        with self.database.transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT id, created_at FROM provider_accounts "
                "WHERE profile_id = ? AND provider = ?",
                (profile_id, provider),
            ).fetchone()
            identifier = str(existing["id"]) if existing else (account_id or new_id())
            created = str(existing["created_at"]) if existing else timestamp(current)
            connection.execute(
                "INSERT INTO provider_accounts "
                "(id, profile_id, provider, credential_ref, scopes_json, status, "
                "expires_at, metadata_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(profile_id, provider) DO UPDATE SET "
                "credential_ref=excluded.credential_ref, "
                "scopes_json=excluded.scopes_json, "
                "status=excluded.status, expires_at=excluded.expires_at, "
                "metadata_json=excluded.metadata_json, updated_at=excluded.updated_at",
                (
                    identifier,
                    profile_id,
                    provider,
                    credential_ref,
                    json.dumps(list(scopes), sort_keys=True),
                    status,
                    timestamp(expires) if expires else None,
                    json_object(metadata or {}),
                    created,
                    timestamp(current),
                ),
            )
        result = self.get_account(profile_id=profile_id, provider=provider)
        assert result is not None
        return result

    def get_account(self, *, profile_id: str, provider: str) -> ProviderAccount | None:
        """Return one provider account metadata record."""
        row = self.database.query_one(
            "SELECT * FROM provider_accounts WHERE profile_id = ? AND provider = ?",
            (profile_id, provider),
        )
        return _provider_account(row) if row else None

    def list_accounts(self, profile_id: str | None = None) -> list[ProviderAccount]:
        """List provider metadata, optionally limited to a profile."""
        if profile_id is None:
            rows = self.database.query(
                "SELECT * FROM provider_accounts ORDER BY provider, id"
            )
        else:
            rows = self.database.query(
                "SELECT * FROM provider_accounts WHERE profile_id = ? "
                "ORDER BY provider, id",
                (profile_id,),
            )
        return [_provider_account(row) for row in rows]

    def set_account_status(
        self, account_id: str, status: str, *, now: dt.datetime | None = None
    ) -> ProviderAccount:
        """Update connection status without touching credentials."""
        current = to_utc(now or utc_now())
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "UPDATE provider_accounts SET status = ?, updated_at = ? WHERE id = ?",
                (status, timestamp(current), account_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown provider account: {account_id}")
        row = self.database.query_one(
            "SELECT * FROM provider_accounts WHERE id = ?", (account_id,)
        )
        assert row is not None
        return _provider_account(row)

    def bind(
        self,
        kind: str,
        profile_id: str,
        alias: str,
        provider_id: str,
        *,
        provider_account_id: str | None = None,
        expires_at: dt.datetime | None = None,
        binding_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> Binding:
        """Bind a normalised local alias to a provider object.

        ``kind`` is one of ``calendar``, ``spotify_device`` or ``shopping_list``.
        """
        if kind not in self._binding_tables:
            raise ValueError(f"unknown binding kind: {kind}")
        normalised = normalise_alias(alias)
        if not normalised:
            raise ValueError("alias must contain at least one alphanumeric character")
        current = to_utc(now or utc_now())
        table, id_column = self._binding_tables[kind]
        item = Binding(
            id=binding_id or new_id(),
            profile_id=profile_id,
            provider_account_id=provider_account_id,
            alias=alias.strip(),
            normalised_alias=normalised,
            provider_id=provider_id,
            created_at=current,
            expires_at=to_utc(expires_at) if expires_at else None,
        )
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                f"INSERT INTO {table} "
                "(id, profile_id, provider_account_id, alias, normalised_alias, "
                f"{id_column}, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.id,
                    item.profile_id,
                    item.provider_account_id,
                    item.alias,
                    item.normalised_alias,
                    item.provider_id,
                    timestamp(current),
                    timestamp(item.expires_at) if item.expires_at else None,
                ),
            )
        return item

    def find_bindings(
        self,
        kind: str,
        *,
        profile_id: str | None = None,
        alias: str | None = None,
        now: dt.datetime | None = None,
        include_expired: bool = False,
    ) -> list[Binding]:
        """Find exact local bindings, optionally by profile and alias.

        Expiring provider resolutions are excluded by default.
        """
        table, id_column = self._binding_table(kind)
        clauses: list[str] = []
        parameters: list[str] = []
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            parameters.append(timestamp(to_utc(now or utc_now())))
        if profile_id is not None:
            clauses.append("profile_id = ?")
            parameters.append(profile_id)
        if alias is not None:
            clauses.append("normalised_alias = ?")
            parameters.append(normalise_alias(alias))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.database.query(
            f"SELECT *, {id_column} AS provider_id FROM {table}{where} "
            "ORDER BY created_at, id",
            tuple(parameters),
        )
        return [_binding(row) for row in rows]

    def unbind(self, kind: str, alias: str, *, profile_id: str) -> bool:
        """Remove one local binding."""
        table, _ = self._binding_table(kind)
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                f"DELETE FROM {table} WHERE profile_id = ? AND normalised_alias = ?",
                (profile_id, normalise_alias(alias)),
            )
        return cursor.rowcount == 1

    def _binding_table(self, kind: str) -> tuple[str, str]:
        try:
            return self._binding_tables[kind]
        except KeyError as error:
            raise ValueError(f"unknown binding kind: {kind}") from error


class ReminderRepository:
    """Persist reminders and move due reminders onto the notification queue."""

    def __init__(self, database: Database) -> None:
        """Create a repository backed by ``database``."""
        self.database = database

    def create(
        self,
        profile_id: str,
        message: str,
        due_at: dt.datetime,
        *,
        name: str | None = None,
        original_timezone: str = "UTC",
        deduplication_key: str | None = None,
        reminder_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> Reminder:
        """Create a scheduled reminder idempotently.

        A deduplication key returns the existing reminder instead of creating a
        second one.
        """
        current = to_utc(now or utc_now())
        due = to_utc(due_at)
        identifier = reminder_id or new_id()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO reminders "
                "(id, profile_id, name, message, due_at, original_timezone, status, "
                "deduplication_key, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, "
                "'scheduled', ?, ?, ?)",
                (
                    identifier,
                    profile_id,
                    name,
                    message,
                    timestamp(due),
                    original_timezone,
                    deduplication_key,
                    timestamp(current),
                    timestamp(current),
                ),
            )
            if deduplication_key is not None:
                row = connection.execute(
                    "SELECT id FROM reminders WHERE deduplication_key = ?",
                    (deduplication_key,),
                ).fetchone()
                if row is not None:
                    identifier = str(row["id"])
        result = self.get(identifier)
        assert result is not None
        return result

    def get(self, reminder_id: str) -> Reminder | None:
        """Return a reminder by ID."""
        row = self.database.query_one(
            "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
        )
        return _reminder(row) if row else None

    def list(
        self, *, profile_id: str | None = None, status: str | None = None
    ) -> b.list[Reminder]:
        """List reminders in due-time order."""
        clauses: list[str] = []
        parameters: list[str] = []
        if profile_id is not None:
            clauses.append("profile_id = ?")
            parameters.append(profile_id)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.database.query(
            f"SELECT * FROM reminders{where} ORDER BY due_at, id", tuple(parameters)
        )
        return [_reminder(row) for row in rows]

    def cancel(self, reminder_id: str, *, now: dt.datetime | None = None) -> bool:
        """Cancel a reminder unless it has already been delivered."""
        current = to_utc(now or utc_now())
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "UPDATE reminders SET status = 'cancelled', updated_at = ? "
                "WHERE id = ? AND status IN ('scheduled', 'queued')",
                (timestamp(current), reminder_id),
            )
            connection.execute(
                "UPDATE notification_queue SET status = 'dismissed', "
                "acknowledged_at = ? WHERE reminder_id = ? "
                "AND status IN ('pending', 'processing')",
                (timestamp(current), reminder_id),
            )
        return cursor.rowcount == 1

    def claim_due(
        self,
        *,
        now: dt.datetime | None = None,
        late_window: dt.timedelta = dt.timedelta(minutes=15),
    ) -> b.list[Reminder]:
        """Claim due reminders and enqueue them atomically.

        Reminders older than ``late_window`` are marked missed so downtime does not
        unexpectedly speak stale notifications.
        """
        current = to_utc(now or utc_now())
        cutoff = current - late_window
        claimed: list[Reminder] = []
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE reminders SET status = 'missed', updated_at = ? "
                "WHERE status = 'scheduled' AND due_at < ?",
                (timestamp(current), timestamp(cutoff)),
            )
            rows = connection.execute(
                "SELECT * FROM reminders WHERE status = 'scheduled' "
                "AND due_at <= ? ORDER BY due_at, id",
                (timestamp(current),),
            ).fetchall()
            for row in rows:
                reminder = _reminder(row)
                connection.execute(
                    "UPDATE reminders SET status = 'queued', claimed_at = ?, "
                    "delivery_attempts = delivery_attempts + 1, updated_at = ? "
                    "WHERE id = ?",
                    (timestamp(current), timestamp(current), reminder.id),
                )
                payload: dict[str, object] = {
                    "message": reminder.message,
                    "name": reminder.name,
                    "due_at": timestamp(reminder.due_at),
                }
                connection.execute(
                    "INSERT INTO notification_queue "
                    "(id, reminder_id, profile_id, kind, payload_json, available_at, "
                    "status, created_at) VALUES (?, ?, ?, 'reminder', ?, ?, "
                    "'pending', ?)",
                    (
                        new_id(),
                        reminder.id,
                        reminder.profile_id,
                        json_object(payload),
                        timestamp(current),
                        timestamp(current),
                    ),
                )
                claimed.append(self.get(reminder.id) or reminder)
        return claimed


class NotificationQueueRepository:
    """Provide transactional claim, acknowledgement and crash recovery."""

    def __init__(self, database: Database) -> None:
        """Create a repository backed by ``database``."""
        self.database = database

    def enqueue(
        self,
        profile_id: str,
        payload: Mapping[str, object],
        *,
        kind: str = "notification",
        available_at: dt.datetime | None = None,
        reminder_id: str | None = None,
        notification_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> Notification:
        """Add a durable notification to the queue."""
        current = to_utc(now or utc_now())
        available = to_utc(available_at or current)
        item_id = notification_id or new_id()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO notification_queue "
                "(id, reminder_id, profile_id, kind, payload_json, available_at, "
                "status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    item_id,
                    reminder_id,
                    profile_id,
                    kind,
                    json_object(payload),
                    timestamp(available),
                    timestamp(current),
                ),
            )
        result = self.get(item_id)
        assert result is not None
        return result

    def get(self, notification_id: str) -> Notification | None:
        """Return a queue item by ID."""
        row = self.database.query_one(
            "SELECT * FROM notification_queue WHERE id = ?", (notification_id,)
        )
        return _notification(row) if row else None

    def recover_expired(self, *, now: dt.datetime | None = None) -> int:
        """Return processing items whose lease expired to the pending queue."""
        current = to_utc(now or utc_now())
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "UPDATE notification_queue SET status = 'pending', claim_token = NULL, "
                "claimed_at = NULL, lease_until = NULL "
                "WHERE status = 'processing' AND lease_until <= ?",
                (timestamp(current),),
            )
        return cursor.rowcount

    def claim(
        self,
        *,
        owner: str,
        limit: int = 10,
        lease_for: dt.timedelta = dt.timedelta(minutes=5),
        now: dt.datetime | None = None,
    ) -> b.list[Notification]:
        """Claim available queue items with an expiring worker lease."""
        if limit < 1:
            raise ValueError("limit must be positive")
        current = to_utc(now or utc_now())
        lease_until = current + lease_for
        result: list[Notification] = []
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE notification_queue SET status = 'pending', claim_token = NULL, "
                "claimed_at = NULL, lease_until = NULL "
                "WHERE status = 'processing' AND lease_until <= ?",
                (timestamp(current),),
            )
            rows = connection.execute(
                "SELECT * FROM notification_queue WHERE status = 'pending' "
                "AND available_at <= ? ORDER BY available_at, created_at, id LIMIT ?",
                (timestamp(current), limit),
            ).fetchall()
            for row in rows:
                token = f"{owner}:{new_id()}"
                connection.execute(
                    "UPDATE notification_queue SET status = 'processing', "
                    "attempts = attempts + 1, claim_token = ?, claimed_at = ?, "
                    "lease_until = ? WHERE id = ?",
                    (token, timestamp(current), timestamp(lease_until), row["id"]),
                )
                refreshed = connection.execute(
                    "SELECT * FROM notification_queue WHERE id = ?", (row["id"],)
                ).fetchone()
                assert refreshed is not None
                result.append(_notification(refreshed))
        return result

    def acknowledge(
        self,
        notification_id: str,
        *,
        claim_token: str | None = None,
        dismissed: bool = False,
        now: dt.datetime | None = None,
    ) -> bool:
        """Acknowledge or dismiss a claimed item, optionally checking its token."""
        current = to_utc(now or utc_now())
        status = "dismissed" if dismissed else "acknowledged"
        conditions = "id = ? AND status = 'processing'"
        parameters: list[str] = [notification_id]
        if claim_token is not None:
            conditions += " AND claim_token = ?"
            parameters.append(claim_token)
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                f"UPDATE notification_queue SET status = ?, acknowledged_at = ?, "
                f"claim_token = NULL, claimed_at = NULL, lease_until = NULL "
                f"WHERE {conditions}",
                [status, timestamp(current), *parameters],
            )
            if cursor.rowcount == 1:
                connection.execute(
                    "UPDATE reminders SET status = 'delivered', delivered_at = ?, "
                    "updated_at = ? WHERE id = (SELECT reminder_id "
                    "FROM notification_queue WHERE id = ?)",
                    (timestamp(current), timestamp(current), notification_id),
                )
        return cursor.rowcount == 1


class OperationRepository:
    """Track idempotent operations independently of provider implementations."""

    def __init__(self, database: Database) -> None:
        """Create a repository backed by ``database``."""
        self.database = database

    def acquire(
        self,
        idempotency_key: str,
        operation_type: str,
        *,
        profile_id: str | None = None,
        provider: str | None = None,
        request: Mapping[str, object] | None = None,
        operation_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> OperationAcquisition:
        """Atomically acquire an operation for at-most-once execution.

        The caller which inserts a new idempotency key owns execution. Every
        later caller receives the same operation with ``owned`` set to false;
        it must use the persisted result rather than invoke the provider.
        """
        current = to_utc(now or utc_now())
        identifier = operation_id or new_id()
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO operations "
                "(id, idempotency_key, profile_id, operation_type, provider, status, "
                "request_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'started', ?, ?, ?)",
                (
                    identifier,
                    idempotency_key,
                    profile_id,
                    operation_type,
                    provider,
                    json_object(_redacted_mapping(request or {})),
                    timestamp(current),
                    timestamp(current),
                ),
            )
            row = connection.execute(
                "SELECT * FROM operations WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            assert row is not None
            return OperationAcquisition(
                operation=_operation(row), owned=cursor.rowcount == 1
            )

    def start(
        self,
        idempotency_key: str,
        operation_type: str,
        *,
        profile_id: str | None = None,
        provider: str | None = None,
        request: Mapping[str, object] | None = None,
        operation_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> Operation:
        """Start an operation or return its existing idempotent record.

        Use :meth:`acquire` when the caller needs to know whether it owns
        execution. This method remains as a compatibility convenience for
        callers that only need the durable operation record.
        """
        return self.acquire(
            idempotency_key,
            operation_type,
            profile_id=profile_id,
            provider=provider,
            request=request,
            operation_id=operation_id,
            now=now,
        ).operation

    def get(self, operation_id: str) -> Operation | None:
        """Return an operation by ID."""
        row = self.database.query_one(
            "SELECT * FROM operations WHERE id = ?", (operation_id,)
        )
        return _operation(row) if row else None

    def get_by_key(self, idempotency_key: str) -> Operation | None:
        """Return an operation by its idempotency key."""
        row = self.database.query_one(
            "SELECT * FROM operations WHERE idempotency_key = ?", (idempotency_key,)
        )
        return _operation(row) if row else None

    def update(
        self,
        operation_id: str,
        status: str,
        *,
        result: Mapping[str, object] | None = None,
        error: str | None = None,
        now: dt.datetime | None = None,
    ) -> Operation:
        """Set an operation outcome.

        ``awaiting_confirmation`` was used by an earlier runtime and is
        accepted as a read/write compatibility alias for the canonical
        ``pending_confirmation`` status.
        """
        current = to_utc(now or utc_now())
        status = _normalise_operation_status(status)
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "UPDATE operations SET status = ?, result_json = ?, error = ?, "
                "updated_at = ? WHERE id = ?",
                (
                    status,
                    json_object(_redacted_mapping(result))
                    if result is not None
                    else None,
                    _redacted_text(error) if error is not None else None,
                    timestamp(current),
                    operation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown operation: {operation_id}")
        found = self.get(operation_id)
        assert found is not None
        return found

    def complete(
        self,
        operation_id: str,
        result: Mapping[str, object] | None = None,
        *,
        now: dt.datetime | None = None,
    ) -> Operation:
        """Mark an operation completed."""
        return self.update(operation_id, "completed", result=result, now=now)

    def fail(
        self, operation_id: str, error: str, *, now: dt.datetime | None = None
    ) -> Operation:
        """Mark an operation failed."""
        return self.update(operation_id, "failed", error=error, now=now)


class ConfirmationRepository:
    """Store confirmations until a runtime accepts or rejects them."""

    def __init__(self, database: Database) -> None:
        """Create a repository backed by ``database``."""
        self.database = database

    def create(
        self,
        profile_id: str,
        action: str,
        payload: Mapping[str, object],
        expires_at: dt.datetime,
        *,
        operation_id: str | None = None,
        confirmation_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> PendingConfirmation:
        """Create a pending confirmation."""
        current = to_utc(now or utc_now())
        expires = to_utc(expires_at)
        item = PendingConfirmation(
            id=confirmation_id or new_id(),
            operation_id=operation_id,
            profile_id=profile_id,
            action=action,
            payload=_redacted_mapping(payload),
            expires_at=expires,
            status="pending",
            created_at=current,
            resolved_at=None,
        )
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO pending_confirmations "
                "(id, operation_id, profile_id, action, payload_json, expires_at, "
                "status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    item.id,
                    item.operation_id,
                    item.profile_id,
                    item.action,
                    json_object(item.payload),
                    timestamp(expires),
                    timestamp(current),
                ),
            )
        return item

    def get(
        self, confirmation_id: str, *, now: dt.datetime | None = None
    ) -> PendingConfirmation | None:
        """Return a confirmation, expiring it first when necessary."""
        self.expire(now=now)
        row = self.database.query_one(
            "SELECT * FROM pending_confirmations WHERE id = ?", (confirmation_id,)
        )
        return _confirmation(row) if row else None

    def list(
        self,
        *,
        profile_id: str | None = None,
        status: str = "pending",
        now: dt.datetime | None = None,
    ) -> b.list[PendingConfirmation]:
        """List confirmations, expiring overdue pending records first."""
        self.expire(now=now)
        if profile_id is None:
            rows = self.database.query(
                "SELECT * FROM pending_confirmations WHERE status = ? "
                "ORDER BY created_at, id",
                (status,),
            )
        else:
            rows = self.database.query(
                "SELECT * FROM pending_confirmations WHERE profile_id = ? "
                "AND status = ? ORDER BY created_at, id",
                (profile_id, status),
            )
        return [_confirmation(row) for row in rows]

    def resolve(
        self, confirmation_id: str, *, accepted: bool, now: dt.datetime | None = None
    ) -> PendingConfirmation | None:
        """Atomically accept or reject a still-pending confirmation."""
        current = to_utc(now or utc_now())
        status = "confirmed" if accepted else "rejected"
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "UPDATE pending_confirmations SET status = ?, resolved_at = ? "
                "WHERE id = ? AND status = 'pending' AND expires_at > ?",
                (status, timestamp(current), confirmation_id, timestamp(current)),
            )
            if cursor.rowcount != 1:
                return None
        return self.get(confirmation_id, now=current)

    def expire(self, *, now: dt.datetime | None = None) -> int:
        """Mark all expired pending confirmations."""
        current = to_utc(now or utc_now())
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "UPDATE pending_confirmations SET status = 'expired', resolved_at = ? "
                "WHERE status = 'pending' AND expires_at <= ?",
                (timestamp(current), timestamp(current)),
            )
        return cursor.rowcount


class SchedulerLeaseRepository:
    """Coordinate one scheduler owner using a renewable database lease."""

    def __init__(self, database: Database) -> None:
        """Create a repository backed by ``database``."""
        self.database = database

    def acquire(
        self,
        name: str,
        owner: str,
        *,
        lease_for: dt.timedelta = dt.timedelta(seconds=30),
        now: dt.datetime | None = None,
    ) -> SchedulerLease | None:
        """Acquire a lease if it is free, expired, or already owned by ``owner``."""
        current = to_utc(now or utc_now())
        expires = current + lease_for
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM scheduler_leases WHERE name = ?", (name,)
            ).fetchone()
            if (
                row
                and row["owner"] != owner
                and parse_timestamp(row["expires_at"]) > current
            ):
                return None
            if row:
                connection.execute(
                    "UPDATE scheduler_leases SET owner = ?, acquired_at = ?, "
                    "expires_at = ?, updated_at = ? WHERE name = ?",
                    (
                        owner,
                        timestamp(current),
                        timestamp(expires),
                        timestamp(current),
                        name,
                    ),
                )
            else:
                connection.execute(
                    "INSERT INTO scheduler_leases "
                    "(name, owner, acquired_at, expires_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        name,
                        owner,
                        timestamp(current),
                        timestamp(expires),
                        timestamp(current),
                    ),
                )
        return self.get(name)

    def get(self, name: str) -> SchedulerLease | None:
        """Return a scheduler lease."""
        row = self.database.query_one(
            "SELECT * FROM scheduler_leases WHERE name = ?", (name,)
        )
        return _lease(row) if row else None

    def renew(
        self,
        name: str,
        owner: str,
        *,
        lease_for: dt.timedelta = dt.timedelta(seconds=30),
        now: dt.datetime | None = None,
    ) -> SchedulerLease | None:
        """Renew an unexpired lease owned by ``owner``."""
        current = to_utc(now or utc_now())
        expires = current + lease_for
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "UPDATE scheduler_leases SET expires_at = ?, updated_at = ? "
                "WHERE name = ? AND owner = ? AND expires_at > ?",
                (
                    timestamp(expires),
                    timestamp(current),
                    name,
                    owner,
                    timestamp(current),
                ),
            )
        return self.get(name) if cursor.rowcount == 1 else None

    def release(self, name: str, owner: str) -> bool:
        """Release a lease owned by ``owner``."""
        with self.database.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "DELETE FROM scheduler_leases WHERE name = ? AND owner = ?",
                (name, owner),
            )
        return cursor.rowcount == 1


class AuditRepository:
    """Append redacted audit events without retaining private provider payloads."""

    def __init__(self, database: Database) -> None:
        """Create a repository backed by ``database``."""
        self.database = database

    def record(
        self,
        event_type: str,
        details: Mapping[str, object],
        *,
        operation_id: str | None = None,
        profile_id: str | None = None,
        event_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> AuditEvent:
        """Redact sensitive fields and append an audit event."""
        current = to_utc(now or utc_now())
        redacted = redact(details)
        assert isinstance(redacted, dict)
        item = AuditEvent(
            id=event_id or new_id(),
            event_type=event_type,
            operation_id=operation_id,
            profile_id=profile_id,
            details=redacted,
            created_at=current,
        )
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO audit_events "
                "(id, event_type, operation_id, profile_id, details_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    item.id,
                    item.event_type,
                    item.operation_id,
                    item.profile_id,
                    json_object(item.details),
                    timestamp(current),
                ),
            )
        return item

    def list(self, *, limit: int = 100) -> b.list[AuditEvent]:
        """Return the newest audit events first."""
        rows = self.database.query(
            "SELECT * FROM audit_events ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [_audit_event(row) for row in rows]


_SENSITIVE_KEY = re.compile(
    r"authorization|cookie|password|secret|token|oauth|credential|email|account.?id|"
    r"body|error|request|response|payload",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_SECRET_TEXT = re.compile(
    r"(?i)(?:bearer|basic)\s+[^\s,;]+|"
    r"(?:access|refresh|api|oauth)[_-]?token[=: ]+[^\s,;]+|"
    r"password[=: ]+[^\s,;]+"
)


def _redacted_text(value: str) -> str:
    """Redact secrets embedded in free-form provider errors."""
    redacted = redact(value)
    assert isinstance(redacted, str)
    return redacted


def _redacted_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return a typed, recursively redacted JSON object."""
    redacted = redact(value)
    assert isinstance(redacted, dict)
    return redacted


def _normalise_operation_status(status: str) -> str:
    """Return the canonical spelling for an operation status."""
    return "pending_confirmation" if status == "awaiting_confirmation" else status


def redact(value: object, *, key: str = "") -> object:
    """Recursively redact credentials, provider payloads and email addresses."""
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(name): redact(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_TEXT.sub("[REDACTED]", _EMAIL.sub("[REDACTED_EMAIL]", value))
    return value


def _profile(row: sqlite3.Row) -> Profile:
    return Profile(
        id=str(row["id"]),
        name=str(row["name"]),
        created_at=parse_timestamp(str(row["created_at"])),
        updated_at=parse_timestamp(str(row["updated_at"])),
    )


def _profile_alias(row: sqlite3.Row) -> ProfileAlias:
    return ProfileAlias(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        alias=str(row["alias"]),
        normalised_alias=str(row["normalised_alias"]),
        created_at=parse_timestamp(str(row["created_at"])),
    )


def _provider_account(row: sqlite3.Row) -> ProviderAccount:
    scopes = json.loads(str(row["scopes_json"]))
    if not isinstance(scopes, list):
        raise ValueError("stored scopes are not a list")
    return ProviderAccount(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        provider=str(row["provider"]),
        credential_ref=str(row["credential_ref"]),
        scopes=tuple(str(scope) for scope in scopes),
        status=str(row["status"]),
        expires_at=(
            parse_timestamp(str(row["expires_at"])) if row["expires_at"] else None
        ),
        metadata=parse_json_object(str(row["metadata_json"])),
        created_at=parse_timestamp(str(row["created_at"])),
        updated_at=parse_timestamp(str(row["updated_at"])),
    )


def _binding(row: sqlite3.Row) -> Binding:
    return Binding(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        provider_account_id=(
            str(row["provider_account_id"]) if row["provider_account_id"] else None
        ),
        alias=str(row["alias"]),
        normalised_alias=str(row["normalised_alias"]),
        provider_id=str(row["provider_id"]),
        created_at=parse_timestamp(str(row["created_at"])),
        expires_at=(
            parse_timestamp(str(row["expires_at"])) if row["expires_at"] else None
        ),
    )


def _reminder(row: sqlite3.Row) -> Reminder:
    return Reminder(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        name=str(row["name"]) if row["name"] is not None else None,
        message=str(row["message"]),
        due_at=parse_timestamp(str(row["due_at"])),
        original_timezone=str(row["original_timezone"]),
        status=str(row["status"]),
        deduplication_key=(
            str(row["deduplication_key"]) if row["deduplication_key"] else None
        ),
        created_at=parse_timestamp(str(row["created_at"])),
        updated_at=parse_timestamp(str(row["updated_at"])),
        delivery_attempts=int(row["delivery_attempts"]),
        claimed_at=parse_timestamp(str(row["claimed_at"]))
        if row["claimed_at"]
        else None,
        delivered_at=(
            parse_timestamp(str(row["delivered_at"])) if row["delivered_at"] else None
        ),
    )


def _notification(row: sqlite3.Row) -> Notification:
    return Notification(
        id=str(row["id"]),
        reminder_id=str(row["reminder_id"]) if row["reminder_id"] else None,
        profile_id=str(row["profile_id"]),
        kind=str(row["kind"]),
        payload=parse_json_object(str(row["payload_json"])),
        available_at=parse_timestamp(str(row["available_at"])),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        claim_token=str(row["claim_token"]) if row["claim_token"] else None,
        claimed_at=parse_timestamp(str(row["claimed_at"]))
        if row["claimed_at"]
        else None,
        lease_until=parse_timestamp(str(row["lease_until"]))
        if row["lease_until"]
        else None,
        created_at=parse_timestamp(str(row["created_at"])),
        acknowledged_at=(
            parse_timestamp(str(row["acknowledged_at"]))
            if row["acknowledged_at"]
            else None
        ),
    )


def _operation(row: sqlite3.Row) -> Operation:
    return Operation(
        id=str(row["id"]),
        idempotency_key=str(row["idempotency_key"]),
        profile_id=str(row["profile_id"]) if row["profile_id"] else None,
        operation_type=str(row["operation_type"]),
        provider=str(row["provider"]) if row["provider"] else None,
        status=str(row["status"]),
        request=parse_json_object(str(row["request_json"])),
        result=(
            parse_json_object(str(row["result_json"])) if row["result_json"] else None
        ),
        error=str(row["error"]) if row["error"] else None,
        created_at=parse_timestamp(str(row["created_at"])),
        updated_at=parse_timestamp(str(row["updated_at"])),
    )


def _confirmation(row: sqlite3.Row) -> PendingConfirmation:
    return PendingConfirmation(
        id=str(row["id"]),
        operation_id=str(row["operation_id"]) if row["operation_id"] else None,
        profile_id=str(row["profile_id"]),
        action=str(row["action"]),
        payload=parse_json_object(str(row["payload_json"])),
        expires_at=parse_timestamp(str(row["expires_at"])),
        status=str(row["status"]),
        created_at=parse_timestamp(str(row["created_at"])),
        resolved_at=(
            parse_timestamp(str(row["resolved_at"])) if row["resolved_at"] else None
        ),
    )


def _lease(row: sqlite3.Row) -> SchedulerLease:
    return SchedulerLease(
        name=str(row["name"]),
        owner=str(row["owner"]),
        acquired_at=parse_timestamp(str(row["acquired_at"])),
        expires_at=parse_timestamp(str(row["expires_at"])),
        updated_at=parse_timestamp(str(row["updated_at"])),
    )


def _audit_event(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        id=str(row["id"]),
        event_type=str(row["event_type"]),
        operation_id=str(row["operation_id"]) if row["operation_id"] else None,
        profile_id=str(row["profile_id"]) if row["profile_id"] else None,
        details=parse_json_object(str(row["details_json"])),
        created_at=parse_timestamp(str(row["created_at"])),
    )
