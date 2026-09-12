"""Tests for the durable SQLite storage foundation."""

from __future__ import annotations

import datetime as dt
import pathlib
import sqlite3

import pytest

from voicebot.storage import CURRENT_SCHEMA_VERSION, Database, Storage, normalise_alias

NOW = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)


def test_migrations_are_idempotent_and_enable_foreign_keys() -> None:
    """Opening an existing database does not rerun or corrupt migrations."""
    with Database(":memory:") as database:
        assert (
            database.connection.execute(
                "SELECT max(version) FROM schema_migrations"
            ).fetchone()[0]
            == CURRENT_SCHEMA_VERSION
        )
        assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        database.connection.execute("PRAGMA user_version")


def test_aliases_are_normalised_and_unique(tmp_path: pathlib.Path) -> None:
    """Aliases match Unicode/case/whitespace variants but never ambiguously."""
    assert normalise_alias("  HéLLO!!! ") == "héllo"
    with Storage(str(tmp_path / "state.sqlite")) as storage:
        profile = storage.profiles.create("Dan", now=NOW)
        storage.profiles.add_alias(profile.id, "  HéLLO!!! ", now=NOW)
        assert storage.profiles.resolve_alias("hello") == []
        assert storage.profiles.resolve_alias(" héllo ") == [profile]
        other = storage.profiles.create("Other", now=NOW)
        storage.profiles.add_alias(other.id, "HéLLO", now=NOW)
        assert len(storage.profiles.resolve_alias("héllo")) == 2
        with pytest.raises(sqlite3.IntegrityError):
            storage.profiles.add_alias(profile.id, "HÉLLO", now=NOW)
        with pytest.raises(sqlite3.IntegrityError):
            storage.profiles.add_alias("missing", "other", now=NOW)


def test_reminder_queue_recovers_after_worker_restart() -> None:
    """An expired claim can be claimed by a replacement worker and acknowledged."""
    with Storage() as storage:
        profile = storage.profiles.create("Household", now=NOW)
        reminder = storage.reminders.create(
            profile.id, "Call home", NOW - dt.timedelta(minutes=2), now=NOW
        )
        storage.reminders.claim_due(now=NOW)
        first = storage.notifications.claim(
            owner="worker-a", lease_for=dt.timedelta(minutes=1), now=NOW
        )
        assert len(first) == 1
        second = storage.notifications.claim(
            owner="worker-b", now=NOW + dt.timedelta(minutes=2)
        )
        assert [item.id for item in second] == [first[0].id]
        assert (
            storage.notifications.acknowledge(
                second[0].id, claim_token=first[0].claim_token, now=NOW
            )
            is False
        )
        assert storage.notifications.acknowledge(
            second[0].id,
            claim_token=second[0].claim_token,
            now=NOW + dt.timedelta(minutes=2),
        )
        delivered = storage.reminders.get(reminder.id)
        assert delivered is not None
        assert delivered.status == "delivered"


def test_operation_confirmation_lease_and_audit() -> None:
    """The non-scheduler records survive normal runtime state transitions."""
    with Storage() as storage:
        profile = storage.profiles.create("Dan", now=NOW)
        operation = storage.operations.start(
            "request-1", "delete_item", profile_id=profile.id, now=NOW
        )
        assert (
            storage.operations.start("request-1", "ignored", now=NOW).id == operation.id
        )
        confirmation = storage.confirmations.create(
            profile.id,
            "delete_item",
            {"item": "milk"},
            NOW + dt.timedelta(minutes=5),
            operation_id=operation.id,
            now=NOW,
        )
        resolved = storage.confirmations.resolve(
            confirmation.id, accepted=True, now=NOW
        )
        assert resolved is not None
        assert resolved.status == "confirmed"
        assert storage.leases.acquire("scheduler", "one", now=NOW) is not None
        assert storage.leases.acquire("scheduler", "two", now=NOW) is None
        event = storage.audit.record(
            "provider.call",
            {
                "Authorization": "Bearer secret",
                "email": "person@example.com",
                "status": "ok",
            },
            operation_id=operation.id,
            profile_id=profile.id,
            now=NOW,
        )
        assert event.details == {
            "Authorization": "[REDACTED]",
            "email": "[REDACTED]",
            "status": "ok",
        }
        assert (
            "secret"
            not in storage.database.connection.execute(
                "SELECT details_json FROM audit_events"
            ).fetchone()[0]
        )
