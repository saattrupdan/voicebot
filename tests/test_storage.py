"""Tests for the durable SQLite storage foundation."""

from __future__ import annotations

import datetime as dt
import pathlib
import sqlite3
import threading

import pytest

from voicebot.storage import (
    CURRENT_SCHEMA_VERSION,
    Database,
    OperationAcquisition,
    Storage,
    normalise_alias,
)

NOW = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)


def test_fresh_filesystem_database_creates_parent_directory(
    tmp_path: pathlib.Path,
) -> None:
    """Opening a new database also creates its nested parent directory."""
    path = tmp_path / "state" / "nested" / "voicebot.sqlite"

    with Database(path) as database:
        assert path.is_file()
        assert database.schema_version == CURRENT_SCHEMA_VERSION
        assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


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


def test_transactions_are_isolated_and_rollback_across_threads() -> None:
    """A concurrent transaction waits and cannot inherit a rollback."""
    errors: list[BaseException] = []
    first_ready = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    second_count: list[int] = []

    with Database() as database:
        database.connection.execute(
            "CREATE TABLE transaction_probe (value TEXT NOT NULL)"
        )

        def rollback_worker() -> None:
            try:
                with database.transaction(immediate=True) as connection:
                    connection.execute(
                        "INSERT INTO transaction_probe(value) VALUES (?)",
                        ("rolled back",),
                    )
                    first_ready.set()
                    if not release_first.wait(timeout=5):
                        raise AssertionError("timed out waiting to roll back")
                    raise RuntimeError("test rollback")
            except RuntimeError:
                pass
            except BaseException as error:
                errors.append(error)

        def commit_worker() -> None:
            second_attempted.set()
            try:
                with database.transaction(immediate=True) as connection:
                    second_entered.set()
                    second_count.append(
                        connection.execute(
                            "SELECT count(*) FROM transaction_probe"
                        ).fetchone()[0]
                    )
                    connection.execute(
                        "INSERT INTO transaction_probe(value) VALUES (?)",
                        ("committed",),
                    )
            except BaseException as error:
                errors.append(error)

        first_thread = threading.Thread(target=rollback_worker)
        second_thread = threading.Thread(target=commit_worker)
        first_thread.start()
        assert first_ready.wait(timeout=5)
        second_thread.start()
        assert second_attempted.wait(timeout=5)
        entered_before_release = second_entered.wait(timeout=0.1)
        release_first.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert errors == []
        assert not entered_before_release
        assert second_count == [0]
        rows = database.connection.execute(
            "SELECT value FROM transaction_probe"
        ).fetchall()
        assert [row[0] for row in rows] == ["committed"]


def test_operation_acquisition_has_one_owner() -> None:
    """Concurrent callers cannot both acquire one idempotency key."""
    barrier = threading.Barrier(8)
    acquisitions: list[OperationAcquisition] = []
    errors: list[BaseException] = []

    with Storage() as storage:

        def acquire() -> None:
            try:
                barrier.wait(timeout=5)
                acquisitions.append(
                    storage.operations.acquire("same-key", "delete_item")
                )
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=acquire) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert sum(item.owned for item in acquisitions) == 1
        assert len({item.operation.id for item in acquisitions}) == 1
        assert sum(item.is_owner for item in acquisitions) == 1


def test_database_queries_do_not_observe_rolled_back_rows() -> None:
    """A read waits for an active transaction instead of seeing dirty data."""
    first_ready = threading.Event()
    release_first = threading.Event()
    read_values: list[list[str]] = []
    errors: list[BaseException] = []

    with Database() as database:
        database.connection.execute(
            "CREATE TABLE dirty_read_probe (value TEXT NOT NULL)"
        )

        def rollback_worker() -> None:
            try:
                with database.transaction(immediate=True) as connection:
                    connection.execute(
                        "INSERT INTO dirty_read_probe(value) VALUES (?)",
                        ("uncommitted",),
                    )
                    first_ready.set()
                    assert release_first.wait(timeout=5)
                    raise RuntimeError("test rollback")
            except RuntimeError:
                pass
            except BaseException as error:
                errors.append(error)

        def read_worker() -> None:
            try:
                assert first_ready.wait(timeout=5)
                read_values.append(
                    [
                        str(row[0])
                        for row in database.query("SELECT value FROM dirty_read_probe")
                    ]
                )
            except BaseException as error:
                errors.append(error)

        writer = threading.Thread(target=rollback_worker)
        reader = threading.Thread(target=read_worker)
        writer.start()
        assert first_ready.wait(timeout=5)
        reader.start()
        release_first.set()
        writer.join(timeout=5)
        reader.join(timeout=5)

        assert errors == []
        assert not writer.is_alive()
        assert not reader.is_alive()
        assert read_values == [[]]


def test_legacy_awaiting_confirmation_status_is_migrated(
    tmp_path: pathlib.Path,
) -> None:
    """Old operation rows remain readable after status spelling is corrected."""
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL
        );
        INSERT INTO schema_migrations VALUES (1, 'legacy');
        INSERT INTO schema_migrations VALUES (2, 'legacy');
        CREATE TABLE profiles (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE operations (
            id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE,
            profile_id TEXT REFERENCES profiles(id), operation_type TEXT NOT NULL,
            provider TEXT,
            status TEXT NOT NULL CHECK(status IN
                ('started', 'awaiting_confirmation', 'completed', 'failed',
                 'cancelled')),
            request_json TEXT NOT NULL, result_json TEXT, error TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        INSERT INTO operations VALUES
            ('operation-1', 'request-1', NULL, 'delete_item', NULL,
             'awaiting_confirmation', '{}', NULL, NULL, '2026-01-01T12:00:00.000Z',
             '2026-01-01T12:00:00.000Z');
        """
    )
    connection.commit()
    connection.close()

    with Storage(path) as storage:
        operation = storage.operations.get("operation-1")
        assert operation is not None
        assert operation.status == "pending_confirmation"
        assert (
            storage.operations.update(
                operation.id, "awaiting_confirmation", now=NOW
            ).status
            == "pending_confirmation"
        )


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
