"""Versioned SQLite schema migrations for voicebot storage."""

from __future__ import annotations

import sqlite3

MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        1,
        (
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profile_aliases (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                alias TEXT NOT NULL,
                normalised_alias TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(profile_id, normalised_alias)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS provider_accounts (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                credential_ref TEXT NOT NULL,
                scopes_json TEXT NOT NULL,
                status TEXT NOT NULL,
                expires_at TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(profile_id, provider)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS calendar_bindings (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                provider_account_id TEXT REFERENCES provider_accounts(id)
                    ON DELETE SET NULL,
                alias TEXT NOT NULL,
                normalised_alias TEXT NOT NULL,
                calendar_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                UNIQUE(profile_id, normalised_alias)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS spotify_device_aliases (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                provider_account_id TEXT REFERENCES provider_accounts(id)
                    ON DELETE SET NULL,
                alias TEXT NOT NULL,
                normalised_alias TEXT NOT NULL,
                device_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                UNIQUE(profile_id, normalised_alias)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS shopping_list_bindings (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                provider_account_id TEXT REFERENCES provider_accounts(id)
                    ON DELETE SET NULL,
                alias TEXT NOT NULL,
                normalised_alias TEXT NOT NULL,
                list_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                UNIQUE(profile_id, normalised_alias)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                name TEXT,
                message TEXT NOT NULL,
                due_at TEXT NOT NULL,
                original_timezone TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN
                    ('scheduled', 'queued', 'delivered', 'cancelled', 'missed')),
                deduplication_key TEXT UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                delivery_attempts INTEGER NOT NULL DEFAULT 0,
                claimed_at TEXT,
                delivered_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS notification_queue (
                id TEXT PRIMARY KEY,
                reminder_id TEXT UNIQUE REFERENCES reminders(id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                available_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN
                    ('pending', 'processing', 'acknowledged', 'dismissed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                claim_token TEXT,
                claimed_at TEXT,
                lease_until TEXT,
                created_at TEXT NOT NULL,
                acknowledged_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                profile_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
                operation_type TEXT NOT NULL,
                provider TEXT,
                status TEXT NOT NULL CHECK(status IN
                    ('started', 'pending_confirmation', 'completed',
                     'failed', 'cancelled')),
                request_json TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS pending_confirmations (
                id TEXT PRIMARY KEY,
                operation_id TEXT REFERENCES operations(id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                action TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN
                    ('pending', 'confirmed', 'rejected', 'expired')),
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS scheduler_leases (
                name TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                acquired_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                operation_id TEXT REFERENCES operations(id) ON DELETE SET NULL,
                profile_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
        ),
    ),
    (
        2,
        (
            "CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, due_at)",
            "CREATE INDEX IF NOT EXISTS idx_notifications_claim "
            "ON notification_queue(status, available_at)",
            "CREATE INDEX IF NOT EXISTS idx_confirmations_pending "
            "ON pending_confirmations(status, expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events(created_at)",
        ),
    ),
)

CURRENT_SCHEMA_VERSION = MIGRATIONS[-1][0]


def migrate(connection: sqlite3.Connection) -> int:
    """Apply all unapplied migrations and return the schema version.

    Migrations are individually transactional and can safely be called every time a
    database connection is opened.
    """
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {
        int(row[0])
        for row in connection.execute("SELECT version FROM schema_migrations")
    }
    for version, statements in MIGRATIONS:
        if version in applied:
            continue
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in statements:
                connection.execute(statement)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) "
                "VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                (version,),
            )
            connection.execute(f"PRAGMA user_version = {version}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        applied.add(version)
    return max(applied, default=0)
