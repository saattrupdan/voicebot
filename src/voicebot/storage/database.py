"""SQLite connection management and transaction helpers."""

from __future__ import annotations

import contextlib
import pathlib
import sqlite3
from collections.abc import Iterator

from .migrations import migrate


class Database:
    """Own one SQLite connection and initialise its schema."""

    def __init__(self, path: str | pathlib.Path = ":memory:") -> None:
        """Open ``path`` and apply all storage migrations."""
        self.path = pathlib.Path(path) if path != ":memory:" else path
        self.connection = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False, timeout=30
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        migrate(self.connection)

    @property
    def schema_version(self) -> int:
        """Return the highest successfully applied migration version."""
        row = self.connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the underlying connection."""
        self.connection.close()

    def __enter__(self) -> Database:
        """Return this database for use as a context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close this database when leaving a context manager."""
        self.close()

    @contextlib.contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Run a block in a transaction, rolling back on errors.

        Existing transactions are reused, allowing a repository operation to be
        composed into a larger transaction without accidentally committing it.
        """
        owns_transaction = not self.connection.in_transaction
        if owns_transaction:
            self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield self.connection
        except Exception:
            if owns_transaction:
                self.connection.rollback()
            raise
        else:
            if owns_transaction:
                self.connection.commit()

    def vacuum(self) -> None:
        """Compact the database after deleting a large amount of data."""
        self.connection.execute("VACUUM")
