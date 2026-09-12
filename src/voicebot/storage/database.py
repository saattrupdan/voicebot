"""SQLite connection management and transaction helpers."""

from __future__ import annotations

import contextlib
import pathlib
import sqlite3
import threading
from collections.abc import Iterator

from .migrations import migrate


class Database:
    """Own one SQLite connection and initialise its schema."""

    def __init__(self, path: str | pathlib.Path = ":memory:") -> None:
        """Open ``path`` and apply all storage migrations."""
        if path == ":memory:":
            self.path = path
        else:
            self.path = pathlib.Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._transaction_lock = threading.RLock()
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
        with self._transaction_lock:
            row = self.connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()
            return int(row[0])

    def close(self) -> None:
        """Close the underlying connection."""
        with self._transaction_lock:
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
        The reentrant lock must cover the entire block: checking transaction state,
        executing statements, and committing or rolling back.
        """
        with self._transaction_lock:
            owns_transaction = not self.connection.in_transaction
            if owns_transaction:
                self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield self.connection
            except BaseException:
                if owns_transaction:
                    self.connection.rollback()
                raise
            else:
                if owns_transaction:
                    try:
                        self.connection.commit()
                    except BaseException:
                        self.connection.rollback()
                        raise

    def vacuum(self) -> None:
        """Compact the database after deleting a large amount of data."""
        with self._transaction_lock:
            self.connection.execute("VACUUM")
