"""Durable local storage for profiles, tools and scheduler workers."""

from __future__ import annotations

import pathlib

from .database import Database
from .migrations import CURRENT_SCHEMA_VERSION, migrate
from .models import (
    AuditEvent,
    Binding,
    Notification,
    Operation,
    PendingConfirmation,
    Profile,
    ProfileAlias,
    ProviderAccount,
    Reminder,
    SchedulerLease,
    normalise_alias,
    utc_now,
)
from .repositories import (
    AuditRepository,
    ConfirmationRepository,
    NotificationQueueRepository,
    OperationRepository,
    ProfileRepository,
    ProviderRepository,
    ReminderRepository,
    SchedulerLeaseRepository,
    redact,
)


class Storage:
    """Convenient aggregate of the storage repositories."""

    def __init__(self, path: str | pathlib.Path = ":memory:") -> None:
        """Open a database and expose all repositories."""
        self.database = Database(path)
        self.profiles = ProfileRepository(self.database)
        self.providers = ProviderRepository(self.database)
        self.reminders = ReminderRepository(self.database)
        self.notifications = NotificationQueueRepository(self.database)
        self.operations = OperationRepository(self.database)
        self.confirmations = ConfirmationRepository(self.database)
        self.leases = SchedulerLeaseRepository(self.database)
        self.audit = AuditRepository(self.database)

    def close(self) -> None:
        """Close the storage database."""
        self.database.close()

    def __enter__(self) -> Storage:
        """Return this storage aggregate for use as a context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close storage when leaving a context manager."""
        self.close()


__all__ = [
    "AuditEvent",
    "AuditRepository",
    "Binding",
    "ConfirmationRepository",
    "CURRENT_SCHEMA_VERSION",
    "Database",
    "migrate",
    "Notification",
    "NotificationQueueRepository",
    "Operation",
    "OperationRepository",
    "PendingConfirmation",
    "Profile",
    "ProfileAlias",
    "ProfileRepository",
    "ProviderAccount",
    "ProviderRepository",
    "Reminder",
    "ReminderRepository",
    "redact",
    "SchedulerLease",
    "SchedulerLeaseRepository",
    "Storage",
    "normalise_alias",
    "utc_now",
]
