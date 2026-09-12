"""Single-owner durable reminder scheduler."""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from collections.abc import Callable

from .storage import Storage
from .storage.models import utc_now

logger = logging.getLogger(__name__)
Clock = Callable[[], dt.datetime]


class ReminderScheduler:
    """Claim due reminders after acquiring the database scheduler lease."""

    LEASE_NAME = "reminder-scheduler"
    MISSED_GRACE = dt.timedelta(minutes=15)

    def __init__(
        self,
        storage: Storage,
        *,
        owner: str | None = None,
        clock: Clock = utc_now,
        lease_for: dt.timedelta = dt.timedelta(seconds=30),
        grace_period: dt.timedelta = MISSED_GRACE,
    ) -> None:
        """Create a scheduler with an injectable clock and stable owner identity."""
        self.storage = storage
        self.owner = owner or uuid.uuid4().hex
        self.clock = clock
        self.lease_for = lease_for
        self.grace_period = grace_period
        self._running = False

    @property
    def is_running(self) -> bool:
        """Return whether this scheduler has been started."""
        return self._running

    def start(self) -> bool:
        """Start this owner if no other scheduler currently holds the lease."""
        acquired = self._acquire()
        self._running = acquired
        return acquired

    def stop(self) -> None:
        """Stop this scheduler and release its lease."""
        if self._running:
            self.storage.leases.release(self.LEASE_NAME, self.owner)
        self._running = False

    def run_once(self) -> int:
        """Claim due reminders once and report how many were queued."""
        return self.tick()

    def tick(self) -> int:
        """Claim due reminders and report how many were moved to the queue."""
        if not self._acquire():
            return 0
        self._running = True
        return len(
            self.storage.reminders.claim_due(
                now=self._now(), late_window=self.grace_period
            )
        )

    def renew(self) -> bool:
        """Renew this owner's lease while it is still valid."""
        lease = self.storage.leases.renew(
            self.LEASE_NAME, self.owner, lease_for=self.lease_for, now=self._now()
        )
        self._running = lease is not None
        return self._running

    def _acquire(self) -> bool:
        lease = self.storage.leases.acquire(
            self.LEASE_NAME, self.owner, lease_for=self.lease_for, now=self._now()
        )
        if lease is None:
            logger.debug("Scheduler lease is held by another owner")
        return lease is not None

    def _now(self) -> dt.datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("the scheduler clock must return an aware datetime")
        return value


# A concise name is useful to applications which treat the scheduler as a service.
Scheduler = ReminderScheduler

__all__ = ["ReminderScheduler", "Scheduler"]
