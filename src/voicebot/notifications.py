"""Bot-neutral durable notification delivery with cooperative cancellation."""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import logging
import threading
import uuid
from collections.abc import Callable

from .storage import Storage
from .storage.models import Notification, utc_now

logger = logging.getLogger(__name__)
Clock = Callable[[], dt.datetime]


class DeliveryStatus(enum.StrEnum):
    """Possible outcomes returned by a notification callback."""

    DELIVERED = "delivered"
    DISMISSED = "dismissed"
    RETRY = "retry"


@dataclasses.dataclass(frozen=True, slots=True)
class DeliveryOutcome:
    """Outcome of one callback invocation."""

    status: DeliveryStatus

    @classmethod
    def delivered(cls) -> DeliveryOutcome:
        """Return a successful delivery outcome."""
        return cls(DeliveryStatus.DELIVERED)

    @classmethod
    def dismissed(cls) -> DeliveryOutcome:
        """Return an explicitly dismissed outcome."""
        return cls(DeliveryStatus.DISMISSED)

    @classmethod
    def retry(cls) -> DeliveryOutcome:
        """Return an outcome which leaves the item recoverable."""
        return cls(DeliveryStatus.RETRY)


NotificationCallback = Callable[[Notification, threading.Event], object]


class NotificationDispatcher:
    """Claim, deliver and acknowledge durable notifications.

    The callback receives the queue item and a cancellation event. It owns all
    speech/audio details and must stop promptly when that event is set.
    """

    def __init__(
        self,
        storage: Storage,
        callback: NotificationCallback,
        *,
        owner: str | None = None,
        clock: Clock = utc_now,
        lease_for: dt.timedelta = dt.timedelta(minutes=5),
    ) -> None:
        """Create a dispatcher around a bot-neutral delivery callback."""
        self.storage = storage
        self.callback = callback
        self.owner = owner or f"notification-worker:{uuid.uuid4().hex}"
        self.clock = clock
        self.lease_for = lease_for
        self._active: dict[str, threading.Event] = {}
        self._dismissed: set[str] = set()
        self._lock = threading.Lock()

    def deliver_once(self, limit: int = 10) -> list[Notification]:
        """Deliver up to ``limit`` available items and return claimed items."""
        now = self._now()
        self.storage.notifications.recover_expired(now=now)
        claimed = self.storage.notifications.claim(
            owner=self.owner, limit=limit, lease_for=self.lease_for, now=now
        )
        for notification in claimed:
            self._deliver_one(notification)
        return claimed

    def run_once(self, limit: int = 10) -> list[Notification]:
        """Alias for :meth:`deliver_once` used by worker loops."""
        return self.deliver_once(limit=limit)

    def cancel(self, notification_id: str) -> bool:
        """Cancel an in-flight delivery and dismiss its durable queue item."""
        with self._lock:
            event = self._active.get(notification_id)
            if event is None:
                return False
            self._dismissed.add(notification_id)
            event.set()
            return True

    def dismiss(self, notification: Notification) -> bool:
        """Dismiss a claimed notification explicitly."""
        return self.storage.notifications.acknowledge(
            notification.id,
            claim_token=notification.claim_token,
            dismissed=True,
            now=self._now(),
        )

    def acknowledge(self, notification: Notification) -> bool:
        """Acknowledge a claimed notification after playback completes."""
        return self.storage.notifications.acknowledge(
            notification.id, claim_token=notification.claim_token, now=self._now()
        )

    def _deliver_one(self, notification: Notification) -> None:
        event = threading.Event()
        with self._lock:
            self._active[notification.id] = event
        try:
            result = self.callback(notification, event)
            with self._lock:
                dismissed = notification.id in self._dismissed
                self._dismissed.discard(notification.id)
            outcome = _outcome(result, dismissed=dismissed)
            if outcome.status is DeliveryStatus.DELIVERED:
                self.acknowledge(notification)
            elif outcome.status is DeliveryStatus.DISMISSED:
                self.dismiss(notification)
            else:
                logger.info(
                    "Leaving notification available for recovery: %s", notification.id
                )
        except Exception:
            # The lease makes an interrupted callback recoverable after a restart.
            logger.exception("Notification callback failed id=%s", notification.id)
        finally:
            with self._lock:
                self._active.pop(notification.id, None)

    def _now(self) -> dt.datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("the notification clock must return an aware datetime")
        return value


# Naming aliases keep the delivery mechanism independent of the bot runtime.
DurableNotificationDispatcher = NotificationDispatcher
NotificationDelivery = NotificationDispatcher
AlarmDelivery = NotificationDispatcher


def _outcome(result: object, *, dismissed: bool) -> DeliveryOutcome:
    if dismissed:
        return DeliveryOutcome.dismissed()
    if isinstance(result, DeliveryOutcome):
        return result
    if isinstance(result, DeliveryStatus):
        return DeliveryOutcome(result)
    if result is False:
        return DeliveryOutcome.retry()
    return DeliveryOutcome.delivered()


__all__ = [
    "AlarmDelivery",
    "DeliveryOutcome",
    "DeliveryStatus",
    "DurableNotificationDispatcher",
    "NotificationCallback",
    "NotificationDelivery",
    "NotificationDispatcher",
]
