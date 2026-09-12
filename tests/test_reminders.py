"""Tests for persistent reminder tools and worker delivery."""

from __future__ import annotations

import datetime as dt
import pathlib
from zoneinfo import ZoneInfo

from voicebot.notifications import DeliveryOutcome, NotificationDispatcher
from voicebot.scheduler import ReminderScheduler
from voicebot.storage import Storage
from voicebot.tool_runtime import ToolContext, ToolStatus
from voicebot.tools.reminders import (
    cancel_reminder_spec,
    create_reminder_spec,
    list_reminders_spec,
)

NOW = dt.datetime(2027, 1, 1, 12, tzinfo=dt.UTC)


def _context(storage: Storage, clock: list[dt.datetime]) -> ToolContext:
    profile = storage.profiles.list()[0]
    return ToolContext(
        state={"storage": storage, "profile_id": profile.id, "clock": lambda: clock[0]}
    )


def test_create_requires_one_time_and_persists_timezone() -> None:
    """Relative times become UTC while retaining the local timezone."""
    clock = [NOW]
    local_now = dt.datetime(2027, 1, 1, 13, tzinfo=dt.timezone(dt.timedelta(hours=1)))
    clock[0] = local_now
    with Storage() as storage:
        storage.profiles.create("Home", now=NOW)
        context = _context(storage, clock)
        result = create_reminder_spec(storage, clock=lambda: clock[0]).handler(
            context,
            {
                "profile_name": None,
                "name": "oven",
                "message": "Turn off the oven",
                "delay_seconds": 60,
                "due_at": None,
            },
        )
        assert result.status is ToolStatus.OK
        reminder = storage.reminders.list()[0]
        assert reminder.due_at == dt.datetime(2027, 1, 1, 12, 1, tzinfo=dt.UTC)
        assert reminder.original_timezone == "+01:00"
        listed = list_reminders_spec(storage, clock=lambda: clock[0]).handler(
            context,
            {
                "profile_name": None,
                "starts_at": None,
                "ends_at": None,
                "status": "pending",
            },
        )
        assert listed.status is ToolStatus.OK
        assert listed.data is not None
        reminders = listed.data["reminders"]
        assert isinstance(reminders, list)
        assert len(reminders) == 1

        invalid = create_reminder_spec(storage, clock=lambda: clock[0]).handler(
            context,
            {
                "profile_name": None,
                "name": None,
                "message": "x",
                "delay_seconds": 1,
                "due_at": "2027-01-01T14:00:00+01:00",
            },
        )
        assert invalid.status is ToolStatus.INVALID_REQUEST


def test_dst_transition_times_are_rejected() -> None:
    """Nonexistent and ambiguous local times cannot become reminders."""
    timezone = ZoneInfo("Europe/Copenhagen")
    with Storage() as storage:
        storage.profiles.create("Home", now=NOW)
        context = _context(storage, [NOW])
        tool = create_reminder_spec(storage, clock=lambda: NOW)
        for due_at in (
            dt.datetime(2027, 3, 28, 2, 30, tzinfo=timezone),
            dt.datetime(2027, 10, 31, 2, 30, tzinfo=timezone),
        ):
            result = tool.handler(
                context,
                {
                    "profile_name": None,
                    "name": None,
                    "message": "transition",
                    "delay_seconds": None,
                    "due_at": due_at,
                },
            )
            assert result.status is ToolStatus.INVALID_REQUEST


def test_scheduler_contention_and_missed_policy(tmp_path: pathlib.Path) -> None:
    """Only one owner claims due work and stale work is marked missed."""
    database = tmp_path / "reminders.sqlite"
    with Storage(database) as storage:
        profile = storage.profiles.create("Home", now=NOW)
        storage.reminders.create(
            profile.id, "fresh", NOW - dt.timedelta(minutes=5), name="fresh", now=NOW
        )
        storage.reminders.create(
            profile.id, "stale", NOW - dt.timedelta(minutes=16), name="stale", now=NOW
        )
        first = ReminderScheduler(storage, owner="one", clock=lambda: NOW)
        second = ReminderScheduler(storage, owner="two", clock=lambda: NOW)
        assert first.start()
        assert not second.start()
        assert first.tick() == 1
        assert storage.reminders.list(status="missed")[0].name == "stale"
        assert len(storage.notifications.claim(owner="worker", now=NOW)) == 1


def test_restart_recovery_and_duplicate_prevention(tmp_path: pathlib.Path) -> None:
    """A leased notification is recovered and is enqueued only once."""
    database = tmp_path / "reminders.sqlite"
    with Storage(database) as storage:
        profile = storage.profiles.create("Home", now=NOW)
        reminder = storage.reminders.create(profile.id, "hello", NOW, now=NOW)
        assert ReminderScheduler(storage, clock=lambda: NOW).tick() == 1
        claimed = storage.notifications.claim(
            owner="crashed", lease_for=dt.timedelta(seconds=1), now=NOW
        )
        assert claimed
        assert ReminderScheduler(storage, clock=lambda: NOW).tick() == 0
    later = NOW + dt.timedelta(seconds=2)
    with Storage(database) as storage:
        recovered = storage.notifications.claim(owner="replacement", now=later)
        assert len(recovered) == 1
        assert recovered[0].reminder_id == reminder.id
        assert storage.reminders.get(reminder.id) is not None


def test_cancel_is_exact_and_dispatch_acknowledges() -> None:
    """Cancellation does not guess, while successful playback is acknowledged."""
    with Storage() as storage:
        profile = storage.profiles.create("Home", now=NOW)
        storage.reminders.create(profile.id, "one", NOW, name="same", now=NOW)
        storage.reminders.create(profile.id, "two", NOW, name="same", now=NOW)
        cancellable = storage.reminders.create(
            profile.id, "cancel me", NOW + dt.timedelta(hours=1), name="cancel", now=NOW
        )
        context = _context(storage, [NOW])
        ambiguous = cancel_reminder_spec(storage, clock=lambda: NOW).handler(
            context, {"profile_name": None, "reminder_name": "same"}
        )
        assert ambiguous.status is ToolStatus.CONFLICT
        cancelled = cancel_reminder_spec(storage, clock=lambda: NOW).handler(
            context, {"profile_name": None, "reminder_name": "cancel"}
        )
        assert cancelled.status is ToolStatus.OK
        cancelled_record = storage.reminders.get(cancellable.id)
        assert cancelled_record is not None
        assert cancelled_record.status == "cancelled"

        storage.reminders.claim_due(now=NOW)
        dispatcher = NotificationDispatcher(
            storage,
            callback=lambda notification, event: DeliveryOutcome.delivered(),
            clock=lambda: NOW,
        )
        dispatcher.deliver_once()
        assert len(storage.reminders.list(status="delivered")) == 2
