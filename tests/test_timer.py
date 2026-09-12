"""Tests for named timers."""

import datetime as dt
import threading
import time
import typing as t

from voicebot.tool_runtime import ToolContext, ToolResult, ToolStatus
from voicebot.tools.timer import Timer, list_timers, set_timer, stop_timer


class FakeClock:
    """Manually controlled monotonic clock."""

    def __init__(self, value: float = 0) -> None:
        """Initialise the clock at a chosen monotonic value."""
        self.value = value

    def __call__(self) -> float:
        """Return the current manually controlled time."""
        return self.value


def test_named_timers_use_unicode_normalisation_and_reject_duplicates() -> None:
    """Equivalent Unicode names conflict while the first name is retained."""
    clock = FakeClock()
    context = ToolContext(state={})

    created = set_timer(
        context,
        {"name": "Cafe\u0301", "duration_seconds": 30},
        clock=clock,
    )
    duplicate = set_timer(
        context,
        {"name": "CAFÉ", "duration_seconds": 10},
        clock=clock,
    )

    assert isinstance(created, ToolResult)
    assert isinstance(duplicate, ToolResult)
    assert created.status is ToolStatus.OK
    assert duplicate.status is ToolStatus.CONFLICT
    assert created.data is not None
    assert created.data["timer"]["name"] == "Café"  # type: ignore[index]


def test_concurrent_timers_are_listed_and_stopped_exactly() -> None:
    """Stopping one named timer leaves concurrently active timers untouched."""
    clock = FakeClock()
    context = ToolContext(state={})
    set_timer(context, {"name": "short", "duration_seconds": 5}, clock=clock)
    set_timer(context, {"name": "long", "duration_seconds": 20}, clock=clock)

    listed = list_timers(context, {"name": None})
    missing = stop_timer(context, {"name": "unknown"})
    stopped = stop_timer(context, {"name": "long"})

    assert isinstance(listed, ToolResult)
    assert isinstance(missing, ToolResult)
    assert isinstance(stopped, ToolResult)
    assert listed.status is ToolStatus.OK
    assert listed.data is not None
    assert [item["name"] for item in listed.data["timers"]] == [  # type: ignore[index]
        "short",
        "long",
    ]
    assert missing.status is ToolStatus.NOT_FOUND
    assert stopped.status is ToolStatus.OK
    remaining = list_timers(context, {"name": None})
    assert isinstance(remaining, ToolResult)
    assert remaining.data is not None
    assert [item["name"] for item in remaining.data["timers"]] == [  # type: ignore[index]
        "short",
    ]


def test_expiry_is_cleaned_up_and_notified() -> None:
    """Due timers disappear from active state before notification."""
    clock = FakeClock()
    notifications: list[str] = []
    state: dict[str, object] = {
        "notification_callback": lambda timer: notifications.append(timer.name),
    }
    context = ToolContext(state=state)
    set_timer(context, {"name": "tea", "duration_seconds": 10}, clock=clock)

    clock.value = 10
    result = list_timers(context, {"name": None})

    assert isinstance(result, ToolResult)
    assert result.status is ToolStatus.OK
    assert result.data == {"timers": []}
    assert notifications == ["tea"]


def test_invalid_zero_and_out_of_range_durations_do_not_create_timers() -> None:
    """Durations outside the inclusive contract are rejected."""
    context = ToolContext(state={})

    zero = set_timer(context, {"name": "zero", "duration_seconds": 0})
    too_long = set_timer(
        context,
        {"name": "too long", "duration_seconds": 86_401},
    )

    assert isinstance(zero, ToolResult)
    assert isinstance(too_long, ToolResult)
    assert zero.status is ToolStatus.INVALID_REQUEST
    assert too_long.status is ToolStatus.INVALID_REQUEST
    assert context.state == {}


def test_duration_boundaries_are_inclusive() -> None:
    """One second and one day are both valid timer durations."""
    clock = FakeClock()
    context = ToolContext(state={})

    shortest = set_timer(
        context,
        {"name": "shortest", "duration_seconds": 1},
        clock=clock,
    )
    longest = set_timer(
        context,
        {"name": "longest", "duration_seconds": 86_400},
        clock=clock,
    )

    assert isinstance(shortest, ToolResult)
    assert isinstance(longest, ToolResult)
    assert shortest.status is ToolStatus.OK
    assert longest.status is ToolStatus.OK
    stop_timer(context, {"name": "shortest"})
    stop_timer(context, {"name": "longest"})
    assert Timer.prettify_timedelta(timedelta=dt.timedelta(0)) == "0 sekunder"


def test_cancellation_does_not_mutate_timer_state() -> None:
    """A cancelled turn cannot create a timer."""
    event = threading.Event()
    event.set()
    state: dict[str, object] = {}
    context = ToolContext(state=state, cancel_event=event)

    result = set_timer(context, {"name": "cancelled", "duration_seconds": 1})

    assert isinstance(result, ToolResult)
    assert result.status is ToolStatus.CANCELLED
    assert state == {}


def test_timer_stop_suppresses_alarm() -> None:
    """Stopping a timer wakes its worker without delivering an alarm."""
    alarmed: list[str] = []
    timer = Timer(
        duration_seconds=1,
        name="quiet",
        alarm_callback=lambda completed: alarmed.append(completed.name),
    ).start()

    timer.stop()
    time.sleep(0.02)

    assert alarmed == []


def test_legacy_duration_callable_remains_supported() -> None:
    """The old duration-only state API remains usable during migration."""
    state: dict[str, object] = {}

    message, updated = t.cast(
        tuple[str, dict[str, object]], set_timer(state=state, duration_seconds=5)
    )
    assert message == ""
    assert len(updated["running_timers"]) == 1  # type: ignore[arg-type]

    _, stopped = t.cast(
        tuple[str, dict[str, object]],
        stop_timer(state=updated, duration="0:00:05"),
    )
    assert stopped["running_timers"] == []
