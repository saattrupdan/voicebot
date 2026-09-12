"""Named, in-process timers for the voicebot."""

import collections.abc as c
import datetime as dt
import logging
import math
import threading
import time
import typing as t
import unicodedata
from dataclasses import dataclass

from ..tool_runtime import ToolContext, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

Clock = c.Callable[[], float]
TimerCallback = c.Callable[["Timer"], None]

_MIN_DURATION_SECONDS = 1
_MAX_DURATION_SECONDS = 86_400
_STATE_LOCKS: dict[int, threading.RLock] = {}
_STATE_LOCKS_GUARD = threading.Lock()


@dataclass
class Timer:
    """An in-process timer with a monotonic deadline.

    Args:
        name:
            The display name of the timer.
        duration_seconds:
            The duration in seconds.
        clock (optional):
            A monotonic clock, mainly useful for deterministic tests. Defaults to
            :func:`time.monotonic`.
        alarm_callback (optional):
            A callback handed the timer when it completes. The callback is never
            called when the timer is stopped.
        completion_callback (optional):
            An internal callback used by the timer collection to remove the timer.
    """

    duration_seconds: int
    name: str = ""
    clock: Clock = time.monotonic
    alarm_callback: TimerCallback | None = None
    completion_callback: TimerCallback | None = None

    def __post_init__(self) -> None:
        """Initialise cancellation and worker state."""
        if not _valid_duration(self.duration_seconds):
            raise ValueError("Timer duration must be between 1 and 86400 seconds")
        self.duration = dt.timedelta(seconds=self.duration_seconds)
        self.deadline: float | None = None
        self.start_time: float | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._completed = False

    def start(self) -> "Timer":
        """Start the timer and return it."""
        with self._lock:
            if self.deadline is not None:
                return self
            self.start_time = self.clock()
            self.deadline = self.start_time + self.duration_seconds
            self._thread = threading.Thread(
                target=self._wait_for_deadline,
                name=f"timer-{self.name}",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> "Timer":
        """Cancel the timer without delivering its alarm."""
        should_join = self._cancel()
        if should_join:
            self._join_worker()
        return self

    def expire(self) -> bool:
        """Complete the timer immediately, delivering its alarm once."""
        return self._finish(completed=True)

    @property
    def active(self) -> bool:
        """Whether the timer is still active."""
        with self._lock:
            return (
                self.deadline is not None
                and not self._stopped
                and not self._completed
            )

    @property
    def expired(self) -> bool:
        """Whether the deadline has passed according to the injected clock."""
        return self.deadline is not None and self.clock() >= self.deadline

    @property
    def remaining(self) -> dt.timedelta:
        """Return the non-negative whole seconds remaining."""
        if self.deadline is None:
            return dt.timedelta(0)
        remaining_seconds = max(math.ceil(self.deadline - self.clock()), 0)
        return dt.timedelta(seconds=remaining_seconds)

    @property
    def remaining_seconds(self) -> int:
        """Return the non-negative whole seconds remaining."""
        return int(self.remaining.total_seconds())

    @property
    def pretty_duration(self) -> str:
        """Return the duration in the historical Danish format."""
        return self.prettify_timedelta(timedelta=self.duration)

    @property
    def pretty_remaining(self) -> str:
        """Return the remaining duration in the historical Danish format."""
        return self.prettify_timedelta(timedelta=self.remaining)

    def __repr__(self) -> str:
        """Return a useful representation without exposing thread details."""
        return f"Timer(name={self.name!r}, duration={self.duration!r})"

    @staticmethod
    def prettify_timedelta(timedelta: dt.timedelta) -> str:
        """Prettify a timedelta in Danish.

        Args:
            timedelta:
                The timedelta to prettify.

        Returns:
            A human-readable duration.
        """
        total_seconds = max(int(timedelta.total_seconds()), 0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_strings: list[str] = []
        if hours:
            time_strings.append(f"{hours} timer")
        if minutes:
            time_strings.append(f"{minutes} minutter")
        if seconds or not time_strings:
            time_strings.append(f"{seconds} sekunder")
        return ", ".join(time_strings[:-1]) + " og " + time_strings[-1] if len(
            time_strings
        ) > 1 else time_strings[0]

    def _wait_for_deadline(self) -> None:
        """Wait without creating a child process, then deliver the alarm."""
        while True:
            deadline = self.deadline
            if deadline is None:
                return
            remaining = deadline - self.clock()
            if remaining <= 0:
                self._finish(completed=True)
                return
            if self._stop_event.wait(timeout=min(remaining, 0.1)):
                return

    def _cancel(self) -> bool:
        """Mark the timer cancelled and wake its worker."""
        with self._lock:
            if self._stopped or self._completed:
                return False
            self._stopped = True
            self._stop_event.set()
            return True

    def _finish(self, *, completed: bool) -> bool:
        """Mark completion once and deliver callbacks outside the lock."""
        with self._lock:
            if self._stopped or self._completed:
                return False
            if not completed and not self.expired:
                return False
            self._completed = True
            self._stop_event.set()
        if self.completion_callback is not None:
            self.completion_callback(self)
        if self.alarm_callback is not None:
            try:
                self.alarm_callback(self)
            except Exception:
                logger.exception("Timer alarm callback failed for %s", self.name)
        return True

    def _join_worker(self) -> None:
        """Wait briefly for a cancelled worker to finish."""
        worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1)


def set_timer(
    context: dict[str, object] | ToolContext | None = None,
    name: str | int | dict[str, object] | None = None,
    duration_seconds: int | None = None,
    *,
    state: dict[str, object] | None = None,
    clock: Clock = time.monotonic,
    alarm_callback: TimerCallback | None = None,
) -> ToolResult | tuple[str, dict[str, object]]:
    """Create a uniquely named timer.

    The typed calling form is ``set_timer(context, arguments)`` (or with ``name``
    and ``duration_seconds`` keyword arguments). The state-based form is retained
    for callers from the pre-runtime tool API.

    Args:
        state:
            A typed tool context or legacy state mapping.
        name:
            The timer name, or the typed argument mapping.
        duration_seconds (optional):
            The duration in seconds.
        clock (optional):
            Clock used to calculate the deadline. Defaults to ``time.monotonic``.
        alarm_callback (optional):
            Callback to receive the completed timer.

    Returns:
        A :class:`ToolResult` for typed calls, or the historical state tuple.
    """
    if context is not None and state is not None:
        raise TypeError("Pass either context or state, not both")
    target = context if context is not None else state
    if target is None:
        raise TypeError("A tool context or state is required")
    typed_context = target if isinstance(target, ToolContext) else None
    arguments = _arguments(name=name, duration_seconds=duration_seconds)
    if typed_context is not None:
        return _set_typed_timer(
            context=typed_context,
            arguments=arguments,
            clock=clock,
            alarm_callback=alarm_callback,
        )

    legacy_state = t.cast(dict[str, object], target)
    legacy_duration = _legacy_duration(name=name, duration_seconds=duration_seconds)
    legacy_name = str(legacy_duration)
    _create_timer(
        state=legacy_state,
        name=legacy_name,
        duration_seconds=legacy_duration,
        clock=clock,
        alarm_callback=alarm_callback,
    )
    return "", dict(legacy_state, running_timers=_timers(legacy_state))


def list_timers(
    context: dict[str, object] | ToolContext | None = None,
    name: str | dict[str, object] | None = None,
    *,
    state: dict[str, object] | None = None,
) -> ToolResult | tuple[str, dict[str, object]]:
    """List active timers, optionally selecting one normalised name.

    Args:
        context:
            A typed tool context or legacy state mapping.
        name:
            A timer name, ``None`` for all timers, or typed arguments.

    Returns:
        A structured result for typed calls, or the historical state tuple.
    """
    if context is not None and state is not None:
        raise TypeError("Pass either context or state, not both")
    target = context if context is not None else state
    if target is None:
        raise TypeError("A tool context or state is required")
    typed_context = target if isinstance(target, ToolContext) else None
    timer_state = t.cast(
        dict[str, object], typed_context.state if typed_context is not None else target
    )
    requested_name = _argument_name(name=name)
    if typed_context is not None and typed_context.cancelled:
        return _cancelled_result(context=typed_context)
    if typed_context is not None and not _valid_name_argument(name=name, optional=True):
        return _invalid_result(context=typed_context)
    timers = _active_timers(state=timer_state)
    if typed_context is not None:
        selected = _select_timers(timers=timers, name=requested_name)
        return ToolResult(
            status=ToolStatus.OK,
            operation_id=typed_context.operation_id,
            data={"timers": [_timer_data(timer=timer) for timer in selected]},
        )

    selected = _select_timers(timers=timers, name=requested_name)
    legacy_state = t.cast(dict[str, object], target)
    return "", dict(legacy_state, running_timers=selected)


def stop_timer(
    context: dict[str, object] | ToolContext | None = None,
    name: str | dict[str, object] | None = None,
    *,
    state: dict[str, object] | None = None,
    duration: str | None = None,
) -> ToolResult | tuple[str, dict[str, object]]:
    """Stop exactly one named timer; never choose a fallback timer.

    Args:
        context:
            A typed tool context or legacy state mapping.
        name:
            The exact timer name, or typed arguments.
        duration (optional):
            Historical alias used by the old duration-based adapter.

    Returns:
        A structured result for typed calls, or the historical state tuple.
    """
    if context is not None and state is not None:
        raise TypeError("Pass either context or state, not both")
    target = context if context is not None else state
    if target is None:
        raise TypeError("A tool context or state is required")
    typed_context = target if isinstance(target, ToolContext) else None
    timer_state = t.cast(
        dict[str, object], typed_context.state if typed_context is not None else target
    )
    requested_name = _argument_name(name=name) if duration is None else duration
    if typed_context is not None and typed_context.cancelled:
        return _cancelled_result(context=typed_context)
    if typed_context is not None and not _valid_name_argument(
        name=name, optional=False
    ):
        return _invalid_result(context=typed_context)
    timers = _active_timers(state=timer_state)

    matching = [
        timer
        for timer in timers
        if _normalise_name(timer.name) == _normalise_name(requested_name or "")
        or (
            typed_context is None
            and isinstance(requested_name, str)
            and _legacy_timer_matches(timer, requested_name)
        )
        or (duration is not None and _legacy_timer_matches(timer, duration))
    ]

    if typed_context is not None:
        if typed_context.cancelled:
            return _cancelled_result(context=typed_context)
        if not isinstance(requested_name, str) or not requested_name.strip():
            return _invalid_result(context=typed_context)
        if not matching:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                operation_id=typed_context.operation_id,
                message_da=f"Der var ingen aktiv timer med navnet {requested_name}.",
                data={"name": requested_name},
            )
        timer = matching[0]
        if not _cancel_and_remove(state=timer_state, timer=timer):
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                operation_id=typed_context.operation_id,
                message_da=f"Der var ingen aktiv timer med navnet {requested_name}.",
                data={"name": requested_name},
            )
        timer.stop()
        return ToolResult(
            status=ToolStatus.OK,
            operation_id=typed_context.operation_id,
            data={"name": timer.name},
        )

    if not isinstance(requested_name, str) or not requested_name.strip():
        return "", dict(timer_state, running_timers=timers)
    if not matching:
        return "", dict(timer_state, running_timers=timers)
    timer = matching[0]
    _cancel_and_remove(state=timer_state, timer=timer)
    return "", dict(timer_state, running_timers=_timers(timer_state))


def _set_typed_timer(
    context: ToolContext,
    arguments: dict[str, object],
    clock: Clock,
    alarm_callback: TimerCallback | None,
) -> ToolResult:
    """Validate and create a timer for the typed runtime."""
    if context.cancelled:
        return _cancelled_result(context=context)
    name = arguments.get("name")
    duration = arguments.get("duration_seconds")
    if not isinstance(name, str) or not name.strip() or not _valid_duration(duration):
        return _invalid_result(context=context)
    return _create_typed_result(
        context=context,
        state=context.state,
        name=name,
        duration_seconds=t.cast(int, duration),
        clock=clock,
        alarm_callback=alarm_callback,
    )


def _create_typed_result(
    context: ToolContext,
    state: dict[str, object],
    name: str,
    duration_seconds: int,
    clock: Clock,
    alarm_callback: TimerCallback | None,
) -> ToolResult:
    """Create a timer and represent the mutation as a tool result."""
    timer = _create_timer(
        state=state,
        name=name,
        duration_seconds=duration_seconds,
        clock=clock,
        alarm_callback=alarm_callback,
    )
    if timer is None:
        return ToolResult(
            status=ToolStatus.CONFLICT,
            operation_id=context.operation_id,
            message_da=f"Der findes allerede en aktiv timer med navnet {name}.",
            data={"name": name},
        )
    return ToolResult(
        status=ToolStatus.OK,
        operation_id=context.operation_id,
        message_da=f"Timeren {timer.name} er sat.",
        data={"timer": _timer_data(timer=timer)},
    )


def _create_timer(
    state: dict[str, object],
    name: str,
    duration_seconds: int,
    clock: Clock,
    alarm_callback: TimerCallback | None,
) -> Timer | None:
    """Add one timer to state unless its normalised name is already active."""
    timers = _active_timers(state=state)
    lock = _state_lock(state=state)
    with lock:
        if any(
            _normalise_name(timer.name) == _normalise_name(name) for timer in timers
        ):
            return None
        callback = alarm_callback or _notification_callback(state=state)
        timer = Timer(
            name=unicodedata.normalize("NFC", name).strip(),
            duration_seconds=duration_seconds,
            clock=clock,
            alarm_callback=callback,
            completion_callback=lambda completed: _complete_timer(
                state=state, timer=completed
            ),
        ).start()
        timers.append(timer)
    return timer


def _active_timers(state: dict[str, object]) -> list[Timer]:
    """Remove expired timers and return the shared active timer list."""
    timers = _timers(state)
    due = [timer for timer in list(timers) if timer.expired]
    for timer in due:
        timer.expire()
    return timers


def _complete_timer(state: dict[str, object], timer: Timer) -> None:
    """Remove a completed timer before its notification is delivered."""
    lock = _state_lock(state=state)
    with lock:
        timers = _timers(state)
        for index, active_timer in enumerate(timers):
            if active_timer is timer:
                del timers[index]
                break


def _cancel_and_remove(state: dict[str, object], timer: Timer) -> bool:
    """Cancel and remove a timer as one state-serialised operation."""
    lock = _state_lock(state=state)
    with lock:
        timers = _timers(state)
        timer_index = next(
            (
                index
                for index, active_timer in enumerate(timers)
                if active_timer is timer
            ),
            None,
        )
        if timer_index is None or not timer._cancel():
            return False
        del timers[timer_index]
    timer._join_worker()
    return True


def _timers(state: dict[str, object]) -> list[Timer]:
    """Return the timer list in state, repairing legacy malformed state."""
    timers = state.get("running_timers")
    if not isinstance(timers, list):
        timers = []
        state["running_timers"] = timers
    invalid_entries = [entry for entry in timers if not isinstance(entry, Timer)]
    for entry in invalid_entries:
        timers.remove(entry)
    return t.cast(list[Timer], timers)


def _select_timers(timers: list[Timer], name: str | None) -> list[Timer]:
    """Select all timers or the one exact normalised name."""
    if name is None:
        return list(timers)
    normalised = _normalise_name(name)
    return [timer for timer in timers if _normalise_name(timer.name) == normalised]


def _timer_data(timer: Timer) -> dict[str, object]:
    """Return the model-safe representation of a timer."""
    return {
        "name": timer.name,
        "duration_seconds": timer.duration_seconds,
        "remaining_seconds": timer.remaining_seconds,
    }


def _notification_callback(state: dict[str, object]) -> TimerCallback | None:
    """Find the notification hook without coupling timers to audio or TTS."""
    for key in (
        "notification_callback",
        "timer_notification_callback",
        "alarm_callback",
    ):
        callback = state.get(key)
        if callable(callback):
            return t.cast(TimerCallback, callback)
    return None


def _state_lock(state: dict[str, object]) -> threading.RLock:
    """Return a stable lock for a state mapping."""
    state_id = id(state)
    with _STATE_LOCKS_GUARD:
        return _STATE_LOCKS.setdefault(state_id, threading.RLock())


def _arguments(
    name: str | int | dict[str, object] | None,
    duration_seconds: int | None,
) -> dict[str, object]:
    """Convert supported typed call forms into an argument mapping."""
    if isinstance(name, dict):
        return name
    return {"name": name, "duration_seconds": duration_seconds}


def _argument_name(name: str | dict[str, object] | None) -> str | None:
    """Extract a name from positional or typed arguments."""
    if isinstance(name, dict):
        value = name.get("name")
        return value if isinstance(value, str) else None
    return name


def _valid_name_argument(
    name: str | dict[str, object] | None,
    optional: bool,
) -> bool:
    """Validate a name or nullable name argument."""
    value: object = name.get("name") if isinstance(name, dict) else name
    if value is None:
        return optional
    return isinstance(value, str)


def _legacy_duration(
    name: str | int | dict[str, object] | None,
    duration_seconds: int | None,
) -> int:
    """Extract the old duration-only call's duration."""
    value: object = duration_seconds if duration_seconds is not None else name
    if not _valid_duration(value):
        raise ValueError("Timer duration must be between 1 and 86400 seconds")
    return t.cast(int, value)


def _legacy_timer_matches(timer: Timer, duration: str) -> bool:
    """Match the historical duration identifier without introducing fallback."""
    return duration in {
        timer.name,
        str(timer.duration),
        str(timer.duration).replace("00:", "0:"),
    }


def _valid_duration(value: object) -> bool:
    """Return whether a duration is an integer in the supported range."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and _MIN_DURATION_SECONDS <= value <= _MAX_DURATION_SECONDS
    )


def _normalise_name(name: str) -> str:
    """Return a Unicode- and case-normalised comparison key."""
    return unicodedata.normalize("NFKC", name).strip().casefold()


def _cancelled_result(context: ToolContext) -> ToolResult:
    """Build the standard cancelled result."""
    return ToolResult(
        status=ToolStatus.CANCELLED,
        operation_id=context.operation_id,
        message_da="Handlingen blev afbrudt.",
    )


def _invalid_result(context: ToolContext) -> ToolResult:
    """Build the standard invalid-request result."""
    return ToolResult(
        status=ToolStatus.INVALID_REQUEST,
        operation_id=context.operation_id,
        message_da="Forespørgslen havde ugyldige argumenter.",
    )
