"""Timer tool."""

import datetime as dt
import logging
import multiprocessing as mp
from time import sleep
from typing import Literal

from pydantic import BaseModel

from ..speech_synthesis import synthesise_speech

logger = logging.getLogger(__name__)


### Setting a timer ###


def set_timer(duration_seconds: int, state: dict) -> tuple[str, dict]:
    """Set a timer for the given duration.

    Args:
        duration_seconds:
            The duration of the timer in seconds.
        state:
            The current state of the text engine.

    Returns:
        A pair (message, state) where message is a message indicating the timer
        was set and state is information that the text engine should store.
    """
    running_timers = state.get("running_timers", [])
    timer = Timer(duration_seconds=duration_seconds).start()
    running_timers.append(timer)
    return (
        f"Startet timer med varighed {timer.duration}.",
        dict(running_timers=[timer for timer in running_timers]),
    )


class SetTimerParameters(BaseModel):
    """The parameters for setting a timer."""

    duration_seconds: int


class SetTimerResponse(BaseModel):
    """A response containing the timer duration."""

    name: Literal["set_timer"]
    parameters: SetTimerParameters


### Stopping a timer ###


def stop_timer(duration: str | None, state: dict) -> tuple[str, dict]:
    """Stop a timer.

    Args:
        duration:
            The duration of the timer to stop. If None, stop the timer with the
            shortest duration.
        state:
            The current state of the text engine.

    Returns:
        A pair (message, state) where message is a message indicating the timer
        was stopped and state is information that the text engine should store.
    """
    running_timers = state.get("running_timers", [])

    if not running_timers:
        return ("Ingen kørende timere.", dict(running_timers=[]))

    timer_to_stop: Timer
    if duration is None:
        timer_to_stop = min(running_timers, key=lambda t: t.duration)
    else:
        valid_timers = [
            timer
            for timer in running_timers
            if str(timer.duration).replace("00:", "0:") == duration.replace("00:", "0:")
        ]
        if not valid_timers:
            return (
                f"Ingen timer med varighed {duration}.",
                dict(running_timers=[timer for timer in running_timers]),
            )
        timer_to_stop = valid_timers[0]

    timer_to_stop.stop()
    running_timers.remove(timer_to_stop)
    return (
        f"Timer med varighed {duration} stoppet.",
        dict(running_timers=[timer for timer in running_timers]),
    )


class StopTimerParameters(BaseModel):
    """The parameters for stopping a timer."""

    duration: str | None


class StopTimerResponse(BaseModel):
    """A response containing the stopped timer duration."""

    name: Literal["stop_timer"]
    parameters: StopTimerParameters


### List timers ###


def list_timers(state: dict) -> tuple[str, dict]:
    """List the running timers.

    Args:
        state:
            The current state of the text engine.

    Returns:
        A pair (message, state) where message is a message listing the running
        timers and state is information that the text engine should store.
    """
    running_timers = state.get("running_timers", [])
    if not running_timers:
        return "Ingen kørende timere.", state

    timers_info = ", ".join(
        f"timer med varighed {timer.duration} ({timer.remaining} tilbage)"
        for timer in running_timers
    )
    noun = "timer" if len(running_timers) == 1 else "timere"
    return f"Der kører {len(running_timers)} {noun}: {timers_info}", state


class ListTimersParameters(BaseModel):
    """The parameters for listing timers."""

    pass


class ListTimersResponse(BaseModel):
    """A response containing the list of running timers."""

    name: Literal["list_timers"]
    parameters: ListTimersParameters


### Timer class ###


class Timer:
    """A timer."""

    def __init__(self, duration_seconds: int) -> None:
        """Initialise the timer.

        Args:
            duration_seconds:
                The duration of the timer in seconds.
        """
        self.duration: dt.timedelta = dt.timedelta(seconds=duration_seconds)
        self.process = mp.Process(target=self._run_timer, args=(duration_seconds,))
        self.start_time: dt.datetime | None = None

    def start(self) -> "Timer":
        """Start the timer."""
        self.process.start()
        self.start_time = dt.datetime.now()
        return self

    def stop(self) -> "Timer":
        """Stop the timer."""
        self.process.terminate()
        return self

    @property
    def remaining(self) -> dt.timedelta:
        """Return the remaining seconds of the timer."""
        if self.start_time is None:
            remaining_seconds = 0
        else:
            remaining = self.start_time + self.duration - dt.datetime.now()
            remaining_seconds = max(int(remaining.total_seconds()), 0)
        return dt.timedelta(seconds=remaining_seconds)

    def __repr__(self) -> str:
        """Return the representation of the timer."""
        out = f"Timer(duration={self.duration}"
        if self.start_time is not None:
            out += f", remaining={self.remaining}"
        return out + ")"

    @staticmethod
    def _run_timer(duration_seconds: int) -> None:
        """Run a timer for the given duration.

        Args:
            duration_seconds:
                The duration to wait in seconds.
        """
        sleep(duration_seconds)
        logging.info("Timer finished! Announcing it...")
        synthesise_speech("Beep beep. Beep beep. Tiden er gået!")
