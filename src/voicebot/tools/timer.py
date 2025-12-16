"""Timer tool."""

import datetime as dt
import logging
import multiprocessing as mp
from time import sleep
from typing import Literal

import chime
from pydantic import BaseModel

from ..speech_synthesis import synthesise_speech

logger = logging.getLogger(__name__)


### Setting a timer ###


def set_timer(duration_seconds: int, state: dict) -> tuple[Literal[""], dict]:
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
    synthesise_speech(
        text=f"Startet timer på {timer.pretty_duration}.",
        synthesiser=state.get("synthesiser"),
    )
    return ("", dict(running_timers=[timer for timer in running_timers]))


class SetTimerParameters(BaseModel):
    """The parameters for setting a timer."""

    duration_seconds: int


class SetTimerResponse(BaseModel):
    """A response containing the timer duration."""

    name: Literal["set_timer"]
    parameters: SetTimerParameters


### Stopping a timer ###


def stop_timer(duration: str | None, state: dict) -> tuple[Literal[""], dict]:
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
        logger.info("No running timers to stop.")
        synthesise_speech(
            text="Der er ingen kørende timere.", synthesiser=state.get("synthesiser")
        )
        return ("", dict(running_timers=[]))

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
            logger.info(f"No timer found with duration {duration}.")
            synthesise_speech(
                text=f"Der var ingen timer med varighed {duration}.",
                synthesiser=state.get("synthesiser"),
            )
            return ("", dict(running_timers=[timer for timer in running_timers]))
        timer_to_stop = valid_timers[0]

    timer_to_stop.stop()
    running_timers.remove(timer_to_stop)

    logger.info(f"Stopped timer: {timer_to_stop!r}")
    synthesise_speech(
        text=f"Stoppet timer på {timer_to_stop.pretty_duration}.",
        synthesiser=state.get("synthesiser"),
    )
    return ("", dict(running_timers=[timer for timer in running_timers]))


class StopTimerParameters(BaseModel):
    """The parameters for stopping a timer."""

    duration: str | None


class StopTimerResponse(BaseModel):
    """A response containing the stopped timer duration."""

    name: Literal["stop_timer"]
    parameters: StopTimerParameters


### List timers ###


def list_timers(state: dict) -> tuple[Literal[""], dict]:
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
        logger.info("No running timers to list.")
        synthesise_speech(
            text="Der er ingen kørende timere.", synthesiser=state.get("synthesiser")
        )
        return "", state

    timers_info = ", ".join(
        f"timer på {timer.pretty_duration} ({timer.pretty_remaining} tilbage)"
        for timer in running_timers
    )
    noun = "timer" if len(running_timers) == 1 else "timere"

    logger.info(f"Listing running timers: {timers_info}")
    synthesise_speech(
        text=f"Der kører {len(running_timers)} {noun}: {timers_info}",
        synthesiser=state.get("synthesiser"),
    )
    return "", state


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

    @property
    def pretty_duration(self) -> str:
        """Return the pretty duration of the timer."""
        return self.prettify_timedelta(timedelta=self.duration)

    @property
    def pretty_remaining(self) -> str:
        """Return the pretty remaining duration of the timer."""
        return self.prettify_timedelta(timedelta=self.remaining)

    def __repr__(self) -> str:
        """Return the representation of the timer."""
        out = f"Timer(duration={self.duration}"
        if self.start_time is not None:
            out += f", remaining={self.remaining}"
        return out + ")"

    @staticmethod
    def prettify_timedelta(timedelta: dt.timedelta) -> str:
        """Prettify a timedelta.

        Args:
            timedelta:
                The timedelta to prettify.

        Returns:
            The prettified timedelta.
        """
        hours = timedelta.seconds // 3600
        minutes = (timedelta.seconds % 3600) // 60
        seconds = timedelta.seconds % 60
        time_strings: list[str] = []
        if hours > 0:
            time_strings.append(f"{hours} timer")
        if minutes > 0:
            time_strings.append(f"{minutes} minutter")
        if seconds > 0:
            time_strings.append(f"{seconds} sekunder")
        return (
            ", ".join(time_strings[:-1]) + " og " + time_strings[-1]
            if len(time_strings) > 1
            else time_strings[0]
        )

    @staticmethod
    def _run_timer(duration_seconds: int) -> None:
        """Run a timer for the given duration.

        Args:
            duration_seconds:
                The duration to wait in seconds.
        """
        sleep(duration_seconds)
        logging.info("Timer finished! Announcing it...")
        while True:
            chime.theme("material")
            chime.info()
            sleep(3)
