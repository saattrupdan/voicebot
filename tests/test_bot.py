"""Tests for voicebot response orchestration."""

import datetime as dt
import threading
from collections.abc import Callable
from unittest.mock import MagicMock, call

import pytest

from voicebot import bot as bot_module
from voicebot.speech_synthesis import PlaybackOutcome
from voicebot.text_engine import TurnAction, TurnResult


def test_response_worker_speaks_streamed_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response worker speaks each model segment and signals completion."""
    voicebot = bot_module.VoiceBot.__new__(bot_module.VoiceBot)
    voicebot.synthesiser = MagicMock()
    voicebot.text_engine = MagicMock()

    def generate_response(
        prompt: str,
        last_response_time: dt.datetime,
        current_response_time: dt.datetime,
        on_segment: Callable[[str], None],
        cancel_event: threading.Event,
    ) -> TurnResult:
        del prompt, last_response_time, current_response_time, cancel_event
        on_segment("Første sætning.")
        on_segment("Anden sætning.")
        return TurnResult(action=TurnAction.RESPOND, text="Hele svaret.")

    voicebot.text_engine.generate_response.side_effect = generate_response
    synthesise = MagicMock()
    monkeypatch.setattr(bot_module, "synthesise_speech", synthesise)
    cancel = threading.Event()
    done = threading.Event()
    state = bot_module._ResponseState()

    voicebot._respond(
        prompt="Fortæl noget.",
        last_response_time=dt.datetime(1900, 1, 1),
        current_response_time=dt.datetime.now(),
        cancel_event=cancel,
        done_event=done,
        state=state,
    )

    assert done.is_set()
    assert state.result == TurnResult(action=TurnAction.RESPOND, text="Hele svaret.")
    assert synthesise.call_args_list == [
        call(
            text="Første sætning.",
            synthesiser=voicebot.synthesiser,
            cancel_event=cancel,
        ),
        call(
            text="Anden sætning.", synthesiser=voicebot.synthesiser, cancel_event=cancel
        ),
    ]


def test_reminder_requires_complete_playback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelled reminder playback remains available for a later retry."""
    voicebot = bot_module.VoiceBot.__new__(bot_module.VoiceBot)
    voicebot.synthesiser = MagicMock()
    notification = type("Notification", (), {"payload": {"message": "Påmindelse"}})()
    cancel_event = threading.Event()
    monkeypatch.setattr(
        bot_module,
        "synthesise_speech",
        MagicMock(return_value=PlaybackOutcome.CANCELLED),
    )

    outcome = voicebot._deliver_notification(notification, cancel_event)

    assert outcome == bot_module.DeliveryOutcome.retry()


def test_timer_alarm_still_runs_through_bot_synthesiser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timer alarms remain automatic and expose their playback outcome."""
    voicebot = bot_module.VoiceBot.__new__(bot_module.VoiceBot)
    voicebot.synthesiser = MagicMock()
    voicebot._timer_alarm_events = {}
    voicebot._timer_alarm_lock = threading.Lock()
    monkeypatch.setattr(
        bot_module,
        "synthesise_speech",
        MagicMock(return_value=PlaybackOutcome.COMPLETE),
    )
    timer = type("Timer", (), {"name": "ovn"})()

    voicebot._deliver_timer_alarm(timer)

    assert voicebot._timer_alarm_events == {}


def test_response_worker_propagates_failure_state() -> None:
    """Worker failures are retained for the main loop and always signal completion."""
    voicebot = bot_module.VoiceBot.__new__(bot_module.VoiceBot)
    voicebot.synthesiser = MagicMock()
    voicebot.text_engine = MagicMock()
    error = RuntimeError("model unavailable")
    voicebot.text_engine.generate_response.side_effect = error
    done = threading.Event()
    state = bot_module._ResponseState()

    voicebot._respond(
        prompt="Fortæl noget.",
        last_response_time=dt.datetime(1900, 1, 1),
        current_response_time=dt.datetime.now(),
        cancel_event=threading.Event(),
        done_event=done,
        state=state,
    )

    assert done.is_set()
    assert state.error is error
