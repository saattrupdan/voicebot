"""A voice bot."""

import datetime as dt
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import openai
import openwakeword as oww
from dotenv import load_dotenv
from omegaconf import DictConfig
from openwakeword.utils import download_models as download_wakeword_models

from .notifications import DeliveryOutcome
from .runtime import build_integration_runtime
from .speech_recognition import transcribe_speech
from .speech_recording import create_voice_activity_detector, record_speech
from .speech_synthesis import PlaybackOutcome, SpeechSynthesiser, synthesise_speech
from .text_engine import TextEngine, TurnAction, TurnResult

load_dotenv()
logger = logging.getLogger(__name__)

_IDLE_TIME = dt.datetime(year=1900, month=1, day=1)


@dataclass
class _ResponseState:
    """Result or failure produced by an assistant-response worker."""

    result: TurnResult | None = None
    error: BaseException | None = None


class VoiceBot:
    """A voice bot."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialise the bot.

        Args:
            cfg:
                The Hydra configuration.
        """
        self.cfg = cfg
        self.speech_detector = create_voice_activity_detector(cfg=self.cfg)

        logger.info("Loading the wake word model...")
        ort.set_default_logger_severity(3)
        download_wakeword_models(model_names=["hey_jarvis"], inference_framework="onnx")
        self.wake_word_model = oww.Model(
            wakeword_models=["hey_jarvis"], inference_framework="onnx"
        )

        logger.info("Configuring the SYV audio client...")
        self.syv_client = openai.OpenAI(
            api_key=os.environ["SYV_API_KEY"], base_url=self.cfg.syv_server
        )
        self.synthesiser = SpeechSynthesiser(
            client=self.syv_client,
            model=self.cfg.tts_model_id,
            voice=self.cfg.tts_voice,
            sample_rate=int(self.cfg.get("tts_sample_rate", 24_000)),
            echo_stream_delay_ms=int(self.cfg.get("barge_in_echo_delay_ms", 80)),
        )

        logger.info("Configuring local integrations...")
        self._timer_alarm_events: dict[str, threading.Event] = {}
        self._timer_alarm_lock = threading.Lock()
        self.integration_runtime = build_integration_runtime(
            self.cfg,
            notification_callback=self._deliver_notification,
            timer_callback=self._deliver_timer_alarm,
        )

        logger.info("Loading the text engine model...")
        self.text_engine = TextEngine(
            cfg=self.cfg,
            runtime=self.integration_runtime.registry,
            state=self.integration_runtime.state,
        )
        self.text_engine.state["synthesiser"] = self.synthesiser
        default_location = str(self.cfg.get("weather_default_location", "")).strip()
        if default_location:
            self.text_engine.state["weather_default_location"] = default_location

    def run(self) -> None:
        """Run the bot and its durable integration workers."""
        last_response_time = _IDLE_TIME
        self.integration_runtime.start()
        try:
            logger.info("Playing welcome message...")
            synthesise_speech(
                text=self.cfg.starting_phrase, synthesiser=self.synthesiser
            )
            while True:
                speech, audio_start = self._record(
                    last_response_time=last_response_time
                )
                while audio_start is not None:
                    text = self._transcribe(speech=speech)
                    if not text:
                        break

                    response_done = threading.Event()
                    cancel_response = threading.Event()
                    response_state = _ResponseState()
                    response_thread = threading.Thread(
                        target=self._respond,
                        kwargs={
                            "prompt": text,
                            "last_response_time": last_response_time,
                            "current_response_time": audio_start,
                            "cancel_event": cancel_response,
                            "done_event": response_done,
                            "state": response_state,
                        },
                        name="voicebot-response",
                    )
                    response_thread.start()

                    def interrupt() -> None:
                        cancel_response.set()
                        for (
                            notification_id
                        ) in self.integration_runtime.dispatcher.active_ids:
                            self.integration_runtime.dispatcher.cancel(notification_id)
                        self.synthesiser.stop()
                        with self._timer_alarm_lock:
                            for event in self._timer_alarm_events.values():
                                event.set()

                    next_speech, next_audio_start = self._record(
                        last_response_time=last_response_time,
                        force_follow_up=True,
                        stop_event=response_done,
                        on_interrupt=interrupt,
                    )
                    response_thread.join()
                    if response_state.error is not None:
                        raise response_state.error

                    if next_audio_start is not None:
                        last_response_time = dt.datetime.now()
                        speech, audio_start = next_speech, next_audio_start
                        continue

                    result = response_state.result
                    if result is not None and result.action is TurnAction.END:
                        last_response_time = _IDLE_TIME
                    elif result is not None and result.action is TurnAction.RESPOND:
                        last_response_time = dt.datetime.now()
                    break
        finally:
            with self._timer_alarm_lock:
                for event in self._timer_alarm_events.values():
                    event.set()
            self.integration_runtime.stop()

    def _deliver_notification(
        self, notification: object, cancel_event: threading.Event
    ) -> DeliveryOutcome:
        """Speak one durable reminder through the bot-owned audio path."""
        payload = getattr(notification, "payload", {})
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, str) or not message.strip():
            return DeliveryOutcome.dismissed()
        playback = synthesise_speech(
            text=message, synthesiser=self.synthesiser, cancel_event=cancel_event
        )
        if playback is PlaybackOutcome.COMPLETE:
            return DeliveryOutcome.delivered()
        return DeliveryOutcome.retry()

    def _deliver_timer_alarm(self, timer: object) -> None:
        """Speak a named timer alarm, allowing barge-in to dismiss it."""
        name = str(getattr(timer, "name", "timer"))
        event = threading.Event()
        with self._timer_alarm_lock:
            self._timer_alarm_events[name] = event
        try:
            synthesise_speech(
                text=f"Timeren {name} er færdig.",
                synthesiser=self.synthesiser,
                cancel_event=event,
            )
        finally:
            with self._timer_alarm_lock:
                self._timer_alarm_events.pop(name, None)

    def _record(
        self,
        last_response_time: dt.datetime,
        force_follow_up: bool = False,
        stop_event: threading.Event | None = None,
        on_interrupt: Callable[[], None] | None = None,
    ) -> tuple[np.ndarray, dt.datetime | None]:
        """Record one utterance through the configured recorder."""
        return record_speech(
            last_response_time=last_response_time,
            detector=self.speech_detector,
            cfg=self.cfg,
            synthesiser=self.synthesiser,
            wake_word_model=self.wake_word_model,
            force_follow_up=force_follow_up,
            stop_event=stop_event,
            on_interrupt=on_interrupt,
        )

    def _transcribe(self, speech: np.ndarray) -> str:
        """Transcribe one recorded utterance."""
        return transcribe_speech(
            speech=speech,
            client=self.syv_client,
            model=self.cfg.asr_model_id,
            language=self.cfg.asr_language,
            manual_fixes=self.cfg.manual_fixes,
        )

    def _respond(
        self,
        prompt: str,
        last_response_time: dt.datetime,
        current_response_time: dt.datetime,
        cancel_event: threading.Event,
        done_event: threading.Event,
        state: _ResponseState,
    ) -> None:
        """Generate and speak one response in a cancellable worker."""
        try:
            state.result = self.text_engine.generate_response(
                prompt=prompt,
                last_response_time=last_response_time,
                current_response_time=current_response_time,
                on_segment=lambda segment: self._speak_segment(
                    segment=segment, cancel_event=cancel_event
                ),
                cancel_event=cancel_event,
            )
        except BaseException as error:
            state.error = error
        finally:
            done_event.set()

    def _speak_segment(self, segment: str, cancel_event: threading.Event) -> None:
        """Speak one generated segment while discarding its playback result."""
        synthesise_speech(
            text=segment, synthesiser=self.synthesiser, cancel_event=cancel_event
        )
