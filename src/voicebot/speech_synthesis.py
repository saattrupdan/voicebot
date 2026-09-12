"""Generation of Danish speech."""

import enum
import io
import logging
import threading
import wave
from pathlib import Path

import numpy as np
import openai
import sounddevice

logger = logging.getLogger(__name__)

PCM_SAMPLE_RATE = 24_000
PCM_CHANNELS = 1
PCM_CHUNK_BYTES = 4096


class PlaybackOutcome(enum.StrEnum):
    """Outcome of one synthesis and playback attempt."""

    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SpeechSynthesiser:
    """A cancellable speech synthesiser backed by the SYV audio API."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        voice: str,
        sample_rate: int = PCM_SAMPLE_RATE,
    ) -> None:
        """Initialise the speech synthesiser.

        Args:
            client:
                OpenAI-compatible client configured for the SYV API.
            model:
                Speech synthesis model identifier.
            voice:
                Voice identifier supported by the model.
            sample_rate (optional):
                PCM output sample rate. Defaults to 24 kHz, Plapre's native rate.
        """
        self.client = client
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self._state_lock = threading.Lock()
        self._serial_lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        self._active_response: object | None = None
        self._active_output: object | None = None
        self._playing = threading.Event()

    @property
    def is_playing(self) -> bool:
        """Return whether synthesised audio is currently playing."""
        return self._playing.is_set()

    def synthesise(
        self, text: str, cancel_event: threading.Event | None = None
    ) -> PlaybackOutcome:
        """Generate and play speech.

        Args:
            text:
                Text to synthesise.
            cancel_event (optional):
                Shared response-cancellation event. Defaults to None.
        """
        if cancel_event is not None and cancel_event.is_set():
            return PlaybackOutcome.CANCELLED

        local_cancel = threading.Event()
        with self._serial_lock:
            with self._state_lock:
                self._active_cancel = local_cancel
            try:
                try:
                    return self._synthesise_streaming(
                        text=text, local_cancel=local_cancel, cancel_event=cancel_event
                    )
                except Exception:
                    if self._is_cancelled(
                        local_cancel=local_cancel, cancel_event=cancel_event
                    ):
                        return PlaybackOutcome.CANCELLED
                    logger.error("Speech synthesis failed during playback")
                    return PlaybackOutcome.FAILED
            finally:
                self._playing.clear()
                with self._state_lock:
                    self._active_cancel = None
                    self._active_response = None
                    self._active_output = None

    def stop(self) -> None:
        """Stop active generation and playback as soon as possible."""
        with self._state_lock:
            cancel = self._active_cancel
            response = self._active_response
            output = self._active_output
        if cancel is not None:
            cancel.set()
        self._close_resource(resource=response, method="close")
        self._close_resource(resource=output, method="abort")

    def _synthesise_streaming(
        self,
        text: str,
        local_cancel: threading.Event,
        cancel_event: threading.Event | None,
    ) -> PlaybackOutcome:
        """Stream raw Plapre PCM, with a WAV fallback before playback starts."""
        emitted_audio = False
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model, input=text, voice=self.voice, response_format="pcm"
            ) as response:
                with self._state_lock:
                    self._active_response = response
                output = sounddevice.RawOutputStream(
                    samplerate=self.sample_rate, channels=PCM_CHANNELS, dtype="int16"
                )
                with self._state_lock:
                    self._active_output = output
                output.start()
                try:
                    remainder = b""
                    for chunk in response.iter_bytes(chunk_size=PCM_CHUNK_BYTES):
                        if self._is_cancelled(
                            local_cancel=local_cancel, cancel_event=cancel_event
                        ):
                            return PlaybackOutcome.CANCELLED
                        audio = remainder + chunk
                        complete_bytes = len(audio) - len(audio) % 2
                        remainder = audio[complete_bytes:]
                        if complete_bytes:
                            self._playing.set()
                            output.write(audio[:complete_bytes])
                            emitted_audio = True
                finally:
                    if self._is_cancelled(
                        local_cancel=local_cancel, cancel_event=cancel_event
                    ):
                        self._close_resource(resource=output, method="abort")
                    else:
                        self._close_resource(resource=output, method="stop")
                    self._close_resource(resource=output, method="close")
        except Exception as error:
            cancelled = self._is_cancelled(
                local_cancel=local_cancel, cancel_event=cancel_event
            )
            if cancelled:
                return PlaybackOutcome.CANCELLED
            if emitted_audio:
                logger.error("Streaming speech failed after playback began")
                return PlaybackOutcome.FAILED
            if not isinstance(error, (AttributeError, TypeError, openai.APIError)):
                return PlaybackOutcome.FAILED
            logger.warning("Streaming speech unavailable; falling back to WAV")
            return self._synthesise_wav(
                text=text, local_cancel=local_cancel, cancel_event=cancel_event
            )

        if self._is_cancelled(local_cancel=local_cancel, cancel_event=cancel_event):
            return PlaybackOutcome.CANCELLED
        return PlaybackOutcome.COMPLETE

    def _synthesise_wav(
        self,
        text: str,
        local_cancel: threading.Event,
        cancel_event: threading.Event | None,
    ) -> PlaybackOutcome:
        """Generate a complete WAV response and play it in cancellable chunks."""
        response = self.client.audio.speech.create(
            model=self.model, input=text, voice=self.voice, response_format="wav"
        )
        if self._is_cancelled(local_cancel=local_cancel, cancel_event=cancel_event):
            return PlaybackOutcome.CANCELLED

        with wave.open(io.BytesIO(response.content), "rb") as wav_file:
            if wav_file.getsampwidth() != 2:
                raise ValueError("Only 16-bit WAV audio is supported.")
            output = sounddevice.RawOutputStream(
                samplerate=wav_file.getframerate(),
                channels=wav_file.getnchannels(),
                dtype="int16",
            )
            with self._state_lock:
                self._active_output = output
            output.start()
            try:
                while not self._is_cancelled(
                    local_cancel=local_cancel, cancel_event=cancel_event
                ):
                    audio = wav_file.readframes(PCM_CHUNK_BYTES // 2)
                    if not audio:
                        break
                    self._playing.set()
                    output.write(audio)
            finally:
                if self._is_cancelled(
                    local_cancel=local_cancel, cancel_event=cancel_event
                ):
                    self._close_resource(resource=output, method="abort")
                else:
                    self._close_resource(resource=output, method="stop")
                self._close_resource(resource=output, method="close")

        if self._is_cancelled(local_cancel=local_cancel, cancel_event=cancel_event):
            return PlaybackOutcome.CANCELLED
        return PlaybackOutcome.COMPLETE

    @staticmethod
    def _is_cancelled(
        local_cancel: threading.Event, cancel_event: threading.Event | None
    ) -> bool:
        """Return whether local or shared cancellation was requested."""
        return local_cancel.is_set() or (
            cancel_event is not None and cancel_event.is_set()
        )

    @staticmethod
    def _close_resource(resource: object | None, method: str) -> None:
        """Best-effort close or abort of an active streaming resource."""
        if resource is None:
            return
        operation = getattr(resource, method, None)
        if callable(operation):
            try:
                operation()
            except Exception:
                logger.debug(
                    "Could not operate on active speech resource method=%s", method
                )


def synthesise_speech(
    text: str,
    synthesiser: SpeechSynthesiser | None,
    cancel_event: threading.Event | None = None,
) -> PlaybackOutcome:
    """Synthesise and play speech.

    Args:
        text:
            Text to synthesise.
        synthesiser:
            Speech synthesiser to use, or None to produce no speech.
        cancel_event (optional):
            Shared response-cancellation event. Defaults to None.
    """
    if synthesiser is None:
        return PlaybackOutcome.COMPLETE
    return synthesiser.synthesise(text=text, cancel_event=cancel_event)


def play_sound(path: str | Path) -> None:
    """Play a WAV file.

    Args:
        path:
            Path to the WAV file.
    """
    path = Path(path)
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Unknown file extension: {path.suffix!r}")

    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError("Only 16-bit WAV audio is supported.")
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels)
    sounddevice.play(data=audio, samplerate=sample_rate)
    sounddevice.wait()
