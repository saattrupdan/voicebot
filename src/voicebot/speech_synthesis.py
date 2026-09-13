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
from pywebrtc_audio import EchoCanceller

logger = logging.getLogger(__name__)

PCM_SAMPLE_RATE = 24_000
PCM_CHANNELS = 1
PCM_CHUNK_BYTES = 3_840
ECHO_SAMPLE_RATE = 16_000
ECHO_READY_CHUNKS = 2
ECHO_MAX_BUFFER_SECONDS = 1


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
        echo_stream_delay_ms: int = 80,
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
            echo_stream_delay_ms (optional):
                Estimated render-to-capture delay for WebRTC AEC. Defaults to 80 ms.
        """
        self.client = client
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self._state_lock = threading.Lock()
        self._serial_lock = threading.Lock()
        self._echo_lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        self._active_response: object | None = None
        self._active_output: object | None = None
        self._far_end_buffer = np.empty(0, dtype=np.int16)
        self._echo_ready_chunks = 0
        self._echo_canceller = EchoCanceller(
            sample_rate=ECHO_SAMPLE_RATE,
            num_channels=PCM_CHANNELS,
            stream_delay_ms=echo_stream_delay_ms,
        )
        self._playing = threading.Event()

    @property
    def is_playing(self) -> bool:
        """Return whether synthesised audio is currently playing."""
        return self._playing.is_set()

    def playback_non_echo_rms(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[float, bool]:
        """Remove queued playback and return remaining RMS plus readiness.

        Args:
            audio:
                Mono microphone samples.
            sample_rate:
                Sample rate of the microphone audio.

        Returns:
            RMS of capture audio which the echo canceller did not attribute to the
            bot's playback, and whether two aligned chunks have been processed.
        """
        near_end = self._resample(
            samples=audio.astype(np.float64),
            source_rate=sample_rate,
            target_rate=ECHO_SAMPLE_RATE,
        )
        near_end = np.clip(near_end, -32_768, 32_767).astype(np.int16)
        with self._echo_lock:
            if self._far_end_buffer.size < near_end.size:
                self._reset_echo_canceller()
                near_end_float = near_end.astype(np.float64)
                rms = (
                    float(np.sqrt(np.mean(np.square(near_end_float))))
                    if near_end_float.size
                    else 0.0
                )
                return rms, False

            far_end = self._far_end_buffer[: near_end.size]
            self._far_end_buffer = self._far_end_buffer[near_end.size :]
            cleaned = np.asarray(
                self._echo_canceller.process(near_end, far_end), dtype=np.int16
            )
            self._echo_ready_chunks += 1
            ready = self._echo_ready_chunks >= ECHO_READY_CHUNKS
        cleaned_float = cleaned.astype(np.float64)
        rms = (
            float(np.sqrt(np.mean(np.square(cleaned_float))))
            if cleaned_float.size
            else 0.0
        )
        return rms, ready

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
            with self._echo_lock:
                self._reset_echo_canceller()
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
                            output_audio = audio[:complete_bytes]
                            self._queue_playback_reference(
                                audio=output_audio,
                                sample_rate=self.sample_rate,
                                channels=PCM_CHANNELS,
                            )
                            self._playing.set()
                            output.write(output_audio)
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
                    self._queue_playback_reference(
                        audio=audio,
                        sample_rate=wav_file.getframerate(),
                        channels=wav_file.getnchannels(),
                    )
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

    def _queue_playback_reference(
        self, audio: bytes, sample_rate: int, channels: int
    ) -> None:
        """Queue chronological far-end audio for WebRTC echo cancellation."""
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float64)
        if channels > 1:
            complete_samples = samples.size - samples.size % channels
            samples = samples[:complete_samples].reshape(-1, channels).mean(axis=1)
        samples = self._resample(
            samples=samples, source_rate=sample_rate, target_rate=ECHO_SAMPLE_RATE
        )
        reference = np.clip(samples, -32_768, 32_767).astype(np.int16)
        max_samples = ECHO_SAMPLE_RATE * ECHO_MAX_BUFFER_SECONDS
        reference_was_truncated = reference.size > max_samples
        reference = reference[-max_samples:]
        with self._echo_lock:
            if (
                reference_was_truncated
                or self._far_end_buffer.size + reference.size > max_samples
            ):
                self._reset_echo_canceller()
            self._far_end_buffer = np.concatenate((self._far_end_buffer, reference))

    def _reset_echo_canceller(self) -> None:
        """Reset AEC state and queued render audio after a discontinuity."""
        self._echo_canceller.reset()
        self._far_end_buffer = np.empty(0, dtype=np.int16)
        self._echo_ready_chunks = 0

    @staticmethod
    def _resample(
        samples: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        """Linearly resample mono audio while preserving its duration."""
        if samples.size == 0 or source_rate == target_rate:
            return samples
        target_size = max(1, round(samples.size * target_rate / source_rate))
        source_positions = np.arange(samples.size, dtype=np.float64)
        target_positions = np.arange(target_size, dtype=np.float64) * (
            source_rate / target_rate
        )
        return np.interp(target_positions, source_positions, samples)

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
