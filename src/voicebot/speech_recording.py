"""Recording speech with adaptive voice activity detection."""

import datetime as dt
import logging
import math
import threading
import typing as t
from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import openwakeword as oww
import sounddevice
import webrtcvad
from omegaconf import DictConfig
from pvrecorder import PvRecorder

from .speech_synthesis import SpeechSynthesiser, synthesise_speech

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_FRAME_DURATION_MS = 20


class _Vad(t.Protocol):
    def is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        """Classify one valid PCM frame."""
        ...


@dataclass(frozen=True)
class VoiceActivity:
    """The result of processing one recorder chunk.

    ``onset_sample_offset`` is relative to the start of the processed chunk. It can
    be negative when the consecutive candidate run began in an earlier chunk.
    ``candidate_sample_offset`` has the same meaning for an onset which is still
    pending confirmation.
    """

    speech_detected: bool
    onset: bool
    ended: bool
    onset_sample_offset: int | None = None
    candidate_sample_offset: int | None = None


class AdaptiveVoiceActivityDetector:
    """Track a changing noise floor and detect speech with WebRTC VAD.

    The detector deliberately updates its noise floor only for frames which WebRTC
    classifies as non-speech.  This prevents a loud utterance from becoming the new
    baseline while still allowing a sustained fan or air-conditioning noise to be
    learnt over time.

    Args:
        vad_mode:
            WebRTC VAD aggressiveness, from 0 (least aggressive) to 3.
        initial_noise_floor:
            Starting RMS amplitude in signed-int16 units.
        noise_floor_min:
            Lower bound for the learned RMS amplitude.
        noise_floor_max:
            Upper bound for the learned RMS amplitude.
        noise_floor_rise_rate:
            Adaptation rate when the measured background is louder.
        noise_floor_fall_rate:
            Adaptation rate when the measured background is quieter.
        speech_start_snr:
            Minimum RMS-to-floor ratio for onset.
        speech_end_snr:
            Minimum RMS-to-floor ratio while recording.
        onset_frames:
            Consecutive 20 ms frames required to confirm onset.
        max_silence_frames:
            Consecutive non-speech frames required to end a recording.
        sample_rate:
            PCM sample rate. WebRTC supports only a small set of rates.
        frame_duration_ms:
            WebRTC frame duration. The recorder chunks are split into these frames.
        pre_roll_seconds:
            Amount of audio retained by the recorder before onset confirmation.
        vad:
            Optional WebRTC-compatible object, useful for deterministic tests.

    Raises:
        ValueError:
            If a setting is outside the supported range or inconsistent.
    """

    def __init__(
        self,
        *,
        vad_mode: int = 2,
        initial_noise_floor: float = 300.0,
        noise_floor_min: float = 40.0,
        noise_floor_max: float = 8_000.0,
        noise_floor_rise_rate: float = 0.08,
        noise_floor_fall_rate: float = 0.01,
        speech_start_snr: float = 2.5,
        speech_end_snr: float = 1.4,
        onset_frames: int = 2,
        max_silence_frames: int = 100,
        sample_rate: int = SAMPLE_RATE,
        frame_duration_ms: int = _FRAME_DURATION_MS,
        pre_roll_seconds: float = 0.16,
        vad: _Vad | None = None,
    ) -> None:
        """Initialise the detector and validate its speech boundaries."""
        if vad_mode not in range(4):
            raise ValueError("vad_mode must be between 0 and 3")
        if sample_rate not in {8_000, 16_000, 32_000, 48_000}:
            raise ValueError("sample_rate is not supported by WebRTC VAD")
        if frame_duration_ms not in {10, 20, 30}:
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        finite_settings = (
            initial_noise_floor,
            noise_floor_min,
            noise_floor_max,
            noise_floor_rise_rate,
            noise_floor_fall_rate,
            speech_start_snr,
            speech_end_snr,
            pre_roll_seconds,
        )
        if not all(math.isfinite(value) for value in finite_settings):
            raise ValueError("detector settings must be finite")
        if (
            noise_floor_min <= 0
            or noise_floor_min > noise_floor_max
            or noise_floor_max > 32_768
        ):
            raise ValueError("noise floor bounds are invalid")
        if not noise_floor_min <= initial_noise_floor <= noise_floor_max:
            raise ValueError(
                "initial_noise_floor must be within the noise floor bounds"
            )
        if not 0 < noise_floor_rise_rate <= 1:
            raise ValueError("noise_floor_rise_rate must be greater than zero")
        if not 0 < noise_floor_fall_rate <= 1:
            raise ValueError("noise_floor_fall_rate must be greater than zero")
        if speech_end_snr < 1 or speech_end_snr >= speech_start_snr:
            raise ValueError(
                "speech_end_snr must be at least 1 and below speech_start_snr"
            )
        if onset_frames < 1 or max_silence_frames < 1:
            raise ValueError("frame counts must be positive")
        if pre_roll_seconds < 0:
            raise ValueError("pre_roll_seconds cannot be negative")

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.samples_per_frame = sample_rate * frame_duration_ms // 1_000
        self.noise_floor_min = noise_floor_min
        self.noise_floor_max = noise_floor_max
        self.noise_floor_rise_rate = noise_floor_rise_rate
        self.noise_floor_fall_rate = noise_floor_fall_rate
        self.speech_start_snr = speech_start_snr
        self.speech_end_snr = speech_end_snr
        self.onset_frames = onset_frames
        self.max_silence_frames = max_silence_frames
        self.pre_roll_seconds = pre_roll_seconds
        self._noise_floor = float(initial_noise_floor)
        self._onset_run = 0
        self._onset_start_offset: int | None = None
        self._silence_run = 0
        self._active = False
        self._vad = vad if vad is not None else webrtcvad.Vad(vad_mode)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "AdaptiveVoiceActivityDetector":
        """Build a detector from the speech recording configuration.

        Args:
            cfg:
                Hydra configuration object.

        Returns:
            A configured adaptive detector.
        """
        chunk_seconds = float(cfg.num_seconds_per_chunk)
        max_silence_seconds = float(cfg.max_seconds_silence)
        max_audio_seconds = float(cfg.max_seconds_audio)
        pre_roll_seconds = float(cfg.pre_roll_seconds)
        if chunk_seconds <= 0 or not math.isclose(
            chunk_seconds * SAMPLE_RATE % 320, 0, abs_tol=1e-6
        ):
            raise ValueError("num_seconds_per_chunk must contain complete 20 ms frames")
        if max_silence_seconds <= 0 or max_audio_seconds <= 0:
            raise ValueError("audio duration settings must be positive")
        if pre_roll_seconds < 0 or pre_roll_seconds > max_audio_seconds:
            raise ValueError("pre_roll_seconds must fit within max_seconds_audio")
        return cls(
            vad_mode=int(cfg.vad_mode),
            initial_noise_floor=float(cfg.noise_floor_initial),
            noise_floor_min=float(cfg.noise_floor_min),
            noise_floor_max=float(cfg.noise_floor_max),
            noise_floor_rise_rate=float(cfg.noise_floor_rise_rate),
            noise_floor_fall_rate=float(cfg.noise_floor_fall_rate),
            speech_start_snr=float(cfg.speech_start_snr),
            speech_end_snr=float(cfg.speech_end_snr),
            onset_frames=int(cfg.onset_frames),
            max_silence_frames=max(
                1, math.ceil(max_silence_seconds / cls._frame_seconds())
            ),
            sample_rate=SAMPLE_RATE,
            frame_duration_ms=_FRAME_DURATION_MS,
            pre_roll_seconds=pre_roll_seconds,
        )

    @staticmethod
    def _frame_seconds() -> float:
        return _FRAME_DURATION_MS / 1_000

    @property
    def noise_floor(self) -> float:
        """Return the current learned RMS noise floor."""
        return self._noise_floor

    @property
    def active(self) -> bool:
        """Return whether an onset has been confirmed."""
        return self._active

    def reset_activity(self) -> None:
        """Reset onset and trailing-silence state without discarding the floor."""
        self._onset_run = 0
        self._onset_start_offset = None
        self._silence_run = 0
        self._active = False

    def process_chunk(
        self, audio: np.ndarray, *, update_noise_floor: bool = True
    ) -> VoiceActivity:
        """Process a recorder chunk split into valid 20 ms PCM frames.

        Args:
            audio:
                Signed-int16-compatible mono PCM samples.
            update_noise_floor (optional):
                Whether idle non-speech frames may update the floor. Set this to
                false for wake acknowledgement audio.

        Returns:
            Speech, onset, and end-of-recording state for the chunk.
        """
        pcm = _as_int16(audio)
        frame_count = len(pcm) // self.samples_per_frame
        if frame_count == 0:
            return VoiceActivity(speech_detected=False, onset=False, ended=False)

        onset = False
        onset_sample_offset: int | None = None
        candidate_sample_offset: int | None = None
        speech_detected = False
        ended = False
        candidate_seen = False
        idle_rms: list[float] = []
        for index in range(frame_count):
            frame = pcm[
                index * self.samples_per_frame : (index + 1) * self.samples_per_frame
            ]
            rms = _rms(frame)
            vad_speech = bool(self._vad.is_speech(frame.tobytes(), self.sample_rate))

            if self._active:
                speech = vad_speech and rms >= self._noise_floor * self.speech_end_snr
                speech_detected |= speech
                if speech:
                    self._silence_run = 0
                else:
                    self._silence_run += 1
                    if self._silence_run >= self.max_silence_frames:
                        self._active = False
                        self._onset_run = 0
                        ended = True
                        break
                continue

            candidate = vad_speech and rms >= self._noise_floor * self.speech_start_snr
            if candidate:
                candidate_seen = True
                if self._onset_run == 0:
                    self._onset_start_offset = index * self.samples_per_frame
                self._onset_run += 1
                speech_detected = True
                # Once a possible onset is present, the floor is frozen until it is
                # either confirmed or abandoned by a non-speech frame.
                if self._onset_run >= self.onset_frames:
                    self._active = True
                    self._silence_run = 0
                    onset = True
                    onset_sample_offset = self._onset_start_offset
                    self._onset_start_offset = None
                    break
            else:
                if not vad_speech:
                    self._onset_run = 0
                    self._onset_start_offset = None
                    if update_noise_floor:
                        idle_rms.append(rms)
                else:
                    self._onset_run = 0
                    self._onset_start_offset = None

        if not self._active and self._onset_run:
            candidate_sample_offset = self._onset_start_offset
            if self._onset_start_offset is not None:
                self._onset_start_offset -= len(pcm)

        if not self._active and update_noise_floor and idle_rms and not candidate_seen:
            # A median across the chunk means one click cannot become a new floor.
            self._update_noise_floor(float(np.median(idle_rms)))

        return VoiceActivity(
            speech_detected=speech_detected,
            onset=onset,
            ended=ended,
            onset_sample_offset=onset_sample_offset,
            candidate_sample_offset=candidate_sample_offset,
        )

    def _update_noise_floor(self, rms: float) -> None:
        rate = (
            self.noise_floor_rise_rate
            if rms > self._noise_floor
            else self.noise_floor_fall_rate
        )
        self._noise_floor += rate * (rms - self._noise_floor)
        self._noise_floor = float(
            np.clip(self._noise_floor, self.noise_floor_min, self.noise_floor_max)
        )


def create_voice_activity_detector(cfg: DictConfig) -> AdaptiveVoiceActivityDetector:
    """Create the single detector used by the bot for its whole lifetime.

    Args:
        cfg:
            Hydra configuration object.

    Returns:
        A stateful adaptive voice activity detector.
    """
    return AdaptiveVoiceActivityDetector.from_config(cfg=cfg)


def record_speech(
    last_response_time: dt.datetime,
    detector: AdaptiveVoiceActivityDetector,
    wake_word_model: oww.Model,
    synthesiser: SpeechSynthesiser,
    cfg: DictConfig,
    force_follow_up: bool = False,
    stop_event: threading.Event | None = None,
    on_interrupt: Callable[[], None] | None = None,
) -> tuple[np.ndarray, dt.datetime | None]:
    """Record confirmed speech and return it with its start time.

    Args:
        last_response_time:
            Time of the last response.
        detector:
            Stateful adaptive detector shared by all calls.
        wake_word_model:
            ONNX wake-word model.
        synthesiser:
            Speech synthesiser used for wake-word acknowledgement.
        cfg:
            Hydra configuration object.
        force_follow_up (optional):
            Whether confirmed speech should be accepted without a wake word. Defaults
            to False.
        stop_event (optional):
            Event that ends idle listening. Defaults to None.
        on_interrupt (optional):
            Callback invoked as soon as a forced follow-up begins. Defaults to None.

    Returns:
        Recorded speech and the time at which recording started, or an empty array
        and None when no speech was recorded.
    """
    rng = np.random.default_rng()
    chunk_seconds = float(cfg.num_seconds_per_chunk)
    chunk_size = int(SAMPLE_RATE * chunk_seconds)
    max_audio_samples = int(float(cfg.max_seconds_audio) * SAMPLE_RATE)
    post_wake_onset_samples = max(
        1,
        math.ceil(
            float(cfg.get("post_wake_max_seconds", cfg.max_seconds_silence))
            * SAMPLE_RATE
        ),
    )
    pre_roll_chunks = math.ceil(float(cfg.pre_roll_seconds) / chunk_seconds)
    pre_roll: deque[np.ndarray] = deque(maxlen=max(1, pre_roll_chunks))
    frames: list[np.ndarray] = []
    frames_left_to_ignore = 0
    post_wake_samples = 0
    audio_start: dt.datetime | None = None
    recording = False
    armed_after_wake = False
    barge_in_candidate = False
    barge_in_samples = 0
    barge_in_confirmation_samples = int(
        float(cfg.get("barge_in_confirmation_seconds", 0.35)) * SAMPLE_RATE
    )
    interrupt_notified = False

    logger.info(
        "Listening for speech..." if force_follow_up else "Listening for wakeword..."
    )
    try:
        while True:
            restart_after_acknowledgement = False
            wake_word_response: str | None = None
            # Close this lifecycle before playback so its buffered audio is discarded.
            with record(chunk_size=chunk_size) as recorder:
                while True:
                    frame = _as_int16(np.asarray(recorder.read()))
                    if frame.size == 0:
                        break
                    if (
                        stop_event is not None
                        and stop_event.is_set()
                        and not recording
                        and not barge_in_candidate
                    ):
                        return np.empty(0, dtype=np.int16), None
                    if barge_in_candidate:
                        barge_in_samples += frame.size
                        if barge_in_samples > barge_in_confirmation_samples:
                            return np.empty(0, dtype=np.int16), None
                    if frames_left_to_ignore:
                        detector.process_chunk(frame, update_noise_floor=False)
                        frames_left_to_ignore -= 1
                        if frames_left_to_ignore == 0:
                            detector.reset_activity()
                            pre_roll.clear()
                        continue

                    post_wake_chunk_start = post_wake_samples
                    if armed_after_wake:
                        post_wake_samples += frame.size

                    activity = detector.process_chunk(frame)
                    if not detector.active and pre_roll_chunks:
                        pre_roll.append(frame)

                    if activity.onset:
                        if armed_after_wake:
                            onset_offset = activity.onset_sample_offset
                            if (
                                onset_offset is None
                                or post_wake_chunk_start + onset_offset
                                > post_wake_onset_samples
                            ):
                                detector.reset_activity()
                                wake_word_model.reset()
                                return np.empty(0, dtype=np.int16), None

                        seconds_since_last_response = (
                            dt.datetime.now() - last_response_time
                        ).total_seconds()
                        follow_up = (
                            force_follow_up
                            or seconds_since_last_response
                            < float(cfg.follow_up_max_seconds)
                        )
                        if follow_up or armed_after_wake:
                            if (
                                force_follow_up
                                and synthesiser.is_playing
                                and not barge_in_candidate
                            ):
                                logger.info(
                                    "Possible barge-in detected, stopping playback."
                                )
                                synthesiser.stop()
                                if on_interrupt is not None and not interrupt_notified:
                                    on_interrupt()
                                    interrupt_notified = True
                                detector.reset_activity()
                                pre_roll.clear()
                                pre_roll.append(frame)
                                barge_in_candidate = True
                                barge_in_samples = 0
                                continue

                            logger.info(
                                "Follow-up detected!"
                                if follow_up
                                else "Speech detected!"
                            )
                            if on_interrupt is not None and not interrupt_notified:
                                on_interrupt()
                                interrupt_notified = True
                            recording = True
                            barge_in_candidate = False
                            armed_after_wake = False
                            wake_word_model.reset()
                            audio_start = dt.datetime.now()
                            frames.extend(pre_roll)
                            frames.append(frame)
                            if sum(len(item) for item in frames) >= max_audio_samples:
                                break
                            continue

                    if armed_after_wake:
                        candidate_offset = activity.candidate_sample_offset
                        if candidate_offset is not None:
                            candidate_start = post_wake_chunk_start + candidate_offset
                            if candidate_start > post_wake_onset_samples:
                                detector.reset_activity()
                                wake_word_model.reset()
                                return np.empty(0, dtype=np.int16), None
                        elif post_wake_samples > post_wake_onset_samples:
                            detector.reset_activity()
                            wake_word_model.reset()
                            return np.empty(0, dtype=np.int16), None
                        continue

                    if not recording:
                        wake_word_prediction_dict = wake_word_model.predict(x=frame)
                        assert isinstance(wake_word_prediction_dict, dict)
                        wake_word_probability = wake_word_prediction_dict["hey_jarvis"]
                        if wake_word_probability >= cfg.wake_word_probability_threshold:
                            logger.info("Wakeword detected!")
                            wake_word_response = str(
                                rng.choice(cfg.wake_word_responses)
                            )
                            restart_after_acknowledgement = True
                            break

                    if activity.ended:
                        break

                    if recording and detector.active:
                        frames.append(frame)
                        if sum(len(item) for item in frames) >= max_audio_samples:
                            logger.info("Max audio length reached, stopping.")
                            break

            if restart_after_acknowledgement:
                assert wake_word_response is not None
                synthesise_speech(text=wake_word_response, synthesiser=synthesiser)
                # Start the post-wake clock only after playback and its discard period.
                wake_word_model.reset()
                detector.reset_activity()
                pre_roll.clear()
                armed_after_wake = True
                post_wake_samples = 0
                frames_left_to_ignore = max(
                    1, math.ceil(float(cfg.wake_word_seconds) / chunk_seconds)
                )
                continue
            break
    finally:
        detector.reset_activity()

    audio_arr = _limit_audio(
        np.concatenate(frames) if frames else np.empty(0, dtype=np.int16),
        max_audio_samples=max_audio_samples,
    )
    if cfg.play_back_audio and audio_arr.size:
        logger.info("Playing back the audio...")
        sounddevice.play(data=audio_arr, samplerate=SAMPLE_RATE)

    return audio_arr, audio_start


def _limit_audio(audio: np.ndarray, *, max_audio_samples: int) -> np.ndarray:
    return _as_int16(audio[:max_audio_samples])


def _as_int16(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio)
    if values.size == 0:
        return np.empty(0, dtype=np.int16)
    values = np.nan_to_num(values, nan=0.0, posinf=32_767.0, neginf=-32_768.0)
    return np.clip(values, -32_768, 32_767).astype(np.int16, copy=False)


def _rms(frame: np.ndarray) -> float:
    values = frame.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(np.square(values))))


@contextmanager
def record(chunk_size: int) -> Generator[PvRecorder, None, None]:
    """Yield a started microphone recorder and clean it up afterwards.

    Args:
        chunk_size:
            The number of samples in each recorder chunk.

    Yields:
        A started recorder.
    """
    recorder = PvRecorder(frame_length=chunk_size)
    recorder.start()
    try:
        yield recorder
    finally:
        try:
            recorder.stop()
        finally:
            recorder.delete()
