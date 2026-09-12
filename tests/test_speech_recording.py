"""Tests for adaptive speech recording without microphone access."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

import voicebot.speech_recording as speech_recording
from voicebot.speech_recording import AdaptiveVoiceActivityDetector


class _ThresholdVad:
    def __init__(self, threshold: float = 1_000.0) -> None:
        self.threshold = threshold

    def is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        del sample_rate
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        return bool(np.sqrt(np.mean(np.square(samples))) >= self.threshold)


def _chunk(value: int, size: int = 1_280) -> np.ndarray:
    return np.full(size, value, dtype=np.int16)


def _detector(
    *, onset_frames: int = 2, max_silence_frames: int = 4
) -> AdaptiveVoiceActivityDetector:
    return AdaptiveVoiceActivityDetector(
        vad=_ThresholdVad(),
        initial_noise_floor=100.0,
        noise_floor_min=50.0,
        noise_floor_max=500.0,
        noise_floor_rise_rate=0.5,
        noise_floor_fall_rate=0.5,
        onset_frames=onset_frames,
        max_silence_frames=max_silence_frames,
    )


def test_noise_floor_adapts_in_both_directions_and_stays_bounded() -> None:
    """The learned floor follows sustained noise but remains bounded."""
    detector = _detector()

    for _ in range(20):
        detector.process_chunk(_chunk(400))
    assert detector.noise_floor == pytest.approx(400, abs=0.01)

    for _ in range(20):
        detector.process_chunk(_chunk(0))
    assert detector.noise_floor == 50


def test_speech_and_impulse_do_not_immediately_raise_noise_floor() -> None:
    """Speech is excluded from floor updates and a single impulse is limited."""
    detector = _detector()
    initial_floor = detector.noise_floor

    impulse = np.zeros(1_280, dtype=np.int16)
    impulse[0] = np.iinfo(np.int16).max
    assert not detector.process_chunk(impulse).onset
    assert detector.noise_floor <= initial_floor

    detector.process_chunk(_chunk(2_000))
    assert detector.noise_floor == initial_floor

    detector.reset_activity()
    detector.process_chunk(_chunk(300))
    assert detector.noise_floor > initial_floor
    assert detector.noise_floor < 300


def test_speech_requires_consecutive_frames_and_preserves_pre_roll() -> None:
    """Onset is confirmed only after the configured consecutive-frame run."""
    detector = _detector(onset_frames=8)

    assert not detector.process_chunk(_chunk(2_000)).onset
    result = detector.process_chunk(_chunk(2_000))

    assert result.onset
    assert detector.active


def test_end_uses_hysteresis_and_trailing_silence() -> None:
    """Quiet audio ends an active recording only after trailing silence."""
    detector = _detector(onset_frames=1, max_silence_frames=8)
    assert detector.process_chunk(_chunk(2_000)).onset
    assert not detector.process_chunk(_chunk(200)).ended
    assert detector.process_chunk(_chunk(0)).ended


def _config(**overrides: object) -> DictConfig:
    values: dict[str, object] = {
        "num_seconds_per_chunk": 0.08,
        "vad_mode": 2,
        "noise_floor_initial": 100.0,
        "noise_floor_min": 50.0,
        "noise_floor_max": 500.0,
        "noise_floor_rise_rate": 0.5,
        "noise_floor_fall_rate": 0.5,
        "speech_start_snr": 2.5,
        "speech_end_snr": 1.4,
        "onset_frames": 2,
        "pre_roll_seconds": 0.16,
        "max_seconds_silence": 0.08,
        "max_seconds_audio": 2.0,
        "follow_up_max_seconds": 5.0,
        "play_back_audio": False,
        "wake_word_probability_threshold": 0.5,
        "wake_word_responses": ["Ja?"],
        "wake_word_seconds": 0.08,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _recorder(frames: list[np.ndarray]) -> object:
    @contextmanager
    def context(chunk_size: int) -> Iterator[MagicMock]:
        del chunk_size
        recorder = MagicMock()
        recorder.read.side_effect = frames
        yield recorder

    return context


def test_follow_up_requires_voice_activity_and_returns_int16_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recent response accepts confirmed speech without invoking wakeword."""
    detector = _detector(onset_frames=2, max_silence_frames=4)
    monkeypatch.setattr(
        speech_recording, "record", _recorder([_chunk(100), _chunk(2_000), _chunk(0)])
    )
    wake_word = MagicMock()
    wake_word.predict.return_value = {"hey_jarvis": 0.0}

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime.now(),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=MagicMock(),
        cfg=_config(),
    )

    assert started is not None
    assert audio.dtype == np.int16
    assert len(audio) == 2_560
    assert wake_word.predict.call_count == 1


def test_wake_word_path_works_outside_follow_up_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wakeword arms a later confirmed utterance outside the follow-up window."""
    detector = _detector(onset_frames=2, max_silence_frames=4)
    frames = [_chunk(2_000), _chunk(0), _chunk(100), _chunk(2_000), _chunk(0)]
    monkeypatch.setattr(speech_recording, "record", _recorder(frames))
    wake_word = MagicMock()
    wake_word.predict.side_effect = [{"hey_jarvis": 1.0}, {"hey_jarvis": 0.0}]
    synthesiser = MagicMock()
    monkeypatch.setattr(speech_recording, "synthesise_speech", synthesiser)

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime(1900, 1, 1),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=MagicMock(),
        cfg=_config(),
    )

    assert started is not None
    assert len(audio) == 2_560
    synthesiser.assert_called_once()


def test_empty_recording_is_safe_and_maximum_length_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty input is safe and recorded audio is clipped to its configured limit."""
    detector = _detector(onset_frames=2, max_silence_frames=4)
    monkeypatch.setattr(speech_recording, "record", _recorder([np.empty(0)]))
    wake_word = MagicMock()
    wake_word.predict.return_value = {"hey_jarvis": 0.0}

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime(1900, 1, 1),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=MagicMock(),
        cfg=_config(max_seconds_audio=0.08),
    )

    assert started is None
    assert audio.dtype == np.int16
    assert audio.size == 0

    detector = _detector(onset_frames=2, max_silence_frames=4)
    monkeypatch.setattr(
        speech_recording, "record", _recorder([_chunk(2_000), _chunk(2_000)])
    )
    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime.now(),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=MagicMock(),
        cfg=_config(max_seconds_audio=0.08),
    )

    assert started is not None
    assert len(audio) == 1_280
