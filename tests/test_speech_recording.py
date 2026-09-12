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


class _AlwaysSpeechVad:
    def is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        del pcm, sample_rate
        return True


def _chunk(value: int, size: int = 1_280) -> np.ndarray:
    return np.full(size, value, dtype=np.int16)


def _detector(
    *,
    onset_frames: int = 2,
    max_silence_frames: int = 4,
    vad: _ThresholdVad | _AlwaysSpeechVad | None = None,
) -> AdaptiveVoiceActivityDetector:
    return AdaptiveVoiceActivityDetector(
        vad=vad or _ThresholdVad(),
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


def test_speech_requires_consecutive_frames() -> None:
    """Onset is confirmed only after the configured consecutive-frame run."""
    detector = _detector(onset_frames=8)

    assert not detector.process_chunk(_chunk(2_000)).onset
    result = detector.process_chunk(_chunk(2_000))

    assert result.onset
    assert detector.active


def test_onset_offset_spans_chunks_and_clears_after_abandonment() -> None:
    """Onset metadata names the first candidate frame and is not retained."""
    detector = _detector(onset_frames=2)

    pending = detector.process_chunk(_chunk(2_000, size=320))
    assert pending.candidate_sample_offset == 0
    confirmed = detector.process_chunk(_chunk(2_000, size=320))
    assert confirmed.onset
    assert confirmed.onset_sample_offset == -320

    detector.reset_activity()
    abandoned = detector.process_chunk(_chunk(2_000, size=320))
    assert abandoned.candidate_sample_offset == 0
    cleared = detector.process_chunk(_chunk(0, size=320))
    assert cleared.onset_sample_offset is None
    assert cleared.candidate_sample_offset is None


def test_start_and_end_snr_thresholds_are_independent() -> None:
    """The lower end threshold retains speech which cannot trigger onset."""
    detector = _detector(onset_frames=1, max_silence_frames=4, vad=_AlwaysSpeechVad())

    assert not detector.process_chunk(_chunk(200, size=320)).onset
    assert detector.process_chunk(_chunk(300, size=320)).onset

    continuing = detector.process_chunk(_chunk(200, size=320))
    assert continuing.speech_detected
    assert not continuing.ended
    assert not detector.process_chunk(_chunk(130, size=960)).ended
    assert detector.process_chunk(_chunk(130, size=320)).ended


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


def _recorder(frames: list[np.ndarray], recorder: MagicMock | None = None) -> object:
    @contextmanager
    def context(chunk_size: int) -> Iterator[MagicMock]:
        del chunk_size
        active_recorder = recorder or MagicMock()
        active_recorder.read.side_effect = frames
        yield active_recorder

    return context


def test_record_speech_preserves_exact_pre_roll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the configured chunks immediately preceding onset are retained."""
    detector = _detector(onset_frames=8, max_silence_frames=4)
    chunks = [_chunk(100), _chunk(120), _chunk(2_000), _chunk(2_100), _chunk(0)]
    monkeypatch.setattr(speech_recording, "record", _recorder(chunks))
    wake_word = MagicMock()
    wake_word.predict.return_value = {"hey_jarvis": 0.0}

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime.now(),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=MagicMock(),
        cfg=_config(pre_roll_seconds=0.16),
    )

    expected = np.concatenate([_chunk(120), _chunk(2_000), _chunk(2_100)])
    assert started is not None
    assert np.array_equal(audio, expected)


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
    detector = _detector(onset_frames=2, max_silence_frames=2)
    frames = [
        _chunk(2_000, size=320),
        _chunk(0, size=320),
        _chunk(2_000, size=320),
        _chunk(2_100, size=320),
        _chunk(0, size=320),
        _chunk(0, size=320),
    ]
    monkeypatch.setattr(speech_recording, "record", _recorder(frames))
    wake_word = MagicMock()
    wake_word.predict.return_value = {"hey_jarvis": 1.0}
    synthesiser = MagicMock()
    monkeypatch.setattr(speech_recording, "synthesise_speech", synthesiser)

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime(1900, 1, 1),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=MagicMock(),
        cfg=_config(
            num_seconds_per_chunk=0.02,
            pre_roll_seconds=0.04,
            max_seconds_silence=0.04,
            wake_word_seconds=0.02,
        ),
    )

    assert started is not None
    assert np.array_equal(
        audio,
        np.concatenate(
            [_chunk(2_000, size=320), _chunk(2_100, size=320), _chunk(0, size=320)]
        ),
    )
    synthesiser.assert_called_once()


def test_post_wake_onset_at_aligned_deadline_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speech may begin exactly where an aligned onset window ends."""
    detector = _detector(onset_frames=2, max_silence_frames=4)
    chunks = [_chunk(2_000), _chunk(0), _chunk(0), _chunk(2_000), _chunk(0)]
    monkeypatch.setattr(speech_recording, "record", _recorder(chunks))
    wake_word = MagicMock()
    wake_word.predict.return_value = {"hey_jarvis": 1.0}
    synthesiser = MagicMock()
    monkeypatch.setattr(speech_recording, "synthesise_speech", synthesiser)

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime(1900, 1, 1),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=MagicMock(),
        cfg=_config(max_seconds_silence=0.08),
    )

    assert started is not None
    assert audio.size > 0
    synthesiser.assert_called_once()


def test_post_wake_onset_just_before_non_aligned_deadline_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate beginning just before the deadline may confirm afterwards."""
    detector = _detector(onset_frames=2, max_silence_frames=4)
    chunks = [
        _chunk(2_000),
        _chunk(0),
        np.concatenate([_chunk(0, size=960), _chunk(2_000, size=320)]),
        np.concatenate([_chunk(2_000, size=320), _chunk(0, size=960)]),
        _chunk(0),
    ]
    monkeypatch.setattr(speech_recording, "record", _recorder(chunks))
    wake_word = MagicMock()
    wake_word.predict.return_value = {"hey_jarvis": 1.0}
    synthesiser = MagicMock()
    monkeypatch.setattr(speech_recording, "synthesise_speech", synthesiser)

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime(1900, 1, 1),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=MagicMock(),
        cfg=_config(max_seconds_silence=0.0625),
    )

    assert started is not None
    assert audio.size > 0
    synthesiser.assert_called_once()


def test_post_wake_onset_just_after_non_aligned_deadline_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate beginning just after the deadline is rejected immediately."""
    detector = _detector(onset_frames=2, max_silence_frames=4)
    recorder = MagicMock()
    chunks = [
        _chunk(2_000),
        _chunk(0),
        _chunk(0),
        np.concatenate(
            [_chunk(0, size=320), _chunk(2_000, size=640), _chunk(0, size=320)]
        ),
    ]
    monkeypatch.setattr(
        speech_recording, "record", _recorder(chunks, recorder=recorder)
    )
    wake_word = MagicMock()
    wake_word.predict.return_value = {"hey_jarvis": 1.0}
    synthesiser = MagicMock()
    monkeypatch.setattr(speech_recording, "synthesise_speech", synthesiser)

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime(1900, 1, 1),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=MagicMock(),
        cfg=_config(max_seconds_silence=0.09375),
    )

    assert audio.dtype == np.int16
    assert audio.size == 0
    assert started is None
    assert recorder.read.call_count == 4
    synthesiser.assert_called_once()


def test_post_wake_onset_window_expires_before_late_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speech after the post-acknowledgement deadline needs a new wake word."""
    detector = _detector(onset_frames=2, max_silence_frames=4)
    recorder = MagicMock()
    frames = [
        _chunk(2_000),
        _chunk(0),
        _chunk(100),
        _chunk(100),
        _chunk(100),
        _chunk(2_000),
    ]
    monkeypatch.setattr(
        speech_recording, "record", _recorder(frames, recorder=recorder)
    )
    wake_word = MagicMock()
    wake_word.predict.return_value = {"hey_jarvis": 1.0}
    synthesiser = MagicMock()
    monkeypatch.setattr(speech_recording, "synthesise_speech", synthesiser)

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime(1900, 1, 1),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=MagicMock(),
        cfg=_config(max_seconds_silence=0.16),
    )

    assert audio.dtype == np.int16
    assert audio.size == 0
    assert started is None
    assert recorder.read.call_count == 5
    assert wake_word.reset.call_count == 2
    synthesiser.assert_called_once()


def test_barge_in_stops_playback_and_records_continued_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmed speech during playback stops output and becomes the next turn."""
    detector = _detector(onset_frames=2, max_silence_frames=4)
    frames = [_chunk(2_000), _chunk(2_100), _chunk(0)]
    monkeypatch.setattr(speech_recording, "record", _recorder(frames))
    wake_word = MagicMock()
    synthesiser = MagicMock()
    synthesiser.is_playing = True
    interrupted = MagicMock()

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime(1900, 1, 1),
        detector=detector,
        wake_word_model=wake_word,
        synthesiser=synthesiser,
        cfg=_config(),
        force_follow_up=True,
        on_interrupt=interrupted,
    )

    assert started is not None
    assert audio.size > 0
    synthesiser.stop.assert_called_once_with()
    interrupted.assert_called_once_with()
    wake_word.predict.assert_not_called()


def test_response_completion_ends_idle_monitoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response monitor exits when generation ends without an interruption."""
    detector = _detector(onset_frames=2, max_silence_frames=4)
    recorder = MagicMock()
    monkeypatch.setattr(
        speech_recording, "record", _recorder([_chunk(0)], recorder=recorder)
    )
    done = speech_recording.threading.Event()
    done.set()

    audio, started = speech_recording.record_speech(
        last_response_time=speech_recording.dt.datetime(1900, 1, 1),
        detector=detector,
        wake_word_model=MagicMock(),
        synthesiser=MagicMock(),
        cfg=_config(),
        force_follow_up=True,
        stop_event=done,
    )

    assert audio.size == 0
    assert started is None
    recorder.read.assert_called_once_with()


def test_recorder_is_deleted_when_stop_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recorder deletion is guaranteed after a stop failure."""
    recorder = MagicMock()
    recorder.stop.side_effect = RuntimeError("stop failed")
    monkeypatch.setattr(
        speech_recording, "PvRecorder", MagicMock(return_value=recorder)
    )

    with pytest.raises(RuntimeError, match="stop failed"):
        with speech_recording.record(chunk_size=1_280):
            pass

    recorder.start.assert_called_once_with()
    recorder.stop.assert_called_once_with()
    recorder.delete.assert_called_once_with()


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
