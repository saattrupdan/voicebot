"""Tests for speech synthesis."""

import collections.abc as c
import io
import wave
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from voicebot import speech_synthesis
from voicebot.speech_synthesis import PlaybackOutcome


def test_synthesiser_streams_plapre_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plapre PCM chunks play before the complete response has arrived."""
    client = MagicMock()
    response = MagicMock()
    response.iter_bytes.return_value = iter([b"\x01\x00", b"\x02\x00"])
    context = MagicMock()
    context.__enter__.return_value = response
    client.audio.speech.with_streaming_response.create.return_value = context
    output = MagicMock()
    monkeypatch.setattr(
        speech_synthesis.sounddevice, "RawOutputStream", MagicMock(return_value=output)
    )
    synthesiser = speech_synthesis.SpeechSynthesiser(
        client=client, model="syvai/plapre-nano", voice="tor"
    )

    assert synthesiser.synthesise(text="Hej, verden.") is PlaybackOutcome.COMPLETE

    client.audio.speech.with_streaming_response.create.assert_called_once_with(
        model="syvai/plapre-nano",
        input="Hej, verden.",
        voice="tor",
        response_format="pcm",
    )
    assert output.write.call_args_list == [call(b"\x01\x00"), call(b"\x02\x00")]
    output.stop.assert_called_once_with()
    output.close.assert_called_once_with()


def test_webrtc_echo_cancellation_preserves_double_talk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WebRTC AEC suppresses delayed playback while retaining user speech."""
    output_sample_rate = 24_000
    capture_sample_rate = 16_000
    output_chunk_size = 1_920
    capture_chunk_size = 1_280
    chunk_count = 24
    output_time = np.arange(output_chunk_size * chunk_count) / output_sample_rate
    capture_time = np.arange(capture_chunk_size * chunk_count) / capture_sample_rate

    def speech_signal(time: np.ndarray) -> np.ndarray:
        return (
            3_000 * np.sin(2 * np.pi * 180 * time)
            + 1_800 * np.sin(2 * np.pi * 530 * time)
            + 900 * np.sin(2 * np.pi * 1_100 * time)
        ) * (0.5 + 0.5 * np.square(np.sin(2 * np.pi * 3 * time)))

    playback_float = speech_signal(output_time)
    far_end = speech_signal(capture_time)
    playback = np.clip(playback_float, -32_768, 32_767).astype(np.int16)
    filtered = np.convolve(far_end, np.array([0.5, 0.3, 0.15, 0.05]), mode="same")
    near_end = np.zeros_like(far_end)
    near_end[capture_chunk_size:] = filtered[:-capture_chunk_size] * 0.5
    user_mask = (capture_time >= 1.04) & (capture_time < 1.36)
    near_end[user_mask] += 5_000 * np.sin(
        2 * np.pi * 240 * capture_time[user_mask]
    ) + 2_500 * np.sin(2 * np.pi * 700 * capture_time[user_mask])
    capture = np.clip(near_end, -32_768, 32_767).astype(np.int16)
    playback_chunks = np.split(playback, chunk_count)
    capture_chunks = np.split(capture, chunk_count)

    client = MagicMock()
    response = MagicMock()
    response.iter_bytes.return_value = iter(
        [chunk.tobytes() for chunk in playback_chunks]
    )
    context = MagicMock()
    context.__enter__.return_value = response
    client.audio.speech.with_streaming_response.create.return_value = context
    output = MagicMock()
    monkeypatch.setattr(
        speech_synthesis.sounddevice, "RawOutputStream", MagicMock(return_value=output)
    )
    synthesiser = speech_synthesis.SpeechSynthesiser(
        client=client,
        model="syvai/plapre-nano",
        voice="tor",
        sample_rate=output_sample_rate,
        echo_stream_delay_ms=80,
    )
    residual_levels: list[float] = []
    readiness: list[bool] = []

    def process_capture(_: bytes) -> None:
        capture_chunk = capture_chunks[len(residual_levels)]
        residual_rms, ready = synthesiser.playback_non_echo_rms(
            audio=capture_chunk, sample_rate=capture_sample_rate
        )
        residual_levels.append(residual_rms)
        readiness.append(ready)

    output.write.side_effect = process_capture

    assert synthesiser.synthesise(text="Hej.") is PlaybackOutcome.COMPLETE
    assert readiness[:3] == [False, True, True]
    assert max(residual_levels[7:12]) < 100.0
    assert min(residual_levels[13:16]) > 500.0
    assert max(residual_levels[18:23]) < 300.0


def test_echo_canceller_readiness_resets_after_queue_discontinuity() -> None:
    """Underflow and overflow require two newly aligned chunks before use."""
    synthesiser = speech_synthesis.SpeechSynthesiser(
        client=MagicMock(), model="syvai/plapre-nano", voice="tor", sample_rate=16_000
    )
    chunk = np.arange(1_280, dtype=np.int16)

    synthesiser._queue_playback_reference(
        audio=chunk.tobytes(), sample_rate=16_000, channels=1
    )
    assert synthesiser.playback_non_echo_rms(chunk, 16_000)[1] is False
    synthesiser._queue_playback_reference(
        audio=chunk.tobytes(), sample_rate=16_000, channels=1
    )
    assert synthesiser.playback_non_echo_rms(chunk, 16_000)[1] is True

    assert synthesiser.playback_non_echo_rms(chunk, 16_000)[1] is False
    synthesiser._queue_playback_reference(
        audio=chunk.tobytes(), sample_rate=16_000, channels=1
    )
    assert synthesiser.playback_non_echo_rms(chunk, 16_000)[1] is False
    synthesiser._queue_playback_reference(
        audio=chunk.tobytes(), sample_rate=16_000, channels=1
    )
    assert synthesiser.playback_non_echo_rms(chunk, 16_000)[1] is True

    oversized = np.tile(chunk, 13)
    synthesiser._queue_playback_reference(
        audio=oversized.tobytes(), sample_rate=16_000, channels=1
    )
    assert synthesiser.playback_non_echo_rms(chunk, 16_000)[1] is False


def test_synthesiser_stops_streamed_playback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stopping playback prevents queued PCM chunks from being written."""
    client = MagicMock()
    response = MagicMock()
    response.iter_bytes.return_value = iter([b"\x01\x00", b"\x02\x00"])
    context = MagicMock()
    context.__enter__.return_value = response
    client.audio.speech.with_streaming_response.create.return_value = context
    output = MagicMock()
    monkeypatch.setattr(
        speech_synthesis.sounddevice, "RawOutputStream", MagicMock(return_value=output)
    )
    synthesiser = speech_synthesis.SpeechSynthesiser(
        client=client, model="syvai/plapre-nano", voice="tor"
    )
    output.write.side_effect = lambda _: synthesiser.stop()

    assert synthesiser.synthesise(text="En lang besked.") is PlaybackOutcome.CANCELLED

    output.write.assert_called_once_with(b"\x01\x00")
    output.abort.assert_called()
    response.close.assert_called_once_with()


def test_synthesiser_falls_back_to_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsupported streaming endpoint falls back before playback starts."""
    client = MagicMock()
    client.audio.speech.with_streaming_response.create.side_effect = AttributeError(
        "streaming unsupported"
    )
    client.audio.speech.create.return_value.content = _wav_bytes(b"\x01\x00")
    output = MagicMock()
    monkeypatch.setattr(
        speech_synthesis.sounddevice, "RawOutputStream", MagicMock(return_value=output)
    )
    synthesiser = speech_synthesis.SpeechSynthesiser(
        client=client, model="syvai/plapre-nano", voice="tor"
    )

    assert synthesiser.synthesise(text="Hej igen.") is PlaybackOutcome.COMPLETE

    client.audio.speech.create.assert_called_once_with(
        model="syvai/plapre-nano", input="Hej igen.", voice="tor", response_format="wav"
    )
    output.write.assert_called_once_with(b"\x01\x00")


def test_synthesiser_reports_partial_stream_failure_without_provider_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed stream after audio starts is retryable and safely logged."""
    client = MagicMock()
    response = MagicMock()

    def chunks() -> c.Iterator[bytes]:
        yield b"\x01\x00"
        raise RuntimeError("provider-secret")

    response.iter_bytes.return_value = chunks()
    context = MagicMock()
    context.__enter__.return_value = response
    client.audio.speech.with_streaming_response.create.return_value = context
    output = MagicMock()
    monkeypatch.setattr(
        speech_synthesis.sounddevice, "RawOutputStream", MagicMock(return_value=output)
    )
    synthesiser = speech_synthesis.SpeechSynthesiser(
        client=client, model="model", voice="voice"
    )

    with caplog.at_level("ERROR"):
        outcome = synthesiser.synthesise(text="En besked.")

    assert outcome is PlaybackOutcome.FAILED
    assert "provider-secret" not in caplog.text


def test_synthesiser_reports_fallback_failure_without_provider_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed WAV fallback is retryable and safely logged."""
    client = MagicMock()
    client.audio.speech.with_streaming_response.create.side_effect = AttributeError(
        "streaming-secret"
    )
    client.audio.speech.create.side_effect = RuntimeError("provider-secret")
    synthesiser = speech_synthesis.SpeechSynthesiser(
        client=client, model="model", voice="voice"
    )

    with caplog.at_level("ERROR"):
        outcome = synthesiser.synthesise(text="En besked.")

    assert outcome is PlaybackOutcome.FAILED
    assert "streaming-secret" not in caplog.text
    assert "provider-secret" not in caplog.text


def _wav_bytes(audio: bytes) -> bytes:
    """Create a mono 16-bit WAV payload."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(audio)
    return buffer.getvalue()
