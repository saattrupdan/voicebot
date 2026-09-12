"""Tests for speech synthesis."""

import collections.abc as c
import io
import wave
from unittest.mock import MagicMock, call

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
