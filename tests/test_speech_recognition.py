"""Tests for speech transcription."""

import io
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from voicebot.speech_recognition import SAMPLE_RATE, transcribe_speech


def test_transcribe_speech_uploads_wav_and_applies_manual_fixes() -> None:
    """Speech is uploaded as WAV and known transcription errors are fixed."""
    client = MagicMock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(
        text="hvor den bliver solig"
    )
    speech = np.array([-1000, 0, 1000], dtype=np.int16)

    result = transcribe_speech(
        speech=speech,
        client=client,
        model="syv-transcribe",
        language="da",
        manual_fixes={"hvor den": "hvordan", "solig": "solrig"},
    )

    assert result == "hvordan bliver solrig"
    call = client.audio.transcriptions.create.call_args
    assert call.kwargs["model"] == "syv-transcribe"
    assert call.kwargs["language"] == "da"

    filename, wav_bytes, media_type = call.kwargs["file"]
    assert filename == "speech.wav"
    assert media_type == "audio/wav"
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnframes() == len(speech)
