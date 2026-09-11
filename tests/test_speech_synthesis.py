"""Tests for speech synthesis."""

from unittest.mock import MagicMock

import pytest

from voicebot import speech_synthesis


def test_synthesiser_uses_plapre_voice_and_plays_wav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesised speech uses the configured SYV model and voice."""
    client = MagicMock()
    client.audio.speech.create.return_value.content = b"RIFFtest"
    play_sound = MagicMock()
    monkeypatch.setattr(speech_synthesis, "play_sound", play_sound)
    synthesiser = speech_synthesis.SpeechSynthesiser(
        client=client, model="syvai/plapre-nano", voice="tor"
    )

    synthesiser.synthesise(text="Hej, verden.")

    client.audio.speech.create.assert_called_once_with(
        model="syvai/plapre-nano",
        input="Hej, verden.",
        voice="tor",
        response_format="wav",
    )
    play_sound.assert_called_once()
    assert play_sound.call_args.kwargs["path"].endswith(".wav")
