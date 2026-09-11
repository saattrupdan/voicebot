"""Generation of Danish speech."""

import tempfile
import wave
from pathlib import Path

import numpy as np
import openai
import sounddevice


class SpeechSynthesiser:
    """A speech synthesiser backed by the SYV audio API."""

    def __init__(self, client: openai.OpenAI, model: str, voice: str) -> None:
        """Initialise the speech synthesiser.

        Args:
            client:
                OpenAI-compatible client configured for the SYV API.
            model:
                Speech synthesis model identifier.
            voice:
                Voice identifier supported by the model.
        """
        self.client = client
        self.model = model
        self.voice = voice

    def synthesise(self, text: str) -> None:
        """Generate and play speech.

        Args:
            text:
                Text to synthesise.
        """
        response = self.client.audio.speech.create(
            model=self.model, input=text, voice=self.voice, response_format="wav"
        )
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
            wav_file.write(response.content)
            wav_file.flush()
            play_sound(path=wav_file.name)


def synthesise_speech(text: str, synthesiser: SpeechSynthesiser | None) -> None:
    """Synthesise and play speech.

    Args:
        text:
            Text to synthesise.
        synthesiser:
            Speech synthesiser to use, or None to produce no speech.
    """
    if synthesiser is not None:
        synthesiser.synthesise(text=text)


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
