"""Generation of Danish speech."""

import os
import tempfile
from pathlib import Path

import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from pydub import AudioSegment
from pydub.playback import play


def synthesise_speech(
    text: str, synthesiser: ChatterboxMultilingualTTS | None = None
) -> None:
    """Synthesise speech from text.

    Args:
        text:
            Text to be spoken.
        synthesiser (optional):
            The speech synthesiser to use. Can be None to just use the MacOS `say`
            command.
    """
    if synthesiser is None:
        cleaned_text = text.replace('"', "'")
        os.system(f'say "{cleaned_text}"')
        return

    generated_speech = synthesiser.generate(
        text=text, language_id="da", audio_prompt_path="mic.wav"
    )
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav_file:
        torchaudio.save(
            uri=temp_wav_file.name,
            src=generated_speech.cpu(),
            sample_rate=synthesiser.sr,
        )
        play_sound(path=temp_wav_file.name)


def play_sound(path: str | Path) -> None:
    """Play a sound file.

    Args:
        path: The path to the sound file.
    """
    path = Path(path)
    match path.suffix.lower():
        case ".wav":
            audio = AudioSegment.from_wav(str(path))
        case ".mp3":
            audio = AudioSegment.from_mp3(str(path))
        case _:
            raise ValueError(f"Unknown file extension: {path.suffix!r}")
    play(audio)
