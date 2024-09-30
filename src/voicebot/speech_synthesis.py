"""Generation of Danish speech."""

import os
from functools import partial
from pathlib import Path
from typing import Literal

import nltk
from gtts import gTTS
from nltk import sent_tokenize
from pydub import AudioSegment
from pydub.playback import play

# Download the NLTK tokenizer model
nltk.download("punkt_tab", quiet=True)


def synthesise_speech(
    text: str, engine: Literal["gtts", "macos", "auto"] = "auto"
) -> None:
    """Synthesise speech from text.

    Args:
        text:
            Text to be spoken.
        engine:
            The TTS engine to use. Can be `gtts` (Google Translate's text-to-speech),
            `macos` (built-in `say` TTS on MacOS devices) or `auto`, to use `macos` if
            it is available and `gtts` otherwise.
    """
    text = text.replace("m/s", "metersekunder").replace("`", "'")

    if engine == "auto":
        macos_available = Path("/usr/bin/say").exists()
        engine = "macos" if macos_available else "gtts"

    match engine:
        case "macos":
            os.system(f'say "{text}"')
        case "gtts":
            tts = gTTS(
                text=text,
                tld="dk",
                lang="da",
                lang_check=False,
                tokenizer_func=partial(sent_tokenize, language="danish"),
            )
            output_path = Path(".temp.mp3")
            tts.save(savefile=output_path)
            play_sound(path=output_path)
            output_path.unlink()


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
