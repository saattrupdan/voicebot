"""Generation of Danish speech."""

from functools import partial
from pathlib import Path
from gtts import gTTS
from nltk import sent_tokenize
import nltk
from pydub import AudioSegment
from pydub.playback import play


nltk.download("punkt", quiet=True)


def synthesise_speech(text: str | None) -> None:
    """Synthesise speech from text.

    Args:
        text: Text to be spoken, or None if nothing should be spoken.
    """
    if text is not None:
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


def play_sound(path: str | Path) -> None:
    """Play a sound file.

    Args:
        filename: Name of the sound file.
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
