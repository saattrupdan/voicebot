"""Generation of Danish speech."""

import logging
import tempfile
from pathlib import Path

import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from huggingface_hub._snapshot_download import snapshot_download
from pydub import AudioSegment
from pydub.playback import play

logger = logging.getLogger(__name__)


class DanishChatterBox(ChatterboxMultilingualTTS):
    """A Danish speech synthesiser using Chatterbox."""

    @classmethod
    def from_pretrained(cls, device: torch.device) -> "ChatterboxMultilingualTTS":
        """Load the pretrained Danish Chatterbox model.

        Args:
            device:
                The device to load the model onto.

        Returns:
            The Danish Chatterbox model.
        """
        ckpt_dir = Path(
            snapshot_download(
                repo_id="CoRal-project/tts-base-compatible", repo_type="model"
            )
        )
        return cls.from_local(ckpt_dir=ckpt_dir, device=device)


def synthesise_speech(text: str, synthesiser: ChatterboxMultilingualTTS) -> None:
    """Synthesise speech from text.

    Args:
        text:
            Text to be spoken.
        synthesiser:
            The speech synthesiser to use.
    """
    generated_speech = synthesiser.generate(text=text, language_id="da")
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav_file:
        logger.info(f"Saving generated speech of shape {generated_speech.shape}...")
        torchaudio.save(
            uri=temp_wav_file.name,
            src=generated_speech.unsqueeze(0).cpu(),
            sample_rate=24_000,
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
