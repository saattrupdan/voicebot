"""Transcription of speech."""

import io
import logging
import wave

import numpy as np
import openai

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def transcribe_speech(
    speech: np.ndarray,
    client: openai.OpenAI,
    model: str,
    language: str,
    manual_fixes: dict[str, str],
) -> str:
    """Transcribe speech using the SYV transcription API.

    Args:
        speech:
            Mono speech samples recorded at 16 kHz.
        client:
            OpenAI-compatible client configured for the SYV API.
        model:
            Transcription model identifier.
        language:
            ISO language code for the speech.
        manual_fixes:
            Manual fixes for the transcription output.

    Returns:
        Transcribed speech.
    """
    logger.info(f"Transcribing speech of length {speech.shape[0]:,}...")
    response = client.audio.transcriptions.create(
        file=("speech.wav", _encode_wav(speech=speech), "audio/wav"),
        model=model,
        language=language,
    )
    transcription = response if isinstance(response, str) else response.text

    for before, after in manual_fixes.items():
        if before in transcription:
            logger.info(f"Fixing {before!r} to {after!r} in the transcription.")
            transcription = transcription.replace(before, after)

    logger.info(f"Heard the following: {transcription!r}")
    return transcription


def _encode_wav(speech: np.ndarray) -> bytes:
    """Encode mono speech samples as a 16 kHz, 16-bit WAV file."""
    if speech.ndim != 1:
        raise ValueError("Speech must be a one-dimensional mono audio array.")

    if speech.dtype == np.int16:
        pcm_speech = speech
    elif np.issubdtype(speech.dtype, np.floating):
        pcm_speech = (np.clip(speech, -1.0, 1.0) * np.iinfo(np.int16).max).astype(
            np.int16
        )
    else:
        raise TypeError(f"Unsupported speech dtype: {speech.dtype}")

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_speech.astype("<i2", copy=False).tobytes())
    return wav_buffer.getvalue()
