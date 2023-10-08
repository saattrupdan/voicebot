"""Transcription of speech."""

import numpy as np
from transformers import Pipeline
import warnings
import logging


logger = logging.getLogger(__name__)


def transcribe_speech(
    speech: np.ndarray,
    asr_pipeline: Pipeline,
) -> str:
    """Transcribe speech.

    Args:
        speech: Speech to transcribe.
        asr_pipeline: Speech recognition pipeline for `transformers`.

    Returns:
        Transcribed speech.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        transcription = asr_pipeline(inputs=speech)["text"]
        logger.info(f"Heard the following: {transcription!r}")
        return transcription
