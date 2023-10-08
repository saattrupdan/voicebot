"""Transcription of speech."""

import numpy as np
import scipy.signal as sps
from transformers import Pipeline
import warnings


def transcribe_speech(
    speech: np.ndarray,
    sample_rate: int,
    asr_pipeline: Pipeline,
) -> str:
    """Transcribe speech.

    Args:
        speech: Speech to transcribe.
        sample_rate: Sample rate.
        asr_pipeline: Speech recognition pipeline for `transformers`.

    Returns:
        Transcribed speech.
    """
    num_samples = speech.shape[0] * 16_000 // sample_rate
    speech = sps.resample(x=speech, num=num_samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return asr_pipeline(inputs=speech)["text"]
