"""Transcription of speech."""

import numpy as np
from transformers import Pipeline
import warnings
import logging
import torch


logger = logging.getLogger(__name__)
logging.getLogger("torch._dynamo.output_graph").setLevel(logging.CRITICAL)


def transcribe_speech(speech: np.ndarray, asr_pipeline: Pipeline) -> str:
    """Transcribe speech.

    Args:
        speech: Speech to transcribe.
        asr_pipeline: Pipeline for automatic speech recognition.

    Returns:
        Transcribed speech.
    """
    logger.info(f"Transcribing speech of length {speech.shape[0]:,}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with torch.inference_mode():
            transcription = asr_pipeline(speech)["text"]
    logger.info(f"Heard the following: {transcription!r}")
    return transcription
