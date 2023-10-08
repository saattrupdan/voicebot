"""Transcription of speech."""

import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings
import logging
import torch


logger = logging.getLogger(__name__)
logging.getLogger("torch._dynamo.output_graph").setLevel(logging.CRITICAL)


def transcribe_speech(
    speech: np.ndarray,
    asr_model: Wav2Vec2ForCTC,
    asr_processor: Wav2Vec2Processor,
) -> str:
    """Transcribe speech.

    Args:
        speech: Speech to transcribe.
        asr_model: Model for automatic speech recognition.
        asr_processor: Processor for automatic speech recognition.

    Returns:
        Transcribed speech.
    """
    logger.info(f"Transcribing speech of length {speech.shape[0]:,}...")
    tokenized = asr_processor(speech, return_tensors="pt").input_values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with torch.inference_mode():
            transcription_logits = asr_model(tokenized).logits[0]
    transcription_token_ids = transcription_logits.argmax(dim=1)
    transcription = asr_processor.decode(transcription_token_ids)
    logger.info(f"Heard the following: {transcription!r}")
    return transcription
