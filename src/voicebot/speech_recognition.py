"""Transcription of speech."""

import logging

import numpy as np
import torch
from samplerate import resample
from transformers import AutomaticSpeechRecognitionPipeline

logger = logging.getLogger(__name__)
logging.getLogger("torch._dynamo.output_graph").setLevel(logging.CRITICAL)


def transcribe_speech(
    speech: np.ndarray,
    input_sample_rate: int,
    asr_sample_rate: int,
    transcriber: AutomaticSpeechRecognitionPipeline,
) -> str:
    """Transcribe speech.

    Args:
        speech:
            Speech to transcribe.
        input_sample_rate:
            Sample rate of the input speech.
        asr_sample_rate:
            Sample rate of the ASR model.
        transcriber:
            Pipeline for automatic speech recognition.

    Returns:
        Transcribed speech.
    """
    # Resample the audio to the ASR sample rate
    speech = resample(speech, asr_sample_rate / input_sample_rate, "sinc_best")

    logger.info(f"Transcribing speech of length {speech.shape[0]:,}...")
    with torch.inference_mode():
        transcription_dict = transcriber(inputs=speech)
        assert isinstance(transcription_dict, dict)
        transcription = transcription_dict["text"]
    logger.info(f"Heard the following: {transcription!r}")
    return transcription
