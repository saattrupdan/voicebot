"""A voice bot."""

import datetime as dt
import logging

import openwakeword as oww
import torch
import transformers.utils.logging as hf_logging
from omegaconf import DictConfig
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    Wav2Vec2ProcessorWithLM,
    pipeline,
)

from .speech_recognition import transcribe_speech
from .speech_recording import calibrate_audio_threshold, record_speech
from .speech_synthesis import synthesise_speech
from .text_engine import TextEngine

logger = logging.getLogger(__name__)


class VoiceBot:
    """A voice bot."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialise the bot.

        Args:
            cfg:
                The Hydra configuration.
        """
        self.cfg = cfg
        hf_logging.set_verbosity_error()

        if cfg.calibrate:
            self.audio_threshold = calibrate_audio_threshold(cfg=self.cfg)
        else:
            self.audio_threshold = cfg.audio_threshold

        logger.info("Loading the wake word model...")
        self.wake_word_model = oww.Model(
            wakeword_models=["hey_jarvis"], inference_framework="onnx"
        )

        logger.info("Loading the speech recognition model...")
        self.transcriber: AutomaticSpeechRecognitionPipeline = pipeline(
            model=self.cfg.asr_model_id,
            device=self.device,
            task="automatic-speech-recognition",
        )

        # Sanity check that the processor is of the correct type
        processor_class = self.transcriber.feature_extractor._processor_class
        assert processor_class == Wav2Vec2ProcessorWithLM.__name__, (
            "Expected the processor to be of type Wav2Vec2ProcessorWithLM, but got "
            f"{processor_class}."
        )

        logger.info("Loading the text engine model...")
        self.text_engine = TextEngine(cfg=self.cfg)

    @property
    def device(self) -> str:
        """Return the device on which the bot is running."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def run(self) -> None:
        """Run the bot."""
        last_response_time = dt.datetime(year=1900, month=1, day=1)

        synthesise_speech(text=self.cfg.starting_phrase)

        while True:
            speech, audio_start = record_speech(
                last_response_time=last_response_time,
                audio_threshold=self.audio_threshold,
                cfg=self.cfg,
            )
            if audio_start is None:
                continue

            text = transcribe_speech(
                speech=speech,
                transcriber=self.transcriber,
                manual_fixes=self.cfg.manual_fixes,
            )
            if text:
                response = self.text_engine.generate_response(
                    prompt=text,
                    last_response_time=last_response_time,
                    current_response_time=audio_start,
                )
                if response:
                    synthesise_speech(text=response)
                    last_response_time = dt.datetime.now()
