"""A voice bot."""

import datetime as dt
import logging

import onnxruntime as ort
import openwakeword as oww
import torch
import transformers.utils.logging as hf_logging
from omegaconf import DictConfig
from openwakeword.utils import download_models as download_wakeword_models
from transformers.pipelines import Pipeline, pipeline

from .speech_recognition import transcribe_speech
from .speech_recording import calibrate_audio_threshold, record_speech
from .speech_synthesis import DanishChatterBox, synthesise_speech
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
        ort.set_default_logger_severity(3)
        download_wakeword_models(model_names=["hey_jarvis"])
        self.wake_word_model = oww.Model(
            wakeword_models=["hey_jarvis"], inference_framework="onnx"
        )

        logger.info("Loading the text engine model...")
        self.text_engine = TextEngine(cfg=self.cfg)

        # TEMP
        # self.text_engine.generate_response(
        #     prompt="Hvad er dit navn?",
        #     last_response_time=dt.datetime(year=1900, month=1, day=1),
        #     current_response_time=dt.datetime.now(),
        # )
        # self.text_engine.generate_response(
        #     prompt="Sæt en timer på 5 minutter.",
        #     last_response_time=dt.datetime(year=1900, month=1, day=1),
        #     current_response_time=dt.datetime.now(),
        # )
        # self.text_engine.generate_response(
        #     prompt="Sæt en timer på 2 minutter.",
        #     last_response_time=dt.datetime(year=1900, month=1, day=1),
        #     current_response_time=dt.datetime.now(),
        # )
        # self.text_engine.generate_response(
        #     prompt="Stop timeren på 5 minutter.",
        #     last_response_time=dt.datetime(year=1900, month=1, day=1),
        #     current_response_time=dt.datetime.now(),
        # )
        # self.text_engine.generate_response(
        #     prompt="Hvilke timere kører?",
        #     last_response_time=dt.datetime(year=1900, month=1, day=1),
        #     current_response_time=dt.datetime.now(),
        # )

        logger.info("Loading the speech synthesis model...")
        self.synthesiser = DanishChatterBox.from_pretrained(device=self.device)

        logger.info("Loading the speech recognition model...")
        self.transcriber: Pipeline = pipeline(
            model=self.cfg.asr_model_id,
            device=self.device,
            task="automatic-speech-recognition",
        )

    @property
    def device(self) -> torch.device:
        """Return the device on which the bot is running."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def run(self) -> None:
        """Run the bot."""
        last_response_time = dt.datetime(year=1900, month=1, day=1)

        logger.info("Playing welcome message...")
        synthesise_speech(text=self.cfg.starting_phrase, synthesiser=self.synthesiser)

        while True:
            speech, audio_start = record_speech(
                last_response_time=last_response_time,
                audio_threshold=self.audio_threshold,
                cfg=self.cfg,
                synthesiser=self.synthesiser,
                wake_word_model=self.wake_word_model,
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
                    synthesise_speech(text=response, synthesiser=self.synthesiser)
                    last_response_time = dt.datetime.now()
