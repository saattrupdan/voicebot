"""A voice bot."""

import datetime as dt
import logging
import os

import onnxruntime as ort
import openai
import openwakeword as oww
from dotenv import load_dotenv
from omegaconf import DictConfig
from openwakeword.utils import download_models as download_wakeword_models

from .speech_recognition import transcribe_speech
from .speech_recording import create_voice_activity_detector, record_speech
from .speech_synthesis import SpeechSynthesiser, synthesise_speech
from .text_engine import TextEngine

load_dotenv()
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

        self.speech_detector = create_voice_activity_detector(cfg=self.cfg)

        logger.info("Loading the wake word model...")
        ort.set_default_logger_severity(3)
        download_wakeword_models(model_names=["hey_jarvis"], inference_framework="onnx")
        self.wake_word_model = oww.Model(
            wakeword_models=["hey_jarvis"], inference_framework="onnx"
        )

        logger.info("Configuring the SYV audio client...")
        self.syv_client = openai.OpenAI(
            api_key=os.environ["SYV_API_KEY"], base_url=self.cfg.syv_server
        )
        self.synthesiser = SpeechSynthesiser(
            client=self.syv_client,
            model=self.cfg.tts_model_id,
            voice=self.cfg.tts_voice,
        )

        logger.info("Loading the text engine model...")
        self.text_engine = TextEngine(cfg=self.cfg)
        self.text_engine.state["synthesiser"] = self.synthesiser

    def run(self) -> None:
        """Run the bot."""
        last_response_time = dt.datetime(year=1900, month=1, day=1)

        logger.info("Playing welcome message...")
        synthesise_speech(text=self.cfg.starting_phrase, synthesiser=self.synthesiser)

        while True:
            speech, audio_start = record_speech(
                last_response_time=last_response_time,
                detector=self.speech_detector,
                cfg=self.cfg,
                synthesiser=self.synthesiser,
                wake_word_model=self.wake_word_model,
            )
            if audio_start is None:
                continue

            text = transcribe_speech(
                speech=speech,
                client=self.syv_client,
                model=self.cfg.asr_model_id,
                language=self.cfg.asr_language,
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
