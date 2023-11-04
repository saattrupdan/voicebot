"""A voice bot."""

from omegaconf import DictConfig
import openwakeword as oww
from .speech_recording import record_speech
from .speech_recognition import transcribe_speech
from .speech_synthesis import synthesise_speech
from .text_engine import TextEngine
from transformers import pipeline
import transformers.utils.logging as hf_logging
import logging
import datetime as dt


logger = logging.getLogger(__name__)


class VoiceBot:
    """A voice bot.

    Args:
        cfg: Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        hf_logging.set_verbosity_error()
        self.wake_word_model = oww.Model(
            wakeword_models=["hey_jarvis"], inference_framework="onnx"
        )
        self.asr_pipeline = pipeline(
            task="automatic-speech-recognition", model=self.cfg.asr_model_id
        )
        self.text_engine = TextEngine(cfg=cfg)

    def run(self) -> None:
        """Run the bot."""
        last_response_time = dt.datetime(year=1900, month=1, day=1)
        while True:
            speech, audio_start = record_speech(
                last_response_time=last_response_time, cfg=self.cfg
            )
            if audio_start is None:
                continue

            text = transcribe_speech(speech=speech, asr_pipeline=self.asr_pipeline)
            if text:
                response = self.text_engine.generate_response(
                    prompt=text,
                    last_response_time=last_response_time,
                    current_response_time=audio_start,
                )
                if response:
                    synthesise_speech(text=response)
                    last_response_time = dt.datetime.now()
