"""A voice bot."""

from .speech_recording import record_speech
from .speech_recognition import transcribe_speech
from .speech_synthesis import apple_synthesis
from .text_engine import TextEngine
from transformers import pipeline
import logging


logger = logging.getLogger(__name__)


class VoiceBot:
    """A voice bot."""

    def __init__(self, temperature: float) -> None:
        self.asr_pipeline = pipeline(
            model="chcaa/xls-r-300m-danish-nst-cv9", task="automatic-speech-recognition"
        )
        self.text_engine = TextEngine(temperature=temperature)

    def run(self) -> None:
        """Run the bot."""
        while True:
            logger.info("Listening...")
            speech = record_speech()
            text = transcribe_speech(
                speech=speech, sample_rate=16_000, asr_pipeline=self.asr_pipeline
            )
            logger.info(f"Heard the following: {text!r}")
            response = self.text_engine.generate_response(prompt=text)
            logger.info(f"Generated the response: {response!r}")
            apple_synthesis(text=response)
