"""A voice bot."""

from .speech_recording import record_speech
from .speech_recognition import transcribe_speech
from .speech_synthesis import apple_synthesis
from .text_engine import TextEngine
from transformers import pipeline
import transformers.utils.logging as hf_logging
import logging


logger = logging.getLogger(__name__)


class VoiceBot:
    """A voice bot."""

    def __init__(
        self,
        asr_model_id: str,
        sample_rate: int,
        temperature: float,
        num_seconds_per_chunk: float,
        min_audio_threshold: float,
    ) -> None:
        hf_logging.set_verbosity_error()
        self.asr_pipeline = pipeline(
            model=asr_model_id, task="automatic-speech-recognition"
        )
        self.text_engine = TextEngine(temperature=temperature)
        self.sample_rate = sample_rate
        self.num_seconds_per_chunk = num_seconds_per_chunk
        self.min_audio_threshold = min_audio_threshold

    def run(self) -> None:
        """Run the bot."""
        while True:
            logger.info("Listening...")
            speech = record_speech(
                sample_rate=self.sample_rate,
                num_seconds_per_chunk=self.num_seconds_per_chunk,
                min_audio_threshold=self.min_audio_threshold,
            )

            text = transcribe_speech(speech=speech, asr_pipeline=self.asr_pipeline)
            logger.info(f"Heard the following: {text!r}")

            response = self.text_engine.generate_response(prompt=text)
            logger.info(f"Generated the response: {response!r}")

            apple_synthesis(text=response)
