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
        text_model_id: str,
        asr_model_id: str,
        sample_rate: int,
        temperature: float,
        num_seconds_per_chunk: float,
        min_audio_threshold: float,
        max_seconds_silence: float,
        wake_word: str,
    ) -> None:
        self.text_model_id = text_model_id
        self.asr_model_id = asr_model_id
        self.sample_rate = sample_rate
        self.temperature = temperature
        self.num_seconds_per_chunk = num_seconds_per_chunk
        self.min_audio_threshold = min_audio_threshold
        self.max_seconds_silence = max_seconds_silence
        self.wake_word = wake_word

        hf_logging.set_verbosity_error()
        self.asr_pipeline = pipeline(
            model=self.asr_model_id, task="automatic-speech-recognition"
        )
        self.text_engine = TextEngine(
            model_id=self.text_model_id,
            temperature=self.temperature,
            wake_word=self.wake_word,
        )

    def run(self) -> None:
        """Run the bot."""
        while True:
            logger.info("Listening...")
            speech = record_speech(
                sample_rate=self.sample_rate,
                num_seconds_per_chunk=self.num_seconds_per_chunk,
                min_audio_threshold=self.min_audio_threshold,
                max_seconds_silence=self.max_seconds_silence,
            )

            text = transcribe_speech(speech=speech, asr_pipeline=self.asr_pipeline)
            logger.info(f"Heard the following: {text!r}")

            response = self.text_engine.generate_response(prompt=text)
            logger.info(f"Generated the response: {response!r}")

            apple_synthesis(text=response)
