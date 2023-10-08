"""A voice bot."""

from .speech_recording import record_speech
from .speech_recognition import transcribe_speech
from .speech_synthesis import apple_synthesis
from .text_engine import TextEngine
from transformers import pipeline
import transformers.utils.logging as hf_logging
import logging
import datetime as dt


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
        min_seconds_audio: float,
        max_seconds_audio: float,
        follow_up_max_seconds: float,
        wake_words: list[str],
    ) -> None:
        self.text_model_id = text_model_id
        self.asr_model_id = asr_model_id
        self.sample_rate = sample_rate
        self.temperature = temperature
        self.num_seconds_per_chunk = num_seconds_per_chunk
        self.min_audio_threshold = min_audio_threshold
        self.max_seconds_silence = max_seconds_silence
        self.min_seconds_audio = min_seconds_audio
        self.max_seconds_audio = max_seconds_audio
        self.follow_up_max_seconds = follow_up_max_seconds
        self.wake_words = wake_words

        hf_logging.set_verbosity_error()

        self.asr_pipeline = pipeline(
            task="automatic-speech-recognition", model=self.asr_model_id
        )
        self.text_engine = TextEngine(
            model_id=self.text_model_id,
            temperature=self.temperature,
            wake_words=self.wake_words,
            follow_up_max_seconds=self.follow_up_max_seconds,
        )

    def run(self) -> None:
        """Run the bot."""
        last_response_time = dt.datetime(year=1900, month=1, day=1)
        while True:
            speech = record_speech(
                sample_rate=self.sample_rate,
                num_seconds_per_chunk=self.num_seconds_per_chunk,
                min_audio_threshold=self.min_audio_threshold,
                max_seconds_silence=self.max_seconds_silence,
                min_seconds_audio=self.min_seconds_audio,
                max_seconds_audio=self.max_seconds_audio,
            )
            text = transcribe_speech(speech=speech, asr_pipeline=self.asr_pipeline)
            if text:
                response = self.text_engine.generate_response(
                    prompt=text, last_response_time=last_response_time
                )
                apple_synthesis(text=response)
                last_response_time = dt.datetime.now()
