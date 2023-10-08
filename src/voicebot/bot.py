"""A voice bot."""

from .speech_recording import record_speech
from .speech_recognition import transcribe_speech
from .speech_synthesis import apple_synthesis
from .text_engine import TextEngine
from transformers import AutoModelForCTC, AutoProcessor
import transformers.utils.logging as hf_logging
import torch
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
        min_seconds_audio: float,
        wake_word: str,
    ) -> None:
        self.text_model_id = text_model_id
        self.asr_model_id = asr_model_id
        self.sample_rate = sample_rate
        self.temperature = temperature
        self.num_seconds_per_chunk = num_seconds_per_chunk
        self.min_audio_threshold = min_audio_threshold
        self.max_seconds_silence = max_seconds_silence
        self.min_seconds_audio = min_seconds_audio
        self.wake_word = wake_word

        logger.info("Loading models...")
        hf_logging.set_verbosity_error()
        self.asr_model = torch.compile(
            model=AutoModelForCTC.from_pretrained(self.asr_model_id).eval(),
        )
        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_id)
        self.text_engine = TextEngine(
            model_id=self.text_model_id,
            temperature=self.temperature,
            wake_word=self.wake_word,
        )

    def run(self) -> None:
        """Run the bot."""
        while True:
            logger.info("Ready. Note that the first transcription may be slow.")
            speech = record_speech(
                sample_rate=self.sample_rate,
                num_seconds_per_chunk=self.num_seconds_per_chunk,
                min_audio_threshold=self.min_audio_threshold,
                max_seconds_silence=self.max_seconds_silence,
                min_seconds_audio=self.min_seconds_audio,
            )
            text = transcribe_speech(
                speech=speech,
                asr_model=self.asr_model,
                asr_processor=self.asr_processor,
            )
            if text:
                response = self.text_engine.generate_response(prompt=text)
                apple_synthesis(text=response)
