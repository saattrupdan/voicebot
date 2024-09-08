"""Recording of speech."""

from collections.abc import Generator
from contextlib import contextmanager
import datetime as dt
import logging
from time import sleep

import numpy as np
import onnxruntime as ort
import openwakeword as oww
from openwakeword.utils import download_models as download_wakeword_models
from omegaconf import DictConfig
from pvrecorder import PvRecorder
import sounddevice

logger = logging.getLogger(__name__)


SAMPLE_RATE = 16_000


# Load the wake word model. This usually produces logs from `onnxruntime`, so we
# suppress them.
ort.set_default_logger_severity(3)
download_wakeword_models(model_names=["hey_jarvis"])
wake_word_model = oww.Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")


def record_speech(
    last_response_time: dt.datetime, audio_threshold: int, cfg: DictConfig
) -> tuple[np.ndarray, dt.datetime | None]:
    """Record speech and return it as text.

    Args:
        last_response_time:
            Time of the last response.
        audio_threshold:
            The minimum audio threshold.
        cfg:
            Hydra configuration object.

    Returns:
        Recorded speech, and the time at which the recording started (or None if no
        speech was recorded).
    """
    chunk_size = int(SAMPLE_RATE * cfg.num_seconds_per_chunk)

    logger.info("Listening for wakeword...")

    has_begun_talking: bool = False
    audio_start: dt.datetime | None = None
    num_silent_frames: int = 0
    frames: list[np.ndarray] = list()

    with record(chunk_size=chunk_size) as recorder:
        while num_silent_frames < cfg.max_seconds_silence // cfg.num_seconds_per_chunk:
            frame = np.asarray(recorder.read(), dtype=np.int16)
            max_value = frame[~np.isnan(frame)].max()

            if not has_begun_talking:
                # Check if it hasn't been too long since the last response
                if max_value >= audio_threshold:
                    response_delay = dt.datetime.now() - last_response_time
                    seconds_since_last_response = response_delay.total_seconds()
                    if seconds_since_last_response < cfg.follow_up_max_seconds:
                        logger.info("Follow-up detected!")
                        audio_start = dt.datetime.now()
                        has_begun_talking = True
                        num_silent_frames = 0
                        frames.append(frame)
                        wake_word_model.reset()
                        continue

                # Check if the wake_word is triggered and that we haven't already started
                # recording
                wake_word_prediction_dict = wake_word_model.predict(x=frame)
                assert isinstance(wake_word_prediction_dict, dict)
                wake_word_probability = wake_word_prediction_dict["hey_jarvis"]
                if wake_word_probability >= cfg.wake_word_probability_threshold:
                    logger.info("Wakeword detected!")
                    audio_start = dt.datetime.now()
                    has_begun_talking = True
                    num_silent_frames = 0
                    frames.append(frame)
                    wake_word_model.reset()
            else:
                frames.append(frame)
                if len(frames) * cfg.num_seconds_per_chunk >= cfg.max_seconds_audio:
                    logger.info("Max audio length reached, stopping.")
                    break
                if max_value < audio_threshold:
                    num_silent_frames += 1
                else:
                    num_silent_frames = 0

    audio_arr = np.concatenate(frames, axis=0)

    if cfg.play_back_audio:
        logger.info("Playing back the audio...")
        sounddevice.play(data=audio_arr, samplerate=SAMPLE_RATE)

    return audio_arr, audio_start


def calibrate_audio_threshold(cfg: DictConfig) -> int:
    """Calibrate the audio threshold.

    Args:
        cfg:
            Hydra configuration object.

    Returns:
        The calibrated audio threshold.
    """
    chunk_size = int(SAMPLE_RATE * cfg.num_seconds_per_chunk)

    logger.info("Calibrating audio threshold...")
    sleep(3)

    logger.info(
        "Here is some text:\n"
        "'Mette Frederiksen er en dansk socialdemokratisk politiker. Hun har været "
        "statsminister siden den 27. juni 2019.'"
    )
    sleep(3)
    logger.info("Please read the text aloud in 3...")
    sleep(1)
    logger.info("2...")
    sleep(1)
    logger.info("1...")
    sleep(1)
    logger.info("Read!")

    loud_values: list[int] = list()
    with record(chunk_size=chunk_size) as stream:
        for _ in range(int(cfg.calibration_duration / cfg.num_seconds_per_chunk)):
            frame = np.asarray(stream.read(), dtype=np.int16)
            max_value = frame[~np.isnan(frame)].max().astype(int)
            loud_values.append(max_value)

    audio_threshold = np.percentile(a=loud_values, q=25).astype(int)
    logger.info(f"Calibrated audio threshold: {audio_threshold}")

    return audio_threshold


@contextmanager
def record(chunk_size: int) -> Generator[PvRecorder, None, None]:
    """Context manager for recording audio.

    Args:
        chunk_size:
            The size of each audio chunk.

    Yields:
        The audio recorder.
    """
    recorder = PvRecorder(frame_length=chunk_size)
    recorder.start()
    yield recorder
    recorder.stop()
    recorder.delete()
