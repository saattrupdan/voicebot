"""Recording of speech."""

import datetime as dt
import logging
from time import sleep

import numpy as np
import onnxruntime as ort
import openwakeword as oww
from openwakeword.utils import download_models as download_wakeword_models
from omegaconf import DictConfig

# This is required for the PyAudio library to work.
# See https://github.com/OpenInterpreter/01/issues/68#issuecomment-2199165891
import sounddevice  # noqa: F401
import pyaudio  # noqa: F401


logger = logging.getLogger(__name__)
sounddevice.query_devices()


# Load the wake word model. This usually produces logs from `onnxruntime`, so we
# suppress them.
ort.set_default_logger_severity(3)
download_wakeword_models(model_names=["hey_jarvis"])
wake_word_model = oww.Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")


def record_speech(
    last_response_time: dt.datetime, min_audio_threshold: int, cfg: DictConfig
) -> tuple[np.ndarray, dt.datetime | None]:
    """Record speech and return it as text.

    Args:
        last_response_time:
            Time of the last response.
        min_audio_threshold:
            The minimum audio threshold.
        cfg:
            Hydra configuration object.

    Returns:
        Recorded speech, and the time at which the recording started (or None if no
        speech was recorded).
    """
    chunk_size = int(cfg.sample_rate * cfg.num_seconds_per_chunk)

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=cfg.sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    logger.info("Listening for wakeword...")

    has_begun_talking: bool = False
    audio_start = None
    num_silent_frames: int = 0
    frames: list[np.ndarray] = list()

    while num_silent_frames < cfg.max_seconds_silence // cfg.num_seconds_per_chunk:
        # Record a chunk of audio
        chunk = stream.read(num_frames=chunk_size, exception_on_overflow=False)
        frame = np.frombuffer(buffer=chunk, dtype=np.int16)
        max_value = frame[~np.isnan(frame)].max()

        if not has_begun_talking:
            # Check if it hasn't been too long since the last response
            if max_value >= min_audio_threshold:
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
            logger.info(f"Wakeword probability: {wake_word_probability:.2f}")
            if wake_word_probability >= cfg.wake_word_probability_threshold:
                logger.info("Wakeword detected!")
                audio_start = dt.datetime.now()
                has_begun_talking = True
                num_silent_frames = 0
                frames.append(frame)
                wake_word_model.reset()

            continue

        frames.append(frame)

        if len(frames) * cfg.num_seconds_per_chunk >= cfg.max_seconds_audio:
            logger.info("Max audio length reached, stopping.")
            break

        # Check if the audio is silent, and stop recording if it has been silent for
        # too long
        if max_value < min_audio_threshold:
            num_silent_frames += 1
        else:
            num_silent_frames = 0

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Concatenate the frames into a single array, and convert the datatype to float32
    audio_arr = np.concatenate(frames, axis=0).astype(np.float32, order="C") / 32768.0

    # Play the audio back
    if cfg.play_back_audio:
        logger.info("Playing back the audio...")
        sounddevice.play(data=audio_arr, samplerate=cfg.sample_rate)

    return audio_arr, audio_start


def calibrate_audio_threshold(cfg: DictConfig) -> int:
    """Calibrate the audio threshold.

    Args:
        cfg:
            Hydra configuration object.

    Returns:
        The calibrated audio threshold.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16, channels=1, rate=cfg.sample_rate, input=True
    )
    chunk_size = int(cfg.sample_rate * cfg.num_seconds_per_chunk)

    num_chunks_in_five_seconds = int(5 / cfg.num_seconds_per_chunk)

    logger.info("Calibrating audio threshold...")
    sleep(3)

    logger.info("Please be quiet in 3...")
    sleep(1)
    logger.info("2...")
    sleep(1)
    logger.info("1...")
    sleep(1)
    logger.info("Quiet!")

    quiet_values: list[int] = list()
    for _ in range(num_chunks_in_five_seconds):
        chunk = stream.read(num_frames=chunk_size, exception_on_overflow=False)
        frame = np.frombuffer(buffer=chunk, dtype=np.int16)
        max_value = frame[~np.isnan(frame)].max().astype(int)
        quiet_values.append(max_value)
    quiet_value = np.median(a=quiet_values).astype(int)
    logger.info(f"The median quiet value was {quiet_value}")

    logger.info(
        "Here is some text:\n"
        "'A chatbot is a software application or web interface that is designed to "
        "mimic human conversation through text or voice interactions.'"
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
    for _ in range(num_chunks_in_five_seconds):
        chunk = stream.read(num_frames=chunk_size, exception_on_overflow=False)
        frame = np.frombuffer(buffer=chunk, dtype=np.int16)
        max_value = frame[~np.isnan(frame)].max().astype(int)
        loud_values.append(max_value)
    loud_value = np.median(a=loud_values).astype(int)
    logger.info(f"The median loud value was {loud_value}")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    min_audio_threshold = (quiet_value + loud_value) // 2
    logger.info(f"Calibrated audio threshold: {min_audio_threshold}")
    return min_audio_threshold
