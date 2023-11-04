"""Recording of speech."""

from omegaconf import DictConfig
import pyaudio
import numpy as np
import logging
import datetime as dt
import openwakeword as oww
import onnxruntime as ort


logger = logging.getLogger(__name__)


# Load the wake word model. This usually produces logs from `onnxruntime`, so we
# suppress them.
ort.set_default_logger_severity(3)
wake_word_model = oww.Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")


def record_speech(
    last_response_time: dt.datetime, cfg: DictConfig
) -> tuple[np.ndarray, dt.datetime | None]:
    """Record speech and return it as text.

    Args:
        last_response_time: Time of the last response.
        cfg: Hydra configuration object.

    Returns:
        Recorded speech, and the time at which the recording started (or None if no
        speech was recorded).
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=cfg.sample_rate,
        input=True,
    )

    max_num_silent_frames = cfg.max_seconds_silence // cfg.num_seconds_per_chunk
    chunk_size = int(cfg.sample_rate * cfg.num_seconds_per_chunk)

    logger.info("Listening...")

    frames: list[np.ndarray] = list()
    num_silent_frames: int = 0
    has_begun_talking: bool = False
    audio_start = None
    while num_silent_frames < max_num_silent_frames:
        # Record a chunk of audio
        chunk = stream.read(num_frames=chunk_size, exception_on_overflow=False)
        frame = np.frombuffer(buffer=chunk, dtype=np.int16)
        max_value = frame[~np.isnan(frame)].max()

        if not has_begun_talking:
            # Check if it hasn't been too long since the last response
            if max_value >= cfg.min_audio_threshold:
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
            wake_word_probability = wake_word_model.predict(frame)["hey_jarvis"]
            if wake_word_probability >= cfg.wake_word_probability_threshold:
                logger.info("Wakeword detected!")
                audio_start = dt.datetime.now()
                has_begun_talking = True
                num_silent_frames = 0
                frames.append(frame)
                wake_word_model.reset()

            continue

        if len(frames) * cfg.num_seconds_per_chunk >= cfg.max_seconds_audio:
            logger.info("Max audio length reached, stopping.")
            break

        # Check if the audio is silent, and stop recording if it has been silent for
        # too long
        if max_value < cfg.min_audio_threshold:
            num_silent_frames += 1
        elif max_value >= cfg.min_audio_threshold:
            num_silent_frames = 0
            frames.append(frame)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Concatenate the frames into a single array, and convert the datatype to float32
    audio = np.concatenate(frames, axis=0).astype(np.float32, order="C") / 32768.0

    return audio, audio_start
