"""Recording of speech."""

import pyaudio
import sys
import numpy as np
import logging


logger = logging.getLogger(__name__)


def record_speech(
    sample_rate: int,
    num_seconds_per_chunk: float,
    min_audio_threshold: float,
    max_seconds_silence: float,
    min_seconds_audio: float,
) -> np.ndarray:
    """Record speech and return it as text.

    Args:
        sample_rate: Sample rate.
        num_seconds_per_chunk: Number of seconds per chunk.
        min_audio_threshold: Minimum amplitude for audio to be considered speech.
        max_seconds_silence: Maximum number of seconds of silence before the recording
        min_seconds_audio: Minimum number of seconds of audio to be considered speech.

    Returns:
        Recorded speech.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1 if sys.platform == "darwin" else 2,
        rate=sample_rate,
        input=True,
    )

    max_num_silent_frames = max_seconds_silence // num_seconds_per_chunk
    chunk_size = int(sample_rate * num_seconds_per_chunk)

    frames: list[np.ndarray] = list()
    num_silent_frames: int = 0
    has_begun_talking: bool = False
    while num_silent_frames < max_num_silent_frames:
        # Record a chunk of audio and append it to the list of frames
        chunk = stream.read(num_frames=chunk_size, exception_on_overflow=False)
        frame = np.frombuffer(buffer=chunk, dtype=np.float32)

        # Stop the stream when the user stops talking
        if frame.max() < min_audio_threshold and has_begun_talking:
            num_silent_frames += 1
            seconds_audio = len(frames) * num_seconds_per_chunk

            # If there's been enough silence and the audio is too short, restart
            if (
                num_silent_frames >= max_num_silent_frames
                and seconds_audio < min_seconds_audio
            ):
                logger.info("Audio too short!")
                frames = list()
                num_silent_frames = 0
                has_begun_talking = False

        elif frame.max() >= min_audio_threshold:
            if not has_begun_talking:
                logger.info("Audio detected!")
            has_begun_talking = True
            num_silent_frames = 0
            frames.append(frame)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return np.concatenate(frames, axis=0)
