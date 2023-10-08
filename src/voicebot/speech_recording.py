"""Recording of speech."""

import pyaudio
import sys
import numpy as np


def record_speech(sample_rate: int = 16_000, chunk_size: int = 4000) -> np.ndarray:
    """Record speech and return it as text.

    Args:
        sample_rate: Sample rate.
        chunk_size: Chunk size.

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

    frames: list[np.ndarray] = list()
    num_silent_frames: int = 0
    has_begun_talking: bool = False
    while num_silent_frames < 5:
        # Record a chunk of audio and append it to the list of frames
        chunk = stream.read(num_frames=chunk_size, exception_on_overflow=False)
        frame = np.frombuffer(buffer=chunk, dtype=np.float32)
        frames.append(frame)

        # Stop the stream when the user stops talking
        if frame.max() < 0.10 and has_begun_talking:
            num_silent_frames += 1
        elif frame.max() >= 0.10:
            has_begun_talking = True
            num_silent_frames = 0

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return np.concatenate(frames, axis=0)
