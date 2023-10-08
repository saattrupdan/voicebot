"""Generation of Danish speech."""

import subprocess


def apple_synthesis(text: str) -> None:
    """Generate speech using Apple's built-in speech synthesis.

    Args:
        text: Text to be spoken.
    """
    subprocess.call(["say", text])
