"""Generation of Danish speech."""

import subprocess


def apple_synthesis(text: str | None) -> None:
    """Generate speech using Apple's built-in speech synthesis.

    Args:
        text: Text to be spoken, or None if nothing should be spoken.
    """
    if text is not None:
        subprocess.call(["say", text])
