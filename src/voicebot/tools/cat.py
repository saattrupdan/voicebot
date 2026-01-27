"""Cat noises tool."""

from pathlib import Path

import httpx
from playsound3 import playsound


def meow(state: dict) -> tuple[str, dict]:
    """Make a meow sound.

    Args:
        state:
            The current state of the text engine.

    Returns:
        A tuple (message, state) where message is a message indicating the sound has
        been played and state is information that the text engine should store.
    """
    # Get the cache path
    cache_dir = Path(".cache", "cat")
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_path = cache_dir / "cat-sound.mp3"

    # If the sound hasn't been downloaded yet then do it
    if not cache_path.exists():
        response = httpx.get(
            "https://cataas.com/cat/says/meow?size=50&color=red&json=true",
            params=dict(filename="cute-cat-meow-472372.mp3"),
        )
        if response.status_code != 200:
            return (
                "Kunne desværre ikke miaue rigtigt, men her kommer et forsøg: Miaauu!",
                state,
            )
        with open(cache_path, "wb") as f:
            f.write(response.content)

    # Play the sound
    playsound(sound=cache_path.as_posix())

    return "", state
