"""A simple Danish voice bot.

.. include:: ../../README.md
"""

import importlib.metadata
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("httpx").setLevel(logging.WARNING)

from .bot import VoiceBot  # noqa: E402

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__ or "voicebot")
