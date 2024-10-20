"""A simple Danish voice bot.

.. include:: ../../README.md
"""

import importlib.metadata
import logging
import warnings

from .bot import VoiceBot

logging.getLogger("httpx").setLevel(logging.WARNING)


# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
