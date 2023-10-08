"""
.. include:: ../../README.md
"""

import importlib.metadata
from .bot import VoiceBot

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)
