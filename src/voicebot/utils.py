"""Utility functions for the project."""

import requests

WEEKDAYS = ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"]
MONTHS = [
    "januar",
    "februar",
    "marts",
    "april",
    "maj",
    "juni",
    "juli",
    "august",
    "september",
    "oktober",
    "november",
    "december",
]


def is_internet_available() -> bool:
    """Check if the internet is available.

    Returns:
        True if the internet is available, False otherwise.
    """
    try:
        requests.get(url="https://www.google.com", timeout=5)
        return True
    except requests.exceptions.RequestException:
        return False
