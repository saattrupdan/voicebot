"""Tests for weather-location lookup."""

from unittest.mock import MagicMock

import pytest

from voicebot.tools import weather


def test_get_current_city(monkeypatch: pytest.MonkeyPatch) -> None:
    """The current city is extracted from the IP-location response."""
    response = MagicMock()
    response.json.return_value = {"city": "Frederiksberg"}
    get = MagicMock(return_value=response)
    monkeypatch.setattr(weather.httpx, "get", get)

    city = weather._get_current_city()

    assert city == "Frederiksberg"
    get.assert_called_once_with("https://ipapi.co/json/", timeout=5)
    response.raise_for_status.assert_called_once_with()


def test_geocode_location(monkeypatch: pytest.MonkeyPatch) -> None:
    """Place names are resolved through Open-Meteo's geocoding API."""
    response = MagicMock()
    response.json.return_value = {"results": [{"latitude": 55.68, "longitude": 12.53}]}
    get = MagicMock(return_value=response)
    monkeypatch.setattr(weather.httpx, "get", get)

    coordinates = weather._geocode_location(location="Frederiksberg")

    assert coordinates == (55.68, 12.53)
    get.assert_called_once_with(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": "Frederiksberg",
            "count": 1,
            "language": "da",
            "format": "json",
        },
        timeout=5,
    )
    response.raise_for_status.assert_called_once_with()
