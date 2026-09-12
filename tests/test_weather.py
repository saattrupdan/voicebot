"""Tests for weather-location lookup."""

from unittest.mock import MagicMock

import httpx
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


def test_get_current_city_falls_back_after_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate-limited primary IP provider falls back to the secondary."""
    request = httpx.Request("GET", "https://ipapi.co/json/")
    primary = MagicMock()
    primary.raise_for_status.side_effect = httpx.HTTPStatusError(
        "rate limited", request=request, response=httpx.Response(429, request=request)
    )
    secondary = MagicMock()
    secondary.json.return_value = {"success": True, "city": "København"}
    get = MagicMock(side_effect=[primary, secondary])
    monkeypatch.setattr(weather.httpx, "get", get)

    city = weather._get_current_city()

    assert city == "København"
    assert get.call_count == 2
    assert get.call_args_list[1].args == ("https://ipwho.is/",)


def test_get_weather_requests_city_when_ip_providers_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable geolocation providers produce a useful non-exception result."""
    monkeypatch.setattr(
        weather.httpx, "get", MagicMock(side_effect=httpx.ConnectError("offline"))
    )

    message, updates = weather.get_weather(state={}, location="")

    assert "hvilken by" in message
    assert updates == {}


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
