"""Weather forecast tool."""

import logging
import re
from pathlib import Path

import httpx
import numpy as np
import requests
import requests_cache
from openmeteo_requests import Client, OpenMeteoRequestsError
from retry_requests import retry

logger = logging.getLogger(__name__)

IP_LOCATION_URLS = ("https://ipapi.co/json/", "https://ipwho.is/")

WEATHER_CODES = {
    0: "Klar himmel",
    1: "Næsten klar himmel",
    2: "Delvist skyet",
    3: "Overskyet",
    45: "Tåge",
    48: "Tåge",
    51: "Lette byger",
    53: "Moderate byger",
    55: "Tætte byger",
    56: "Lette isslag",
    57: "Tætte isslag",
    61: "Lette regnbyger",
    63: "Moderate regnbyger",
    65: "Tunge regnbyger",
    66: "Lette isslag",
    67: "Tunge isslag",
    71: "Lette snebyger",
    73: "Moderate snebyger",
    75: "Tunge snebyger",
    77: "Snefnug",
    80: "Lette regnbyger",
    81: "Moderate regnbyger",
    82: "Tunge regnbyger",
    85: "Lette snebyger",
    86: "Tunge snebyger",
    95: "Tordenvejr",
    96: "Tordenvejr med let hagl",
    99: "Tordenvejr med tungt hagl",
}


def get_weather(state: dict, location: str) -> tuple[str, dict]:
    """Get the weather forecast for today.

    Args:
        state:
            The current state of the text engine.
        location:
            The location to get the weather forecast for. Can be an empty string to use
            the current IP location.

    Returns:
        A pair (message, state) where message is the weather forecast and state is
        information that the text engine should store.
    """
    location = location.strip()
    if not location:
        remembered_location = state.get("weather_location")
        if isinstance(remembered_location, str) and remembered_location:
            location = remembered_location
            logger.info(f"Using the remembered weather location: {location!r}")
        else:
            try:
                location = _get_current_city()
                logger.info(f"Using the current IP location: {location!r}")
            except ValueError as error:
                logger.warning(f"Could not resolve the current IP location: {error}")
                default_location = state.get("weather_default_location")
                if isinstance(default_location, str) and default_location:
                    location = default_location
                    logger.info(f"Using the default weather location: {location!r}")
                else:
                    return (
                        "Jeg kunne ikke finde din placering. Spørg brugeren hvilken by "
                        "vejrudsigten skal gælde for.",
                        dict(),
                    )

    try:
        latitude, longitude = _geocode_location(location=location)
    except (httpx.HTTPError, TypeError, ValueError) as error:
        logger.error(f"Could not resolve weather location: {error}")
        return "Ingen vejrudsigt tilgængelig.", dict()

    weather_cache = Path(".cache", "weather")
    weather_cache.mkdir(exist_ok=True, parents=True)
    cache_name = re.sub(r"[ ,_]+", "-", location).lower()
    cache_path = weather_cache / cache_name

    openmeteo = Client(
        session=retry(  # pyrefly: ignore[bad-argument-type]
            session=requests_cache.CachedSession(
                cache_name=cache_path.as_posix(), expire_after=3600
            ),
            retries=5,
            backoff_factor=0.2,
        )
    )

    try:
        response = openmeteo.weather_api(
            url="https://api.open-meteo.com/v1/forecast",
            params=dict(
                latitude=latitude,
                longitude=longitude,
                wind_speed_unit="ms",
                hourly=[
                    "weather_code",
                    "temperature_2m",
                    "precipitation",
                    "wind_speed_10m",
                ],
                forecast_days=2,
            ),
        )[0].Hourly()
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OpenMeteoRequestsError,
        requests.RequestException,
    ) as error:
        logger.error(f"Could not retrieve the weather forecast: {error}")
        return "Ingen vejrudsigt tilgængelig.", dict()
    if response is None:
        return "Ingen vejrudsigt tilgængelig.", dict()

    forecast = {
        "Vejrtype": response.Variables(0),
        "Temperatur (i celcius)": response.Variables(1),
        "Nedbør (i millimeter)": response.Variables(2),
        "Vindhastighed (i meter per sekund)": response.Variables(3),
    }

    out = f"Vejrdata for {location}:\n\n"
    for variable_name, variable in forecast.items():
        out += f"{variable_name}:\n"
        if variable is None:
            out += "Ingen data tilgængelig.\n\n"
            continue

        values_arr = variable.ValuesAsNumpy()
        assert isinstance(values_arr, np.ndarray), "Values should be a NumPy array."

        if variable_name == "Vejrtype":
            values = [WEATHER_CODES.get(value, "Ukendt vejr") for value in values_arr]
        else:
            values = [str(int(round(value, 0))) for value in values_arr]

        intervals = {
            "I dag kl. 1-12": (0, 12),
            "I dag kl. 13-24": (12, 24),
            "I morgen kl. 1-12": (24, 36),
            "I morgen kl. 13-24": (36, 48),
        }
        for interval_name, (start, end) in intervals.items():
            interval_values = ", ".join(str(value) for value in values[start:end])
            out += f"{interval_name}: {interval_values}\n"
        out += "\n"

    return out, {"weather_location": location}


def _get_current_city() -> str:
    """Resolve the current city from the public IP address."""
    errors: list[str] = []
    for url in IP_LOCATION_URLS:
        try:
            response = httpx.get(url, timeout=5)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("the response was not an object")
            if payload.get("success") is False:
                raise ValueError(str(payload.get("message", "provider failure")))
            city = payload.get("city")
            if not isinstance(city, str) or not city.strip():
                raise ValueError("the response did not contain a city")
            return city.strip()
        except (httpx.HTTPError, TypeError, ValueError) as error:
            logger.warning(f"IP location provider {url!r} failed: {error}")
            errors.append(f"{url}: {error}")

    raise ValueError("; ".join(errors))


def _geocode_location(location: str) -> tuple[float, float]:
    """Resolve a place name to latitude and longitude."""
    response = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "da", "format": "json"},
        timeout=5,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        raise ValueError(f"No coordinates found for {location!r}.")
    return float(results[0]["latitude"]), float(results[0]["longitude"])
