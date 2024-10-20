"""Weather-related functions."""

import logging
import os
from typing import Literal

import geocoder
import numpy as np
import requests_cache
from openmeteo_requests import Client
from pydantic import BaseModel
from retry_requests import retry

from ..utils import is_internet_available

logging.getLogger("geocoder.base").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


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


def get_weather(location: str | None = None) -> str:
    """Get the weather forecast for today.

    Args:
        location:
            The location to get the weather forecast for. Can be None to use the
            current IP location.

    Returns:
        The weather forecast for today.
    """
    if not is_internet_available():
        return "Ingen vejrudsigt, da internettet ikke er tilgængeligt."

    if location is None:
        location = geocoder.ip("me").address
        logger.info(
            f"No location provided, using the current IP location: {location!r}"
        )

    openmeteo = Client(
        session=retry(
            session=requests_cache.CachedSession(".cache", expire_after=3600),
            retries=5,
            backoff_factor=0.2,
        )
    )

    coordinates = geocoder.geonames(
        location=location, key=os.getenv("GEONAMES_USERNAME")
    ).json

    response = openmeteo.weather_api(
        url="https://api.open-meteo.com/v1/forecast",
        params=dict(
            latitude=coordinates["lat"],
            longitude=coordinates["lng"],
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
    if response is None:
        return "Ingen vejrudsigt tilgængelig."

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

    return out


class GetWeatherParameters(BaseModel):
    """The parameters for the get_weather function."""

    location: str | None


class GetWeatherResponse(BaseModel):
    """A response containing the weather forecast."""

    name: Literal["get_weather"]
    parameters: GetWeatherParameters
