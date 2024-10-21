"""Tools that the text engine can use."""

from pydantic import BaseModel

from .timer import (
    ListTimersResponse,
    SetTimerResponse,
    StopTimerResponse,
    list_timers,
    set_timer,
    stop_timer,
)
from .weather import GetWeatherResponse, get_weather


class NoFunctionCallResponse(BaseModel):
    """The response when no function is called."""

    answer: str


class LLMResponse(BaseModel):
    """The response from the language model."""

    response: (
        NoFunctionCallResponse
        | GetWeatherResponse
        | SetTimerResponse
        | StopTimerResponse
        | ListTimersResponse
    )
