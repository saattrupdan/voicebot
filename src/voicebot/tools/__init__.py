"""Tools that the text engine can use."""

from pydantic import BaseModel

from .news import GetNewsResponse, get_news
from .timer import (
    ListTimersResponse,
    SetTimerResponse,
    StopTimerResponse,
    list_timers,
    set_timer,
    stop_timer,
)
from .weather import GetWeatherResponse, get_weather
from .web_search import SearchWebParameters, SearchWebResponse, search_web


class NonToolAnswer(BaseModel):
    """A response that is not a tool call."""

    response: str


class LLMResponse(BaseModel):
    """The response from the language model."""

    answer: (
        NonToolAnswer
        | GetWeatherResponse
        | SetTimerResponse
        | StopTimerResponse
        | ListTimersResponse
        | GetNewsResponse
        | SearchWebResponse
    )
