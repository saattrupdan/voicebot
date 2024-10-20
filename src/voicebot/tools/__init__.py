"""Skills that the text engine can use."""

from pydantic import BaseModel

from .weather import GetWeatherResponse, get_weather


class NoFunctionCallResponse(BaseModel):
    """The response when no function is called."""

    answer: str


class LLMResponse(BaseModel):
    """The response from the language model."""

    response: GetWeatherResponse | NoFunctionCallResponse
