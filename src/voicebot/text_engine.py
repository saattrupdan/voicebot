"""The engine that produces new responses."""

import datetime as dt
import logging
import os
import re

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from .utils import get_weather_forecast

load_dotenv()
logger = logging.getLogger(__name__)


class TextEngine:
    """The engine that produces new responses."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialise the engine.

        Args:
            cfg:
                The Hydra configuration.
        """
        self.cfg = cfg
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), base_url=cfg.server
        )
        self.conversation: list[ChatCompletionMessageParam] = list()

    def generate_response(
        self,
        prompt: str,
        last_response_time: dt.datetime,
        current_response_time: dt.datetime,
    ) -> str | None:
        """Generate a new response from a prompt.

        Args:
            prompt:
                Prompt to generate a response from.
            last_response_time:
                Time of the last response.
            current_response_time:
                Time of the current response.

        Returns:
            Generated response, or None if prompt should not be responded to.
        """
        if len(prompt.strip()) <= 1:
            logger.info("The prompt is too short, ignoring it.")
            return None

        mentions_weather = re.search(
            pattern=r"\b(vejret|været|erhvervet)\b", string=prompt, flags=re.IGNORECASE
        )
        if mentions_weather:
            weather_forecast = get_weather_forecast(location="Copenhagen")
            prompt = f"{weather_forecast}\n\n{prompt}"
            logger.info(
                "Mentioned weather, appending the weather forecast: "
                f"{weather_forecast!r}"
            )

        response_delay = current_response_time - last_response_time
        seconds_since_last_response = response_delay.total_seconds()
        if seconds_since_last_response > self.cfg.follow_up_max_seconds:
            self.conversation = [
                ChatCompletionSystemMessageParam(
                    role="system", content=self.cfg.system_prompt.strip()
                )
            ]

        self.conversation.append(
            ChatCompletionUserMessageParam(role="user", content=prompt)
        )
        llm_answer = self.client.chat.completions.create(
            model=self.cfg.text_model_id,
            messages=self.conversation,
            temperature=self.cfg.temperature,
        )
        response = llm_answer.choices[0].message.content
        if response is None:
            logger.info("The response is empty, ignoring it.")
            return None
        response = response.strip()

        self.conversation.append(
            ChatCompletionAssistantMessageParam(role="assistant", content=response)
        )
        logger.info(f"Generated the response: {response!r}")
        return response
