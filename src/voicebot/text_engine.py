"""The engine that produces new responses."""

import datetime as dt
import logging
import os

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from .utils import MONTHS, WEEKDAYS
from .weather import get_weather_forecast

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
        self.weather_forecast = get_weather_forecast()

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
        if len(prompt.strip()) <= self.cfg.min_prompt_length:
            logger.info("The prompt is too short, ignoring it.")
            return None

        response_delay = current_response_time - last_response_time
        seconds_since_last_response = response_delay.total_seconds()
        if seconds_since_last_response > self.cfg.follow_up_max_seconds:
            system_prompt = self.cfg.system_prompt.strip().format(
                weekday=WEEKDAYS[dt.datetime.now().weekday()],
                day=dt.datetime.now().day,
                month=MONTHS[dt.datetime.now().month - 1],
                year=dt.datetime.now().year,
                time=dt.datetime.now().strftime("%H:%M"),
                weather_forecast=self.weather_forecast,
            )
            self.conversation = [
                ChatCompletionSystemMessageParam(role="system", content=system_prompt)
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

        # Fix some consistent typos
        for before, after in self.cfg.manual_fixes.items():
            response = response.replace(before, after)

        self.conversation.append(
            ChatCompletionAssistantMessageParam(role="assistant", content=response)
        )
        logger.info(f"Generated the response: {response!r}")
        return response
