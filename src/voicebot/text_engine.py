"""The engine that produces new responses."""

import datetime as dt
import logging
import os

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig

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
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def reset_conversation(self) -> None:
        """Reset the conversation, only keeping the system prompt."""
        self.conversation = [
            dict(role="system", content=self.cfg.system_prompt.strip())
        ]

    def generate_response(
        self,
        prompt: str,
        last_response_time: dt.datetime,
        current_response_time: dt.datetime,
    ) -> str | None:
        """Generate a new response from a prompt.

        Args:
            prompt: Prompt to generate a response from.
            last_response_time: Time of the last response.
            current_response_time: Time of the current response.

        Returns:
            Generated response, or None if prompt should not be responded to.
        """
        response_delay = current_response_time - last_response_time
        seconds_since_last_response = response_delay.total_seconds()
        if seconds_since_last_response > self.cfg.follow_up_max_seconds:
            self.reset_conversation()

        self.conversation.append(dict(role="user", content=prompt))
        llm_answer = openai.chat.completions.create(
            model=self.cfg.text_model_id,
            messages=self.conversation,
            temperature=self.cfg.temperature,
        )
        response: str = llm_answer.choices[0].message.content.strip()
        self.conversation.append(dict(role="assistant", content=response))
        logger.info(f"Generated the response: {response!r}")
        return response
