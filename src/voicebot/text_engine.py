"""The engine that produces new responses."""

import openai
from dotenv import load_dotenv
import os
import logging
import datetime as dt


load_dotenv()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Du hedder Jarvis og er en dansk stemmerobot. Du er sød, rar og hjælpsom, og dine svar
er altid super korte og præcise.
"""


class TextEngine:
    """The engine that produces new responses.

    Args:
        model_id: OpenAI model ID of the model to use.
        temperature: Temperature to use for generation.
        wake_words: Words that should trigger a new conversation.
        follow_up_max_seconds: Maximum number of seconds to wait for a follow-up.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float,
        wake_words: list[str],
        follow_up_max_seconds: float,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.wake_words = wake_words
        self.follow_up_max_seconds = follow_up_max_seconds
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def reset_conversation(self) -> None:
        """Reset the conversation, only keeping the system prompt."""
        self.conversation = [dict(role="system", content=SYSTEM_PROMPT.strip())]

    def generate_response(
        self, prompt: str, last_response_time: dt.datetime
    ) -> str | None:
        """Generate a new response from a prompt.

        Args:
            prompt: Prompt to generate a response from.
            last_response_time: Time of the last response.

        Returns:
            Generated response, or None if prompt should not be responded to.
        """
        now = dt.datetime.now()
        seconds_since_last_response = (now - last_response_time).total_seconds()
        if seconds_since_last_response > self.follow_up_max_seconds:
            self.reset_conversation()
            if all(word not in prompt for word in self.wake_words):
                logger.info("Prompt does not contain any of the wake words, skipping.")
                return None

        # Remove all the wake words from the prompt, to prevent them from influencing
        # the response.
        for word in self.wake_words:
            prompt = prompt.replace(word, "").strip()

        self.conversation.append(dict(role="user", content=prompt))
        llm_answer = openai.ChatCompletion.create(
            model=self.model_id,
            messages=self.conversation,
            temperature=self.temperature,
        )
        response: str = llm_answer.choices[0].message.content.strip()
        self.conversation.append(dict(role="assistant", content=response))
        logger.info(f"Generated the response: {response!r}")
        return response
