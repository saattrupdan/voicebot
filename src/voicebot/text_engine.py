"""The engine that produces new responses."""

import openai
from dotenv import load_dotenv
import os
import logging


load_dotenv()


logger = logging.getLogger(__name__)


class TextEngine:
    """The engine that produces new responses.

    Args:
        model_id: ID of the model to use.
        temperature: Temperature to use for generation.
    """

    def __init__(self, model_id: str, temperature: float, wake_word: str) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.wake_word = wake_word
        self.conversation: list[dict[str, str]] = [
            dict(role="system", content="Du er en dansk stemmerobot, og er sød og rar.")
        ]
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_response(self, prompt: str) -> str | None:
        """Generate a new response from a prompt.

        Args:
            prompt: Prompt to generate a response from.

        Returns:
            Generated response, or None if prompt is empty.
        """
        if self.wake_word not in prompt:
            logger.info(
                f"Prompt does not contain wake word ({self.wake_word!r}), skipping."
            )
            return None

        self.conversation.append(dict(role="user", content=prompt))
        llm_answer = openai.ChatCompletion.create(
            model=self.model_id,
            messages=self.conversation,
            temperature=self.temperature,
        )
        response: str = llm_answer.choices[0].message.content.strip()
        self.conversation.append(dict(role="assistant", content=response))
        return response
