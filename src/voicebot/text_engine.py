"""The engine that produces new responses."""

import openai
from dotenv import load_dotenv
import os


load_dotenv()


class TextEngine:
    """The engine that produces new responses."""

    def __init__(self, temperature: float) -> None:
        self.temperature = temperature
        self.conversation: list[dict[str, str]] = [
            dict(role="system", content="Du er en dansk stemmerobot, og er sød og rar.")
        ]
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_response(self, prompt: str) -> str:
        """Generate a new response from a prompt.

        Args:
            prompt: Prompt to generate a response from.

        Returns:
            Generated response.
        """
        self.conversation.append(dict(role="user", content=prompt))
        llm_answer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.conversation,
            temperature=self.temperature,
        )
        response: str = llm_answer.choices[0].message.content.strip()
        self.conversation.append(dict(role="assistant", content=response))
        return response
