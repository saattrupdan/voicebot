"""The engine that produces new responses."""

import datetime as dt
import json
import logging
import os

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from openai.types.responses import (
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)

from . import tools as tool_module
from .utils import MONTHS, WEEKDAYS

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
            api_key=os.getenv("OPENAI_API_KEY", "not-set"), base_url=cfg.server
        )
        self.conversation: list[ResponseInputItemParam] = list()
        self.tools: list[dict] = OmegaConf.to_object(self.cfg.tools)  # type: ignore[bad-assignment]
        self.state: dict = dict()

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

        logger.info(f"Generating a response from the prompt: {prompt!r}...")

        response_delay = current_response_time - last_response_time
        seconds_since_last_response = response_delay.total_seconds()
        if seconds_since_last_response > self.cfg.follow_up_max_seconds:
            system_prompt = self.cfg.system_prompt.strip().format(
                weekday=WEEKDAYS[dt.datetime.now().weekday()],
                day=dt.datetime.now().day,
                month=MONTHS[dt.datetime.now().month - 1],
                year=dt.datetime.now().year,
                time=dt.datetime.now().strftime("%H:%M"),
            )
            self.conversation = [dict(role="system", content=system_prompt)]

        self.conversation.append(dict(role="user", content=prompt))

        llm_answer = (
            self.client.responses.create(  # pyrefly: ignore[no-matching-overload]
                model=str(self.cfg.text_model_id),
                input=self.conversation,
                temperature=float(self.cfg.temperature),
                tools=self.tools,
            )
        )
        self.conversation.extend(llm_answer.output)

        # Call any tools that were requested
        needs_followup = False
        for item in llm_answer.output:
            if item.type == "function_call":
                logger.info(
                    f"Using the tool {item.name!r} with parameters "
                    f"{item.arguments!r}..."
                )
                tool_response, state = getattr(tool_module, item.name)(
                    state=self.state, **json.loads(item.arguments)
                )
                logger.info(f"Tool {item.name!r} response: {tool_response!r}")
                if tool_response:
                    needs_followup = True
                    self.conversation.append(
                        dict(
                            type="function_call_output",
                            call_id=item.call_id,
                            output=json.dumps({item.name: tool_response}),
                        )
                    )

        # If we called a tool, we need to call the LLM again to get the final response
        if needs_followup:
            llm_answer = (
                self.client.responses.create(  # pyrefly: ignore[no-matching-overload]
                    model=self.cfg.text_model_id,
                    input=self.conversation,
                    instructions=(
                        "Respond only with an answer to the user's question, based on "
                        "the information provided by the tools."
                    ),
                    temperature=self.cfg.temperature,
                    tools=self.tools,
                )
            )
            self.conversation.extend(llm_answer.output)

        # Extract the final answer
        final_response = self.conversation[-1]
        assert isinstance(final_response, ResponseOutputMessage), (
            "The final response is not a ResponseOutputMessage, it's a "
            f"{type(final_response)}"
        )
        final_answer = final_response.content[0]
        if isinstance(final_answer, ResponseOutputRefusal):
            final_answer = final_answer.refusal
        elif isinstance(final_answer, ResponseOutputText):
            final_answer = final_answer.text

        # Fix some consistent typos
        for before, after in self.cfg.manual_fixes.items():
            if before in final_answer:
                logger.info(f"Fixing {before!r} to {after!r} in the response.")
                final_answer = final_answer.replace(before, after)

        logger.info(f"Generated the response: {final_answer!r}")
        return final_answer
