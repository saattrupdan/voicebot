"""The engine that produces new responses."""

import datetime as dt
import json
import logging
import os

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from . import tools
from .tools import LLMResponse, NonToolAnswer
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
        self.conversation: list[ChatCompletionMessageParam] = list()
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
                tools="\n".join([json.dumps(tool, indent=4) for tool in self.tools]),
            )
            self.conversation = [
                ChatCompletionSystemMessageParam(role="system", content=system_prompt)
            ]

        self.conversation.append(
            ChatCompletionUserMessageParam(role="user", content=prompt)
        )

        llm_answer = self.client.beta.chat.completions.parse(
            model=self.cfg.text_model_id,
            messages=self.conversation,
            temperature=self.cfg.temperature,
            response_format=LLMResponse,
        )
        response_or_null = llm_answer.choices[0].message.parsed
        if response_or_null is None:
            logger.info("The response is empty, ignoring it.")
            return None
        response_obj = response_or_null.answer

        if isinstance(response_obj, NonToolAnswer):
            response = response_obj.response
        else:
            function_name = response_obj.name
            if response_obj.parameters is not None:
                function_parameters = response_obj.parameters.model_dump()
            else:
                function_parameters = dict()

            logger.info(
                f"Using the tool {function_name!r} with parameters "
                f"{function_parameters!r}..."
            )
            tool_response, state = getattr(tools, function_name)(
                state=self.state, **function_parameters
            )
            self.state.update(state)
            logger.info(f"Tool response: {tool_response!r}")

            if not tool_response:
                logger.info(
                    "The tool does not require a response, so we do not continue."
                )
                return None

            self.conversation.extend(
                [
                    ChatCompletionToolMessageParam(
                        role="tool", tool_call_id=function_name, content=tool_response
                    ),
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content="Use the above information to answer the original "
                        "question. The user has not seen the above information, so "
                        "include relevant details in your response. You must not "
                        "mention the name of the tool you just used.",
                    ),
                ]
            )
            llm_answer = self.client.beta.chat.completions.parse(
                model=self.cfg.text_model_id,
                messages=self.conversation,
                temperature=self.cfg.temperature,
                response_format=LLMResponse,
            )
            response_or_null = llm_answer.choices[0].message.parsed
            if response_or_null is None:
                logger.info("The response is empty, ignoring it.")
                return None
            answer = response_or_null.answer
            if not isinstance(answer, NonToolAnswer):
                logger.info("The answer to a tool can't be another tool call.")
                return None
            response = answer.response

        # Fix some consistent typos
        for before, after in self.cfg.manual_fixes.items():
            if before in response:
                logger.info(f"Fixing {before!r} to {after!r} in the response.")
                response = response.replace(before, after)

        self.conversation.append(
            ChatCompletionAssistantMessageParam(role="assistant", content=response)
        )
        logger.info(f"Generated the response: {response!r}")
        return response
