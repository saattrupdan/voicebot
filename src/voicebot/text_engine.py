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
    ChatCompletionUserMessageParam,
)

from . import tools
from .tools import LLMResponse, NoFunctionCallResponse
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
            api_key=os.getenv("OPENAI_API_KEY"), base_url=cfg.server
        )
        self.conversation: list[ChatCompletionMessageParam] = list()
        self.tools: list[dict] = OmegaConf.to_object(self.cfg.tools)
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
                state=str(self.state),
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
        response_obj = response_or_null.response

        if isinstance(response_obj, NoFunctionCallResponse):
            response = response_obj.answer
        else:
            function_name = response_obj.name
            function_parameters = response_obj.parameters.model_dump()

            logger.info(
                f"Using the tool {function_name!r} with parameters "
                f"{function_parameters!r}..."
            )
            tool_response, state = getattr(tools, function_name)(
                state=self.state, **function_parameters
            )
            self.state.update(state)
            logger.info(f"Tool response: {tool_response!r}")

            self.conversation.extend(
                [
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=f"I want to call the {function_name!r} function with "
                        f"the parameters {function_parameters!r}.",
                    ),
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=f"The response from the {function_name!r} function is "
                        f"{tool_response!r}. Now answer the query as "
                        '{"response": {"answer": your answer}}.',
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
            response_obj = response_or_null.response
            if not isinstance(response_obj, NoFunctionCallResponse):
                logger.error(
                    "The response after using a tool is not a NoFunctionCallResponse."
                )
                return None
            response = response_obj.answer

        # Fix some consistent typos
        for before, after in self.cfg.manual_fixes.items():
            response = response.replace(before, after)

        self.conversation.append(
            ChatCompletionAssistantMessageParam(role="assistant", content=response)
        )
        logger.info(f"Generated the response: {response!r}")
        return response
