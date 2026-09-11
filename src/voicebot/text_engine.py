"""The engine that produces new responses."""

import datetime as dt
import json
import logging
import os
import re
import typing as t

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from . import tools as tool_module
from .utils import MONTHS, WEEKDAYS

load_dotenv()
logger = logging.getLogger(__name__)

MAX_TOOL_STEPS = 5


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
            api_key=os.environ["MELIOUS_API_KEY"], base_url=cfg.text_server
        )
        self.conversation: list[ChatCompletionMessageParam] = list()
        raw_tools = t.cast(list[dict[str, object]], OmegaConf.to_object(self.cfg.tools))
        self.tools = self._format_tools(tools=raw_tools)
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
        final_answer = self._complete_conversation()

        final_answer = re.sub(
            r"https?://(www\.)[^ ]+", "", final_answer, flags=re.IGNORECASE
        ).replace("()", "")

        for before, after in self.cfg.manual_fixes.items():
            if before in final_answer:
                logger.info(f"Fixing {before!r} to {after!r} in the response.")
                final_answer = final_answer.replace(before, after)

        if final_answer:
            logger.info(f"Generated the response: {final_answer!r}")

        return final_answer

    def _complete_conversation(self) -> str:
        """Run Chat Completions until the model returns text."""
        for _ in range(MAX_TOOL_STEPS):
            completion = self.client.chat.completions.create(
                model=str(self.cfg.text_model_id),
                messages=self.conversation,
                temperature=float(self.cfg.temperature),
                tools=self.tools,
            )
            message = completion.choices[0].message
            self.conversation.append(
                t.cast(
                    ChatCompletionMessageParam, message.model_dump(exclude_none=True)
                )
            )

            if not message.tool_calls:
                return message.content or message.refusal or ""

            needs_followup = False
            for tool_call in message.tool_calls:
                if tool_call.type != "function":
                    raise RuntimeError(f"Unsupported tool call type: {tool_call.type}")
                function_tool_call = tool_call
                tool_response = self._call_tool(
                    name=function_tool_call.function.name,
                    arguments_json=function_tool_call.function.arguments,
                )
                needs_followup = needs_followup or bool(tool_response)
                self.conversation.append(
                    dict(
                        role="tool",
                        tool_call_id=function_tool_call.id,
                        content=json.dumps(
                            {function_tool_call.function.name: tool_response},
                            ensure_ascii=False,
                        ),
                    )
                )

            if not needs_followup:
                self.conversation.append(dict(role="assistant", content=""))
                return ""

        raise RuntimeError(f"The model exceeded {MAX_TOOL_STEPS} tool-calling steps.")

    def _call_tool(self, name: str, arguments_json: str) -> str:
        """Call one model-requested tool and update the engine state."""
        parsed_arguments = json.loads(arguments_json)
        if not isinstance(parsed_arguments, dict):
            message = f"Tool arguments must be an object, got {parsed_arguments!r}"
            raise TypeError(message)
        arguments = {key: value for key, value in parsed_arguments.items() if key != ""}
        logger.info(f"Using the tool {name!r} with parameters {arguments!r}...")

        try:
            tool_response, state_updates = getattr(tool_module, name)(
                state=self.state, **arguments
            )
        except TypeError as error:
            logger.error(f"Error calling tool {name!r}: {error}")
            logger.info(f"Trying to use the tool {name!r} without arguments...")
            tool_response, state_updates = getattr(tool_module, name)(state=self.state)
        self.state.update(state_updates)

        if tool_response:
            logger.info(f"Tool {name!r} response: {tool_response!r}")
        return tool_response

    @staticmethod
    def _format_tools(tools: list[dict[str, object]]) -> list[ChatCompletionToolParam]:
        """Convert Responses API function schemas to Chat Completions schemas."""
        return [
            t.cast(
                ChatCompletionToolParam,
                {
                    "type": "function",
                    "function": {
                        key: value for key, value in tool.items() if key != "type"
                    },
                },
            )
            for tool in tools
        ]
