"""The engine that produces new responses."""

import collections.abc as c
import datetime as dt
import json
import logging
import os
import re
import threading
import typing as t
from dataclasses import dataclass
from enum import StrEnum

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionToolParam,
)

from . import tools as tool_module
from .intents import confirmation_decision, is_end_conversation
from .tool_runtime import ToolContext, ToolResult, ToolRuntime, ToolStatus
from .utils import MONTHS, WEEKDAYS

load_dotenv()
logger = logging.getLogger(__name__)

MAX_TOOL_STEPS = 5
END_CONVERSATION_MARKER = "[[END_CONVERSATION]]"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class TurnAction(StrEnum):
    """The action requested by a completed user turn."""

    RESPOND = "respond"
    SILENT = "silent"
    END = "end"


@dataclass(frozen=True)
class TurnResult:
    """The outcome of processing one user turn."""

    action: TurnAction
    text: str = ""


@dataclass
class _ToolCallParts:
    """Fragments of one streamed function tool call."""

    identifier: str = ""
    name: str = ""
    arguments: str = ""


class TextEngine:
    """The engine that produces new responses."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        runtime: ToolRuntime | None = None,
        state: dict[str, object] | None = None,
    ) -> None:
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
        assembled = None
        if runtime is None and state is None and "integrations" in self.cfg:
            from .runtime import build_integration_runtime

            assembled = build_integration_runtime(self.cfg)
            runtime = assembled.registry
            state = assembled.state
        self.runtime = runtime or ToolRuntime(
            registry=tool_module.build_registry(tools=raw_tools),
            mutations_enabled=bool(self.cfg.get("mutations_enabled", True)),
        )
        # The validated registry is always the source of model-visible contracts.
        self.tools = t.cast(list[ChatCompletionToolParam], self.runtime.schemas())
        self.state = state if state is not None else {}
        self.integration_runtime = assembled

    def generate_response(
        self,
        prompt: str,
        last_response_time: dt.datetime,
        current_response_time: dt.datetime,
        on_segment: c.Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> TurnResult:
        """Generate a new response from a prompt.

        Args:
            prompt:
                Prompt to generate a response from.
            last_response_time:
                Time of the last response.
            current_response_time:
                Time of the current response.
            on_segment (optional):
                Callback receiving speakable response segments as they arrive. Defaults
                to None.
            cancel_event (optional):
                Event used to cancel streamed generation. Defaults to None.

        Returns:
            The completed turn outcome.
        """
        # Results are scoped to one turn; a cancellation consumes this hand-off.
        self.state.pop("committed_operation_results", None)
        if is_end_conversation(text=prompt):
            logger.info("The user ended the conversation.")
            self.state.pop("committed_history_notice", None)
            self.reset_conversation()
            return TurnResult(action=TurnAction.END)

        decision = confirmation_decision(prompt)
        confirmation_handler = self.state.get("confirmation_handler")
        if decision is not None and callable(confirmation_handler):
            result = confirmation_handler(decision)
            if isinstance(result, ToolResult):
                message = self._clean_response(result.message_da)
                return TurnResult(
                    action=TurnAction.RESPOND if message else TurnAction.SILENT,
                    text=message,
                )
            if isinstance(result, TurnResult):
                return result

        if len(prompt.strip()) <= self.cfg.min_prompt_length:
            logger.info("The prompt is too short, ignoring it.")
            return TurnResult(action=TurnAction.SILENT)

        logger.info("Generating response prompt_length=%d", len(prompt))
        self._start_conversation_if_needed(
            last_response_time=last_response_time,
            current_response_time=current_response_time,
        )
        self._inject_committed_notice()
        self.conversation.append(dict(role="user", content=prompt))
        conversation_with_user = list(self.conversation)

        active_cancel = cancel_event or threading.Event()
        if on_segment is None:
            result = self._complete_conversation()
        else:
            self.state["cancel_event"] = active_cancel
            try:
                result = self._complete_conversation_stream(
                    on_segment=on_segment, cancel_event=active_cancel
                )
            finally:
                self.state.pop("cancel_event", None)

        if active_cancel.is_set():
            self.conversation = conversation_with_user
            committed = self.state.pop("committed_operation_results", [])
            if isinstance(committed, list) and committed:
                self.state["committed_history_notice"] = committed
        if result.action is TurnAction.END:
            self.reset_conversation()
        elif result.text:
            logger.info(
                "Generated response action=%s text_length=%d",
                result.action,
                len(result.text),
            )
        return result

    def reset_conversation(self) -> None:
        """Clear the current conversational history."""
        self.conversation.clear()

    def _inject_committed_notice(self) -> None:
        """Tell the next model turn about mutations completed before barge-in."""
        notices = self.state.pop("committed_history_notice", [])
        if not isinstance(notices, list):
            return
        for operation in notices:
            if isinstance(operation, dict):
                status = operation.get("status")
                if status == ToolStatus.OUTCOME_UNKNOWN.value:
                    prefix = (
                        "Resultatet af denne handling er uklart; den kan være "
                        "gennemført og må ikke gentages. "
                    )
                else:
                    prefix = "Denne handling er allerede gennemført. "
                self.conversation.append(
                    {
                        "role": "system",
                        "content": prefix + json.dumps(operation, ensure_ascii=False),
                    }
                )

    def _start_conversation_if_needed(
        self, last_response_time: dt.datetime, current_response_time: dt.datetime
    ) -> None:
        """Start a fresh model conversation after the follow-up window expires."""
        response_delay = current_response_time - last_response_time
        if response_delay.total_seconds() <= self.cfg.follow_up_max_seconds:
            return

        now = dt.datetime.now()
        system_prompt = self.cfg.system_prompt.strip().format(
            weekday=WEEKDAYS[now.weekday()],
            day=now.day,
            month=MONTHS[now.month - 1],
            year=now.year,
            time=now.strftime("%H:%M"),
        )
        self.conversation = [dict(role="system", content=system_prompt)]

    def _complete_conversation(self) -> TurnResult:
        """Run non-streamed Chat Completions until the model returns text."""
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
                text = self._clean_response(message.content or message.refusal or "")
                if text == END_CONVERSATION_MARKER:
                    return TurnResult(action=TurnAction.END)
                action = TurnAction.RESPOND if text else TurnAction.SILENT
                return TurnResult(action=action, text=text)

            needs_followup = self._call_tools(tool_calls=message.tool_calls)
            if not needs_followup:
                self.conversation.append(dict(role="assistant", content=""))
                return TurnResult(action=TurnAction.SILENT)

        raise RuntimeError(f"The model exceeded {MAX_TOOL_STEPS} tool-calling steps.")

    def _complete_conversation_stream(
        self, on_segment: c.Callable[[str], None], cancel_event: threading.Event
    ) -> TurnResult:
        """Stream Chat Completions and emit complete speakable segments."""
        for _ in range(MAX_TOOL_STEPS):
            received_delta = False
            try:
                stream = self.client.chat.completions.create(
                    model=str(self.cfg.text_model_id),
                    messages=self.conversation,
                    temperature=float(self.cfg.temperature),
                    tools=self.tools,
                    stream=True,
                )
                content = ""
                refusal = ""
                pending_speech = ""
                end_marker_possible = True
                tool_parts: dict[int, _ToolCallParts] = {}
                try:
                    for chunk in stream:
                        if cancel_event.is_set():
                            return TurnResult(action=TurnAction.SILENT)
                        if not chunk.choices:
                            continue
                        received_delta = True
                        delta = chunk.choices[0].delta
                        if delta.content:
                            content += delta.content
                            pending_speech += delta.content
                        if delta.refusal:
                            refusal += delta.refusal
                            pending_speech += delta.refusal
                        if pending_speech and end_marker_possible:
                            candidate = pending_speech.lstrip()
                            end_marker_possible = END_CONVERSATION_MARKER.startswith(
                                candidate
                            )
                        if pending_speech and not end_marker_possible:
                            pending_speech = self._emit_complete_segments(
                                text=pending_speech, on_segment=on_segment
                            )
                        for tool_call in delta.tool_calls or []:
                            parts = tool_parts.setdefault(
                                tool_call.index, _ToolCallParts()
                            )
                            if tool_call.id:
                                parts.identifier += tool_call.id
                            if tool_call.function is not None:
                                if tool_call.function.name:
                                    parts.name += tool_call.function.name
                                if tool_call.function.arguments:
                                    parts.arguments += tool_call.function.arguments
                finally:
                    stream.close()
            except AttributeError, TypeError, openai.APIError:
                if received_delta:
                    raise
                logger.warning("Streaming completion unavailable; falling back")
                result = self._complete_conversation()
                if result.text and not cancel_event.is_set():
                    on_segment(result.text)
                return result

            if cancel_event.is_set():
                return TurnResult(action=TurnAction.SILENT)

            if not tool_parts:
                final_text = content or refusal
                if final_text.strip() == END_CONVERSATION_MARKER:
                    return TurnResult(action=TurnAction.END)
                pending_speech = self._emit_complete_segments(
                    text=pending_speech, on_segment=on_segment
                )
                if pending_speech.strip() and not cancel_event.is_set():
                    segment = self._clean_response(pending_speech)
                    if segment:
                        on_segment(segment)
                if cancel_event.is_set():
                    return TurnResult(action=TurnAction.SILENT)
                self.conversation.append(dict(role="assistant", content=final_text))
                cleaned_text = self._clean_response(final_text)
                action = TurnAction.RESPOND if cleaned_text else TurnAction.SILENT
                return TurnResult(action=action, text=cleaned_text)

            tool_calls: list[dict[str, object]] = [
                {
                    "id": parts.identifier,
                    "type": "function",
                    "function": {"name": parts.name, "arguments": parts.arguments},
                }
                for _, parts in sorted(tool_parts.items())
            ]
            self.conversation.append(
                t.cast(
                    ChatCompletionMessageParam,
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": tool_calls,
                    },
                )
            )
            needs_followup = self._call_streamed_tools(tool_calls=tool_calls)
            if not needs_followup:
                self.conversation.append(dict(role="assistant", content=""))
                return TurnResult(action=TurnAction.SILENT)

        raise RuntimeError(f"The model exceeded {MAX_TOOL_STEPS} tool-calling steps.")

    def _emit_complete_segments(
        self, text: str, on_segment: c.Callable[[str], None]
    ) -> str:
        """Emit complete sentence segments and return the unfinished suffix."""
        parts = _SENTENCE_BOUNDARY.split(text)
        if len(parts) == 1:
            return text
        for part in parts[:-1]:
            segment = self._clean_response(part)
            if segment:
                on_segment(segment)
        return parts[-1]

    def _call_tools(
        self, tool_calls: c.Sequence[ChatCompletionMessageToolCallUnion]
    ) -> bool:
        """Call non-streamed model-requested tools."""
        needs_followup = False
        for tool_call in tool_calls:
            if tool_call.type != "function" or tool_call.function is None:
                response = ToolResult(
                    status=ToolStatus.INVALID_REQUEST,
                    message_da="Forespørgslen havde et ugyldigt funktionskald.",
                )
                identifier = str(tool_call.id)
                name = "unknown"
            else:
                name = str(tool_call.function.name)
                response = self._call_tool(
                    name=name,
                    arguments_json=str(tool_call.function.arguments),
                    operation_id=str(tool_call.id),
                )
                identifier = str(tool_call.id)
            needs_followup = needs_followup or response.needs_followup
            self._append_tool_response(
                identifier=identifier, name=name, response=response
            )
        return needs_followup

    def _call_streamed_tools(self, tool_calls: list[dict[str, object]]) -> bool:
        """Call streamed model-requested tools."""
        needs_followup = False
        for tool_call in tool_calls:
            function = t.cast(dict[str, str], tool_call["function"])
            name = function["name"]
            tool_response = self._call_tool(
                name=name,
                arguments_json=function["arguments"],
                operation_id=str(tool_call["id"]),
            )
            needs_followup = needs_followup or tool_response.needs_followup
            self._append_tool_response(
                identifier=str(tool_call["id"]), name=name, response=tool_response
            )
        return needs_followup

    def _append_tool_response(
        self, identifier: str, name: str, response: ToolResult
    ) -> None:
        """Append a structured tool response to the model conversation."""
        content = (
            json.dumps({name: response.legacy_message}, ensure_ascii=False)
            if response.legacy_message is not None
            else response.to_json()
        )
        self.conversation.append(
            dict(role="tool", tool_call_id=identifier, content=content)
        )

    def _call_tool(
        self, name: str, arguments_json: str, operation_id: str | None = None
    ) -> ToolResult:
        """Parse, validate, and invoke one model-requested tool exactly once."""
        try:
            arguments: object = json.loads(arguments_json)
        except json.JSONDecodeError, TypeError:
            arguments = None
        cancel_event = self.state.get("cancel_event")
        event = cancel_event if isinstance(cancel_event, threading.Event) else None
        configured_device = self.state.get("device_id")
        device_id = configured_device if isinstance(configured_device, str) else None
        context = ToolContext(
            state=self.state,
            cancel_event=event,
            operation_id=operation_id,
            device_id=device_id,
        )
        return self.runtime.invoke(name=name, arguments=arguments, context=context)

    def _clean_response(self, text: str) -> str:
        """Prepare response text for speech synthesis."""
        text = re.sub(r"https?://(?:www\.)?[^ ]+", "", text, flags=re.IGNORECASE)
        text = text.replace("()", "").strip()
        for before, after in self.cfg.manual_fixes.items():
            if before in text:
                logger.info("Applied a configured response text fix")
                text = text.replace(before, after)
        return text

    @staticmethod
    def _format_tools(tools: list[dict[str, object]]) -> list[ChatCompletionToolParam]:
        """Convert Responses API function schemas to Chat Completions schemas."""
        allowed_names = tool_module.MODEL_TOOL_NAMES | {"lookup"}
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
            if tool.get("name") in allowed_names
        ]
