"""Tests for the Melious text engine."""

import datetime as dt
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from voicebot import text_engine


def test_text_engine_uses_melious_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine sends prompts through Melious Chat Completions."""
    client = MagicMock()
    message = MagicMock(content="Et kort svar.", tool_calls=None)
    message.model_dump.return_value = {"role": "assistant", "content": "Et kort svar."}
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )
    openai_factory = MagicMock(return_value=client)
    monkeypatch.setattr(text_engine.openai, "OpenAI", openai_factory)
    monkeypatch.setenv("MELIOUS_API_KEY", "test-key")
    engine = text_engine.TextEngine(cfg=_config())

    result = engine.generate_response(
        prompt="Hvordan går det?",
        last_response_time=dt.datetime(year=1900, month=1, day=1),
        current_response_time=dt.datetime.now(),
    )

    assert result == "Et kort svar."
    openai_factory.assert_called_once_with(
        api_key="test-key", base_url="https://api.melious.ai/v1"
    )
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "deepseek-v4-flash-0731"
    assert request["messages"][-1] == {"role": "assistant", "content": "Et kort svar."}
    assert request["tools"][0] == {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_text_engine_preserves_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal is returned when the completion has no regular content."""
    client = MagicMock()
    message = MagicMock(content=None, refusal="Det kan jeg ikke.", tool_calls=None)
    message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "refusal": "Det kan jeg ikke.",
    }
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )
    monkeypatch.setattr(text_engine.openai, "OpenAI", MagicMock(return_value=client))
    monkeypatch.setenv("MELIOUS_API_KEY", "test-key")
    engine = text_engine.TextEngine(cfg=_config())

    result = engine.generate_response(
        prompt="Afvis denne forespørgsel.",
        last_response_time=dt.datetime(year=1900, month=1, day=1),
        current_response_time=dt.datetime.now(),
    )

    assert result == "Det kan jeg ikke."


def test_text_engine_returns_tool_result_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Information-returning tool calls are sent back for a final answer."""
    client = MagicMock()
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="lookup", arguments='{"query": "vejret"}'),
    )
    tool_message = MagicMock(content=None, tool_calls=[tool_call])
    tool_message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "vejret"}'},
            }
        ],
    }
    final_message = MagicMock(content="Det bliver solrigt.", tool_calls=None)
    final_message.model_dump.return_value = {
        "role": "assistant",
        "content": "Det bliver solrigt.",
    }
    client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=tool_message)]),
        SimpleNamespace(choices=[SimpleNamespace(message=final_message)]),
    ]
    monkeypatch.setattr(text_engine.openai, "OpenAI", MagicMock(return_value=client))
    monkeypatch.setenv("MELIOUS_API_KEY", "test-key")
    monkeypatch.setattr(
        text_engine.tool_module,
        "lookup",
        lambda state, query: (f"Resultat for {query}", {"tool_was_called": True}),
        raising=False,
    )
    engine = text_engine.TextEngine(cfg=_config())
    synthesiser = object()
    engine.state["synthesiser"] = synthesiser

    result = engine.generate_response(
        prompt="Hvordan bliver vejret?",
        last_response_time=dt.datetime(year=1900, month=1, day=1),
        current_response_time=dt.datetime.now(),
    )

    assert result == "Det bliver solrigt."
    assert engine.state == {"synthesiser": synthesiser, "tool_was_called": True}
    assert client.chat.completions.create.call_count == 2
    second_request = client.chat.completions.create.call_args_list[1].kwargs
    assert second_request["messages"][-2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"lookup": "Resultat for vejret"}',
    }


def _config() -> DictConfig:
    """Create the minimal configuration required by the text engine."""
    return OmegaConf.create(
        {
            "text_server": "https://api.melious.ai/v1",
            "text_model_id": "deepseek-v4-flash-0731",
            "temperature": 0.6,
            "min_prompt_length": 5,
            "follow_up_max_seconds": 5.0,
            "system_prompt": (
                "I dag er {weekday} {day}. {month} {year}, og klokken er {time}."
            ),
            "manual_fixes": {},
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look something up.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
    )
