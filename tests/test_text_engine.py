"""Tests for the Melious text engine."""

import collections.abc as c
import datetime as dt
import logging
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

    assert result == text_engine.TurnResult(
        action=text_engine.TurnAction.RESPOND, text="Et kort svar."
    )
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


def test_text_engine_logs_only_safe_response_metadata(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Prompt and generated response contents never enter operational logs."""
    client = MagicMock()
    message = MagicMock(content="generated-secret", tool_calls=None)
    message.model_dump.return_value = {
        "role": "assistant",
        "content": "generated-secret",
    }
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )
    monkeypatch.setattr(text_engine.openai, "OpenAI", MagicMock(return_value=client))
    monkeypatch.setenv("MELIOUS_API_KEY", "test-key")
    engine = text_engine.TextEngine(cfg=_config())

    with caplog.at_level(logging.INFO):
        engine.generate_response(
            prompt="prompt-secret",
            last_response_time=dt.datetime(year=1900, month=1, day=1),
            current_response_time=dt.datetime.now(),
        )

    assert "prompt-secret" not in caplog.text
    assert "generated-secret" not in caplog.text
    assert "prompt_length=" in caplog.text
    assert "text_length=" in caplog.text


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

    assert result == text_engine.TurnResult(
        action=text_engine.TurnAction.RESPOND, text="Det kan jeg ikke."
    )


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

    assert result == text_engine.TurnResult(
        action=text_engine.TurnAction.RESPOND, text="Det bliver solrigt."
    )
    assert engine.state == {"synthesiser": synthesiser, "tool_was_called": True}
    assert client.chat.completions.create.call_count == 2
    second_request = client.chat.completions.create.call_args_list[1].kwargs
    assert second_request["messages"][-2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"lookup": "Resultat for vejret"}',
    }


def test_text_engine_streams_complete_speech_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete sentences are emitted before the full answer is received."""
    segments: list[str] = []

    def chunks() -> c.Iterator[SimpleNamespace]:
        yield _chunk(content="Første sætning. ")
        assert segments == ["Første sætning."]
        yield _chunk(content="Anden sætning.")

    client = MagicMock()
    stream = MagicMock()
    stream.__iter__.return_value = chunks()
    client.chat.completions.create.return_value = stream
    monkeypatch.setattr(text_engine.openai, "OpenAI", MagicMock(return_value=client))
    monkeypatch.setenv("MELIOUS_API_KEY", "test-key")
    engine = text_engine.TextEngine(cfg=_config())

    result = engine.generate_response(
        prompt="Fortæl mig noget.",
        last_response_time=dt.datetime(year=1900, month=1, day=1),
        current_response_time=dt.datetime.now(),
        on_segment=segments.append,
    )

    assert segments == ["Første sætning.", "Anden sætning."]
    assert result == text_engine.TurnResult(
        action=text_engine.TurnAction.RESPOND, text="Første sætning. Anden sætning."
    )
    assert client.chat.completions.create.call_args.kwargs["stream"] is True
    stream.close.assert_called_once_with()


def test_clear_end_intent_is_silent_and_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear stop phrase ends the session without a model call or response."""
    client = MagicMock()
    monkeypatch.setattr(text_engine.openai, "OpenAI", MagicMock(return_value=client))
    monkeypatch.setenv("MELIOUS_API_KEY", "test-key")
    engine = text_engine.TextEngine(cfg=_config())
    engine.conversation.append({"role": "user", "content": "Tidligere"})

    result = engine.generate_response(
        prompt="Mange tak for hjælpen!",
        last_response_time=dt.datetime.now(),
        current_response_time=dt.datetime.now(),
    )

    assert result == text_engine.TurnResult(action=text_engine.TurnAction.END)
    assert engine.conversation == []
    client.chat.completions.create.assert_not_called()


def test_streamed_end_marker_remains_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fragmented end marker is recognised before any text is spoken."""
    client = MagicMock()
    stream = MagicMock()
    stream.__iter__.return_value = iter(
        [_chunk(content="[[END_"), _chunk(content="CONVERSATION]]")]
    )
    client.chat.completions.create.return_value = stream
    monkeypatch.setattr(text_engine.openai, "OpenAI", MagicMock(return_value=client))
    monkeypatch.setenv("MELIOUS_API_KEY", "test-key")
    engine = text_engine.TextEngine(cfg=_config())
    segments: list[str] = []

    result = engine.generate_response(
        prompt="Jeg tror, vi er ved vejs ende.",
        last_response_time=dt.datetime(year=1900, month=1, day=1),
        current_response_time=dt.datetime.now(),
        on_segment=segments.append,
    )

    assert result == text_engine.TurnResult(action=text_engine.TurnAction.END)
    assert segments == []


def test_cancelled_stream_rolls_back_turn_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted answer retains the user prompt for the follow-up."""
    client = MagicMock()
    cancel = text_engine.threading.Event()

    def chunks() -> c.Iterator[SimpleNamespace]:
        yield _chunk(content="Et afbrudt svar. ")
        cancel.set()
        yield _chunk(content="Dette må ikke gemmes.")

    stream = MagicMock()
    stream.__iter__.return_value = chunks()
    client.chat.completions.create.return_value = stream
    monkeypatch.setattr(text_engine.openai, "OpenAI", MagicMock(return_value=client))
    monkeypatch.setenv("MELIOUS_API_KEY", "test-key")
    engine = text_engine.TextEngine(cfg=_config())
    engine.conversation.append({"role": "assistant", "content": "Tidligere svar."})

    result = engine.generate_response(
        prompt="Et spørgsmål, der bliver afbrudt.",
        last_response_time=dt.datetime.now(),
        current_response_time=dt.datetime.now(),
        on_segment=MagicMock(),
        cancel_event=cancel,
    )

    assert result == text_engine.TurnResult(action=text_engine.TurnAction.SILENT)
    assert engine.conversation == [
        {"role": "assistant", "content": "Tidligere svar."},
        {"role": "user", "content": "Et spørgsmål, der bliver afbrudt."},
    ]

    cancel.clear()
    message = MagicMock(content="I morgen også.", refusal=None, tool_calls=None)
    message.model_dump.return_value = {"role": "assistant", "content": "I morgen også."}
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )
    follow_up = engine.generate_response(
        prompt="Og i morgen?",
        last_response_time=dt.datetime.now(),
        current_response_time=dt.datetime.now(),
    )

    assert follow_up.action is text_engine.TurnAction.RESPOND
    request_messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert request_messages[:3] == [
        {"role": "assistant", "content": "Tidligere svar."},
        {"role": "user", "content": "Et spørgsmål, der bliver afbrudt."},
        {"role": "user", "content": "Og i morgen?"},
    ]


def test_non_streamed_end_marker_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model can silently end a less literal conversation without streaming."""
    client = MagicMock()
    message = MagicMock(
        content=text_engine.END_CONVERSATION_MARKER, refusal=None, tool_calls=None
    )
    message.model_dump.return_value = {
        "role": "assistant",
        "content": text_engine.END_CONVERSATION_MARKER,
    }
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )
    monkeypatch.setattr(text_engine.openai, "OpenAI", MagicMock(return_value=client))
    monkeypatch.setenv("MELIOUS_API_KEY", "test-key")
    engine = text_engine.TextEngine(cfg=_config())

    result = engine.generate_response(
        prompt="Jeg tror ikke, vi behøver tale mere.",
        last_response_time=dt.datetime(year=1900, month=1, day=1),
        current_response_time=dt.datetime.now(),
    )

    assert result == text_engine.TurnResult(action=text_engine.TurnAction.END)
    assert engine.conversation == []


def _chunk(
    content: str | None = None, tool_calls: list[SimpleNamespace] | None = None
) -> SimpleNamespace:
    """Create a minimal streamed Chat Completions chunk."""
    delta = SimpleNamespace(content=content, refusal=None, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


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
