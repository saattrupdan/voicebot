"""Tests for local conversational intent detection."""

import pytest

from voicebot.intents import is_end_conversation


@pytest.mark.parametrize(
    "utterance",
    [
        "stop",
        "Ti stille!",
        "tak",
        "Mange tak for hjælpen.",
        "farvel",
        "Det var alt",
        "Jeg er færdig",
        "Du kan godt stoppe nu",
    ],
)
def test_end_conversation_phrases(utterance: str) -> None:
    """Clear ending phrases are detected across common formulations."""
    assert is_end_conversation(text=utterance)


@pytest.mark.parametrize(
    "utterance",
    [
        "Stop timeren",
        "Tak, hvad bliver vejret?",
        "Hvornår er filmen færdig?",
        "Sig farvel til katten",
    ],
)
def test_requests_containing_end_words_continue(utterance: str) -> None:
    """End words inside a substantive request do not stop the conversation."""
    assert not is_end_conversation(text=utterance)
