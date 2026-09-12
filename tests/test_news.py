"""Tests for cancellable news playback."""

import threading
from unittest.mock import MagicMock

import pytest

from voicebot.tools import news

_RSS = """\
<rss><channel><item>
<title>En overskrift</title>
<description>En beskrivelse</description>
<pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
</item></channel></rss>
"""


def test_news_stops_after_interrupted_intro(monkeypatch: pytest.MonkeyPatch) -> None:
    """No headline, chime, or closing phrase follows an interrupted introduction."""
    response = MagicMock(text=_RSS)
    monkeypatch.setattr(news.httpx, "get", MagicMock(return_value=response))
    monkeypatch.setattr(news.chime, "theme", MagicMock())
    chime = MagicMock()
    monkeypatch.setattr(news.chime, "info", chime)
    cancel = threading.Event()

    def interrupt(**kwargs: object) -> None:
        del kwargs
        cancel.set()

    synthesise = MagicMock(side_effect=interrupt)
    monkeypatch.setattr(news, "synthesise_speech", synthesise)

    message, _ = news.get_news(state={"synthesiser": object(), "cancel_event": cancel})

    assert message == ""
    synthesise.assert_called_once()
    assert synthesise.call_args.kwargs["cancel_event"] is cancel
    chime.assert_not_called()
