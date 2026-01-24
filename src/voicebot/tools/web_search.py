"""Search the web for a given query."""

import json

from webscout import DuckDuckGoSearch, TextResult


def search_web(state: dict, keywords: str) -> tuple[str, dict]:
    """Search the web for a given query.

    Args:
        state:
            The current state of the text engine.
        keywords:
            The keywords to search for.

    Returns:
        A tuple (message, state) where message is a message with all the web results and
        state is information that the text engine should store.
    """
    results: list[TextResult] = DuckDuckGoSearch().text(
        keywords=keywords, max_results=10
    )
    return json.dumps([result.to_dict() for result in results]), state
