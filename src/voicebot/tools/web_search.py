"""Search the web for a given query."""

import typing as t

from pydantic import BaseModel
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
    result_str: str = "\n\n".join(
        f"# {result.title}\n\n{result.body}" for result in results
    )
    return result_str, state


class SearchWebParameters(BaseModel):
    """The parameters for the search_web function."""

    keywords: str


class SearchWebResponse(BaseModel):
    """A response containing the web results."""

    name: t.Literal["search_web"]
    parameters: SearchWebParameters
