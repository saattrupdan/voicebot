"""Get the latest news from DR."""

import datetime as dt
import logging
from time import sleep
from typing import Literal
from xml.etree import ElementTree

import chime
import httpx
from pydantic.main import BaseModel

from ..speech_synthesis import synthesise_speech

logger = logging.getLogger(__name__)


def get_news(state: dict) -> tuple[Literal[""], dict]:
    """Get the current news headlines from DR.

    Args:
        state:
            The current state of the text engine.

    Returns:
        A tuple (message, state) where message is a message indicating the news
        have been read and state is information that the text engine should store.
    """
    all_news_items: list[NewsItem] = list()

    base_url = "https://www.dr.dk/nyheder/service/feeds/{}"
    for category in ["indland", "udland", "politik"]:
        rss_feed = httpx.get(base_url.format(category)).text
        root = ElementTree.fromstring(rss_feed)
        channel = root.find("channel")
        if channel is None:
            continue

        for item in channel.findall("item"):
            title = item.find("title")
            description = item.find("description")
            pub_date = item.find("pubDate")
            if (
                title is None
                or description is None
                or pub_date is None
                or pub_date.text is None
            ):
                continue
            published_at = dt.datetime.strptime(
                pub_date.text, "%a, %d %b %Y %H:%M:%S GMT"
            )
            news_item = NewsItem(
                title=title.text or "",
                description=description.text or "",
                published_at=published_at,
            )
            all_news_items.append(news_item)

    all_news_items.sort(key=lambda x: x.published_at, reverse=True)

    # Take the top-5 news items, ignoring duplicates
    top_news_items = []
    for news in all_news_items:
        if news.title not in [item.title for item in top_news_items]:
            top_news_items.append(news)
        if len(top_news_items) >= 5:
            break

    logger.info("Reading out the latest news headlines...")
    synthesise_speech(text="Her er seneste nyt.", synthesiser=state.get("synthesiser"))

    for news in top_news_items:
        chime.theme("pokemon")
        chime.info()
        sleep(0.5)
        logger.info(f"Reading news item: {news.title!r}...")
        synthesise_speech(
            text=news.title + ". " + news.description,
            synthesiser=state.get("synthesiser"),
        )

    synthesise_speech(
        text="Det var alt for denne gang.", synthesiser=state.get("synthesiser")
    )
    logger.info("Finished reading the news.")

    return "", state


class NewsItem(BaseModel):
    """A news item."""

    title: str
    description: str
    published_at: dt.datetime
