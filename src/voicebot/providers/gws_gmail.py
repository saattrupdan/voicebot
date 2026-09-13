"""Hardened Gmail access through the already-authenticated local ``gws`` CLI."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import html.parser
import re

from ..tool_runtime import ToolContext
from .gws_transport import (
    GwsForbidden,
    GwsInvalidResponse,
    GwsRateLimited,
    GwsTransport,
    GwsTransportError,
    GwsUnauthenticated,
    GwsUnavailable,
    json_argument,
)

MAX_MESSAGES = 10
MAX_QUERY_LENGTH = 200
MAX_BODY_CHARS = 6000
MAX_OUTPUT_CHARS = 8000
MAX_ATTACHMENTS = 10
MAX_ATTACHMENT_NAME = 160
_MAX_HEADER = 500

_ALLOWED_COMMANDS = {
    ("gmail", "users", "messages", "list"),
    ("gmail", "users", "messages", "get"),
    ("gmail", "users", "drafts", "create"),
}


class GmailError(Exception):
    """Base class for safe Gmail provider failures."""


class GmailUnauthenticated(GmailError):
    """The local Gmail session is not authenticated."""


class GmailForbidden(GmailError):
    """The authenticated account cannot access Gmail."""


class GmailRateLimited(GmailError):
    """Gmail rate limiting prevents this request."""


class GmailUnavailable(GmailError):
    """The local CLI or Gmail is temporarily unavailable."""


class GmailInvalidResponse(GmailError):
    """Gmail returned an unusable response."""


@dataclasses.dataclass(frozen=True, slots=True)
class GmailMessageSummary:
    """Private provider message data used to make a safe summary."""

    message_id: str
    thread_id: str
    subject: str
    sender: str
    received_at: str
    snippet: str


@dataclasses.dataclass(frozen=True, slots=True)
class GmailAttachment:
    """Safe attachment metadata; attachment content is never downloaded."""

    filename: str
    mime_type: str
    size: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class GmailMessage:
    """Private provider message data after bounded MIME parsing."""

    message_id: str
    thread_id: str
    subject: str
    sender: str
    recipients: str
    received_at: str
    body: str
    attachments: tuple[GmailAttachment, ...]


class GwsGmailProvider:
    """Invoke only Gmail list, get, and draft-create commands."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        timeout: float = 15.0,
        max_output_bytes: int = 256 * 1024,
    ) -> None:
        """Initialise the shared bounded local CLI transport."""
        self.transport = GwsTransport(
            executable=executable, timeout=timeout, max_output_bytes=max_output_bytes
        )

    def search_messages(
        self,
        *,
        query: str,
        max_results: int = MAX_MESSAGES,
        context: ToolContext | None = None,
    ) -> list[GmailMessageSummary]:
        """Search Gmail and return at most ten safe message summaries."""
        _validate_query(query)
        _validate_max_results(max_results)
        return self._list_summaries(
            query=f"({query}) -in:spam -in:trash",
            max_results=max_results,
            context=context,
        )

    def list_latest_messages(
        self, *, max_results: int = MAX_MESSAGES, context: ToolContext | None = None
    ) -> list[GmailMessageSummary]:
        """Return at most ten newest non-spam, non-trash message summaries."""
        _validate_max_results(max_results)
        return self._list_summaries(
            query="-in:spam -in:trash", max_results=max_results, context=context
        )

    def read_message(
        self, *, message_id: str, context: ToolContext | None = None
    ) -> GmailMessage:
        """Read one message using a private ID supplied by the handle store."""
        _validate_private_id(message_id)
        payload = self._invoke(
            [
                "gmail",
                "users",
                "messages",
                "get",
                "--params",
                json_argument(
                    {
                        "userId": "me",
                        "id": message_id,
                        "format": "full",
                        "fields": (
                            "id,threadId,internalDate,payload(headers,mimeType,"
                            "filename,body(data,size),parts)"
                        ),
                    }
                ),
                "--format",
                "json",
            ],
            context=context,
        )
        return _message_from_payload(payload=payload)

    def create_draft(
        self, *, raw_message: str, context: ToolContext | None = None
    ) -> None:
        """Save one RFC822 message as an unsent Gmail draft."""
        if not raw_message or len(raw_message.encode()) > 64 * 1024:
            raise ValueError("draft is too large")
        self._invoke(
            [
                "gmail",
                "users",
                "drafts",
                "create",
                "--params",
                json_argument({"userId": "me"}),
                "--json",
                json_argument({"message": {"raw": raw_message}}),
                "--format",
                "json",
            ],
            context=context,
        )

    def close(self) -> None:
        """Release no resources; every request is an isolated process."""

    def _invoke(
        self, arguments: list[str], *, context: ToolContext | None
    ) -> dict[str, object]:
        try:
            return self.transport.invoke(
                arguments, allowed_commands=_ALLOWED_COMMANDS, context=context
            )
        except GwsTransportError as error:
            raise translate_gws_error(error) from None

    def _list_summaries(
        self, *, query: str, max_results: int, context: ToolContext | None
    ) -> list[GmailMessageSummary]:
        payload = self._invoke(
            [
                "gmail",
                "users",
                "messages",
                "list",
                "--params",
                json_argument(
                    {
                        "userId": "me",
                        "q": query,
                        "maxResults": max_results,
                        "includeSpamTrash": False,
                        "fields": "messages(id,threadId,snippet)",
                    }
                ),
                "--format",
                "json",
            ],
            context=context,
        )
        items = payload.get("messages")
        if not isinstance(items, list):
            raise GmailInvalidResponse("invalid Gmail message list")
        summaries: list[GmailMessageSummary] = []
        for item in items[:max_results]:
            if not isinstance(item, dict):
                raise GmailInvalidResponse("invalid Gmail message list")
            message_id, thread_id = item.get("id"), item.get("threadId")
            if not isinstance(message_id, str) or not isinstance(thread_id, str):
                raise GmailInvalidResponse("invalid Gmail message list")
            message = self.read_message(message_id=message_id, context=context)
            summaries.append(
                GmailMessageSummary(
                    message_id=message.message_id,
                    thread_id=message.thread_id,
                    subject=message.subject,
                    sender=message.sender,
                    received_at=message.received_at,
                    snippet=_clean_text(str(item.get("snippet", "")))[:500],
                )
            )
        return summaries


def _message_from_payload(*, payload: dict[str, object]) -> GmailMessage:
    message_id = _required_string(payload, "id")
    thread_id = _required_string(payload, "threadId")
    root = payload.get("payload")
    if not isinstance(root, dict):
        raise GmailInvalidResponse("invalid Gmail message")
    headers = _headers(root.get("headers"))
    body, attachments = _extract_parts(root)
    if not body:
        body = "(Denne besked indeholder ingen læsbar tekst.)"
    received_at = _clean_header(headers.get("date", ""))
    if not received_at:
        raw_date = payload.get("internalDate")
        received_at = _clean_text(str(raw_date))[:80] if raw_date is not None else ""
    return GmailMessage(
        message_id=message_id,
        thread_id=thread_id,
        subject=_clean_header(headers.get("subject", "")),
        sender=_clean_header(headers.get("from", "")),
        recipients=_clean_header(headers.get("to", "")),
        received_at=received_at,
        body=body[:MAX_BODY_CHARS],
        attachments=tuple(attachments[:MAX_ATTACHMENTS]),
    )


def _extract_parts(root: dict[str, object]) -> tuple[str, list[GmailAttachment]]:
    plain: list[str] = []
    html: list[str] = []
    attachments: list[GmailAttachment] = []

    def visit(part: dict[str, object]) -> None:
        mime_type = str(part.get("mimeType", "")).casefold()
        filename = part.get("filename")
        body = part.get("body")
        if isinstance(filename, str) and filename.strip():
            size = body.get("size") if isinstance(body, dict) else None
            attachments.append(
                GmailAttachment(
                    filename=_clean_text(filename)[:MAX_ATTACHMENT_NAME],
                    mime_type=_clean_text(mime_type)[:100],
                    size=size if isinstance(size, int) and size >= 0 else None,
                )
            )
        if (
            not (isinstance(filename, str) and filename.strip())
            and isinstance(body, dict)
            and isinstance(body.get("data"), str)
        ):
            decoded = _decode_body(body["data"])
            if mime_type == "text/plain":
                plain.append(decoded)
            elif mime_type == "text/html":
                html.append(decoded)
        parts = part.get("parts")
        if isinstance(parts, list):
            for child in parts:
                if isinstance(child, dict):
                    visit(child)

    visit(root)
    if plain:
        return _clean_text("\n\n".join(plain))[:MAX_BODY_CHARS], attachments
    return _html_to_text("\n\n".join(html))[:MAX_BODY_CHARS], attachments


def _decode_body(value: str) -> str:
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode(
            "utf-8", errors="replace"
        )
    except ValueError, binascii.Error:
        return ""


def _html_to_text(value: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(value[:MAX_OUTPUT_CHARS])
        parser.close()
    except ValueError, AssertionError:
        return ""
    return _clean_text("".join(parser.text))


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.text.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip = True
        if not self._skip and tag in {"br", "p", "div", "li", "tr"}:
            self.text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip = False


def _headers(value: object) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    result: dict[str, str] = {}
    for item in value:
        if isinstance(item, dict):
            name, value = item.get("name"), item.get("value")
            if isinstance(name, str) and isinstance(value, str):
                result[name.casefold()] = value
    return result


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise GmailInvalidResponse("invalid Gmail message")
    return value


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
        raise ValueError("query is invalid")
    if "\r" in query or "\n" in query:
        raise ValueError("query contains a header break")


def _validate_max_results(value: int) -> None:
    if isinstance(value, bool) or not 1 <= value <= MAX_MESSAGES:
        raise ValueError("max_results must be between 1 and 10")


def _validate_private_id(value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError("message ID is invalid")


def _clean_header(value: str) -> str:
    return _clean_text(value.replace("\r", " ").replace("\n", " "))[:_MAX_HEADER]


def _clean_text(value: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value).strip()


GmailProvider = GwsGmailProvider


def translate_gws_error(error: GwsTransportError) -> GmailError:
    """Translate shared transport errors to Gmail-specific safe errors."""
    mapping: list[tuple[type[GwsTransportError], type[GmailError]]] = [
        (GwsUnauthenticated, GmailUnauthenticated),
        (GwsForbidden, GmailForbidden),
        (GwsRateLimited, GmailRateLimited),
        (GwsInvalidResponse, GmailInvalidResponse),
        (GwsUnavailable, GmailUnavailable),
    ]
    for source, target in mapping:
        if isinstance(error, source):
            return target(str(error))
    return GmailUnavailable("Gmail is unavailable")


__all__ = [
    "GmailAttachment",
    "GmailError",
    "GmailForbidden",
    "GmailInvalidResponse",
    "GmailMessage",
    "GmailMessageSummary",
    "GmailProvider",
    "GmailRateLimited",
    "GmailUnauthenticated",
    "GmailUnavailable",
    "GwsGmailProvider",
    "MAX_MESSAGES",
    "MAX_OUTPUT_CHARS",
    "translate_gws_error",
]
