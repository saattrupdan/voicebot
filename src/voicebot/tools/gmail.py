"""Safe model-facing Gmail tools with process-local opaque message handles."""

from __future__ import annotations

import base64
import collections.abc as c
import datetime as dt
import email.message
import re
import secrets

from ..providers.gws_gmail import (
    GmailError,
    GmailForbidden,
    GmailMessage,
    GmailMessageSummary,
    GmailOutcomeUnknown,
    GmailRateLimited,
    GmailUnauthenticated,
    GmailUnavailable,
    GwsGmailProvider,
    translate_gws_error,
)
from ..providers.gws_transport import GwsTransportError
from ..resolution import AliasResolver, ResolutionStatus
from ..tool_runtime import ToolContext, ToolResult, ToolSpec, ToolStatus

MAX_HANDLES = 100
HANDLE_TTL = dt.timedelta(minutes=15)
MAX_RECIPIENTS = 10
MAX_SUBJECT = 200
MAX_DRAFT_BODY = 20_000
_HEADER_BREAK = re.compile(r"[\r\n]")
_HANDLE = re.compile(r"^[A-Za-z0-9_-]{16,100}$")
_LOCAL_PART = re.compile(r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+$")
_DOMAIN_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


class GmailHandleStore:
    """Keep raw Gmail IDs only in process memory and scope them locally."""

    def __init__(
        self,
        *,
        max_handles: int = MAX_HANDLES,
        ttl: dt.timedelta = HANDLE_TTL,
        clock: c.Callable[[], dt.datetime] | None = None,
    ) -> None:
        """Create a bounded expiring handle store."""
        if max_handles < 1 or ttl <= dt.timedelta(0):
            raise ValueError("handle limits must be positive")
        self.max_handles = max_handles
        self.ttl = ttl
        self._clock = clock or (lambda: dt.datetime.now(dt.UTC))
        self._values: dict[str, tuple[str, str, str, str, dt.datetime]] = {}

    def put(self, *, message_id: str, thread_id: str, profile: str, device: str) -> str:
        """Store private IDs and return a random opaque handle."""
        self._prune()
        while len(self._values) >= self.max_handles:
            self._values.pop(next(iter(self._values)))
        handle = secrets.token_urlsafe(24)
        now = self._clock()
        self._values[handle] = (message_id, thread_id, profile, device, now + self.ttl)
        return handle

    def resolve(
        self, *, handle: str, profile: str, device: str
    ) -> tuple[str, str] | None:
        """Resolve a handle only for its original profile and device."""
        self._prune()
        value = self._values.get(handle)
        if value is None or value[2] != profile or value[3] != device:
            return None
        return value[0], value[1]

    def _prune(self) -> None:
        now = self._clock()
        for handle, value in list(self._values.items()):
            if value[4] <= now:
                del self._values[handle]


class GmailToolAdapter:
    """Resolve profiles and expose bounded Gmail operations."""

    def __init__(
        self,
        provider: GwsGmailProvider,
        *,
        profiles: c.Iterable[str],
        profile_aliases: c.Mapping[str, str] | None = None,
        default_profile: str | None = None,
        handles: GmailHandleStore | None = None,
    ) -> None:
        """Initialise profile routing without persisting provider IDs."""
        self.provider = provider
        self._profiles = set(profiles)
        aliases = [
            (alias, reference) for alias, reference in (profile_aliases or {}).items()
        ]
        aliases.extend((profile, profile) for profile in self._profiles)
        self._resolver = AliasResolver(
            aliases=aliases, default_reference=default_profile
        )
        self.handles = handles or GmailHandleStore()

    def search(self, context: ToolContext, arguments: dict[str, object]) -> ToolResult:
        """Search Gmail and return handle-based summaries."""
        profile = self._profile(arguments.get("profile_name"))
        if isinstance(profile, ToolResult):
            return profile
        query, limit = arguments.get("query"), arguments.get("max_results")
        if (
            not isinstance(query, str)
            or not isinstance(limit, int)
            or isinstance(limit, bool)
        ):
            return _invalid()
        try:
            messages = self.provider.search_messages(
                query=query, max_results=limit, context=context
            )
        except (ValueError, GmailError, GwsTransportError) as error:
            return _error(error)
        return self._summaries(context=context, profile=profile, messages=messages)

    def latest(self, context: ToolContext, arguments: dict[str, object]) -> ToolResult:
        """Return handle-based summaries for the newest Gmail messages."""
        profile = self._profile(arguments.get("profile_name"))
        if isinstance(profile, ToolResult):
            return profile
        limit = arguments.get("max_results")
        if not isinstance(limit, int) or isinstance(limit, bool):
            return _invalid()
        try:
            messages = self.provider.list_latest_messages(
                max_results=limit, context=context
            )
        except (ValueError, GmailError, GwsTransportError) as error:
            return _error(error)
        return self._summaries(context=context, profile=profile, messages=messages)

    def read(self, context: ToolContext, arguments: dict[str, object]) -> ToolResult:
        """Read one message addressed by a profile/device-bound opaque handle."""
        profile = self._profile(arguments.get("profile_name"))
        if isinstance(profile, ToolResult):
            return profile
        handle = arguments.get("message_handle")
        if not isinstance(handle, str) or not _HANDLE.fullmatch(handle):
            return _invalid()
        resolved = self.handles.resolve(
            handle=handle, profile=profile, device=_device(context)
        )
        if resolved is None:
            return ToolResult(
                ToolStatus.NOT_FOUND, message_da="Beskeden blev ikke fundet."
            )
        try:
            message = self.provider.read_message(
                message_id=resolved[0], context=context
            )
        except (ValueError, GmailError, GwsTransportError) as error:
            return _error(error)
        return ToolResult(ToolStatus.OK, data=_message_data(message))

    def create_draft(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """Create an unsent RFC822 Gmail draft without retaining its private fields."""
        profile = self._profile(arguments.get("profile_name"))
        if isinstance(profile, ToolResult):
            return profile
        recipients = arguments.get("recipients")
        subject, body = arguments.get("subject"), arguments.get("body")
        try:
            clean_recipients = _validate_recipients(recipients)
            clean_subject = _validate_header(subject, MAX_SUBJECT, "subject")
            clean_body = _validate_body(body)
            message = email.message.EmailMessage()
            message["To"] = ", ".join(clean_recipients)
            message["Subject"] = clean_subject
            message.set_content(clean_body)
            raw = (
                base64.urlsafe_b64encode(message.as_bytes())
                .rstrip(b"=")
                .decode("ascii")
            )
            self.provider.create_draft(raw_message=raw, context=context)
        except (ValueError, GmailError, GwsTransportError) as error:
            return _error(error)
        return ToolResult(ToolStatus.OK, data={"saved": True, "sent": False})

    def _summaries(
        self, *, context: ToolContext, profile: str, messages: list[GmailMessageSummary]
    ) -> ToolResult:
        device = _device(context)
        data = []
        for message in messages[:10]:
            data.append(
                {
                    "message_handle": self.handles.put(
                        message_id=message.message_id,
                        thread_id=message.thread_id,
                        profile=profile,
                        device=device,
                    ),
                    "subject": message.subject,
                    "sender": message.sender,
                    "received_at": message.received_at,
                    "snippet": message.snippet,
                }
            )
        return ToolResult(ToolStatus.OK, data={"messages": data})

    def _profile(self, value: object) -> str | ToolResult:
        if value is not None and not isinstance(value, str):
            return _invalid()
        result = self._resolver.resolve(value)
        if (
            result.status is ResolutionStatus.MATCHED
            and result.reference in self._profiles
        ):
            return result.reference
        if result.status is ResolutionStatus.NEEDS_CLARIFICATION:
            return ToolResult(
                ToolStatus.NEEDS_CLARIFICATION,
                message_da="Hvilken person mener du?",
                candidates=list(result.candidates),
            )
        return ToolResult(ToolStatus.NOT_FOUND, message_da="Personen blev ikke fundet.")


def make_gmail_tool_specs(adapter: GmailToolAdapter) -> tuple[ToolSpec, ...]:
    """Create exactly the four strict Gmail tool specifications."""
    profile = {"type": ["string", "null"]}
    return (
        ToolSpec(
            name="search_gmail_messages",
            description="Search Gmail messages without opening attachments.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "profile_name": profile,
                    "query": {"type": "string", "maxLength": 200},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["profile_name", "query", "max_results"],
            },
            handler=adapter.search,
        ),
        ToolSpec(
            name="list_latest_gmail_messages",
            description="List the newest Gmail messages without opening attachments.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "profile_name": profile,
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["profile_name", "max_results"],
            },
            handler=adapter.latest,
        ),
        ToolSpec(
            name="read_gmail_message",
            description="Read a Gmail message selected by a message handle.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "profile_name": profile,
                    "message_handle": {
                        "type": "string",
                        "minLength": 16,
                        "maxLength": 100,
                    },
                },
                "required": ["profile_name", "message_handle"],
            },
            handler=adapter.read,
        ),
        ToolSpec(
            name="create_gmail_draft",
            description="Save an unsent Gmail draft; this never sends email.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "profile_name": profile,
                    "recipients": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 320},
                        "minItems": 1,
                        "maxItems": MAX_RECIPIENTS,
                    },
                    "subject": {"type": "string", "maxLength": MAX_SUBJECT},
                    "body": {"type": "string", "maxLength": MAX_DRAFT_BODY},
                },
                "required": ["profile_name", "recipients", "subject", "body"],
            },
            handler=adapter.create_draft,
            mutates=True,
            persist_arguments=False,
        ),
    )


def _device(context: ToolContext) -> str:
    value = context.device_id or context.state.get("device_id")
    return value if isinstance(value, str) and value else "local-device"


def _message_data(message: GmailMessage) -> dict[str, object]:
    return {
        "subject": message.subject,
        "sender": message.sender,
        "recipients": message.recipients,
        "received_at": message.received_at,
        "body": message.body[:8000],
        "attachments": [
            {
                "filename": attachment.filename,
                "mime_type": attachment.mime_type,
                "size": attachment.size,
            }
            for attachment in message.attachments[:10]
        ],
    }


def _validate_recipients(value: object) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_RECIPIENTS:
        raise ValueError("recipients are invalid")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _valid_mailbox(item):
            raise ValueError("recipients are invalid")
        result.append(item)
    return result


def _valid_mailbox(value: str) -> bool:
    if (
        not value
        or value != value.strip()
        or len(value) > 254
        or _HEADER_BREAK.search(value)
        or value.count("@") != 1
    ):
        return False
    local, domain = value.rsplit("@", 1)
    if (
        not 1 <= len(local) <= 64
        or not _LOCAL_PART.fullmatch(local)
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
        or len(domain) > 253
    ):
        return False
    labels = domain.split(".")
    return (
        len(labels) >= 2
        and all(_DOMAIN_LABEL.fullmatch(label) for label in labels)
        and (labels[-1].isalpha() or labels[-1].casefold().startswith("xn--"))
    )


def _validate_header(value: object, limit: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > limit
        or _HEADER_BREAK.search(value)
    ):
        raise ValueError(f"{name} is invalid")
    return value.strip()


def _validate_body(value: object) -> str:
    if not isinstance(value, str) or len(value) > MAX_DRAFT_BODY or "\x00" in value:
        raise ValueError("body is invalid")
    return value


def _invalid() -> ToolResult:
    return ToolResult(
        ToolStatus.INVALID_REQUEST, message_da="Ugyldige Gmail-argumenter."
    )


def _error(error: Exception) -> ToolResult:
    if isinstance(error, GwsTransportError):
        error = translate_gws_error(error)
    if isinstance(error, ValueError):
        return _invalid()
    statuses: list[tuple[type[GmailError], ToolStatus, str, bool]] = [
        (
            GmailUnauthenticated,
            ToolStatus.UNAUTHENTICATED,
            "Gmail er ikke forbundet.",
            False,
        ),
        (GmailForbidden, ToolStatus.FORBIDDEN, "Gmail-adgang blev afvist.", False),
        (
            GmailRateLimited,
            ToolStatus.RATE_LIMITED,
            "Gmail har bedt om en pause.",
            True,
        ),
        (
            GmailOutcomeUnknown,
            ToolStatus.OUTCOME_UNKNOWN,
            "Kladden kan være gemt og må ikke oprettes igen.",
            False,
        ),
        (
            GmailUnavailable,
            ToolStatus.UNAVAILABLE,
            "Gmail er midlertidigt utilgængelig.",
            True,
        ),
    ]
    for kind, status, message, retryable in statuses:
        if isinstance(error, kind):
            return ToolResult(status, message_da=message, retryable=retryable)
    return ToolResult(
        ToolStatus.UNAVAILABLE, message_da="Gmail-svaret kunne ikke læses."
    )


__all__ = ["GmailHandleStore", "GmailToolAdapter", "make_gmail_tool_specs"]
