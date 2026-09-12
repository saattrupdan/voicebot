"""Typed, side-effect-safe runtime for model-requested tools."""

import collections.abc as c
import datetime as dt
import json
import logging
import re
import threading
import typing as t
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class ToolStatus(StrEnum):
    """Statuses that can be returned by a tool."""

    OK = "ok"
    NEEDS_CLARIFICATION = "needs_clarification"
    CONFIRMATION_REQUIRED = "confirmation_required"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    CANCELLED = "cancelled"
    INVALID_REQUEST = "invalid_request"


@dataclass(frozen=True)
class ToolResult:
    """The safe, structured result of a tool invocation."""

    status: ToolStatus
    operation_id: str | None = None
    message_da: str = ""
    data: dict[str, object] | None = None
    candidates: list[object] = field(default_factory=list)
    retryable: bool = False
    legacy_message: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the result in the model-facing wire format."""
        return {
            "status": self.status.value,
            "operation_id": self.operation_id,
            "message_da": self.message_da,
            "data": self.data,
            "candidates": self.candidates,
            "retryable": self.retryable,
        }

    def to_json(self) -> str:
        """Serialise the result without leaking implementation exceptions."""
        return json.dumps(self.as_dict(), ensure_ascii=False)

    @property
    def needs_followup(self) -> bool:
        """Whether the model should receive this result and decide what to say."""
        return self.status is not ToolStatus.OK or bool(
            self.message_da or self.data or self.candidates
        )


@dataclass
class ToolContext:
    """Context passed to an adapter, including cooperative cancellation."""

    state: dict[str, object]
    cancel_event: threading.Event | None = None
    operation_id: str | None = None
    device_id: str | None = None

    @property
    def cancelled(self) -> bool:
        """Return whether the current turn has been interrupted."""
        return self.cancel_event is not None and self.cancel_event.is_set()

    @property
    def cancellation_event(self) -> threading.Event | None:
        """Return the event used to cancel this operation."""
        return self.cancel_event

    def is_cancelled(self) -> bool:
        """Return whether the current turn has been interrupted."""
        return self.cancelled

    def check_cancelled(self) -> None:
        """Raise ``CancelledError`` when the turn has been interrupted."""
        self.raise_if_cancelled()

    def raise_if_cancelled(self) -> None:
        """Raise ``CancelledError`` before cancellable work begins."""
        if self.cancelled:
            raise CancelledError


class CancelledError(Exception):
    """Internal signal used when an adapter observes turn cancellation."""


ToolHandler = c.Callable[[ToolContext, dict[str, object]], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    """Allow-listed model-visible tool definition."""

    name: str
    description: str
    parameters: dict[str, object]
    handler: ToolHandler

    def as_openai_tool(self) -> dict[str, object]:
        """Return this definition in Chat Completions format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """A closed registry of typed tool specifications."""

    def __init__(self, specs: c.Iterable[ToolSpec] = ()) -> None:
        """Initialise a registry with optional tool specifications."""
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec=spec)

    def register(self, spec: ToolSpec) -> None:
        """Register one tool, rejecting duplicate names."""
        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        """Return an allow-listed tool, if present."""
        return self._specs.get(name)

    def replace(self, spec: ToolSpec) -> None:
        """Replace a registered adapter, for a later provider registration."""
        if spec.name not in self._specs:
            raise KeyError(f"Tool is not allow-listed: {spec.name}")
        self._specs[spec.name] = spec

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in registration order."""
        return tuple(self._specs)

    def schemas(self) -> list[dict[str, object]]:
        """Return registered model-visible schemas."""
        return [spec.as_openai_tool() for spec in self._specs.values()]


class LegacyToolAdapter:
    """Adapt the existing state-based tools to the typed runtime."""

    def __init__(
        self,
        function: c.Callable[
            [dict[str, object], dict[str, object]], tuple[str, dict[str, object]]
        ],
    ) -> None:
        """Initialise an adapter around a legacy state-based function."""
        self.function = function

    def __call__(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """Invoke the legacy function exactly once."""
        context.raise_if_cancelled()
        had_cancel_event = "cancel_event" in context.state
        if context.cancel_event is not None:
            context.state["cancel_event"] = context.cancel_event
        try:
            response, updates = self.function(context.state, arguments)
        finally:
            if context.cancel_event is not None and not had_cancel_event:
                context.state.pop("cancel_event", None)
        context.state.update(updates)
        return ToolResult(
            status=ToolStatus.OK,
            operation_id=context.operation_id,
            message_da=response,
            legacy_message=response,
        )


class ToolRuntime:
    """Validate and invoke tools without dynamic dispatch or retries."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        """Initialise the runtime with a closed registry."""
        self.registry = registry or ToolRegistry()

    def register(self, spec: ToolSpec) -> None:
        """Register an allow-listed tool before serving requests."""
        self.registry.register(spec=spec)

    @property
    def names(self) -> tuple[str, ...]:
        """Return the names in the live allow-list."""
        return self.registry.names

    def schemas(self) -> list[dict[str, object]]:
        """Return the live model-visible schemas."""
        return self.registry.schemas()

    def replace(self, spec: ToolSpec) -> None:
        """Replace an unavailable adapter with a provider implementation."""
        self.registry.replace(spec=spec)

    def invoke(
        self, name: str, arguments: object, context: ToolContext | None = None
    ) -> ToolResult:
        """Validate and invoke an allow-listed tool once."""
        context = context or ToolContext(state={})
        spec = self.registry.get(name=name)
        if spec is None:
            logger.warning("Rejected unknown tool name=%s", _safe_name(name))
            return ToolResult(
                status=ToolStatus.INVALID_REQUEST,
                message_da="Den ønskede funktion er ikke tilgængelig.",
            )

        operation_id = context.operation_id or uuid.uuid4().hex
        context.operation_id = operation_id
        error = validate_arguments(schema=spec.parameters, arguments=arguments)
        if error is not None:
            logger.warning(
                "Rejected invalid tool request name=%s operation_id=%s reason=%s",
                _safe_name(name),
                operation_id,
                error,
            )
            return ToolResult(
                status=ToolStatus.INVALID_REQUEST,
                operation_id=operation_id,
                message_da="Forespørgslen havde ugyldige argumenter.",
            )
        if context.cancelled:
            return ToolResult(
                status=ToolStatus.CANCELLED,
                operation_id=operation_id,
                message_da="Handlingen blev afbrudt.",
            )

        logger.info(
            "Invoking tool name=%s operation_id=%s", _safe_name(name), operation_id
        )
        typed_arguments = arguments if isinstance(arguments, dict) else {}
        try:
            result = spec.handler(context, typed_arguments)
        except CancelledError:
            return ToolResult(
                status=ToolStatus.CANCELLED,
                operation_id=operation_id,
                message_da="Handlingen blev afbrudt.",
            )
        except Exception:
            logger.error(
                "Tool failed name=%s operation_id=%s", _safe_name(name), operation_id
            )
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                operation_id=operation_id,
                message_da="Funktionen er midlertidigt utilgængelig.",
                retryable=True,
            )

        if not isinstance(result, ToolResult):
            logger.error(
                "Tool returned an invalid result name=%s operation_id=%s",
                _safe_name(name),
                operation_id,
            )
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                operation_id=operation_id,
                message_da="Funktionen er midlertidigt utilgængelig.",
                retryable=True,
            )

        return ToolResult(
            status=result.status,
            operation_id=operation_id,
            message_da=result.message_da,
            data=result.data,
            candidates=result.candidates,
            retryable=result.retryable,
            legacy_message=result.legacy_message,
        )


def validate_arguments(schema: dict[str, object], arguments: object) -> str | None:
    """Validate an argument object against a strict JSON schema.

    The runtime intentionally implements the small JSON Schema subset used by the
    model-visible tools, keeping validation deterministic and dependency-free.
    """
    return _validate_value(schema=schema, value=arguments, path="arguments")


def _validate_value(schema: dict[str, object], value: object, path: str) -> str | None:
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        valid_branches = sum(
            _validate_value(branch, value, path) is None
            for branch in one_of
            if isinstance(branch, dict)
        )
        if valid_branches != 1:
            return f"{path} does not satisfy exactly one constraint"

    types = schema.get("type")
    allowed_types = [types] if isinstance(types, str) else types
    if isinstance(allowed_types, list):
        if value is None and "null" in allowed_types:
            return None
        if not any(
            _matches_type(value=value, type_name=item) for item in allowed_types
        ):
            return f"{path} has the wrong type"
    if value is None:
        return None

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"{path} is not an allowed value"
    if isinstance(value, str):
        if "minLength" in schema and len(value) < t.cast(int, schema["minLength"]):
            return f"{path} is too short"
        if schema.get("format") == "date-time" and not _has_offset(value):
            return f"{path} must include a timezone offset"
    if isinstance(value, int) and not isinstance(value, bool):
        if "minimum" in schema and value < t.cast(int, schema["minimum"]):
            return f"{path} is below the minimum"
        if "maximum" in schema and value > t.cast(int, schema["maximum"]):
            return f"{path} is above the maximum"
    if isinstance(value, float):
        if "minimum" in schema and value < t.cast(float, schema["minimum"]):
            return f"{path} is below the minimum"
        if "maximum" in schema and value > t.cast(float, schema["maximum"]):
            return f"{path} is above the maximum"

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return f"{path} has an invalid schema"
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                return f"{path} contains unknown properties"
        required = schema.get("required", [])
        if isinstance(required, list):
            missing = set(required) - set(value)
            if missing:
                return f"{path} is missing required properties"
        for key, child in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                error = _validate_value(
                    schema=child_schema, value=child, path=f"{path}.{key}"
                )
                if error is not None:
                    return error
    if isinstance(value, list):
        if "minItems" in schema and len(value) < t.cast(int, schema["minItems"]):
            return f"{path} has too few items"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _validate_value(
                    schema=item_schema, value=item, path=f"{path}[{index}]"
                )
                if error is not None:
                    return error
    return None


def _matches_type(value: object, type_name: object) -> bool:
    return (
        (type_name == "object" and isinstance(value, dict))
        or (type_name == "array" and isinstance(value, list))
        or (type_name == "string" and isinstance(value, str))
        or (
            type_name == "integer"
            and isinstance(value, int)
            and not isinstance(value, bool)
        )
        or (
            type_name == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
        or (type_name == "boolean" and isinstance(value, bool))
        or (type_name == "null" and value is None)
    )


def _has_offset(value: str) -> bool:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _safe_name(name: object) -> str:
    """Return a short, non-sensitive log representation of a tool name."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,80}", name):
        return "<invalid>"
    return name
