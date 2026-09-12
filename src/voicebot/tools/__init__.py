"""Explicit adapters for tools that the text engine can use."""

import typing as t

from ..tool_runtime import (
    LegacyToolAdapter,
    ToolContext,
    ToolHandler,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from .cat import meow
from .news import get_news
from .timer import Timer, list_timers, set_timer, stop_timer
from .weather import get_weather
from .web_search import search_web


def _set_timer(
    state: dict[str, object], arguments: dict[str, object]
) -> tuple[str, dict[str, object]]:
    return t.cast(
        tuple[str, dict[str, object]],
        set_timer(
            state=state, duration_seconds=t.cast(int, arguments["duration_seconds"])
        ),
    )


def _list_timers(
    state: dict[str, object], arguments: dict[str, object]
) -> tuple[str, dict[str, object]]:
    del arguments
    return t.cast(tuple[str, dict[str, object]], list_timers(state=state))


def _stop_timer(
    state: dict[str, object], arguments: dict[str, object]
) -> tuple[str, dict[str, object]]:
    name = t.cast(str, arguments["name"])
    running_timers = state.get("running_timers", [])
    if not isinstance(running_timers, list):
        return f"Der var ingen timer med navnet {name}.", state
    matching_timers = [
        timer
        for timer in running_timers
        if isinstance(timer, Timer)
        and str(timer.duration).replace("00:", "0:") == name.replace("00:", "0:")
    ]
    if not matching_timers:
        return f"Der var ingen timer med navnet {name}.", state
    return t.cast(tuple[str, dict[str, object]], stop_timer(state=state, duration=name))


def _weather(
    state: dict[str, object], arguments: dict[str, object]
) -> tuple[str, dict[str, object]]:
    return get_weather(state=state, location=t.cast(str, arguments["location"]))


def _news(
    state: dict[str, object], arguments: dict[str, object]
) -> tuple[str, dict[str, object]]:
    del arguments
    return get_news(state=state)


def _web_search(
    state: dict[str, object], arguments: dict[str, object]
) -> tuple[str, dict[str, object]]:
    return search_web(state=state, keywords=t.cast(str, arguments["keywords"]))


def _cat(
    state: dict[str, object], arguments: dict[str, object]
) -> tuple[str, dict[str, object]]:
    del arguments
    return meow(state=state)


def lookup(state: dict[str, object], query: str) -> tuple[str, dict[str, object]]:
    """Compatibility hook used by the historical text-engine test fixture."""
    del query
    return "Funktionen er ikke tilgængelig.", state


def _lookup(
    state: dict[str, object], arguments: dict[str, object]
) -> tuple[str, dict[str, object]]:
    implementation = lookup
    return implementation(state=state, query=t.cast(str, arguments["query"]))


def _unavailable(context: ToolContext, arguments: dict[str, object]) -> ToolResult:
    del arguments
    return ToolResult(
        status=ToolStatus.UNAVAILABLE,
        operation_id=context.operation_id,
        message_da="Denne integration er ikke tilgængelig endnu.",
        retryable=False,
    )


LEGACY_TOOL_ADAPTERS: dict[str, LegacyToolAdapter] = {
    "get_weather": LegacyToolAdapter(function=_weather),
    "set_timer": LegacyToolAdapter(function=_set_timer),
    "stop_timer": LegacyToolAdapter(function=_stop_timer),
    "list_timers": LegacyToolAdapter(function=_list_timers),
    "get_news": LegacyToolAdapter(function=_news),
    "search_web": LegacyToolAdapter(function=_web_search),
    "meow": LegacyToolAdapter(function=_cat),
    "lookup": LegacyToolAdapter(function=_lookup),
}

MODEL_TOOL_NAMES = frozenset(
    {
        "set_timer",
        "list_timers",
        "stop_timer",
        "create_reminder",
        "list_reminders",
        "cancel_reminder",
        "list_calendar_events",
        "get_calendar_availability",
        "spotify_play",
        "spotify_control",
        "spotify_set_volume",
        "spotify_now_playing",
        "spotify_list_devices",
        "list_shopping_lists",
        "get_shopping_list",
        "add_shopping_items",
        "set_shopping_item_checked",
        "remove_shopping_item",
    }
)

_LEGACY_SCHEMAS: dict[str, dict[str, object]] = {
    "get_weather": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
        "additionalProperties": False,
    },
    "get_news": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "search_web": {
        "type": "object",
        "properties": {"keywords": {"type": "string"}},
        "required": ["keywords"],
        "additionalProperties": False,
    },
    "meow": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "lookup": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}


def build_registry(
    tools: list[dict[str, object]],
    *,
    integration_specs: t.Iterable[ToolSpec] = (),
    include_integrations: bool = False,
    include_legacy: bool = True,
) -> ToolRegistry:
    """Build a closed registry from schemas and bound integration adapters.

    ``integration_specs`` is intentionally separate from configuration. Configuration
    controls which tools are model-visible; the application assembly supplies the
    already-bound handlers. This prevents an unavailable placeholder from replacing a
    real provider adapter.
    """
    specs: list[ToolSpec] = []
    configured_names: set[str] = set()
    bound = {spec.name: spec for spec in integration_specs}
    allowed = MODEL_TOOL_NAMES | _LEGACY_SCHEMAS.keys() | {"lookup"}
    for tool in tools:
        name = tool.get("name")
        parameters = tool.get("parameters")
        if (
            not isinstance(name, str)
            or name not in allowed
            or not isinstance(parameters, dict)
        ):
            continue
        configured_names.add(name)
        spec = bound.get(name)
        if spec is not None:
            specs.append(spec)
            continue
        adapter = LEGACY_TOOL_ADAPTERS.get(name)
        handler: ToolHandler = adapter if adapter is not None else _unavailable
        description = tool.get("description", "")
        specs.append(
            ToolSpec(
                name=name,
                description=description if isinstance(description, str) else "",
                parameters=parameters,
                handler=handler,
            )
        )

    if include_integrations:
        for name, spec in bound.items():
            if name not in configured_names:
                specs.append(spec)
                configured_names.add(name)

    if include_legacy:
        for name, schema in _LEGACY_SCHEMAS.items():
            if name not in configured_names:
                specs.append(
                    ToolSpec(
                        name=name,
                        description="Legacy compatibility tool.",
                        parameters=schema,
                        handler=LEGACY_TOOL_ADAPTERS[name],
                    )
                )
    return ToolRegistry(specs=specs)
