"""Shared helpers for cancellable voice tools."""

import threading

from ..tool_runtime import ToolContext


def get_cancel_event(state: dict[str, object] | ToolContext) -> threading.Event | None:
    """Return the active turn's cancellation event, if present.

    Args:
        state:
            The current text-engine state or a typed tool context.

    Returns:
        The active cancellation event, or None outside a streamed turn.
    """
    if isinstance(state, ToolContext):
        return state.cancel_event
    event = state.get("cancel_event")
    return event if isinstance(event, threading.Event) else None


def is_cancelled(state: dict[str, object] | ToolContext) -> bool:
    """Return whether the active turn has been cancelled.

    Args:
        state:
            The current text-engine state or a typed tool context.

    Returns:
        Whether cancellation has been requested.
    """
    event = get_cancel_event(state=state)
    return event is not None and event.is_set()
