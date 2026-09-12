"""Shared helpers for cancellable voice tools."""

import threading


def get_cancel_event(state: dict) -> threading.Event | None:
    """Return the active turn's cancellation event, if present.

    Args:
        state:
            The current text-engine state.

    Returns:
        The active cancellation event, or None outside a streamed turn.
    """
    event = state.get("cancel_event")
    return event if isinstance(event, threading.Event) else None


def is_cancelled(state: dict) -> bool:
    """Return whether the active turn has been cancelled.

    Args:
        state:
            The current text-engine state.

    Returns:
        Whether cancellation has been requested.
    """
    event = get_cancel_event(state=state)
    return event is not None and event.is_set()
