"""Local conversational intent detection."""

import re

_END_PATTERNS = (
    r"(?:okay |ok )?(?:stop|ti stille|farvel|hej hej|vi ses|slut)(?: nu)?",
    r"(?:mange )?tak(?: for (?:hjælpen|hjaelpen|i dag|nu))?",
    r"(?:det var|det er) (?:alt|det)",
    r"jeg (?:er )?(?:færdig|faerdig)",
    r"du kan (?:godt )?(?:stoppe|ti stille)(?: nu)?",
    r"ikke mere(?:,? tak)?",
)


def confirmation_decision(text: str) -> bool | None:
    """Return a local yes/no decision, or ``None`` for an unrelated utterance."""
    normalised = re.sub(r"[^\\wæøå]+", " ", text.casefold()).strip()
    normalised = re.sub(r"\\s+", " ", normalised)
    if normalised in {"ja", "jep", "yes", "okay", "ok", "bekræft", "bekræfter"}:
        return True
    if normalised in {"nej", "no", "ellers ikke", "annuller", "drop det"}:
        return False
    return None


def is_end_conversation(text: str) -> bool:
    """Return whether an utterance clearly ends the conversation.

    Args:
        text:
            Transcribed user utterance.

    Returns:
        Whether the complete utterance expresses an intent to stop.
    """
    normalised = re.sub(r"[^\wæøå]+", " ", text.casefold()).strip()
    normalised = re.sub(r"\s+", " ", normalised)
    return any(re.fullmatch(pattern, normalised) for pattern in _END_PATTERNS)
