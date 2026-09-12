"""Deterministic resolution of spoken local aliases."""

import collections.abc as c
import string
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class ResolutionStatus(StrEnum):
    """Outcomes of exact alias resolution."""

    MATCHED = "matched"
    NOT_FOUND = "not_found"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass(frozen=True)
class ResolutionResult:
    """A resolved local reference and safe clarification candidates."""

    status: ResolutionStatus
    reference: str | None = None
    candidates: tuple[str, ...] = ()


class AliasResolver:
    """Resolve normalised aliases using exact, deterministic matching only."""

    def __init__(
        self,
        aliases: c.Mapping[str, str] | c.Iterable[tuple[str, str]],
        default_reference: str | None = None,
    ) -> None:
        """Initialise the resolver with alias-to-reference pairs."""
        values = aliases.items() if isinstance(aliases, c.Mapping) else aliases
        normalised: dict[str, list[str]] = {}
        for pair in values:
            alias, reference = pair
            if not isinstance(alias, str) or not isinstance(reference, str):
                continue
            key = normalise_alias(alias)
            if not key:
                continue
            normalised.setdefault(key, [])
            if reference not in normalised[key]:
                normalised[key].append(reference)
        self._aliases = normalised
        self.default_reference = default_reference

    def resolve(self, spoken: str | None) -> ResolutionResult:
        """Resolve a spoken alias without fuzzy matching.

        Args:
            spoken:
                The spoken value, or None when the caller did not name a target.

        Returns:
            An exact match, an ambiguity result, or a not-found result.
        """
        key = normalise_alias(spoken) if spoken is not None else ""
        if not key:
            if self.default_reference is None:
                return ResolutionResult(status=ResolutionStatus.NOT_FOUND)
            return ResolutionResult(
                status=ResolutionStatus.MATCHED, reference=self.default_reference
            )

        references = self._aliases.get(key, [])
        if not references:
            return ResolutionResult(status=ResolutionStatus.NOT_FOUND)
        if len(references) > 1:
            return ResolutionResult(
                status=ResolutionStatus.NEEDS_CLARIFICATION,
                candidates=tuple(sorted(references)),
            )
        return ResolutionResult(
            status=ResolutionStatus.MATCHED, reference=references[0]
        )


def resolve_alias(
    spoken: str | None,
    aliases: c.Mapping[str, str] | c.Iterable[tuple[str, str]],
    default_reference: str | None = None,
) -> ResolutionResult:
    """Resolve one value using the exact alias algorithm.

    Args:
        spoken:
            The spoken alias, or None to request the default.
        aliases:
            Alias-to-opaque-reference pairs.
        default_reference (optional):
            Reference to use when no target was spoken. Defaults to None.

    Returns:
        The deterministic resolution result.
    """
    return AliasResolver(aliases=aliases, default_reference=default_reference).resolve(
        spoken=spoken
    )


def normalise_alias(value: str) -> str:
    """Unicode-normalise, case-fold, and trim a spoken alias."""
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    punctuation = string.punctuation + "\u2018\u2019\u201c\u201d\u2013\u2014\u2026"
    value = value.strip(punctuation)
    while value and unicodedata.category(value[0]).startswith("P"):
        value = value[1:]
    while value and unicodedata.category(value[-1]).startswith("P"):
        value = value[:-1]
    return " ".join(value.split())
