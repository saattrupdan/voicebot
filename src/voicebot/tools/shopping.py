"""Model-facing Listonic shopping tools with exact local resolution."""

from __future__ import annotations

import collections.abc as c
import typing as t

from ..auth.credentials import ProviderAccount
from ..providers.listonic import (
    ListonicAuthenticationError,
    ListonicConflictError,
    ListonicContractDriftError,
    ListonicError,
    ListonicForbiddenError,
    ListonicInvalidRequestError,
    ListonicNotFoundError,
    ListonicProvider,
    ListonicRateLimitError,
    ShoppingItem,
    ShoppingList,
)
from ..resolution import AliasResolver, ResolutionResult, ResolutionStatus
from ..tool_runtime import CancelledError, ToolContext, ToolResult, ToolSpec, ToolStatus

AliasMap = c.Mapping[str, str]
NestedAliasMap = c.Mapping[str, c.Mapping[str, str]]


class ShoppingTools:
    """Expose Listonic operations after profile, list, and item resolution."""

    def __init__(
        self,
        provider: ListonicProvider,
        *,
        profile_aliases: AliasMap | None = None,
        list_aliases: NestedAliasMap | AliasMap | None = None,
        item_aliases: NestedAliasMap | AliasMap | None = None,
        default_profile: str | None = None,
        default_lists: c.Mapping[str, str] | None = None,
        confirmation_checker: c.Callable[[ToolContext, dict[str, object]], bool]
        | None = None,
    ) -> None:
        """Create bound shopping tool handlers.

        Args:
            provider:
                The gated Listonic provider adapter.
            profile_aliases (optional):
                Exact spoken profile aliases to local profile references.
            list_aliases (optional):
                Exact list aliases, either per profile or shared by all profiles.
            item_aliases (optional):
                Exact item aliases, either per list/profile or shared by all lists.
            default_profile (optional):
                Local profile reference used only when no profile was spoken.
            default_lists (optional):
                Local profile reference to provider list ID defaults.
            confirmation_checker (optional):
                Local runtime callback for an already-confirmed removal.
        """
        self.provider = provider
        self.profile_aliases = dict(profile_aliases or {})
        self.list_aliases = list_aliases or {}
        self.item_aliases = item_aliases or {}
        self.default_profile = default_profile
        self.default_lists = dict(default_lists or {})
        self.confirmation_checker = confirmation_checker

    def specs(self) -> tuple[ToolSpec, ...]:
        """Return all strict model-visible Listonic tool specifications."""
        return (
            ToolSpec(
                name="list_shopping_lists",
                description="List configured shopping-list aliases.",
                parameters=_LIST_LISTS_SCHEMA,
                handler=self.list_shopping_lists,
            ),
            ToolSpec(
                name="get_shopping_list",
                description="Return a bounded shopping list.",
                parameters=_GET_LIST_SCHEMA,
                handler=self.get_shopping_list,
            ),
            ToolSpec(
                name="add_shopping_items",
                description="Add one or more items.",
                parameters=_ADD_ITEMS_SCHEMA,
                handler=self.add_shopping_items,
                mutates=True,
            ),
            ToolSpec(
                name="set_shopping_item_checked",
                description="Mark exactly one shopping item as bought or not bought.",
                parameters=_SET_CHECKED_SCHEMA,
                handler=self.set_shopping_item_checked,
                mutates=True,
            ),
            ToolSpec(
                name="remove_shopping_item",
                description="Remove exactly one shopping item after confirmation.",
                parameters=_REMOVE_ITEM_SCHEMA,
                handler=self.remove_shopping_item,
                mutates=True,
            ),
        )

    def list_shopping_lists(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """List provider lists without exposing provider IDs."""
        profile, account, failure = self._resolve_profile(context, arguments)
        if failure is not None:
            return failure
        assert profile is not None and account is not None
        try:
            context.raise_if_cancelled()
            lists = self.provider.list_lists(account=account)
        except Exception as error:
            return _provider_failure(context=context, error=error)
        return _ok(
            context, data={"lists": [self._list_data(profile, item) for item in lists]}
        )

    def get_shopping_list(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """Return exact-list items, optionally excluding checked items."""
        profile, account, failure = self._resolve_profile(context, arguments)
        if failure is not None:
            return failure
        assert profile is not None and account is not None
        list_result, listed, failure = self._resolve_list(
            context=context,
            account=account,
            profile=profile,
            spoken=arguments.get("list_name"),
        )
        if failure is not None:
            return failure
        assert list_result is not None and listed is not None
        try:
            context.raise_if_cancelled()
            items = self.provider.list_items(account=account, list_id=list_result)
        except Exception as error:
            return _provider_failure(context=context, error=error)
        include_checked = arguments.get("include_checked")
        assert isinstance(include_checked, bool)
        if not include_checked:
            items = [item for item in items if not item.checked]
        return _ok(
            context,
            data={
                "list_name": listed,
                "items": [self._item_data(item) for item in items],
            },
        )

    def add_shopping_items(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """Add each requested item once, checking cancellation before each POST."""
        profile, account, failure = self._resolve_profile(context, arguments)
        if failure is not None:
            return failure
        assert profile is not None and account is not None
        list_result, listed, failure = self._resolve_list(
            context=context,
            account=account,
            profile=profile,
            spoken=arguments.get("list_name"),
        )
        if failure is not None:
            return failure
        assert list_result is not None and listed is not None
        items = arguments.get("items")
        if not isinstance(items, list) or not items:
            return _result(
                context, ToolStatus.INVALID_REQUEST, "Der skal være mindst én vare."
            )
        prepared: list[tuple[str, float | None, str | None]] = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                return _result(
                    context, ToolStatus.INVALID_REQUEST, "Varen var ugyldig."
                )
            name = t.cast(str, item["name"]).strip()
            if not name:
                return _result(
                    context, ToolStatus.INVALID_REQUEST, "Varen var ugyldig."
                )
            try:
                prepared.append(
                    (name, _as_number(item.get("amount")), _as_string(item.get("unit")))
                )
            except ValueError:
                return _result(
                    context, ToolStatus.INVALID_REQUEST, "Varen var ugyldig."
                )
        added: list[ShoppingItem] = []
        for name, amount, unit in prepared:
            try:
                context.raise_if_cancelled()
                added.append(
                    self.provider.add_item(
                        account=account,
                        list_id=list_result,
                        name=name,
                        amount=amount,
                        unit=unit,
                    )
                )
            except Exception as error:
                if added:
                    return _ok(
                        context,
                        message="De første varer blev tilføjet.",
                        data={
                            "list_name": listed,
                            "items": [self._item_data(value) for value in added],
                        },
                    )
                return _provider_failure(context=context, error=error)
        return _ok(
            context,
            data={
                "list_name": listed,
                "items": [self._item_data(value) for value in added],
            },
        )

    def set_shopping_item_checked(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """Check or uncheck one exact item."""
        profile, account, failure = self._resolve_profile(context, arguments)
        if failure is not None:
            return failure
        assert profile is not None and account is not None
        list_id, list_name, failure = self._resolve_list(
            context=context,
            account=account,
            profile=profile,
            spoken=arguments.get("list_name"),
        )
        if failure is not None:
            return failure
        assert list_id is not None and list_name is not None
        item_id, item_name, failure = self._resolve_item(
            context=context,
            account=account,
            profile=profile,
            list_id=list_id,
            spoken=t.cast(str, arguments.get("item_name")),
        )
        if failure is not None:
            return failure
        assert item_id is not None and item_name is not None
        checked = arguments.get("checked")
        assert isinstance(checked, bool)
        try:
            context.raise_if_cancelled()
            item = self.provider.set_item_checked(
                account=account, list_id=list_id, item_id=item_id, checked=checked
            )
        except Exception as error:
            return _provider_failure(context=context, error=error)
        return _ok(
            context,
            data={"list_name": list_name, "item": self._item_data(item, item_name)},
        )

    def remove_shopping_item(
        self, context: ToolContext, arguments: c.Mapping[str, object]
    ) -> ToolResult:
        """Request confirmation, then remove one exact item."""
        typed_arguments = dict(arguments)
        confirmed_target = (
            context.state.get("_confirmed_target")
            if context.confirmation_resume
            else None
        )
        if _valid_removal_target(confirmed_target):
            profile = confirmed_target["profile_name"]
            account = self.provider.accounts.get(profile)
            if account is None:
                return _result(
                    context,
                    ToolStatus.UNAUTHENTICATED,
                    "Listonic er ikke forbundet for profilen.",
                )
            list_id = confirmed_target["list_id"]
            list_name = confirmed_target["list_name"]
            item_id = confirmed_target["item_id"]
            item_name = confirmed_target["item_name"]
        else:
            profile, account, failure = self._resolve_profile(context, typed_arguments)
            if failure is not None:
                return failure
            assert profile is not None and account is not None
            list_id, list_name, failure = self._resolve_list(
                context=context,
                account=account,
                profile=profile,
                spoken=arguments.get("list_name"),
            )
            if failure is not None:
                return failure
            assert list_id is not None and list_name is not None
            item_id, item_name, failure = self._resolve_item(
                context=context,
                account=account,
                profile=profile,
                list_id=list_id,
                spoken=t.cast(str, arguments.get("item_name")),
            )
            if failure is not None:
                return failure
            assert item_id is not None and item_name is not None
        resolved: dict[str, object] = {
            "profile_name": self._profile_label(profile),
            "list_name": list_name,
            "item_name": item_name,
            "action": "remove_shopping_item",
        }
        context.state["_confirmation_arguments"] = {
            "profile_name": resolved["profile_name"],
            "list_name": resolved["list_name"],
            "item_name": resolved["item_name"],
        }
        context.state["_confirmation_target"] = {
            "profile_name": profile,
            "list_id": list_id,
            "list_name": list_name,
            "item_id": item_id,
            "item_name": item_name,
        }
        if not self._confirmed(context=context, resolved=resolved):
            return ToolResult(
                status=ToolStatus.CONFIRMATION_REQUIRED,
                operation_id=context.operation_id,
                message_da=f"Skal {item_name} fjernes fra {list_name}?",
                data=resolved,
            )
        try:
            context.raise_if_cancelled()
            self.provider.remove_item(account=account, list_id=list_id, item_id=item_id)
        except Exception as error:
            return _provider_failure(context=context, error=error)
        return _ok(
            context,
            data={"list_name": list_name, "item_name": item_name, "removed": True},
        )

    def _resolve_profile(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> tuple[str | None, ProviderAccount | None, ToolResult | None]:
        if not self.provider.available:
            return None, None, _unavailable(context)
        spoken = arguments.get("profile_name")
        if spoken is not None and not isinstance(spoken, str):
            return (
                None,
                None,
                _result(context, ToolStatus.INVALID_REQUEST, "Profilen var ugyldig."),
            )
        resolver = AliasResolver(
            aliases=self.profile_aliases, default_reference=self.default_profile
        )
        result = resolver.resolve(spoken=t.cast(str | None, spoken))
        failure = _resolution_failure(
            context=context,
            result=result,
            labels={
                reference: alias for alias, reference in self.profile_aliases.items()
            },
        )
        if failure is not None:
            return None, None, failure
        assert result.reference is not None
        account = self.provider.accounts.get(result.reference)
        if account is None:
            return (
                None,
                None,
                _result(
                    context, ToolStatus.UNAUTHENTICATED, "Profilen er ikke tilsluttet."
                ),
            )
        return result.reference, account, None

    def _resolve_list(
        self,
        *,
        context: ToolContext,
        account: ProviderAccount,
        profile: str,
        spoken: object,
    ) -> tuple[str | None, str | None, ToolResult | None]:
        if spoken is not None and not isinstance(spoken, str):
            return (
                None,
                None,
                _result(context, ToolStatus.INVALID_REQUEST, "Listen var ugyldig."),
            )
        try:
            context.raise_if_cancelled()
            provider_lists = self.provider.list_lists(account=account)
        except Exception as error:
            return None, None, _provider_failure(context=context, error=error)
        aliases = _aliases_for(self.list_aliases, profile)
        labels = {value: key for key, value in aliases.items()}
        entries = list(aliases.items())
        for item in provider_lists:
            if item.id not in labels:
                entries.append((item.name, item.id))
                labels[item.id] = item.name
        default = self.default_lists.get(profile) if spoken is None else None
        result = AliasResolver(aliases=entries, default_reference=default).resolve(
            spoken=t.cast(str | None, spoken)
        )
        failure = _resolution_failure(context=context, result=result, labels=labels)
        if failure is not None:
            return None, None, failure
        assert result.reference is not None
        if result.reference not in labels:
            return (
                None,
                None,
                _result(context, ToolStatus.NOT_FOUND, "Listen blev ikke fundet."),
            )
        return result.reference, labels[result.reference], None

    def _resolve_item(
        self,
        *,
        context: ToolContext,
        account: ProviderAccount,
        profile: str,
        list_id: str,
        spoken: str,
    ) -> tuple[str | None, str | None, ToolResult | None]:
        try:
            context.raise_if_cancelled()
            items = self.provider.list_items(account=account, list_id=list_id)
        except Exception as error:
            return None, None, _provider_failure(context=context, error=error)
        aliases = _aliases_for(self.item_aliases, list_id)
        if not aliases:
            aliases = _aliases_for(self.item_aliases, profile)
        entries = list(aliases.items())
        labels = {value: key for key, value in entries}
        for item in items:
            if item.id not in labels:
                entries.append((item.name, item.id))
                labels[item.id] = item.name
        result = AliasResolver(aliases=entries).resolve(spoken=spoken)
        failure = _resolution_failure(context=context, result=result, labels=labels)
        if failure is not None:
            return None, None, failure
        assert result.reference is not None
        return result.reference, labels.get(result.reference, spoken), None

    def _confirmed(self, context: ToolContext, resolved: dict[str, object]) -> bool:
        if self.confirmation_checker is not None:
            return self.confirmation_checker(context, resolved)
        state = context.state
        if state.get("confirmed") is True or state.get("confirm_removal") is True:
            return True
        confirmed_operation = state.get("confirmed_operation_id")
        return (
            context.operation_id is not None
            and confirmed_operation == context.operation_id
        )

    def _profile_label(self, profile: str) -> str:
        return next(
            (
                alias
                for alias, reference in self.profile_aliases.items()
                if reference == profile
            ),
            profile,
        )

    def _list_data(self, profile: str, item: ShoppingList) -> dict[str, object]:
        aliases = _aliases_for(self.list_aliases, profile)
        label = next(
            (alias for alias, reference in aliases.items() if reference == item.id),
            item.name,
        )
        data: dict[str, object] = {"name": label}
        if item.item_count is not None:
            data["item_count"] = item.item_count
        return data

    def _item_data(
        self, item: ShoppingItem, label: str | None = None
    ) -> dict[str, object]:
        data: dict[str, object] = {"name": label or item.name, "checked": item.checked}
        if item.amount is not None:
            data["amount"] = item.amount
        if item.unit is not None:
            data["unit"] = item.unit
        return data


def build_shopping_tool_specs(
    provider: ListonicProvider,
    *,
    profile_aliases: AliasMap | None = None,
    list_aliases: NestedAliasMap | AliasMap | None = None,
    item_aliases: NestedAliasMap | AliasMap | None = None,
    default_profile: str | None = None,
    default_lists: c.Mapping[str, str] | None = None,
    confirmation_checker: c.Callable[[ToolContext, dict[str, object]], bool]
    | None = None,
) -> tuple[ToolSpec, ...]:
    """Create Listonic tool specs for final registry assembly."""
    return ShoppingTools(
        provider=provider,
        profile_aliases=profile_aliases or {},
        list_aliases=list_aliases,
        item_aliases=item_aliases,
        default_profile=default_profile,
        default_lists=default_lists,
        confirmation_checker=confirmation_checker,
    ).specs()


shopping_tool_specs = build_shopping_tool_specs


_LIST_LISTS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"profile_name": {"type": ["string", "null"]}},
    "required": ["profile_name"],
}
_GET_LIST_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "profile_name": {"type": ["string", "null"]},
        "list_name": {"type": ["string", "null"]},
        "include_checked": {"type": "boolean"},
    },
    "required": ["profile_name", "list_name", "include_checked"],
}
_ADD_ITEMS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "profile_name": {"type": ["string", "null"]},
        "list_name": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "amount": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                },
                "required": ["name", "amount", "unit"],
            },
        },
    },
    "required": ["profile_name", "list_name", "items"],
}
_SET_CHECKED_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "profile_name": {"type": ["string", "null"]},
        "list_name": {"type": ["string", "null"]},
        "item_name": {"type": "string"},
        "checked": {"type": "boolean"},
    },
    "required": ["profile_name", "list_name", "item_name", "checked"],
}
_REMOVE_ITEM_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "profile_name": {"type": ["string", "null"]},
        "list_name": {"type": ["string", "null"]},
        "item_name": {"type": "string"},
    },
    "required": ["profile_name", "list_name", "item_name"],
}


def _aliases_for(value: NestedAliasMap | AliasMap, key: str) -> dict[str, str]:
    if not value:
        return {}
    nested = value.get(key)
    if isinstance(nested, c.Mapping):
        return {str(alias): str(reference) for alias, reference in nested.items()}
    if all(isinstance(item, str) for item in value.values()):
        return {str(alias): str(reference) for alias, reference in value.items()}
    return {}


def _resolution_failure(
    *, context: ToolContext, result: ResolutionResult, labels: c.Mapping[str, str] = {}
) -> ToolResult | None:
    if result.status is ResolutionStatus.MATCHED:
        return None
    if result.status is ResolutionStatus.NEEDS_CLARIFICATION:
        return _result(
            context,
            ToolStatus.NEEDS_CLARIFICATION,
            "Der er flere mulige valg.",
            candidates=[labels.get(value, value) for value in result.candidates],
        )
    return _result(context, ToolStatus.NOT_FOUND, "Jeg fandt ikke det ønskede.")


def _provider_failure(*, context: ToolContext, error: Exception) -> ToolResult:
    if isinstance(error, CancelledError):
        return _result(context, ToolStatus.CANCELLED, "Handlingen blev afbrudt.")
    if isinstance(error, ListonicAuthenticationError):
        return _result(
            context, ToolStatus.UNAUTHENTICATED, "Listonic skal tilsluttes igen."
        )
    if isinstance(error, ListonicForbiddenError):
        return _result(context, ToolStatus.FORBIDDEN, "Listonic afviste handlingen.")
    if isinstance(error, ListonicConflictError):
        return _result(
            context, ToolStatus.CONFLICT, "Listonic-handlingen stødte sammen."
        )
    if isinstance(error, ListonicInvalidRequestError):
        return _result(
            context, ToolStatus.INVALID_REQUEST, "Forespørgslen var ugyldig."
        )
    if isinstance(error, ListonicNotFoundError):
        return _result(context, ToolStatus.NOT_FOUND, "Jeg fandt ikke det ønskede.")
    if isinstance(error, ListonicRateLimitError):
        return _result(
            context,
            ToolStatus.RATE_LIMITED,
            "Listonic har bedt om en pause.",
            retryable=True,
        )
    if isinstance(error, ListonicContractDriftError):
        return _result(
            context, ToolStatus.UNAVAILABLE, "Listonic er midlertidigt utilgængelig."
        )
    if isinstance(error, ListonicError):
        return _result(
            context,
            ToolStatus.UNAVAILABLE,
            "Listonic er midlertidigt utilgængelig.",
            retryable=True,
        )
    return _result(
        context,
        ToolStatus.UNAVAILABLE,
        "Listonic er midlertidigt utilgængelig.",
        retryable=True,
    )


def _valid_removal_target(value: object) -> t.TypeGuard[dict[str, str]]:
    required = {"profile_name", "list_id", "list_name", "item_id", "item_name"}
    return (
        isinstance(value, dict)
        and set(value) == required
        and all(isinstance(item, str) and item for item in value.values())
    )


def _unavailable(context: ToolContext) -> ToolResult:
    return _result(context, ToolStatus.UNAVAILABLE, "Listonic er ikke aktiveret.")


def _ok(
    context: ToolContext, *, data: dict[str, object] | None = None, message: str = ""
) -> ToolResult:
    return ToolResult(
        status=ToolStatus.OK,
        operation_id=context.operation_id,
        message_da=message,
        data=data,
    )


def _result(
    context: ToolContext,
    status: ToolStatus,
    message: str,
    *,
    data: dict[str, object] | None = None,
    candidates: list[object] | None = None,
    retryable: bool = False,
) -> ToolResult:
    return ToolResult(
        status=status,
        operation_id=context.operation_id,
        message_da=message,
        data=data,
        candidates=candidates or [],
        retryable=retryable,
    )


def _as_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _as_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("amount must be a number or null")
    return float(value)


__all__ = ["ShoppingTools", "build_shopping_tool_specs", "shopping_tool_specs"]
