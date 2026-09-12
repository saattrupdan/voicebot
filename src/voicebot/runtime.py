"""Application assembly for the durable local integrations."""

from __future__ import annotations

import collections.abc as c
import datetime as dt
import os
import pathlib
import threading
import typing as t
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from .auth import CredentialStore, MemoryCredentialBackend, ProviderAccount
from .notifications import NotificationCallback, NotificationDispatcher
from .providers.google_calendar import GoogleCalendarProvider
from .providers.listonic import ListonicProvider
from .providers.spotify import SpotifyProvider
from .scheduler import ReminderScheduler
from .storage import Storage
from .tool_runtime import ToolContext, ToolResult, ToolRuntime, ToolSpec, ToolStatus
from .tools import build_registry
from .tools.calendar import (
    CalendarBinding,
    build_calendar_tools,
    make_calendar_tool_specs,
)
from .tools.reminders import (
    cancel_reminder_spec,
    create_reminder_spec,
    list_reminders_spec,
)
from .tools.shopping import ShoppingTools
from .tools.spotify import build_spotify_tools
from .tools.timer import list_timers, set_timer, stop_timer


@dataclass
class IntegrationRuntime:
    """All process-owned services used by one voicebot instance."""

    storage: Storage
    credentials: CredentialStore
    registry: ToolRuntime
    scheduler: ReminderScheduler
    dispatcher: NotificationDispatcher
    state: dict[str, object]
    providers: tuple[object, ...]
    _stop_event: threading.Event
    _worker: threading.Thread | None = None

    def start(self) -> None:
        """Start the scheduler and durable notification worker."""
        if self._worker is not None and self._worker.is_alive():
            return
        self.scheduler.start()
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._run_worker, name="voicebot-integrations", daemon=True
        )
        self._worker.start()

    def stop(self) -> None:
        """Stop workers, release leases, and close provider resources."""
        for notification_id in self.dispatcher.active_ids:
            self.dispatcher.cancel(notification_id)
        self._stop_event.set()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=2)
        self._worker = None
        self.scheduler.stop()
        timers = self.state.get("running_timers", [])
        if isinstance(timers, list):
            for timer in list(timers):
                stop = getattr(timer, "stop", None)
                if callable(stop):
                    stop()
            timers.clear()
        for provider in self.providers:
            close = getattr(provider, "close", None)
            if callable(close):
                close()
        self.storage.close()

    def _run_worker(self) -> None:
        poll = _number(self.state.get("scheduler_poll_seconds"), 1.0)
        while not self._stop_event.is_set():
            self.scheduler.tick()
            self.dispatcher.deliver_once()
            self._stop_event.wait(timeout=max(poll, 0.05))


def build_integration_runtime(
    cfg: DictConfig,
    *,
    notification_callback: NotificationCallback | None = None,
    timer_callback: c.Callable[[object], None] | None = None,
) -> IntegrationRuntime:
    """Build storage, providers, strict tool handlers, and worker services.

    All values read from configuration are routing metadata or feature flags. Provider
    hosts, OAuth scopes, and credentials remain constants or environment/keyring data.
    """
    storage = _build_storage(cfg)
    credentials = _build_credentials(cfg)
    _, profile_aliases, default_profile = _ensure_profiles(cfg, storage)
    accounts = _load_accounts(storage, credentials)
    integrations = _mapping(_value(cfg, "integrations", {}))

    google_enabled = _enabled(integrations, "google_calendar")
    spotify_enabled = _enabled(integrations, "spotify")
    listonic_config = _mapping(integrations.get("listonic", {}))
    listonic_enabled = bool(listonic_config.get("enabled", False))
    listonic_allowed = bool(listonic_config.get("allow_unofficial", False))
    google_config = _mapping(integrations.get("google_calendar", {}))
    spotify_config = _mapping(integrations.get("spotify", {}))
    list_aliases = _merge_aliases(
        t.cast(dict[str, object], _list_aliases(storage)),
        _mapping(listonic_config.get("list_aliases", {})),
        _mapping(_value(cfg, "shopping_list_aliases", {})),
    )

    google = GoogleCalendarProvider(
        credential_store=credentials,
        client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "disabled-client"),
    )
    spotify = SpotifyProvider(
        credential_store=credentials,
        accounts={
            profile: account for profile, account in accounts.get("spotify", {}).items()
        },
        profile_aliases=profile_aliases,
        default_profile=default_profile,
        device_aliases=t.cast(
            dict[str, str],
            _merge_aliases(
                t.cast(
                    dict[str, object],
                    _device_aliases(storage, accounts.get("spotify", {})),
                ),
                _mapping(spotify_config.get("device_aliases", {})),
                _mapping(_value(cfg, "spotify_device_aliases", {})),
            ),
        ),
    )
    listonic = ListonicProvider(
        credentials,
        accounts=accounts.get("listonic", {}),
        enabled=listonic_enabled,
        allow_unofficial=listonic_allowed,
    )

    state: dict[str, object] = {
        "storage": storage,
        "profile_id": _default_profile_id(storage, default_profile),
        "clock": lambda: dt.datetime.now(dt.UTC),
        "scheduler_poll_seconds": _number(
            _mapping(_value(cfg, "scheduler", {})).get("poll_seconds"), 1.0
        ),
    }
    if timer_callback is not None:
        state["notification_callback"] = timer_callback

    specs: list[ToolSpec] = [
        ToolSpec(
            name="set_timer",
            description="Create a uniquely named local timer.",
            parameters=_timer_set_schema(),
            handler=lambda context, arguments: t.cast(
                ToolResult, set_timer(context, arguments)
            ),
        ),
        ToolSpec(
            name="list_timers",
            description="List active timers, optionally matching one name.",
            parameters=_timer_list_schema(),
            handler=lambda context, arguments: t.cast(
                ToolResult, list_timers(context, arguments)
            ),
        ),
        ToolSpec(
            name="stop_timer",
            description="Stop exactly one named timer.",
            parameters=_timer_stop_schema(),
            handler=lambda context, arguments: t.cast(
                ToolResult, stop_timer(context, arguments)
            ),
        ),
        create_reminder_spec(storage),
        list_reminder_spec(storage),
        cancel_reminder_spec(storage),
    ]
    calendar = build_calendar_tools(
        google,
        credentials,
        profile_aliases=profile_aliases,
        calendar_bindings=_calendar_bindings(
            storage,
            accounts.get("google_calendar", {}),
            aliases=_mapping(google_config.get("calendar_aliases", {})),
        ),
        profile_accounts={
            profile: account
            for profile, account in accounts.get("google_calendar", {}).items()
        },
        default_profile=default_profile,
    )
    specs.extend(
        _gate_specs(
            make_calendar_tool_specs(calendar),
            google_enabled,
            "Kalenderen er ikke aktiveret.",
        )
    )
    specs.extend(
        _gate_specs(
            build_spotify_tools(spotify), spotify_enabled, "Spotify er ikke aktiveret."
        )
    )
    shopping = ShoppingTools(
        listonic,
        profile_aliases=profile_aliases,
        list_aliases=t.cast(dict[str, str] | dict[str, dict[str, str]], list_aliases),
        default_profile=default_profile,
        default_lists=_default_lists(storage),
        confirmation_checker=None,
    )
    specs.extend(
        _gate_specs(
            shopping.specs(),
            listonic_enabled and listonic_allowed,
            "Listonic er ikke aktiveret.",
        )
    )

    scheduler_config = _mapping(_value(cfg, "scheduler", {}))
    scheduler = ReminderScheduler(
        storage,
        lease_for=dt.timedelta(
            seconds=_number(scheduler_config.get("lease_seconds"), 30)
        ),
    )
    dispatcher = NotificationDispatcher(
        storage,
        callback=notification_callback or _discard_notification,
        lease_for=dt.timedelta(minutes=5),
    )
    registry = build_registry(
        tools=_tools(cfg),
        integration_specs=specs,
        include_integrations=True,
        include_legacy=False,
    )
    runtime = ToolRuntime(registry=registry)
    result = IntegrationRuntime(
        storage=storage,
        credentials=credentials,
        registry=runtime,
        scheduler=scheduler,
        dispatcher=dispatcher,
        state=state,
        providers=(google, spotify, listonic),
        _stop_event=threading.Event(),
    )
    _install_confirmation(result, shopping)
    return result


def _install_confirmation(runtime: IntegrationRuntime, shopping: ShoppingTools) -> None:
    """Install local yes/no handling for destructive shopping operations."""
    original_checker = shopping.confirmation_checker

    def checker(context: ToolContext, resolved: dict[str, object]) -> bool:
        if context.state.get("confirmed") is True:
            return True
        arguments = context.state.get("_confirmation_arguments")
        profile_id = context.state.get("profile_id")
        if not isinstance(profile_id, str) or not isinstance(arguments, dict):
            return bool(original_checker and original_checker(context, resolved))
        pending = runtime.storage.confirmations.create(
            profile_id=profile_id,
            action="remove_shopping_item",
            payload={"arguments": arguments, "summary": resolved},
            expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(minutes=2),
            operation_id=context.operation_id,
        )
        context.state["pending_confirmation_id"] = pending.id
        return False

    shopping.confirmation_checker = checker
    remove_handler = shopping.remove_shopping_item

    def resolve_confirmation(accepted: bool) -> ToolResult:
        profile_id = runtime.state.get("profile_id")
        pending_id = runtime.state.get("pending_confirmation_id")
        if not isinstance(profile_id, str) or not isinstance(pending_id, str):
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                message_da="Der er ingen afventende bekræftelse.",
            )
        pending = runtime.storage.confirmations.get(pending_id)
        if pending is None or pending.profile_id != profile_id:
            return ToolResult(
                status=ToolStatus.NOT_FOUND, message_da="Bekræftelsen er udløbet."
            )
        resolved = runtime.storage.confirmations.resolve(pending.id, accepted=accepted)
        runtime.state.pop("pending_confirmation_id", None)
        if resolved is None:
            return ToolResult(
                status=ToolStatus.CONFLICT, message_da="Bekræftelsen er udløbet."
            )
        if not accepted:
            return ToolResult(
                status=ToolStatus.OK, message_da="Handlingen er annulleret."
            )
        arguments = pending.payload.get("arguments")
        if not isinstance(arguments, dict):
            return ToolResult(
                status=ToolStatus.INVALID_REQUEST,
                message_da="Bekræftelsen var ugyldig.",
            )
        runtime.state["confirmed"] = True
        try:
            return remove_handler(
                ToolContext(state=runtime.state, operation_id=pending.operation_id),
                arguments,
            )
        finally:
            runtime.state.pop("confirmed", None)

    runtime.state["confirmation_handler"] = resolve_confirmation


def _build_storage(cfg: DictConfig) -> Storage:
    value = _mapping(_value(cfg, "storage", {})).get("database_path", ":memory:")
    return Storage(pathlib.Path(str(value)))


def _build_credentials(cfg: DictConfig) -> CredentialStore:
    value = _mapping(_value(cfg, "storage", {})).get("credential_backend", "keyring")
    if str(value).casefold() in {"memory", "memory_only", "in-memory"}:
        return CredentialStore(backend=MemoryCredentialBackend())
    return CredentialStore()


def _ensure_profiles(
    cfg: DictConfig, storage: Storage
) -> tuple[list[str], dict[str, str], str | None]:
    configured = _mapping(_value(cfg, "profiles", {}))
    default = str(_mapping(_value(cfg, "device", {})).get("default_profile", "dan"))
    if not configured:
        configured = {default: {"aliases": [default]}}
    profile_names: list[str] = []
    aliases: dict[str, str] = {}
    for name, value in configured.items():
        if not isinstance(name, str):
            continue
        profile_names.append(name)
        profile = next(
            (item for item in storage.profiles.list() if item.name == name), None
        )
        if profile is None:
            profile = storage.profiles.create(name)
        profile_aliases = _mapping(value).get("aliases", [])
        if isinstance(profile_aliases, str):
            profile_aliases = [profile_aliases]
        if isinstance(profile_aliases, list):
            for alias in profile_aliases:
                if isinstance(alias, str):
                    aliases[alias] = name
                    if not storage.profiles.find_aliases(alias):
                        storage.profiles.add_alias(profile.id, alias)
    top_aliases = _mapping(_value(cfg, "profile_aliases", {}))
    aliases.update(
        {str(alias): str(reference) for alias, reference in top_aliases.items()}
    )
    return profile_names, aliases, default if default in profile_names else None


def _load_accounts(
    storage: Storage, credentials: CredentialStore
) -> dict[str, dict[str, ProviderAccount]]:
    result: dict[str, dict[str, ProviderAccount]] = {}
    profiles = {profile.id: profile.name for profile in storage.profiles.list()}
    for record in storage.providers.list_accounts():
        profile = profiles.get(record.profile_id)
        if profile is None:
            continue
        account = credentials.connect(
            record.provider,
            profile,
            credential_ref=record.credential_ref,
            scopes=record.scopes,
        )
        result.setdefault(record.provider, {})[profile] = account
    return result


def _calendar_bindings(
    storage: Storage,
    accounts: dict[str, ProviderAccount],
    *,
    aliases: dict[str, object] | None = None,
) -> list[CalendarBinding]:
    profile_names = {profile.id: profile.name for profile in storage.profiles.list()}
    values: list[CalendarBinding] = []
    for binding in storage.providers.find_bindings("calendar"):
        profile = profile_names.get(binding.profile_id)
        if profile is not None:
            account = accounts.get(profile)
            values.append(
                CalendarBinding(
                    profile,
                    binding.alias,
                    binding.provider_id,
                    account.credential_ref if account else None,
                )
            )
    for profile, values_for_profile in (aliases or {}).items():
        if isinstance(values_for_profile, c.Mapping):
            for alias, calendar_id in values_for_profile.items():
                if isinstance(alias, str) and isinstance(calendar_id, str):
                    account = accounts.get(profile)
                    values.append(
                        CalendarBinding(
                            profile,
                            alias,
                            calendar_id,
                            account.credential_ref if account else None,
                        )
                    )
    return values


def _device_aliases(
    storage: Storage, accounts: dict[str, ProviderAccount]
) -> dict[str, str]:
    del accounts
    values: dict[str, str] = {}
    for binding in storage.providers.find_bindings("spotify_device"):
        values[binding.alias] = binding.provider_id
    return values


def _list_aliases(storage: Storage) -> dict[str, dict[str, str]]:
    profiles = {profile.id: profile.name for profile in storage.profiles.list()}
    result: dict[str, dict[str, str]] = {}
    for binding in storage.providers.find_bindings("shopping_list"):
        profile = profiles.get(binding.profile_id)
        if profile is not None:
            result.setdefault(profile, {})[binding.alias] = binding.provider_id
    return result


def _default_lists(storage: Storage) -> dict[str, str]:
    return {
        profile.name: binding.provider_id
        for profile in storage.profiles.list()
        for binding in storage.providers.find_bindings(
            "shopping_list", profile_id=profile.id
        )[:1]
    }


def _merge_aliases(*values: c.Mapping[str, object]) -> dict[str, object]:
    """Merge setup-owned alias maps, with later values taking precedence."""
    merged: dict[str, object] = {}
    for value in values:
        merged.update(value)
    return merged


def _default_profile_id(storage: Storage, name: str | None) -> str | None:
    profile = next(
        (item for item in storage.profiles.list() if item.name == name), None
    )
    return profile.id if profile else None


def _tools(cfg: DictConfig) -> list[dict[str, object]]:
    value = OmegaConf.to_container(cfg.get("tools", []), resolve=True)
    return value if isinstance(value, list) else []


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, c.Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _value(cfg: DictConfig, key: str, default: object) -> object:
    value = cfg.get(key, default)
    return default if value is None else value


def _enabled(values: dict[str, object], name: str) -> bool:
    return bool(_mapping(values.get(name, {})).get("enabled", False))


def _number(value: object, default: float) -> float:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else default
    )


def _gate_specs(
    specs: c.Iterable[ToolSpec], enabled: bool, message: str
) -> list[ToolSpec]:
    if enabled:
        return list(specs)
    result: list[ToolSpec] = []
    for spec in specs:

        def handler(
            context: ToolContext, arguments: dict[str, object], wrapped: ToolSpec = spec
        ) -> ToolResult:
            del arguments
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                operation_id=context.operation_id,
                message_da=message,
            )

        result.append(ToolSpec(spec.name, spec.description, spec.parameters, handler))
    return result


def _discard_notification(notification: object, event: threading.Event) -> object:
    del notification, event
    return False


def _timer_set_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "duration_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
        },
        "required": ["name", "duration_seconds"],
    }


def _timer_list_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"name": {"type": ["string", "null"]}},
        "required": ["name"],
    }


def _timer_stop_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }


def list_reminder_spec(storage: Storage) -> ToolSpec:
    """Return the reminder-list specification."""
    return list_reminders_spec(storage)


__all__ = ["IntegrationRuntime", "build_integration_runtime"]
