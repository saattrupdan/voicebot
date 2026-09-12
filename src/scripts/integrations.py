"""Safe local onboarding entry point for provider integrations.

The commands route OAuth and isolated-browser onboarding through provider handlers.
Providers without local OAuth client configuration remain harmlessly unavailable rather
than prompting for or accepting credentials on the command line.
"""

from __future__ import annotations

import argparse
import collections.abc as c
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf

from voicebot.auth import (
    CredentialError,
    CredentialStore,
    ProviderAccount,
    ProviderAuthHandler,
    redact_text,
)
from voicebot.auth.google import GoogleCalendarAuthHandler
from voicebot.auth.listonic import (
    IsolatedBrowserFactory,
    IsolatedBrowserSession,
    ListonicAuthHandler,
)
from voicebot.auth.spotify import SpotifyAuthHandler
from voicebot.storage import Storage

DEFAULT_PROVIDERS = ("google-calendar", "spotify", "listonic")


class BrowserHelperUnavailable(CredentialError):
    """Raised when disposable Listonic browser support is not installed."""


class _IsolatedBrowserSession:
    """Small, secret-free adapter around an operator-installed browser helper."""

    def __init__(self, helper: str) -> None:
        self.helper = helper
        self.profile = pathlib.Path(tempfile.mkdtemp(prefix="voicebot-listonic-"))

    def open_login(self, url: str) -> None:
        subprocess.run(
            [self.helper, "--profile", str(self.profile), "open", url],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # The prompt is only a synchronization point; the password stays in the
        # browser and is never read by this process.
        try:
            input("Complete Listonic login in the isolated browser, then press Enter: ")
        except EOFError as error:
            raise BrowserHelperUnavailable(
                "Listonic login needs an interactive terminal"
            ) from error

    def export_tokens(self) -> object:
        completed = subprocess.run(
            [
                self.helper,
                "--profile",
                str(self.profile),
                "eval",
                "JSON.stringify({localStorage:Object.fromEntries(Object.entries(localStorage)),sessionStorage:Object.fromEntries(Object.entries(sessionStorage))})",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        exported: object = json.loads(completed.stdout)
        try:
            cookies = subprocess.run(
                [
                    self.helper,
                    "--profile",
                    str(self.profile),
                    "cookies",
                    "get",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            cookie_state: object = json.loads(cookies.stdout)
        except OSError, subprocess.SubprocessError, ValueError:
            cookie_state = {}
        if isinstance(exported, dict) and isinstance(cookie_state, dict):
            return {**exported, "cookies": cookie_state}
        return exported

    def destroy(self) -> None:
        try:
            subprocess.run(
                [self.helper, "--profile", str(self.profile), "close"],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            shutil.rmtree(self.profile, ignore_errors=True)


def _isolated_browser_factory() -> IsolatedBrowserSession:
    configured = os.environ.get("LISTONIC_BROWSER_HELPER")
    helper = shutil.which(configured) if configured else shutil.which("agent-browser")
    if configured and helper is None and pathlib.Path(configured).is_file():
        helper = configured
    if not helper:
        raise BrowserHelperUnavailable(
            "isolated Listonic browser helper is unavailable; install agent-browser "
            "or set LISTONIC_BROWSER_HELPER"
        )
    return _IsolatedBrowserSession(helper)


class ProviderNotConfiguredError(RuntimeError):
    """Raised when no provider-specific onboarding handler is registered."""


@dataclasses.dataclass(frozen=True, slots=True)
class IntegrationStatus:
    """Non-sensitive status displayed by the setup command."""

    provider: str
    profile: str | None
    state: str


class NotConfiguredProvider:
    """Placeholder handler used until a provider module registers an implementation."""

    def __init__(self, provider: str) -> None:
        """Create a placeholder for one provider name.

        Args:
            provider: Canonical provider name used for status output.
        """
        self.provider = provider

    def login(self, profile: str) -> ProviderAccount | None:
        """Refuse onboarding without a provider implementation."""
        raise ProviderNotConfiguredError(self.provider)

    def status(self, profile: str | None = None) -> IntegrationStatus:
        """Report that this provider has no onboarding implementation yet."""
        return IntegrationStatus(
            provider=self.provider, profile=profile, state="not configured"
        )

    def disconnect(self, profile: str) -> None:
        """Refuse disconnect when there is no provider account registry."""
        raise ProviderNotConfiguredError(self.provider)


class _PersistedAuthHandler:
    """Persist provider metadata while leaving secrets in CredentialStore."""

    def __init__(
        self,
        provider: str,
        handler: ProviderAuthHandler,
        *,
        storage: Storage,
        credentials: CredentialStore,
    ) -> None:
        self.provider = provider
        self.handler = handler
        self.storage = storage
        self.credentials = credentials
        self._rehydrate()

    def login(self, profile: str) -> ProviderAccount | None:
        account = self.handler.login(profile)
        if account is None:
            return None
        self._ensure_profile(profile)
        self._persist(account)
        return account

    def status(self, profile: str | None = None) -> object:
        records = self.storage.providers.list_accounts()
        relevant = [record for record in records if record.provider == self.provider]
        profiles = {item.id: item.name for item in self.storage.profiles.list()}
        if profile is not None:
            record = next(
                (item for item in relevant if profiles.get(item.profile_id) == profile),
                None,
            )
            state = "disconnected"
            if record is not None and record.status == "connected":
                state = "connected"
            return IntegrationStatus(_normalise_provider(self.provider), profile, state)
        return [
            IntegrationStatus(
                _normalise_provider(self.provider),
                profiles.get(record.profile_id),
                "connected" if record.status == "connected" else "disconnected",
            )
            for record in relevant
            if profiles.get(record.profile_id) is not None
        ]

    def disconnect(self, profile: str) -> None:
        profiles = {item.id: item.name for item in self.storage.profiles.list()}
        record = next(
            (
                item
                for item in self.storage.providers.list_accounts()
                if item.provider == self.provider
                and profiles.get(item.profile_id) == profile
            ),
            None,
        )
        if record is None:
            return
        account = self.credentials.connect(
            self.provider,
            profile,
            credential_ref=record.credential_ref,
            scopes=record.scopes,
        )
        if record.status != "connected":
            self.credentials.mark_disconnected(account.credential_ref)
        try:
            disconnect = getattr(self.handler, "disconnect", None)
            if callable(disconnect) and profile in getattr(
                self.handler, "_accounts", {}
            ):
                disconnect(profile)
            else:
                revoke = getattr(self.handler, "revoke_token", None)
                if revoke is None:
                    revoke = getattr(self.handler, "revoke", None)
                self.credentials.disconnect(account, revoke=revoke)
        finally:
            self.storage.providers.set_account_status(record.id, "disconnected")

    def _ensure_profile(self, name: str) -> str:
        profile = next(
            (item for item in self.storage.profiles.list() if item.name == name), None
        )
        if profile is None:
            profile = self.storage.profiles.create(name)
        if not self.storage.profiles.find_aliases(name):
            self.storage.profiles.add_alias(profile.id, name)
        return profile.id

    def _persist(self, account: ProviderAccount) -> None:
        profile_id = self._ensure_profile(account.profile)
        self.storage.providers.upsert_account(
            profile_id,
            account.provider,
            account.credential_ref,
            scopes=account.scopes,
            status=account.status.value,
        )

    def _rehydrate(self) -> None:
        profiles = {item.id: item.name for item in self.storage.profiles.list()}
        for record in self.storage.providers.list_accounts():
            if record.provider != self.provider or record.profile_id not in profiles:
                continue
            account = self.credentials.connect(
                record.provider,
                profiles[record.profile_id],
                credential_ref=record.credential_ref,
                scopes=record.scopes,
            )
            if record.status != "connected":
                self.credentials.mark_disconnected(record.credential_ref)
            accounts = getattr(self.handler, "_accounts", None)
            if isinstance(accounts, dict):
                accounts[account.profile] = account


def _persisted_handler(
    provider: str,
    handler: ProviderAuthHandler,
    *,
    storage: Storage | None,
    credentials: CredentialStore,
) -> ProviderAuthHandler:
    if storage is None:
        return handler
    return _PersistedAuthHandler(
        provider, handler, storage=storage, credentials=credentials
    )


class ProviderRegistry:
    """Route setup operations to registered provider handlers."""

    def __init__(self, providers: c.Iterable[str] = DEFAULT_PROVIDERS) -> None:
        """Create a registry populated with safe not-configured handlers.

        Args:
            providers (optional): Provider names to make routable. Defaults to the
                providers described by the integration plan.
        """
        self._handlers: dict[str, ProviderAuthHandler] = {
            name: NotConfiguredProvider(name)
            for name in (_normalise_provider(provider) for provider in providers)
        }

    def register(self, provider: str, handler: ProviderAuthHandler) -> None:
        """Register or replace a provider onboarding handler."""
        name = _normalise_provider(provider)
        if not name:
            raise ValueError("provider must not be empty")
        self._handlers[name] = handler

    def handler(self, provider: str) -> ProviderAuthHandler:
        """Return a handler, using a harmless placeholder for unknown providers."""
        name = _normalise_provider(provider)
        return self._handlers.get(name, NotConfiguredProvider(name))

    def providers(self) -> tuple[str, ...]:
        """Return registered provider names in stable order."""
        return tuple(self._handlers)


def register_provider(provider: str, handler: ProviderAuthHandler) -> None:
    """Register a provider handler for the command-line entry point."""
    _default_registry.register(provider, handler)


def register_integration_handlers(
    registry: ProviderRegistry | None = None,
    *,
    credential_store: CredentialStore | None = None,
    storage: Storage | None = None,
    google_client_id: str | None = None,
    google_client_secret: str | None = None,
    spotify_client_id: str | None = None,
    listonic_browser_factory: IsolatedBrowserFactory | None = None,
) -> ProviderRegistry:
    """Register the three local onboarding implementations.

    Client IDs are read from the environment by default. Missing IDs leave OAuth
    providers safely unconfigured rather than prompting for a secret on stdin.
    """
    active = registry or _default_registry
    store = credential_store or CredentialStore()
    google_id = google_client_id or os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    if google_id:
        active.register(
            "google-calendar",
            _persisted_handler(
                "google_calendar",
                GoogleCalendarAuthHandler(
                    credential_store=store,
                    client_id=google_id,
                    client_secret=google_client_secret
                    or os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"),
                ),
                storage=storage,
                credentials=store,
            ),
        )
    spotify_id = spotify_client_id or os.environ.get("SPOTIFY_CLIENT_ID")
    if spotify_id:
        active.register(
            "spotify",
            _persisted_handler(
                "spotify",
                SpotifyAuthHandler(client_id=spotify_id, credential_store=store),
                storage=storage,
                credentials=store,
            ),
        )
    active.register(
        "listonic",
        _persisted_handler(
            "listonic",
            ListonicAuthHandler(
                credential_store=store,
                browser_factory=listonic_browser_factory or _isolated_browser_factory,
            ),
            storage=storage,
            credentials=store,
        ),
    )
    return active


def main(
    argv: list[str] | None = None,
    *,
    registry: ProviderRegistry | None = None,
    output: c.Callable[[str], None] | None = None,
) -> int:
    """Run the integrations setup command.

    Args:
        argv (optional): Arguments excluding the executable name. Defaults to None.
        registry (optional): Handler registry, useful for embedding and tests.
            Defaults to the process-wide registry.
        output (optional): Line sink for user-facing, non-secret output. Defaults to
            standard output.

    Returns:
        Zero for a successful operation, or two when a provider is not configured.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    active_registry = registry or _default_registry
    if registry is None:
        storage = Storage(_database_path())
        register_integration_handlers(
            active_registry, credential_store=CredentialStore(), storage=storage
        )
    write_line = output or _write_line
    provider = getattr(args, "provider", None)

    if args.command == "status":
        return _status(
            active_registry,
            provider=provider,
            profile=args.profile,
            write_line=write_line,
        )

    assert provider is not None
    handler = active_registry.handler(provider)
    try:
        if args.command == "login":
            handler.login(args.profile)
            write_line(f"{_normalise_provider(provider)}: login complete")
        else:
            handler.disconnect(args.profile)
            write_line(f"{_normalise_provider(provider)}: disconnected")
    except ProviderNotConfiguredError:
        write_line(f"{_normalise_provider(provider)}: not configured")
        return 2
    except BrowserHelperUnavailable as error:
        write_line(f"{_normalise_provider(provider)}: {redact_text(str(error))}")
        return 2
    except Exception:
        # Provider exceptions may echo request bodies or credentials.  The CLI exposes
        # a stable generic error rather than forwarding provider text.
        write_line(f"{_normalise_provider(provider)}: operation failed")
        return 1
    return 0


def _missing_browser_factory() -> IsolatedBrowserSession:
    """Compatibility alias for callers that explicitly request fail-closed setup."""
    raise BrowserHelperUnavailable(
        "isolated Listonic browser helper is unavailable; install agent-browser "
        "or set LISTONIC_BROWSER_HELPER"
    )


def _status(
    registry: ProviderRegistry,
    *,
    provider: str | None,
    profile: str | None,
    write_line: c.Callable[[str], None],
) -> int:
    names = (_normalise_provider(provider),) if provider else registry.providers()
    result = 0
    for name in names:
        handler = registry.handler(name)
        try:
            status = handler.status(profile)
        except ProviderNotConfiguredError:
            write_line(f"{name}: not configured")
            result = 2
            continue
        except Exception:
            write_line(f"{name}: operation failed")
            result = 1
            continue
        write_line(_format_status(name, status, profile))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicebot integrations")
    commands = parser.add_subparsers(dest="command", required=True)

    login = commands.add_parser("login", help="open provider onboarding")
    login.add_argument("provider")
    login.add_argument("--profile", required=True)

    status = commands.add_parser("status", help="show connection status")
    status.add_argument("provider", nargs="?")
    status.add_argument("--profile")

    disconnect = commands.add_parser("disconnect", help="remove a provider connection")
    disconnect.add_argument("provider")
    disconnect.add_argument("--profile", required=True)
    return parser


def _format_status(provider: str, status: object, requested_profile: str | None) -> str:
    if isinstance(status, list):
        lines = [
            _format_status(provider, item, requested_profile)
            for item in status
            if isinstance(item, IntegrationStatus)
        ]
        return "\n".join(lines) if lines else f"{provider}: disconnected"
    if isinstance(status, dict):
        lines = [
            _format_status(provider, item, requested_profile)
            for item in status.values()
        ]
        return "\n".join(lines) if lines else f"{provider}: disconnected"
    if isinstance(status, IntegrationStatus):
        profile = status.profile or requested_profile
        route = f"/{redact_text(profile)}" if profile else ""
        return f"{status.provider}{route}: {redact_text(status.state)}"
    if isinstance(status, str):
        route = f"/{redact_text(requested_profile)}" if requested_profile else ""
        return f"{provider}{route}: {redact_text(status)}"
    return f"{provider}: status available"


def _normalise_provider(provider: str) -> str:
    return provider.strip().casefold().replace("_", "-")


def _write_line(line: str) -> None:
    sys.stdout.write(f"{line}\n")


def _database_path() -> pathlib.Path:
    """Return the same database location used by the bot configuration."""
    configured = os.environ.get("VOICEBOT_DATABASE_PATH") or os.environ.get(
        "VOICEBOT_DB_PATH"
    )
    if configured:
        return pathlib.Path(configured)
    config_override = os.environ.get("VOICEBOT_CONFIG")
    config_path = (
        pathlib.Path(config_override)
        if config_override
        else (pathlib.Path(__file__).resolve().parents[2] / "config" / "config.yaml")
    )
    try:
        config = OmegaConf.load(config_path)
        configured_path = str(config.storage.database_path)
    except OSError, AttributeError, TypeError:
        configured_path = ".local/state/voicebot.sqlite"
    path = pathlib.Path(configured_path)
    return path if path.is_absolute() else config_path.parent.parent / path


_default_registry = ProviderRegistry()


if __name__ == "__main__":
    raise SystemExit(main())
