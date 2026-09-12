"""Raw HTTP Spotify Connect adapter with deterministic local resolution."""

from __future__ import annotations

import collections.abc as c
import logging
import math
import re
import typing as t
from dataclasses import dataclass

import httpx

from ..auth.credentials import (
    CredentialDisconnectedError,
    CredentialError,
    CredentialReference,
    CredentialStore,
    PermanentRefreshError,
    ProviderAccount,
    TokenSet,
    redact_text,
)
from ..auth.spotify import SPOTIFY_SCOPES, SpotifyAuthHandler
from ..resolution import AliasResolver, ResolutionStatus, normalise_alias
from ..tool_runtime import CancelledError, ToolContext, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

_MEDIA_TYPES = ("track", "album", "artist", "playlist", "show", "episode")
_MEDIA_KEYS = {name: f"{name}s" for name in _MEDIA_TYPES}
_MAX_CANDIDATES = 5


@dataclass(frozen=True, slots=True)
class SpotifyDevice:
    """Safe, provider-independent representation of a Connect device."""

    device_id: str
    name: str
    is_active: bool = False
    device_type: str | None = None
    volume_percent: int | None = None


@dataclass(frozen=True, slots=True)
class SpotifyCandidate:
    """A safe search candidate; provider URIs never cross this boundary."""

    name: str
    media_type: str
    artist: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return candidate data suitable for the model."""
        result: dict[str, object] = {"name": self.name, "media_type": self.media_type}
        if self.artist:
            result["artist"] = self.artist
        return result


class SpotifyProvider:
    """Implement safe Spotify playback operations.

    Device IDs and Spotify URIs are accepted only from the immediately preceding
    provider response. Model-facing arguments contain names and aliases, never IDs.
    """

    def __init__(
        self,
        credential_store: CredentialStore | None = None,
        *,
        credentials: CredentialStore | None = None,
        store: CredentialStore | None = None,
        accounts: c.Mapping[str, ProviderAccount | CredentialReference] | None = None,
        profile_aliases: c.Mapping[str, str] | c.Iterable[tuple[str, str]] = (),
        default_profile: str | None = None,
        device_aliases: (
            c.Mapping[str, str]
            | c.Mapping[str, c.Mapping[str, str]]
            | c.Iterable[tuple[str, str]]
        ) = (),
        default_device: str | None = None,
        http_client: httpx.Client | None = None,
        auth_handler: SpotifyAuthHandler | None = None,
        refresh_callback: c.Callable[[str], TokenSet] | None = None,
        base_url: str = "https://api.spotify.com/v1",
        timeout: float = 15.0,
    ) -> None:
        """Initialise the adapter and its local routing metadata."""
        if (
            sum(value is not None for value in (credential_store, credentials, store))
            > 1
        ):
            raise ValueError("provide only one credential store")
        self.credential_store = (
            credential_store or credentials or store or CredentialStore()
        )
        self.http_client = http_client or httpx.Client(timeout=timeout)
        self.auth_handler = auth_handler
        self.refresh_callback = refresh_callback
        self.base_url = base_url.rstrip("/")
        self._accounts: dict[str, ProviderAccount | CredentialReference] = dict(
            accounts or {}
        )
        self._profile_aliases = t.cast(
            tuple[tuple[str, str], ...],
            tuple(
                profile_aliases.items()
                if isinstance(profile_aliases, c.Mapping)
                else t.cast(c.Iterable[tuple[str, str]], profile_aliases)
            ),
        )
        self.default_profile = default_profile
        self._device_aliases: dict[str, tuple[tuple[str, str], ...]] = {}
        if isinstance(device_aliases, c.Mapping) and all(
            isinstance(value, c.Mapping) for value in device_aliases.values()
        ):
            self._device_aliases = {
                str(profile): tuple(
                    (str(alias), str(target)) for alias, target in aliases.items()
                )
                for profile, aliases in device_aliases.items()
                if isinstance(aliases, c.Mapping)
            }
        else:
            profile = default_profile
            if profile is None and len(self._accounts) == 1:
                profile = next(iter(self._accounts))
            if profile is not None:
                values = (
                    device_aliases.items()
                    if isinstance(device_aliases, c.Mapping)
                    else t.cast(c.Iterable[tuple[str, str]], device_aliases)
                )
                self._device_aliases[profile] = tuple(
                    (str(alias), str(target)) for alias, target in values
                )
        self.default_device = default_device
        self._known_secrets: set[str] = set()

    def register_account(self, account: ProviderAccount) -> None:
        """Make a connected account available to this provider."""
        if account.provider != "spotify":
            raise ValueError("account is not a Spotify account")
        self._accounts[account.profile] = account

    def play(
        self,
        context: ToolContext,
        query: str,
        media_type: str | None = None,
        device_name: str | None = None,
        profile_name: str | None = None,
    ) -> ToolResult:
        """Search internally and start an unambiguous Spotify item."""
        try:
            context.raise_if_cancelled()
            if not query.strip() or query.strip().casefold().startswith("spotify:"):
                return self._result(
                    context,
                    ToolStatus.INVALID_REQUEST,
                    "Søgeteksten skal være et navn, ikke en Spotify URI.",
                )
            account = self._account_result(profile_name=profile_name)
            if isinstance(account, ToolResult):
                return self._with_operation(account, context)
            selected_type = media_type if media_type in _MEDIA_TYPES else None
            if media_type is not None and selected_type is None:
                return self._result(
                    context, ToolStatus.INVALID_REQUEST, "Medietypen er ikke gyldig."
                )
            response, error = self._request(
                context=context,
                account=account,
                method="GET",
                path="/search",
                params={
                    "q": query,
                    "type": selected_type or "track,album,artist,playlist,show,episode",
                    "limit": "10",
                },
            )
            if error is not None:
                return self._with_operation(error, context)
            assert response is not None
            payload = self._json_object(response)
            match = self._select_search_result(
                payload=payload, query=query, media_type=selected_type
            )
            if match is None:
                candidates = self._search_candidates(
                    payload=payload, media_type=selected_type
                )
                return self._with_operation(
                    ToolResult(
                        status=ToolStatus.NEEDS_CLARIFICATION
                        if candidates
                        else ToolStatus.NOT_FOUND,
                        message_da=(
                            "Jeg fandt flere mulige resultater."
                            if candidates
                            else "Jeg fandt ikke noget passende."
                        ),
                        candidates=self._candidate_values(
                            item.as_dict() for item in candidates
                        ),
                    ),
                    context,
                )
            device, device_error = self._resolve_device(
                context=context, account=account, spoken=device_name
            )
            if device_error is not None:
                return self._with_operation(device_error, context)
            assert device is not None
            context.raise_if_cancelled()
            uri = match.get("uri")
            if not isinstance(uri, str) or not self._safe_uri(uri):
                return self._result(
                    context,
                    ToolStatus.UNAVAILABLE,
                    "Spotify kunne ikke starte resultatet.",
                )
            item_type = self._string(match.get("type")) or selected_type or "track"
            body: dict[str, object]
            if item_type in {"track", "episode"}:
                body = {"uris": [uri]}
            else:
                body = {"context_uri": uri}
            _, mutation_error = self._request(
                context=context,
                account=account,
                method="PUT",
                path="/me/player/play",
                params={"device_id": device.device_id},
                json_body=body,
                mutation=True,
            )
            if mutation_error is not None:
                return self._with_operation(mutation_error, context)
            return self._with_operation(
                ToolResult(
                    status=ToolStatus.OK,
                    message_da="Afspilningen er startet.",
                    data={
                        "name": self._safe_text(match.get("name")),
                        "media_type": item_type,
                        "device": self._device_label(device, account.profile),
                    },
                ),
                context,
            )
        except CancelledError:
            return self._result(
                context, ToolStatus.CANCELLED, "Handlingen blev afbrudt."
            )

    def control(
        self,
        context: ToolContext,
        action: str,
        device_name: str | None = None,
        profile_name: str | None = None,
    ) -> ToolResult:
        """Pause, resume, skip, or go back on a resolved device."""
        if action not in {"pause", "resume", "next", "previous"}:
            return self._result(
                context, ToolStatus.INVALID_REQUEST, "Handlingen er ikke gyldig."
            )
        try:
            context.raise_if_cancelled()
            account = self._account_result(profile_name=profile_name)
            if isinstance(account, ToolResult):
                return self._with_operation(account, context)
            device, device_error = self._resolve_device(
                context=context, account=account, spoken=device_name
            )
            if device_error is not None:
                return self._with_operation(device_error, context)
            assert device is not None
            path = {
                "pause": "/me/player/pause",
                "resume": "/me/player/play",
                "next": "/me/player/next",
                "previous": "/me/player/previous",
            }[action]
            method = "POST" if action in {"next", "previous"} else "PUT"
            params = {"device_id": device.device_id}
            _, error = self._request(
                context=context,
                account=account,
                method=method,
                path=path,
                params=params,
                mutation=True,
            )
            if error is not None:
                return self._with_operation(error, context)
            return self._with_operation(
                ToolResult(
                    status=ToolStatus.OK,
                    message_da="Afspilningen er opdateret.",
                    data={
                        "action": action,
                        "device": self._device_label(device, account.profile),
                    },
                ),
                context,
            )
        except CancelledError:
            return self._result(
                context, ToolStatus.CANCELLED, "Handlingen blev afbrudt."
            )

    def pause(
        self,
        context: ToolContext,
        device_name: str | None = None,
        profile_name: str | None = None,
    ) -> ToolResult:
        """Pause playback on a resolved device."""
        return self.control(
            context=context,
            action="pause",
            device_name=device_name,
            profile_name=profile_name,
        )

    def resume(
        self,
        context: ToolContext,
        device_name: str | None = None,
        profile_name: str | None = None,
    ) -> ToolResult:
        """Resume playback on a resolved device."""
        return self.control(
            context=context,
            action="resume",
            device_name=device_name,
            profile_name=profile_name,
        )

    def next(
        self,
        context: ToolContext,
        device_name: str | None = None,
        profile_name: str | None = None,
    ) -> ToolResult:
        """Skip to the next item on a resolved device."""
        return self.control(
            context=context,
            action="next",
            device_name=device_name,
            profile_name=profile_name,
        )

    def previous(
        self,
        context: ToolContext,
        device_name: str | None = None,
        profile_name: str | None = None,
    ) -> ToolResult:
        """Return to the previous item on a resolved device."""
        return self.control(
            context=context,
            action="previous",
            device_name=device_name,
            profile_name=profile_name,
        )

    def set_volume(
        self,
        context: ToolContext,
        volume_percent: int,
        device_name: str | None = None,
        profile_name: str | None = None,
    ) -> ToolResult:
        """Set volume from zero through one hundred percent."""
        if isinstance(volume_percent, bool) or not 0 <= volume_percent <= 100:
            return self._result(
                context,
                ToolStatus.INVALID_REQUEST,
                "Lydstyrken skal være 0 til 100 procent.",
            )
        try:
            context.raise_if_cancelled()
            account = self._account_result(profile_name=profile_name)
            if isinstance(account, ToolResult):
                return self._with_operation(account, context)
            device, device_error = self._resolve_device(
                context=context, account=account, spoken=device_name
            )
            if device_error is not None:
                return self._with_operation(device_error, context)
            assert device is not None
            if volume_percent > 80 and not context.confirmation_resume:
                manager = context.state.get("confirmation_manager")
                request = getattr(manager, "request", None)
                resolved = {
                    "volume_percent": volume_percent,
                    "device_name": self._device_label(device, account.profile),
                    "action": "spotify_set_volume",
                }
                if callable(request) and not request(
                    context,
                    action="spotify_set_volume",
                    arguments={
                        "profile_name": profile_name,
                        "volume_percent": volume_percent,
                        "device_name": self._device_label(device, account.profile),
                    },
                    resolved=resolved,
                ):
                    pass
                return self._with_operation(
                    ToolResult(
                        status=ToolStatus.CONFIRMATION_REQUIRED,
                        message_da="Bekræft lydstyrke over 80 procent.",
                        data=t.cast(dict[str, object], resolved),
                    ),
                    context,
                )
            _, error = self._request(
                context=context,
                account=account,
                method="PUT",
                path="/me/player/volume",
                params={
                    "device_id": device.device_id,
                    "volume_percent": str(volume_percent),
                },
                mutation=True,
            )
            if error is not None:
                return self._with_operation(error, context)
            return self._with_operation(
                ToolResult(
                    status=ToolStatus.OK,
                    message_da="Lydstyrken er ændret.",
                    data={
                        "volume_percent": volume_percent,
                        "device": self._device_label(device, account.profile),
                    },
                ),
                context,
            )
        except CancelledError:
            return self._result(
                context, ToolStatus.CANCELLED, "Handlingen blev afbrudt."
            )

    def now_playing(
        self, context: ToolContext, profile_name: str | None = None
    ) -> ToolResult:
        """Return a filtered description of the current playback."""
        try:
            context.raise_if_cancelled()
            account = self._account_result(profile_name=profile_name)
            if isinstance(account, ToolResult):
                return self._with_operation(account, context)
            response, error = self._request(
                context=context, account=account, method="GET", path="/me/player"
            )
            if error is not None:
                return self._with_operation(error, context)
            assert response is not None
            payload = self._json_object(response)
            item = payload.get("item")
            device_payload = payload.get("device")
            data: dict[str, object] = {
                "is_playing": payload.get("is_playing") is True,
                "progress_ms": payload.get("progress_ms")
                if isinstance(payload.get("progress_ms"), int)
                else None,
            }
            if isinstance(item, dict):
                data["item"] = self._safe_item(item)
            if isinstance(device_payload, dict):
                device = self._device_from_payload(device_payload)
                if device is not None:
                    data["device"] = self._device_label(device, account.profile)
            return self._with_operation(
                ToolResult(status=ToolStatus.OK, data=data), context
            )
        except CancelledError:
            return self._result(
                context, ToolStatus.CANCELLED, "Handlingen blev afbrudt."
            )

    def list_devices(
        self, context: ToolContext, profile_name: str | None = None
    ) -> ToolResult:
        """List currently available devices without exposing provider IDs."""
        try:
            context.raise_if_cancelled()
            account = self._account_result(profile_name=profile_name)
            if isinstance(account, ToolResult):
                return self._with_operation(account, context)
            response, error = self._request(
                context=context,
                account=account,
                method="GET",
                path="/me/player/devices",
            )
            if error is not None:
                return self._with_operation(error, context)
            assert response is not None
            payload = self._json_object(response)
            devices = [
                device
                for raw in self._list(payload.get("devices"))
                if isinstance(raw, dict)
                for device in [self._device_from_payload(raw)]
                if device is not None
            ]
            return self._with_operation(
                ToolResult(
                    status=ToolStatus.OK,
                    data={
                        "devices": [
                            {
                                "name": self._safe_text(device.name),
                                "active": device.is_active,
                                "type": self._safe_text(device.device_type),
                            }
                            for device in devices
                        ]
                    },
                ),
                context,
            )
        except CancelledError:
            return self._result(
                context, ToolStatus.CANCELLED, "Handlingen blev afbrudt."
            )

    get_devices = list_devices

    def _account_result(self, profile_name: str | None) -> ProviderAccount | ToolResult:
        profile = self._resolve_profile(profile_name=profile_name)
        if isinstance(profile, ToolResult):
            return profile
        value = self._accounts.get(profile)
        if value is None and self.auth_handler is not None:
            account = next(
                (
                    candidate
                    for candidate in self.auth_handler._accounts.values()
                    if candidate.profile == profile
                ),
                None,
            )
            value = account
        if isinstance(value, ProviderAccount):
            account = value
        elif isinstance(value, str):
            account = self.credential_store.account(value)
        else:
            account = self._find_store_account(profile=profile)
        if account is None or account.provider != "spotify":
            return ToolResult(
                status=ToolStatus.UNAUTHENTICATED,
                message_da="Spotify er ikke forbundet. Kør lokal opsætning først.",
            )
        if set(account.scopes) != set(SPOTIFY_SCOPES):
            return ToolResult(
                status=ToolStatus.UNAUTHENTICATED,
                message_da="Spotify-forbindelsen skal godkendes igen.",
            )
        return account

    def _resolve_profile(self, profile_name: str | None) -> str | ToolResult:
        known = set(self._accounts)
        known.update(
            account.profile
            for account in self._store_accounts()
            if account.provider == "spotify"
        )
        aliases = list(self._profile_aliases)
        aliases.extend((profile, profile) for profile in sorted(known))
        default = self.default_profile
        if default is None and profile_name is None and len(known) > 1:
            return ToolResult(
                status=ToolStatus.NEEDS_CLARIFICATION,
                message_da="Hvilken Spotify-profil mener du?",
                candidates=self._candidate_values(sorted(known)),
            )
        if default is None and len(known) == 1 and profile_name is None:
            default = next(iter(known))
        result = AliasResolver(aliases=aliases, default_reference=default).resolve(
            profile_name
        )
        if result.status is ResolutionStatus.NEEDS_CLARIFICATION:
            return ToolResult(
                status=ToolStatus.NEEDS_CLARIFICATION,
                message_da="Hvilken Spotify-profil mener du?",
                candidates=self._candidate_values(result.candidates),
            )
        if result.status is ResolutionStatus.NOT_FOUND or result.reference is None:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                message_da="Spotify-profilen blev ikke fundet.",
            )
        return result.reference

    def _resolve_device(
        self, context: ToolContext, account: ProviderAccount, spoken: str | None
    ) -> tuple[SpotifyDevice | None, ToolResult | None]:
        response, error = self._request(
            context=context, account=account, method="GET", path="/me/player/devices"
        )
        if error is not None:
            return None, error
        assert response is not None
        payload = self._json_object(response)
        devices = [
            device
            for raw in self._list(payload.get("devices"))
            if isinstance(raw, dict)
            for device in [self._device_from_payload(raw)]
            if device is not None
        ]
        if not devices:
            return None, ToolResult(
                status=ToolStatus.UNAVAILABLE,
                message_da="Ingen Spotify-enheder er aktive.",
                retryable=True,
            )
        requested = spoken if spoken is not None else self.default_device
        if requested is None and self.default_device is None:
            if len(devices) == 1:
                return devices[0], None
            labels = sorted(
                {self._device_label(device, account.profile) for device in devices}
            )
            return None, ToolResult(
                status=ToolStatus.NEEDS_CLARIFICATION,
                message_da="Hvilken Spotify-enhed mener du?",
                candidates=self._candidate_values(labels),
            )
        device_aliases = self._device_aliases.get(account.profile, ())
        configured = AliasResolver(aliases=device_aliases).resolve(requested)
        if configured.status is ResolutionStatus.NEEDS_CLARIFICATION:
            labels = self._labels_for_references(
                devices=devices,
                references=configured.candidates,
                profile=account.profile,
            )
            return None, ToolResult(
                status=ToolStatus.NEEDS_CLARIFICATION,
                message_da="Hvilken Spotify-enhed mener du?",
                candidates=self._candidate_values(labels),
            )
        aliases = list(device_aliases)
        aliases.extend((device.name, device.device_id) for device in devices)
        result = (
            configured
            if configured.status is ResolutionStatus.MATCHED
            else AliasResolver(aliases=aliases).resolve(requested)
        )
        if result.status is ResolutionStatus.NEEDS_CLARIFICATION:
            labels = self._labels_for_references(
                devices=devices, references=result.candidates, profile=account.profile
            )
            return None, ToolResult(
                status=ToolStatus.NEEDS_CLARIFICATION,
                message_da="Hvilken Spotify-enhed mener du?",
                candidates=self._candidate_values(labels),
            )
        if result.status is ResolutionStatus.NOT_FOUND or result.reference is None:
            # An explicitly spoken device must be a configured/current exact alias.
            return None, ToolResult(
                status=ToolStatus.NOT_FOUND,
                message_da="Spotify-enheden blev ikke fundet.",
            )
        matches = [
            device
            for device in devices
            if device.device_id == result.reference
            or normalise_alias(device.name) == normalise_alias(result.reference)
        ]
        if len(matches) != 1:
            candidates = sorted(
                {self._device_label(device, account.profile) for device in matches}
            )
            return None, ToolResult(
                status=ToolStatus.NEEDS_CLARIFICATION
                if candidates
                else ToolStatus.NOT_FOUND,
                message_da="Spotify-enheden er ikke entydig.",
                candidates=self._candidate_values(candidates),
            )
        return matches[0], None

    def _request(
        self,
        context: ToolContext,
        account: ProviderAccount,
        method: str,
        path: str,
        *,
        params: c.Mapping[str, str] | None = None,
        json_body: dict[str, object] | None = None,
        mutation: bool = False,
    ) -> tuple[httpx.Response | None, ToolResult | None]:
        """Make one safe request, refreshing once after an expired access token."""
        try:
            context.raise_if_cancelled()
            refresh_credential = self.credential_store.load_refresh_token(
                account.credential_ref
            )
            if refresh_credential is not None and len(refresh_credential) >= 8:
                self._known_secrets.add(refresh_credential)
            token = self.credential_store.get_access_token(
                account.credential_ref, self._refresh
            )
            if len(token) >= 8:
                self._known_secrets.add(token)
            response = self._send(
                context=context,
                token=token,
                method=method,
                path=path,
                params=params,
                json_body=json_body,
                mutation=mutation,
            )
            if response.status_code == 401:
                self.credential_store.clear_access_token(account.credential_ref)
                token = self.credential_store.get_access_token(
                    account.credential_ref, self._refresh
                )
                if len(token) >= 8:
                    self._known_secrets.add(token)
                response = self._send(
                    context=context,
                    token=token,
                    method=method,
                    path=path,
                    params=params,
                    json_body=json_body,
                    mutation=mutation,
                )
        except CancelledError:
            raise
        except CredentialDisconnectedError, CredentialError, PermanentRefreshError:
            return None, ToolResult(
                status=ToolStatus.UNAUTHENTICATED,
                message_da="Spotify-forbindelsen skal godkendes igen.",
            )
        except httpx.TimeoutException, httpx.NetworkError:
            return None, ToolResult(
                status=ToolStatus.UNAVAILABLE,
                message_da="Spotify er midlertidigt utilgængelig.",
                retryable=True,
            )
        except Exception:
            logger.warning("Spotify request failed method=%s path=%s", method, path)
            return None, ToolResult(
                status=ToolStatus.UNAVAILABLE,
                message_da="Spotify er midlertidigt utilgængelig.",
                retryable=True,
            )
        return self._classify_response(response=response)

    def _send(
        self,
        context: ToolContext,
        token: str,
        method: str,
        path: str,
        params: c.Mapping[str, str] | None,
        json_body: dict[str, object] | None,
        mutation: bool,
    ) -> httpx.Response:
        context.raise_if_cancelled()
        response = self.http_client.request(
            method,
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            json=json_body,
        )
        if not mutation:
            context.raise_if_cancelled()
        return response

    def _refresh(self, refresh_token: str) -> TokenSet:
        if self.refresh_callback is not None:
            return self.refresh_callback(refresh_token)
        if self.auth_handler is not None:
            return self.auth_handler.refresh(refresh_token)
        raise CredentialError("Spotify authentication is unavailable")

    def _classify_response(
        self, response: httpx.Response
    ) -> tuple[httpx.Response | None, ToolResult | None]:
        status = response.status_code
        if status < 400:
            return response, None
        if status == 401:
            return None, ToolResult(
                status=ToolStatus.UNAUTHENTICATED,
                message_da="Spotify-forbindelsen skal godkendes igen.",
            )
        if status == 403:
            return None, ToolResult(
                status=ToolStatus.FORBIDDEN,
                message_da="Spotify-afspilning kræver en Premium-konto.",
            )
        if status == 404:
            return None, ToolResult(
                status=ToolStatus.UNAVAILABLE,
                message_da="Spotify-enheden er ikke længere aktiv.",
                retryable=True,
            )
        if status == 429:
            retry_after = response.headers.get("Retry-After")
            seconds: int | None = None
            if retry_after is not None:
                try:
                    parsed = float(retry_after)
                    if math.isfinite(parsed) and 0 <= parsed <= 86400:
                        seconds = int(parsed)
                except ValueError:
                    pass
            data: dict[str, object] | None = (
                {"retry_after_seconds": seconds} if seconds is not None else None
            )
            return None, ToolResult(
                status=ToolStatus.RATE_LIMITED,
                message_da="Spotify har bedt os prøve igen senere.",
                data=data,
                retryable=True,
            )
        return None, ToolResult(
            status=ToolStatus.UNAVAILABLE,
            message_da="Spotify er midlertidigt utilgængelig.",
            retryable=status >= 500,
        )

    def _select_search_result(
        self, payload: dict[str, object], query: str, media_type: str | None
    ) -> dict[str, object] | None:
        candidates = self._raw_search_items(payload=payload, media_type=media_type)
        wanted = normalise_alias(query)
        exact = [
            item
            for item in candidates
            if normalise_alias(self._string(item.get("name")) or "") == wanted
        ]
        if len(exact) == 1:
            return exact[0]
        return None

    def _search_candidates(
        self, payload: dict[str, object], media_type: str | None
    ) -> list[SpotifyCandidate]:
        result: list[SpotifyCandidate] = []
        seen: set[tuple[str, str, str | None]] = set()
        for item in self._raw_search_items(payload=payload, media_type=media_type):
            name = self._safe_text(item.get("name"))
            item_type = self._string(item.get("type")) or media_type or "track"
            artist = self._first_artist(item)
            key = (name, item_type, artist)
            if name and key not in seen:
                seen.add(key)
                result.append(
                    SpotifyCandidate(name=name, media_type=item_type, artist=artist)
                )
            if len(result) >= _MAX_CANDIDATES:
                break
        return result

    def _raw_search_items(
        self, payload: dict[str, object], media_type: str | None
    ) -> list[dict[str, object]]:
        if media_type is not None:
            keys = [_MEDIA_KEYS[media_type]]
        else:
            keys = [_MEDIA_KEYS[name] for name in _MEDIA_TYPES]
        items: list[dict[str, object]] = []
        for key in keys:
            container = payload.get(key)
            if isinstance(container, dict):
                items.extend(
                    item
                    for item in self._list(container.get("items"))
                    if isinstance(item, dict)
                )
        return items

    def _safe_item(self, item: dict[str, object]) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self._safe_text(item.get("name")),
            "media_type": self._string(item.get("type")),
        }
        artist = self._first_artist(item)
        if artist:
            result["artist"] = artist
        album = item.get("album")
        if isinstance(album, dict) and isinstance(album.get("name"), str):
            result["album"] = self._safe_text(album["name"])
        return result

    def _find_store_account(self, profile: str) -> ProviderAccount | None:
        return next(
            (
                account
                for account in self._store_accounts()
                if account.profile == profile
            ),
            None,
        )

    def _store_accounts(self) -> list[ProviderAccount]:
        values = getattr(self.credential_store, "_accounts", {})
        return [
            account
            for account in values.values()
            if isinstance(account, ProviderAccount)
        ]

    def _device_from_payload(self, value: dict[str, object]) -> SpotifyDevice | None:
        device_id = value.get("id")
        name = value.get("name")
        if (
            not isinstance(device_id, str)
            or not device_id
            or not isinstance(name, str)
            or not name
        ):
            return None
        volume = value.get("volume_percent")
        return SpotifyDevice(
            device_id=device_id,
            name=self._safe_text(name),
            is_active=value.get("is_active") is True,
            device_type=self._string(value.get("type")),
            volume_percent=volume if isinstance(volume, int) else None,
        )

    def _device_label(self, device: SpotifyDevice, profile: str) -> str:
        for alias, target in self._device_aliases.get(profile, ()):
            if normalise_alias(target) in {
                normalise_alias(device.name),
                normalise_alias(device.device_id),
            }:
                return self._safe_text(alias)
        return self._safe_text(device.name)

    def _labels_for_references(
        self, devices: list[SpotifyDevice], references: c.Iterable[str], profile: str
    ) -> list[str]:
        reference_set = set(references)
        return sorted(
            {
                self._device_label(device, profile)
                for device in devices
                if device.device_id in reference_set
                or device.name in reference_set
                or normalise_alias(device.name)
                in {normalise_alias(reference) for reference in reference_set}
            }
        )

    def _with_operation(self, result: ToolResult, context: ToolContext) -> ToolResult:
        return ToolResult(
            status=result.status,
            operation_id=context.operation_id,
            message_da=self._safe_text(result.message_da),
            data=self._clean_data(result.data),
            candidates=self._clean_candidates(result.candidates),
            retryable=result.retryable,
        )

    def _clean_data(self, value: dict[str, object] | None) -> dict[str, object] | None:
        if value is None:
            return None
        cleaned = self._clean_value(value)
        return cleaned if isinstance(cleaned, dict) else None

    def _clean_candidates(self, value: c.Iterable[object]) -> list[object]:
        cleaned = self._clean_value(list(value))
        return cleaned if isinstance(cleaned, list) else []

    @staticmethod
    def _candidate_values(values: c.Iterable[object]) -> list[object]:
        return list(values)

    def _result(
        self, context: ToolContext, status: ToolStatus, message: str
    ) -> ToolResult:
        return ToolResult(
            status=status,
            operation_id=context.operation_id,
            message_da=self._safe_text(message),
        )

    def _safe_uri(self, value: str) -> bool:
        return value.startswith("spotify:") and "\n" not in value and "\r" not in value

    def _safe_text(self, value: object) -> str:
        text = value if isinstance(value, str) else ""
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text).strip()
        return redact_text(text=text, secrets_to_redact=self._known_secrets)[:200]

    def _clean_value(self, value: object) -> object:
        if isinstance(value, str):
            return self._safe_text(value)
        if isinstance(value, list):
            return [self._clean_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._clean_value(item) for key, item in value.items()}
        return value

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, object]:
        try:
            payload = response.json()
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _list(value: object) -> list[object]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _string(value: object) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _first_artist(item: dict[str, object]) -> str | None:
        artists = item.get("artists")
        if (
            not isinstance(artists, list)
            or not artists
            or not isinstance(artists[0], dict)
        ):
            return None
        value = artists[0].get("name")
        return value if isinstance(value, str) else None


# Names used by early integrations and final assembly code.
SpotifyClient = SpotifyProvider
SpotifyAdapter = SpotifyProvider

__all__ = [
    "SpotifyAdapter",
    "SpotifyCandidate",
    "SpotifyClient",
    "SpotifyDevice",
    "SpotifyProvider",
]
