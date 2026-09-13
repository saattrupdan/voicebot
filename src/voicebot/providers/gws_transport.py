"""Bounded, no-shell transport for the local Google Workspace CLI."""

from __future__ import annotations

import collections.abc as c
import json
import os
import pathlib
import shutil
import subprocess

from ..tool_runtime import ToolContext

GWS_EXECUTABLE = "gws"
GWS_TIMEOUT_SECONDS = 15.0
GWS_MAX_OUTPUT_BYTES = 256 * 1024


class GwsTransportError(Exception):
    """Base class for safe local CLI failures."""


class GwsUnauthenticated(GwsTransportError):
    """The local CLI has no usable authenticated session."""


class GwsForbidden(GwsTransportError):
    """The authenticated account cannot access the requested resource."""


class GwsRateLimited(GwsTransportError):
    """The service asked the caller to slow down."""


class GwsUnavailable(GwsTransportError):
    """The local CLI or service is temporarily unavailable."""


class GwsInvalidResponse(GwsTransportError):
    """The local CLI returned an unusable response."""


class GwsTransport:
    """Run a fixed Google Workspace CLI executable with bounded I/O."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        timeout: float = GWS_TIMEOUT_SECONDS,
        max_output_bytes: int = GWS_MAX_OUTPUT_BYTES,
    ) -> None:
        """Resolve the executable once and enforce process limits."""
        if timeout <= 0 or max_output_bytes <= 0:
            raise ValueError("transport limits must be positive")
        self.executable = resolve_executable(executable)
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes

    def invoke(
        self,
        arguments: list[str],
        *,
        allowed_commands: c.Collection[tuple[str, ...]],
        context: ToolContext | None = None,
    ) -> dict[str, object]:
        """Invoke one allow-listed command and decode a bounded JSON object."""
        command = tuple(arguments[:4])
        if command not in allowed_commands:
            raise GwsInvalidResponse("Google Workspace command is not allowed")
        if context is not None:
            context.check_cancelled()
        if self.executable is None:
            raise GwsUnavailable("Google Workspace CLI is unavailable")
        try:
            completed = subprocess.run(
                [self.executable, *arguments],
                capture_output=True,
                check=False,
                shell=False,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise GwsUnavailable("Google Workspace CLI timed out") from None
        except OSError, subprocess.SubprocessError:
            raise GwsUnavailable("Google Workspace CLI is unavailable") from None
        if context is not None:
            context.check_cancelled()
        stdout = _as_bytes(completed.stdout)
        stderr = _as_bytes(completed.stderr)
        if len(stdout) > self.max_output_bytes or len(stderr) > self.max_output_bytes:
            raise GwsInvalidResponse("Google Workspace response is too large")
        if completed.returncode != 0:
            raise cli_failure(stderr)
        try:
            payload = json.loads(stdout)
        except UnicodeDecodeError, json.JSONDecodeError:
            raise GwsInvalidResponse("invalid Google Workspace CLI response") from None
        if not isinstance(payload, dict):
            raise GwsInvalidResponse("invalid Google Workspace CLI response")
        return payload


def resolve_executable(executable: str | None) -> str | None:
    """Resolve an absolute executable path without accepting PATH input."""
    if executable is not None:
        path = pathlib.Path(executable)
        if not path.is_absolute():
            raise ValueError("gws executable must be an absolute path")
        return str(path)
    found = shutil.which(GWS_EXECUTABLE)
    return os.path.realpath(found) if found is not None else None


def cli_failure(stderr: bytes) -> GwsTransportError:
    """Map untrusted diagnostics to a safe status without returning diagnostics."""
    text = stderr[:4096].decode("utf-8", errors="ignore").casefold()
    if any(
        marker in text
        for marker in (
            "401",
            "unauthorised",
            "unauthorized",
            "authentication",
            "not authenticated",
            "auth required",
            "no credentials",
            "not logged in",
            "login",
        )
    ):
        return GwsUnauthenticated("Google account is not authorised")
    if any(marker in text for marker in ("403", "forbidden", "permission denied")):
        return GwsForbidden("Google Workspace access was denied")
    if "429" in text or "rate limit" in text:
        return GwsRateLimited("Google Workspace rate limit reached")
    return GwsUnavailable("Google Workspace CLI request failed")


def json_argument(value: object) -> str:
    """Encode CLI JSON arguments without whitespace or unstable ordering."""
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _as_bytes(value: object) -> bytes:
    return value if isinstance(value, bytes) else str(value).encode()


__all__ = [
    "GWS_MAX_OUTPUT_BYTES",
    "GWS_TIMEOUT_SECONDS",
    "GwsForbidden",
    "GwsInvalidResponse",
    "GwsRateLimited",
    "GwsTransport",
    "GwsTransportError",
    "GwsUnauthenticated",
    "GwsUnavailable",
    "json_argument",
]
