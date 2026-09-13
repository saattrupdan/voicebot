"""Bounded, no-shell transport for the local Google Workspace CLI."""

from __future__ import annotations

import collections.abc as c
import json
import os
import pathlib
import selectors
import shutil
import subprocess
import time

from ..tool_runtime import ToolContext

GWS_EXECUTABLE = "gws"
GWS_TIMEOUT_SECONDS = 15.0
GWS_MAX_OUTPUT_BYTES = 256 * 1024
_READ_SIZE = 64 * 1024
_POLL_SECONDS = 0.05


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


class GwsOutcomeUnknown(GwsTransportError):
    """A launched mutation may have succeeded and must not be retried."""


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

    def invoke_read(
        self,
        arguments: list[str],
        *,
        allowed_commands: c.Collection[tuple[str, ...]],
        context: ToolContext | None = None,
    ) -> dict[str, object]:
        """Invoke one cancellable allow-listed read command."""
        return self._invoke(
            arguments,
            allowed_commands=allowed_commands,
            context=context,
            mutation=False,
        )

    def invoke_mutation(
        self,
        arguments: list[str],
        *,
        allowed_commands: c.Collection[tuple[str, ...]],
        context: ToolContext | None = None,
    ) -> dict[str, object]:
        """Invoke a mutation once, reporting uncertainty after it is launched."""
        return self._invoke(
            arguments, allowed_commands=allowed_commands, context=context, mutation=True
        )

    def _invoke(
        self,
        arguments: list[str],
        *,
        allowed_commands: c.Collection[tuple[str, ...]],
        context: ToolContext | None,
        mutation: bool,
    ) -> dict[str, object]:
        command = tuple(arguments[:4])
        if command not in allowed_commands:
            raise GwsInvalidResponse("Google Workspace command is not allowed")
        if context is not None:
            context.check_cancelled()
        if self.executable is None:
            raise GwsUnavailable("Google Workspace CLI is unavailable")
        try:
            process = subprocess.Popen(
                [self.executable, *arguments],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except OSError, subprocess.SubprocessError:
            raise GwsUnavailable("Google Workspace CLI is unavailable") from None

        try:
            stdout, stderr, returncode = self._collect(
                process=process, context=context, cancellable=not mutation
            )
        except _OutputOverflow:
            if mutation:
                raise GwsOutcomeUnknown(
                    "Google Workspace mutation outcome is unknown"
                ) from None
            raise GwsInvalidResponse("Google Workspace response is too large") from None
        except _ProcessTimeout:
            if mutation:
                raise GwsOutcomeUnknown(
                    "Google Workspace mutation outcome is unknown"
                ) from None
            raise GwsUnavailable("Google Workspace CLI timed out") from None
        except OSError, subprocess.SubprocessError:
            _terminate(process)
            if mutation:
                raise GwsOutcomeUnknown(
                    "Google Workspace mutation outcome is unknown"
                ) from None
            raise GwsUnavailable("Google Workspace CLI is unavailable") from None

        if returncode != 0:
            failure = cli_failure(stderr)
            if mutation and isinstance(failure, GwsUnavailable):
                raise GwsOutcomeUnknown(
                    "Google Workspace mutation outcome is unknown"
                ) from None
            raise failure
        try:
            payload = json.loads(stdout)
        except UnicodeDecodeError, json.JSONDecodeError:
            if mutation:
                raise GwsOutcomeUnknown(
                    "Google Workspace mutation outcome is unknown"
                ) from None
            raise GwsInvalidResponse("invalid Google Workspace CLI response") from None
        if not isinstance(payload, dict):
            if mutation:
                raise GwsOutcomeUnknown("Google Workspace mutation outcome is unknown")
            raise GwsInvalidResponse("invalid Google Workspace CLI response")
        return payload

    def _collect(
        self,
        *,
        process: subprocess.Popen[bytes],
        context: ToolContext | None,
        cancellable: bool,
    ) -> tuple[bytes, bytes, int]:
        stdout = bytearray()
        stderr = bytearray()
        streams = ((process.stdout, stdout), (process.stderr, stderr))
        selector = selectors.DefaultSelector()
        try:
            for stream, target in streams:
                if stream is None:
                    raise OSError("missing process pipe")
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, target)
            deadline = time.monotonic() + self.timeout
            while selector.get_map():
                if cancellable and context is not None and context.cancelled:
                    _terminate(process)
                    context.check_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate(process)
                    raise _ProcessTimeout
                for key, _ in selector.select(timeout=min(remaining, _POLL_SECONDS)):
                    target = key.data
                    assert isinstance(target, bytearray)
                    allowance = self.max_output_bytes - len(target)
                    chunk = os.read(key.fd, min(_READ_SIZE, allowance + 1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if len(chunk) > allowance:
                        _terminate(process)
                        stdout.clear()
                        stderr.clear()
                        raise _OutputOverflow
                    target.extend(chunk)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate(process)
                raise _ProcessTimeout
            returncode = process.wait(timeout=remaining)
            return bytes(stdout), bytes(stderr), returncode
        except subprocess.TimeoutExpired:
            _terminate(process)
            raise _ProcessTimeout from None
        finally:
            selector.close()
            for stream, _ in streams:
                if stream is not None:
                    stream.close()


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Terminate a child and reap it without retaining any more output."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    except OSError:
        try:
            process.kill()
            process.wait()
        except OSError:
            pass


class _OutputOverflow(Exception):
    """Internal signal that a process pipe exceeded its cap."""


class _ProcessTimeout(Exception):
    """Internal signal that a launched process exceeded its deadline."""


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
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


__all__ = [
    "GWS_EXECUTABLE",
    "GWS_MAX_OUTPUT_BYTES",
    "GWS_TIMEOUT_SECONDS",
    "GwsForbidden",
    "GwsInvalidResponse",
    "GwsOutcomeUnknown",
    "GwsRateLimited",
    "GwsTransport",
    "GwsTransportError",
    "GwsUnauthenticated",
    "GwsUnavailable",
    "json_argument",
]
