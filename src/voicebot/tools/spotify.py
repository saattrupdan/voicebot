"""Model-facing Spotify tool specifications and explicit handlers."""

from __future__ import annotations

from ..providers.spotify import SpotifyProvider
from ..tool_runtime import ToolContext, ToolResult, ToolSpec

_NULL_STRING = {"type": ["string", "null"]}


class SpotifyToolFactory:
    """Create the closed set of Spotify tools for one provider instance."""

    def __init__(self, provider: SpotifyProvider) -> None:
        """Bind handlers to a configured Spotify provider."""
        self.provider = provider

    def specs(self) -> list[ToolSpec]:
        """Return all Spotify model-visible tool specifications."""
        return [
            ToolSpec(
                name="spotify_play",
                description="Find and play one clearly identified Spotify item.",
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "profile_name": _NULL_STRING,
                        "query": {"type": "string", "minLength": 1},
                        "media_type": {
                            "type": ["string", "null"],
                            "enum": [
                                "track",
                                "album",
                                "artist",
                                "playlist",
                                "show",
                                "episode",
                                None,
                            ],
                        },
                        "device_name": _NULL_STRING,
                    },
                    "required": ["profile_name", "query", "media_type", "device_name"],
                },
                handler=self.play,
                mutates=True,
            ),
            ToolSpec(
                name="spotify_control",
                description=(
                    "Pause, resume, skip, or replay the current Spotify playback."
                ),
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "profile_name": _NULL_STRING,
                        "action": {
                            "type": "string",
                            "enum": ["pause", "resume", "next", "previous"],
                        },
                        "device_name": _NULL_STRING,
                    },
                    "required": ["profile_name", "action", "device_name"],
                },
                handler=self.control,
                mutates=True,
            ),
            ToolSpec(
                name="spotify_set_volume",
                description="Set Spotify volume from 0 through 100 percent.",
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "profile_name": _NULL_STRING,
                        "volume_percent": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "device_name": _NULL_STRING,
                    },
                    "required": ["profile_name", "volume_percent", "device_name"],
                },
                handler=self.set_volume,
                mutates=True,
            ),
            ToolSpec(
                name="spotify_now_playing",
                description="Return the current Spotify playback and device.",
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"profile_name": _NULL_STRING},
                    "required": ["profile_name"],
                },
                handler=self.now_playing,
            ),
            ToolSpec(
                name="spotify_list_devices",
                description="List the available Spotify playback devices.",
                parameters={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"profile_name": _NULL_STRING},
                    "required": ["profile_name"],
                },
                handler=self.list_devices,
            ),
        ]

    def play(self, context: ToolContext, arguments: dict[str, object]) -> ToolResult:
        """Handle ``spotify_play``."""
        return self.provider.play(
            context=context,
            profile_name=_optional_string(arguments.get("profile_name")),
            query=_required_string(arguments.get("query")),
            media_type=_optional_string(arguments.get("media_type")),
            device_name=_optional_string(arguments.get("device_name")),
        )

    def control(self, context: ToolContext, arguments: dict[str, object]) -> ToolResult:
        """Handle ``spotify_control``."""
        return self.provider.control(
            context=context,
            profile_name=_optional_string(arguments.get("profile_name")),
            action=_required_string(arguments.get("action")),
            device_name=_optional_string(arguments.get("device_name")),
        )

    def set_volume(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """Handle ``spotify_set_volume``."""
        volume = arguments.get("volume_percent")
        return self.provider.set_volume(
            context=context,
            profile_name=_optional_string(arguments.get("profile_name")),
            volume_percent=volume
            if isinstance(volume, int) and not isinstance(volume, bool)
            else -1,
            device_name=_optional_string(arguments.get("device_name")),
        )

    def now_playing(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """Handle ``spotify_now_playing``."""
        return self.provider.now_playing(
            context=context,
            profile_name=_optional_string(arguments.get("profile_name")),
        )

    def list_devices(
        self, context: ToolContext, arguments: dict[str, object]
    ) -> ToolResult:
        """Handle ``spotify_list_devices``."""
        return self.provider.list_devices(
            context=context,
            profile_name=_optional_string(arguments.get("profile_name")),
        )


def create_spotify_tool_specs(provider: SpotifyProvider) -> list[ToolSpec]:
    """Return Spotify tool specs bound to ``provider``."""
    return SpotifyToolFactory(provider=provider).specs()


def spotify_tool_specs(provider: SpotifyProvider) -> list[ToolSpec]:
    """Alias for :func:`create_spotify_tool_specs`."""
    return create_spotify_tool_specs(provider=provider)


def build_spotify_tools(provider: SpotifyProvider) -> list[ToolSpec]:
    """Alias used by integration assembly code."""
    return create_spotify_tool_specs(provider=provider)


build_spotify_tool_specs = create_spotify_tool_specs
get_spotify_tool_specs = create_spotify_tool_specs


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _required_string(value: object) -> str:
    return value if isinstance(value, str) else ""


__all__ = [
    "SpotifyToolFactory",
    "build_spotify_tools",
    "create_spotify_tool_specs",
    "get_spotify_tool_specs",
    "spotify_tool_specs",
    "build_spotify_tool_specs",
]
