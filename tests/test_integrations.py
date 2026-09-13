"""Tests for the safe integrations setup command."""

import pytest

from scripts.integrations import IntegrationStatus, ProviderRegistry, main


class FakeProvider:
    """Record safe provider handler calls for routing assertions."""

    def __init__(self) -> None:
        """Initialise empty call records."""
        self.logins: list[str] = []
        self.disconnections: list[str] = []

    def login(self, profile: str) -> None:
        """Record a profile login."""
        self.logins.append(profile)

    def status(self, profile: str | None = None) -> IntegrationStatus:
        """Return non-sensitive connected status."""
        return IntegrationStatus("fake", profile, "connected")

    def disconnect(self, profile: str) -> None:
        """Record a profile disconnect."""
        self.disconnections.append(profile)


def test_unconfigured_provider_is_harmless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave provider-specific onboarding disabled until a handler is registered."""
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    output: list[str] = []

    result = main(["login", "spotify", "--profile", "household"], output=output.append)

    assert result == 2
    assert output == ["spotify: not configured"]


def test_registered_provider_routes_profile_without_accepting_a_secret() -> None:
    """Route login, status, and disconnect to the selected local profile."""
    provider = FakeProvider()
    registry = ProviderRegistry()
    registry.register("fake", provider)
    output: list[str] = []

    assert (
        main(
            ["login", "fake", "--profile", "dan"],
            registry=registry,
            output=output.append,
        )
        == 0
    )
    assert (
        main(
            ["status", "fake", "--profile", "dan"],
            registry=registry,
            output=output.append,
        )
        == 0
    )
    assert (
        main(
            ["disconnect", "fake", "--profile", "dan"],
            registry=registry,
            output=output.append,
        )
        == 0
    )
    assert provider.logins == ["dan"]
    assert provider.disconnections == ["dan"]
    assert output == [
        "fake: login complete",
        "fake/dan: connected",
        "fake: disconnected",
    ]
