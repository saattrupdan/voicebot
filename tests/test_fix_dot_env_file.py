"""Tests for environment-file setup."""

import stat
from pathlib import Path

from click.testing import CliRunner

from scripts.fix_dot_env_file import fix_dot_env_file


def test_blank_api_keys_are_replaced_securely() -> None:
    """Blank API keys are prompted for, masked, and stored owner-only."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        env_path = Path(".env")
        env_path.write_text(
            "GIT_NAME=Test User\n"
            "GIT_EMAIL=test@example.com\n"
            "SYV_API_KEY=\n"
            "MELIOUS_API_KEY=\n"
            "OTHER_SETTING=preserved\n"
        )

        result = runner.invoke(fix_dot_env_file, input="syv-secret\nmelious-secret\n")

        assert result.exit_code == 0
        assert "syv-secret" not in result.output
        assert "melious-secret" not in result.output
        assert env_path.read_text() == (
            "GIT_NAME=Test User\n"
            "GIT_EMAIL=test@example.com\n"
            "SYV_API_KEY=syv-secret\n"
            "MELIOUS_API_KEY=melious-secret\n"
            "OTHER_SETTING=preserved\n"
        )
        assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
