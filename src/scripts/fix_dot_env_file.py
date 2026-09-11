"""Checks related to the .env file in the repository.

Usage:
    uv run python src/scripts/fix_dot_env_file.py [--non-interactive]
"""

from pathlib import Path

import click

DESIRED_ENVIRONMENT_VARIABLES = dict(
    GIT_NAME="Enter your full name, to be shown in Git commits",
    GIT_EMAIL="Enter your email, as registered on your GitHub account",
    SYV_API_KEY="Enter your SYV API key",
    MELIOUS_API_KEY="Enter your Melious API key",
)
SECRET_ENVIRONMENT_VARIABLES = {"SYV_API_KEY", "MELIOUS_API_KEY"}


@click.command()
@click.option(
    "--non-interactive",
    is_flag=True,
    default=False,
    help="If set, the script will not ask for user input.",
)
def fix_dot_env_file(non_interactive: bool) -> None:
    """Ensure that the .env file contains all desired variables.

    Args:
        non_interactive:
            If set, the script will not ask for user input.
    """
    env_path = Path(".env")
    name_and_email_path = Path(".name_and_email")

    env_path.touch(mode=0o600, exist_ok=True)
    env_path.chmod(0o600)
    name_and_email_path.touch(exist_ok=True)

    env_file_lines = env_path.read_text().splitlines()
    name_and_email_file_lines = name_and_email_path.read_text().splitlines()
    env_vars, env_line_indices = _parse_env_lines(lines=env_file_lines)
    name_and_email_vars, _ = _parse_env_lines(lines=name_and_email_file_lines)

    missing_env_vars = [
        name for name in DESIRED_ENVIRONMENT_VARIABLES if not env_vars.get(name, "")
    ]
    for name in missing_env_vars:
        value = name_and_email_vars.get(name, "")
        if not value and not non_interactive:
            value = click.prompt(
                DESIRED_ENVIRONMENT_VARIABLES[name],
                hide_input=name in SECRET_ENVIRONMENT_VARIABLES,
                show_default=False,
            )

        line = f"{name}={value}"
        if name in env_line_indices:
            env_file_lines[env_line_indices[name]] = line
        else:
            env_line_indices[name] = len(env_file_lines)
            env_file_lines.append(line)

    env_path.write_text("\n".join(env_file_lines) + "\n")
    env_path.chmod(0o600)
    name_and_email_path.unlink()


def _parse_env_lines(lines: list[str]) -> tuple[dict[str, str], dict[str, int]]:
    """Parse environment values and their line indices."""
    values: dict[str, str] = dict()
    line_indices: dict[str, int] = dict()
    for index, line in enumerate(lines):
        name, separator, value = line.partition("=")
        if separator:
            values[name] = value
            line_indices[name] = index
    return values, line_indices


if __name__ == "__main__":
    fix_dot_env_file()
