# Voicebot

A simple Danish voice bot.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/saattrupdan/voicebot/tree/main/tests)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://saattrupdan.github.io/voicebot/voicebot.html)
[![License](https://img.shields.io/github/license/saattrupdan/voicebot)](https://github.com/saattrupdan/voicebot/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/voicebot)](https://github.com/saattrupdan/voicebot/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/saattrupdan/voicebot/blob/main/CODE_OF_CONDUCT.md)

Developer:

- Dan Saattrup Nielsen (saattrupdan@gmail.com)


## Quick Start

### From Docker

Run `make docker` to build a Docker image and run the Docker container.

### From Source

1. Run `make install`, which sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.
3. (Optional) Run `make install-pre-commit`, which installs pre-commit hooks for linting, formatting and type checking.
4. Run `python src/scripts/run_bot.py` to start the bot.


## All Built-in Commands

The project includes the following convenience commands:

- `make install`: Install the project and its dependencies in a virtual environment.
- `make install-pre-commit`: Install pre-commit hooks for linting, formatting and type checking.
- `make lint`: Lint the code using `ruff`.
- `make format`: Format the code using `ruff`.
- `make type-check`: Type check the code using `mypy`.
- `make test`: Run tests using `pytest` and update the coverage badge in the readme.
- `make docker`: Build a Docker image and run the Docker container.
- `make docs`: Generate HTML documentation using `pdoc`.
- `make view-docs`: View the generated HTML documentation in a browser.
- `make tree`: Show the project structure as a tree.
