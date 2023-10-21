# Voicebot

A simple Danish voice bot.

______________________________________________________________________
[![PyPI Status](https://badge.fury.io/py/voicebot.svg)](https://pypi.org/project/voicebot/)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://saattrupdan.github.io/voicebot/voicebot.html)
[![License](https://img.shields.io/github/license/saattrupdan/voicebot)](https://github.com/saattrupdan/voicebot/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/voicebot)](https://github.com/saattrupdan/voicebot/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-40%25-orange.svg)](https://github.com/saattrupdan/voicebot/tree/main/tests)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/saattrupdan/voicebot/blob/main/CODE_OF_CONDUCT.md)


Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)


## Setup

### Set up the environment

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Install new packages

To install new PyPI packages, run:

```
$ poetry add <package-name>
```

### Get an overview of the available commands

Simply write `make` to display a list of the commands available. This includes the
above-mentioned `make install` command, as well as building and viewing documentation,
publishing the code as a package and more.


## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management
* [hydra](https://hydra.cc/): Manage configuration files
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project


## Project structure
```
.
├── .github
│   └── workflows
│       ├── ci.yaml
│       └── docs.yaml
├── .gitignore
├── .pre-commit-config.yaml
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── config
│   ├── __init__.py
│   ├── config.yaml
│   └── hydra
│       └── job_logging
│           └── custom.yaml
├── makefile
├── poetry.toml
├── pyproject.toml
├── src
│   ├── scripts
│   │   ├── fix_dot_env_file.py
│   │   └── run_bot.py
│   └── voicebot
│       ├── __init__.py
│       ├── bot.py
│       ├── speech_recognition.py
│       ├── speech_recording.py
│       ├── speech_synthesis.py
│       └── text_engine.py
└── tests
    ├── __init__.py
    └── test_dummy.py
```
