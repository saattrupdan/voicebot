# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: docs help

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Create poetry env file if it does not already exist
ifeq (,$(wildcard ${HOME}/.poetry/env))
	$(shell mkdir ${HOME}/.poetry)
	$(shell touch ${HOME}/.poetry/env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Ensure that `pipx` and `poetry` will be able to run, since `pip` and `brew` put these
# in the following folders on Unix systems
export PATH := ${HOME}/.local/bin:/opt/homebrew/bin:$(PATH)

# Prevent DBusErrorResponse during `poetry install`.
# See https://stackoverflow.com/a/75098703 for more information
export PYTHON_KEYRING_BACKEND := keyring.backends.null.Keyring

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the 'voicebot' project..."
	@$(MAKE) --quiet install-brew
	@$(MAKE) --quiet install-pipx
	@$(MAKE) --quiet install-poetry
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet setup-environment-variables
	@$(MAKE) --quiet setup-git
	@$(MAKE) --quiet install-pre-commit
	@$(MAKE) --quiet add-repo-to-git
	@echo "Installed the 'voicebot' project! You can now activate your virtual environment with 'source .venv/bin/activate'."
	@echo "Note that this is a Poetry project. Use 'poetry add <package>' to install new dependencies and 'poetry remove <package>' to remove them."

install-brew:
	@if [ $$(uname) = "Darwin" ] && [ "$(shell which brew)" = "" ]; then \
		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		echo "Installed Homebrew."; \
	fi

install-pipx:
	@if [ "$(shell which pipx)" = "" ]; then \
		uname=$$(uname); \
			case $${uname} in \
				(*Darwin*) installCmd='brew install pipx';; \
				(*CYGWIN*) installCmd='py -3 -m pip install --upgrade --user pipx';; \
				(*) installCmd='python3 -m pip install --upgrade --user pipx';; \
			esac; \
			$${installCmd}; \
		pipx ensurepath --force; \
		echo "Installed pipx."; \
	fi

install-poetry:
	@if [ ! "$(shell poetry --version)" = "Poetry (version 1.8.2)" ]; then \
		python3 -m pip uninstall -y poetry poetry-core poetry-plugin-export; \
		pipx install --force poetry==1.8.2; \
		echo "Installed Poetry."; \
	fi

install-dependencies:
	@poetry env use python3.11 && poetry install

setup-environment-variables:
	poetry run python src/scripts/fix_dot_env_file.py; \

setup-environment-variables-non-interactive:
	poetry run python src/scripts/fix_dot_env_file.py --non-interactive; \

setup-git:
	@git config --global init.defaultBranch main
	@git init
	@git config --local user.name "${GIT_NAME}"
	@git config --local user.email "${GIT_EMAIL}"

add-repo-to-git:
	@if [ ! "$(shell git status --short)" = "" ] && [ "$(shell git --no-pager log --all | sed 's/`//g')" = "" ]; then \
		git add .; \
		git commit --quiet -m "Initial commit"; \
	fi
	@if [ "$(shell git remote)" = "" ]; then \
		git remote add origin git@github.com:alexandrainst/voicebot.git; \
	fi

install-pre-commit:  ## Install pre-commit hooks
	@poetry run pre-commit install

docs:  ## Generate documentation
	@poetry run pdoc --docformat google src/voicebot -o docs
	@echo "Saved documentation."

view-docs:  ## View documentation
	@echo "Viewing API documentation..."
	@uname=$$(uname); \
		case $${uname} in \
			(*Linux*) openCmd='xdg-open';; \
			(*Darwin*) openCmd='open';; \
			(*CYGWIN*) openCmd='cygstart';; \
			(*) echo 'Error: Unsupported platform: $${uname}'; exit 2;; \
		esac; \
		"$${openCmd}" docs/voicebot.html

test:  ## Run tests
	@poetry run pytest && poetry run readme-cov

docker:  ## Build Docker image and run container
	@docker build -t voicebot .
	@docker run -it --rm voicebot

tree:  ## Print directory tree
	@tree -a --gitignore -I .git .

lint:  ## Lint the project
	poetry run ruff check . --fix

format:  ## Format the project
	poetry run ruff format .

type-check:  ## Type-check the project
	@poetry run mypy . --install-types --non-interactive --ignore-missing-imports --show-error-codes --check-untyped-defs

add-rag:  ## Install the Ragger package, for RAG projects
	@poetry add git+ssh://git@github.com/alexandrainst/ragger.git --extras all
