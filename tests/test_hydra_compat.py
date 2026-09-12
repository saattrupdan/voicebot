"""Tests for Hydra's Python 3.14 compatibility workaround."""

import argparse
import importlib
import typing as t

import pytest

from voicebot.hydra_compat import install_hydra_argparse_compatibility


def test_hydra_parser_is_built_and_argparse_validation_is_restored() -> None:
    """Hydra's parser builds without leaving argparse validation disabled."""
    hydra_main = importlib.import_module("hydra.main")
    original_check_help = getattr(argparse.ArgumentParser, "_check_help")

    install_hydra_argparse_compatibility()
    install_hydra_argparse_compatibility()
    parser = hydra_main.get_args_parser()

    assert getattr(argparse.ArgumentParser, "_check_help") is original_check_help
    with pytest.raises(ValueError, match="badly formed help string"):
        argparse.ArgumentParser().add_argument("--invalid", help=t.cast(str, object()))
    assert parser is not None
