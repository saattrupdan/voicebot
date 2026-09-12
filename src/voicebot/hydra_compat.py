"""Compatibility helpers for Hydra's command-line parser."""

import argparse
import collections.abc as c
import functools
import importlib
import typing as t

_PATCH_MARKER = "__voicebot_hydra_argparse_compatibility__"


def install_hydra_argparse_compatibility() -> None:
    """Work around Hydra's lazy help text on Python 3.14.

    Python 3.14 validates help values while adding arguments, but Hydra 1.3.6
    uses an object whose ``__repr__`` supplies the help text lazily.  The
    validation is skipped only for Hydra's shell-completion argument and is
    restored before the parser is returned.
    """
    hydra_main = importlib.import_module("hydra.main")
    get_args_parser = t.cast(
        c.Callable[[], argparse.ArgumentParser], getattr(hydra_main, "get_args_parser")
    )
    if getattr(get_args_parser, _PATCH_MARKER, False):
        return

    @functools.wraps(get_args_parser)
    def get_args_parser_with_compatibility() -> argparse.ArgumentParser:
        original_add_argument = t.cast(
            c.Callable[..., argparse.Action], argparse.ArgumentParser.add_argument
        )

        def add_argument(
            parser: argparse.ArgumentParser, *arguments: str, **kwargs: object
        ) -> argparse.Action:
            if "--shell-completion" not in arguments:
                return original_add_argument(parser, *arguments, **kwargs)

            original_check_help = t.cast(
                c.Callable[[argparse.ArgumentParser, argparse.Action], None],
                getattr(argparse.ArgumentParser, "_check_help"),
            )
            setattr(argparse.ArgumentParser, "_check_help", _skip_help_check)
            try:
                return original_add_argument(parser, *arguments, **kwargs)
            finally:
                setattr(argparse.ArgumentParser, "_check_help", original_check_help)

        setattr(argparse.ArgumentParser, "add_argument", add_argument)
        try:
            return get_args_parser()
        finally:
            setattr(argparse.ArgumentParser, "add_argument", original_add_argument)

    setattr(get_args_parser_with_compatibility, _PATCH_MARKER, True)
    setattr(hydra_main, "get_args_parser", get_args_parser_with_compatibility)


def _skip_help_check(parser: argparse.ArgumentParser, action: argparse.Action) -> None:
    del parser, action
