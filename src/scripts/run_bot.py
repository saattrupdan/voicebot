"""Runs the voicebot."""

import hydra
from omegaconf import DictConfig

from voicebot import VoiceBot
from voicebot.hydra_compat import install_hydra_argparse_compatibility


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Runs the voicebot.

    Args:
        cfg: Hydra configuration object.
    """
    VoiceBot(cfg=cfg).run()


if __name__ == "__main__":
    install_hydra_argparse_compatibility()
    main()
