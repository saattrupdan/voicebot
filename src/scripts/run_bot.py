"""Runs the voicebot."""

import hydra
from omegaconf import DictConfig

from voicebot import VoiceBot


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Runs the voicebot.

    Args:
        cfg: Hydra configuration object.
    """
    VoiceBot(cfg=cfg).run()


if __name__ == "__main__":
    main()
