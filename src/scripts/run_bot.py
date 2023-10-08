"""Runs the voicebot."""

from voicebot import VoiceBot
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Runs the voicebot.

    Args:
        cfg: Hydra configuration object.
    """
    bot = VoiceBot(temperature=cfg.temperature)
    bot.run()


if __name__ == "__main__":
    main()
