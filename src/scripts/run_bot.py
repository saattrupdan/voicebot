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
    bot = VoiceBot(
        asr_model_id=cfg.asr_model_id,
        sample_rate=cfg.sample_rate,
        temperature=cfg.temperature,
        num_seconds_per_chunk=cfg.num_seconds_per_chunk,
        min_audio_threshold=cfg.min_audio_threshold,
    )
    bot.run()


if __name__ == "__main__":
    main()
