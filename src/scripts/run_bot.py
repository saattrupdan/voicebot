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
        text_model_id=cfg.text_model_id,
        asr_model_id=cfg.asr_model_id,
        sample_rate=cfg.sample_rate,
        temperature=cfg.temperature,
        num_seconds_per_chunk=cfg.num_seconds_per_chunk,
        min_audio_threshold=cfg.min_audio_threshold,
        max_seconds_silence=cfg.max_seconds_silence,
        min_seconds_audio=cfg.min_seconds_audio,
        max_seconds_audio=cfg.max_seconds_audio,
        wake_words=cfg.wake_words,
    )
    bot.run()


if __name__ == "__main__":
    main()
