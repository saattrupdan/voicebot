"""Run a TTS example."""

from pathlib import Path

import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from huggingface_hub import hf_hub_download

REPO_ID = "CoRal-project/tts-base-compatible"


def main() -> None:
    """Run the TTS example."""
    local_path = ""
    for fpath in [
        "ve.pt",
        "t3_23lang.safetensors",
        "mtl_tokenizer.json",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
    ]:
        local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

    # Load the model
    checkpoint_dir = Path(local_path).parent  # Same for all files
    model = ChatterboxMultilingualTTS.from_local(ckpt_dir=checkpoint_dir, device="cuda")

    # Generate speech to a wav file
    text = "Dette er en test!"
    wav = model.generate(
        text=text, language_id="da", exaggeration=1.0, cfg_weight=0.5, temperature=0.4
    )
    ta.save(uri="test.wav", src=wav, sample_rate=model.sr)


if __name__ == "__main__":
    main()
