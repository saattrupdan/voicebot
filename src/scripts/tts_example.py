"""Run a SYV TTS example."""

import os
from pathlib import Path

import openai
from dotenv import load_dotenv


def main() -> None:
    """Generate example speech and save it as a WAV file."""
    load_dotenv()
    client = openai.OpenAI(
        api_key=os.environ["SYV_API_KEY"], base_url="https://platform.syv.ai/v1"
    )
    response = client.audio.speech.create(
        model="syvai/plapre-nano",
        input="Hej, verden.",
        voice="tor",
        response_format="wav",
    )
    response.write_to_file(Path("speech.wav"))


if __name__ == "__main__":
    main()
