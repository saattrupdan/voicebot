# Voicebot

A simple Danish voice bot.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/saattrupdan/voicebot/tree/main/tests)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://saattrupdan.github.io/voicebot/voicebot.html)
[![License](https://img.shields.io/github/license/saattrupdan/voicebot)](https://github.com/saattrupdan/voicebot/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/voicebot)](https://github.com/saattrupdan/voicebot/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/saattrupdan/voicebot/blob/main/CODE_OF_CONDUCT.md)

Developer:

- Dan Saattrup Smart (<saattrupdan@gmail.com>)

## Quick Start

1. Run `make install`, which sets up a virtual environment and all Python dependencies therein.
2. Add `SYV_API_KEY` and `MELIOUS_API_KEY` to `.env` when prompted.
3. Run `make bot` to start the bot.

Speech detection uses WebRTC VAD and an adaptive noise floor, so no manual microphone
calibration is required. The detector adapts to sustained background noise while the bot
is listening.

## Conversation behaviour

Model responses and Plapre audio use streaming APIs, and each response is synthesised in
sentence-sized pieces so playback does not wait for all TTS audio. The microphone remains
active during generation and playback; confirmed speech
stops the current response and is processed as a follow-up. Speaker echo can still cause
false interruptions without acoustic echo cancellation, so headphones or a directional
microphone work best.

Clear endings such as `stop`, `ti stille`, `tak`, and `farvel` end the interaction without
a spoken reply. A new wake word is then required.

Weather lookup remembers a successful location and falls back from ipapi.co to ipwho.is.
Set `weather_default_location` in `config/config.yaml` to provide a final fallback when
neither IP service is available.

The configured transcription endpoint accepts complete audio files rather than a realtime
microphone stream. Transcription therefore starts as soon as an utterance ends, but true
on-the-go ASR requires a realtime endpoint from the provider.
