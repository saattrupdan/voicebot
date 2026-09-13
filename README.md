# Voicebot

A simple Danish voice bot with local timers, durable reminders, and opt-in provider
integrations.

## Quick Start

1. Run `make install` to create the environment and install dependencies.
2. Add `SYV_API_KEY` and `MELIOUS_API_KEY` to `.env` when prompted.
3. Run `make bot`.

The default configuration enables Google Calendar through the local `gws` CLI and
keeps Spotify and Listonic disabled. Named
timers are local to the process; reminders are stored in
`.local/state/voicebot.sqlite` and are delivered by the scheduler embedded in the bot.
The scheduler uses a SQLite lease, so a restart recovers due notifications. Delivery is
at-least-once: a crash after playback but before acknowledgement can repeat a reminder.

## Local setup and integration status

Run the setup CLI without entering credentials on the command line:

```sh
uv run src/scripts/integrations.py status
uv run src/scripts/integrations.py login spotify --profile dan
uv run src/scripts/integrations.py disconnect spotify --profile dan

# Optional: override the shared SQLite location for smoke tests or another installation
VOICEBOT_DATABASE_PATH=/path/to/voicebot.sqlite \\
  uv run src/scripts/integrations.py status
```

Status output contains only provider, profile, and connection state. Login creates or
rehydrates the profile and persists account metadata in the configured SQLite database;
the bot and this CLI therefore use the same state. Disconnect revokes where supported,
erases the keyring secret, and records disconnected status. Refresh credentials live
in the operating-system keyring by default; use `storage.credential_backend: memory`
only for disposable tests. Listonic's expiring
imported access state is also stored only in that credential backend, separately from
its refresh credential, so it survives CLI exit. The database stores opaque credential
references and aliases, not keys or tokens. Back up the database before migrations,
but never copy `.env` or a keyring export into source control.

Add spoken profile aliases under `profiles` (and select one installation default with
`device.default_profile`). Calendar aliases, Spotify device aliases, and Listonic list
aliases are setup-owned local mappings under their respective integration sections.
Provider IDs in these mappings are never accepted as model arguments or spoken back.
Set `mutations_enabled: false` to centrally disable every tool that changes local or
provider state while retaining read-only tools.

### Google Calendar through `gws`

The shipped configuration uses the already-authenticated local Google Workspace CLI;
it does not create a second OAuth client, persist a Calendar account, or handle tokens.
Install `gws` and authenticate it once (for example with `gws auth login`). Calendar
reads are limited to event listing and free/busy queries. The profile `dan` is bound to
Google's `primary` calendar under the spoken alias `min kalender`.

An authenticated `gws` installation is the sole Calendar and Gmail prerequisite. Gmail
is enabled for the `dan` profile in the shipped configuration. Gmail tools can search or
read bounded message summaries and save unsent drafts; they never send mail. Message
handles are short-lived, device-scoped process-local values, and attachments are never
downloaded. The setup CLI does not log in to or disconnect the `gws` session. If `gws` is
missing, unauthenticated, times out, or returns invalid JSON, Calendar tools fail closed
with their normal unavailable or not-connected status.

### Spotify OAuth setup

Set the Spotify client ID before running Spotify setup:

```sh
export SPOTIFY_CLIENT_ID=your-spotify-client-id
```

Spotify playback control requires a Spotify Premium account and an account with
playback permission; inactive or ambiguous devices are reported instead of guessed.

### Listonic warning

Listonic support is an unofficial, contract-pinned integration. It is disabled unless
both `integrations.listonic.enabled` and
`integrations.listonic.allow_unofficial` are true. Login uses a disposable isolated
browser helper and never asks the bot for a Listonic password. Install `agent-browser`
or set `LISTONIC_BROWSER_HELPER` before Listonic login; the command reports a local
setup error when neither is available. The helper must expose the authenticated browser
state; refresh is deliberately fail-closed because Listonic has no verified refresh
contract.
An expired access state reports that re-onboarding is required. Keep the integration
disabled unless the operational risk of an undocumented API is acceptable; contract
drift fails closed.

Shopping requests which omit `list_name` require an explicit per-profile entry under
`integrations.listonic.default_lists`; the runtime never chooses the first persisted
binding.

Item removal is not verified and is unavailable by default even when Listonic reads and
adds are enabled. Enabling `allow_unverified_item_removal` requires prior live testing
against a disposable list; this project does not claim that endpoint is verified. When
enabled, each removal still requires a local, device-bound yes/no confirmation. List
creation, sharing, whole-list deletion, and other unsupported operations remain
unavailable.

## Conversation and audio behaviour

Model responses and Plapre audio use streaming APIs, and each response is synthesised in
sentence-sized pieces. The microphone remains active during generation and playback;
confirmed speech stops the current response and is processed as a follow-up. Reminder
and timer alarms use the same bot-owned synthesiser and can be dismissed by barge-in.
A clear ending such as `stop`, `ti stille`, `tak`, or `farvel` ends the interaction
silently and requires a new wake word. This rule remains unchanged when no alarm is
active.

Weather lookup remembers a successful location and falls back from ipapi.co to ipwho.is.
Set `weather_default_location` in `config/config.yaml` to provide a final fallback when
neither IP service is available.

The configured transcription endpoint accepts complete audio files rather than a
realtime microphone stream. Transcription therefore starts as soon as an utterance ends,
but true on-the-go ASR requires a realtime endpoint from the provider.
