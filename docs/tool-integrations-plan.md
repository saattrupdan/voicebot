# Voicebot tool integrations plan

Status: Implemented defaults; provider connections remain opt-in

This document proposes named timers, persistent reminders, read-only Google Calendar,
Spotify playback control, and Listonic shopping-list tools for the Danish voicebot.

The design keeps provider credentials and identifiers outside the language model. It
also treats spoken names as routing hints, not proof of identity.

## Goals

- Support multiple named cooking timers.
- Deliver reminders at relative or absolute times, including after a restart.
- Read selected calendars for configured household profiles.
- Control Spotify playback on configured accounts and devices.
- Read and update selected Listonic shopping lists.
- Keep tool selection, ambiguity, confirmation, and cancellation deterministic.
- Prevent credentials, private provider data, and raw errors from reaching logs or the
  model.

## Non-goals

- Voice biometrics or speaker recognition as authentication.
- Writing, deleting, or responding to calendar events.
- Sending credentials or OAuth codes through speech.
- Streaming Spotify audio through the voicebot.
- Exposing arbitrary provider API calls to the model.
- Recurring reminders in the first release.
- Synchronising cooking timers between voicebot installations.
- Using unverified Listonic endpoints as automatic fallbacks.

## Design principles

### Keep the model-visible surface small

Provider searches, OAuth refreshes, alias matching, and retries are implementation
concerns. The model should call a small set of tools corresponding to things a person
would ask the bot to do.

### Resolve before acting

Every mutation follows this order:

1. Validate the arguments.
2. Resolve the profile and target to exactly one configured object.
3. Return a clarification request if resolution is ambiguous.
4. Determine whether confirmation is required.
5. Check cancellation immediately before the provider call.
6. Execute the operation at most once.
7. Store the committed result before producing speech.

The current dynamic `getattr()` dispatch and `TypeError` retry should be replaced with a
typed allow-list registry. Retrying a side-effecting tool without arguments could
execute the wrong operation or execute it twice.

### Treat voice identity as convenience, not security

A spoken name selects a profile or calendar alias. It does not prove who is speaking.
Anyone near the device may be able to issue a command.

Each physical voicebot installation has an optional default profile. Explicit names
override that default only when they resolve uniquely. The bot never falls back to
another person's account when an account is unavailable.

### Keep provider content untrusted

Calendar titles, playlist names, artist names, device names, and shopping-list items are
data, not instructions. They must be represented as structured tool results and must not
be concatenated into system prompts.

## Proposed user experience

### Named timers

Examples:

- “Sæt en pastatimer på ni minutter.”
- “Sæt en kartoffeltimer på tyve minutter.”
- “Hvor lang tid er der tilbage på pastaen?”
- “Stop pastatimeren.”
- “Hvilke timere kører?”

Active timer names must be unique after normalisation. If “pasta” already exists,
the bot reports the conflict instead of silently replacing it.

Timers remain local and in-memory. They use monotonic deadlines and are intended for
short durations. Persistent, wall-clock-based requests use reminders instead.

### Reminders

Examples:

- “Mind mig om at slukke ovnen om en time.”
- “Mind mig om tandlægen i morgen klokken otte.”
- “Hvilke påmindelser har jeg?”
- “Slet påmindelsen om tandlægen.”

Relative times are converted to an absolute timestamp when the reminder is created.
Absolute times must include a resolvable date, local time, and timezone. The bot asks a
short clarification question when “on Tuesday” or a similar phrase has more than one
reasonable interpretation.

### Read-only calendars

Examples:

- “Hvad står der i min kalender i morgen?”
- “Hvad har Anna i kalenderen på fredag?”
- “Hvad er der i familiekalenderen i weekenden?”
- “Hvornår er vi ledige på søndag?”

Suggested routing:

- “Min kalender” uses the device's default profile.
- An explicit person name uses that person's configured calendar aliases.
- “Familiekalenderen” uses a separately configured shared calendar.
- A request with several matching calendars produces a clarification question.
- A device without a default profile requires a spoken person or calendar name.

The integration returns event title, start and end, all-day state, calendar alias, and
optionally location. It omits descriptions, attendees, conference links, and
attachments by default. Private events are spoken as “Optaget” unless explicitly
enabled for that calendar.

### Spotify

Examples:

- “Spil Kind of Blue.”
- “Spil Agnes Obel i køkkenet.”
- “Sæt musikken på pause.”
- “Spring den her sang over.”
- “Skru ned til 30 procent.”
- “Hvad spiller?”

Spotify controls existing Spotify Connect devices. It does not send music through the
voicebot's audio pipeline. A requested device is fetched and resolved from the account's
current device list rather than from a permanent Spotify device ID.

The adapter performs search internally. If a query has no clear match, it returns a few
safe candidate names and asks the user to clarify instead of playing an arbitrary
result.

### Listonic

Examples:

- “Hvad står der på indkøbslisten?”
- “Tilføj mælk og rugbrød til indkøbslisten.”
- “Tilføj to liter mælk til sommerhuslisten.”
- “Marker mælk som købt.”
- “Fjern ketchup fra indkøbslisten.”

A configured default list handles unqualified requests. An explicit list name overrides
the default only when it resolves uniquely.

Listonic is an unofficial integration. It must have an independent feature flag and a
circuit breaker. Contract drift disables Listonic without affecting other tools.

## Model-visible tools

All schemas should use strict JSON Schema with `additionalProperties: false`. Every
property is required; optional values are represented as nullable fields. The Python
runtime validates semantic constraints before dispatch.

Provider IDs, OAuth tokens, raw URLs, and email addresses are never accepted as tool
arguments or returned to the model. Human names are mapped to local opaque references by
the runtime.

### Timer tools

#### `set_timer`

Creates a uniquely named local timer.

Arguments:

```yaml
name: string
duration_seconds: integer # 1 through 86400
```

#### `list_timers`

Lists all active timers or one matching timer.

Arguments:

```yaml
name: string | null
```

#### `stop_timer`

Stops exactly one named timer. It must never fall back to the shortest timer.

Arguments:

```yaml
name: string
```

### Reminder tools

#### `create_reminder`

Creates a persistent reminder. Exactly one of `delay_seconds` and `due_at` must be
non-null.

Arguments:

```yaml
profile_name: string | null
name: string | null
message: string
delay_seconds: integer | null
due_at: RFC3339 timestamp with offset | null
```

#### `list_reminders`

Lists reminders in a bounded interval.

Arguments:

```yaml
profile_name: string | null
starts_at: RFC3339 timestamp with offset | null
ends_at: RFC3339 timestamp with offset | null
status: pending | delivered | missed | all
```

#### `cancel_reminder`

Cancels one reminder resolved by a spoken name or a reference from `list_reminders`.

Arguments:

```yaml
profile_name: string | null
reminder_name: string
```

### Calendar tools

#### `list_calendar_events`

Returns a chronological, bounded event list.

Arguments:

```yaml
profile_name: string | null
calendar_name: string | null
starts_at: RFC3339 timestamp with offset
ends_at: RFC3339 timestamp with offset
query: string | null
max_results: integer # 1 through 50
```

#### `get_calendar_availability`

Returns busy intervals without exposing event details. This is useful for combined
household questions.

Arguments:

```yaml
profile_names: array[string]
starts_at: RFC3339 timestamp with offset
ends_at: RFC3339 timestamp with offset
```

### Spotify tools

#### `spotify_play`

Searches and starts one track, album, artist, playlist, show, or episode.

Arguments:

```yaml
profile_name: string | null
query: string
media_type: track | album | artist | playlist | show | episode | null
device_name: string | null
```

#### `spotify_control`

Controls the current playback context.

Arguments:

```yaml
profile_name: string | null
action: pause | resume | next | previous
device_name: string | null
```

#### `spotify_set_volume`

Sets the volume on one resolved device.

Arguments:

```yaml
profile_name: string | null
volume_percent: integer # 0 through 100
device_name: string | null
```

#### `spotify_now_playing`

Returns the current item, playback state, and device alias.

Arguments:

```yaml
profile_name: string | null
```

#### `spotify_list_devices`

Lists currently available device aliases when a spoken device is ambiguous or missing.

Arguments:

```yaml
profile_name: string | null
```

Spotify search is deliberately not a separate model-visible tool. Search and safe result
binding happen inside `spotify_play`, reducing round trips and preventing the model from
inventing Spotify URIs.

### Listonic tools

#### `list_shopping_lists`

Lists configured shopping-list aliases and item counts.

Arguments:

```yaml
profile_name: string | null
```

#### `get_shopping_list`

Returns a bounded list of shopping items.

Arguments:

```yaml
profile_name: string | null
list_name: string | null
include_checked: boolean
```

#### `add_shopping_items`

Adds one or more items in one operation.

Arguments:

```yaml
profile_name: string | null
list_name: string | null
items:
    - name: string
      amount: number | null
      unit: string | null
```

#### `set_shopping_item_checked`

Marks exactly one resolved item as bought or not bought.

Arguments:

```yaml
profile_name: string | null
list_name: string | null
item_name: string
checked: boolean
```

#### `remove_shopping_item`

Removes exactly one resolved item after confirmation.

Arguments:

```yaml
profile_name: string | null
list_name: string | null
item_name: string
```

Creating, deleting, renaming, sharing, or clearing whole Listonic lists is excluded from
the first version. Those operations are less common, more destructive, and not all have
verified production contracts.

## Common tool result contract

Tools return structured data to the text engine rather than speaking directly.

```yaml
status:
    ok | needs_clarification | confirmation_required | not_found | conflict |
    unauthenticated | forbidden | unavailable | rate_limited | outcome_unknown |
    cancelled | invalid_request
operation_id: string | null
message_da: string
data: object | null
candidates: array
retryable: boolean
```

Rules:

- `needs_clarification` contains only safe local aliases.
- `confirmation_required` creates a short-lived pending action bound to all resolved
  arguments and the originating device.
- Mutations receive an idempotency key and are never automatically replayed after an
  uncertain provider response. An `outcome_unknown` result is shown to the next model
  turn as non-repeatable, even when speech cancellation rolls back the current turn.
- `unauthenticated` instructs the user to run local setup. It never speaks a token,
  login URL, provider identifier, or provider error body.
- Provider exceptions and response bodies do not pass directly to the model.
- A tool that has committed a mutation returns the committed result even if speech was
  interrupted afterward.

## Confirmation and ambiguity policy

### Execute immediately

- Creating a clearly named timer.
- Stopping one uniquely matched timer.
- Creating a reminder with an unambiguous time.
- Reading calendar data.
- Normal Spotify play, pause, resume, skip, and volume changes up to 80 percent.
- Adding items to one uniquely resolved shopping list.
- Checking or unchecking one uniquely resolved shopping item.

### Require confirmation

- Removing a shopping item.
- Any bulk deletion or clearing operation added later.
- Spotify volume above 80 percent.
- An operation that explicitly targets another profile when device policy requires it.
- Replacing an existing object with the same normalised name.

### Never guess

- Multiple profiles, calendars, Spotify devices, lists, items, timers, or reminders
  match.
- A mutation target only matches fuzzily.
- An absolute reminder time is ambiguous or falls in a daylight-saving transition.
- A provider account is disconnected or belongs to another profile.

Confirmations are handled by the local conversation runtime, not by provider-specific
model tools. “Yes” executes only the most recent unexpired action on that device. A
changed request creates a new pending action.

## Authentication architecture

### Shared rules

Authentication never happens through speech. A local setup command performs onboarding:

```text
voicebot integrations login spotify --profile household
voicebot integrations login listonic --profile household
voicebot integrations status
voicebot integrations disconnect <provider> --profile <name>
```

The commands are illustrative; the repository can expose them through its existing
script layout rather than adding a separate package entry point.

Credential handling:

1. Store refresh credentials in the operating-system keyring where available.
2. Store access tokens in memory and refresh them shortly before expiry.
3. Store only credential references, scopes, expiry, and connection status in SQLite.
4. Serialise refreshes per account so concurrent tools do not rotate the same token.
5. Atomically replace rotated refresh tokens.
6. Mark the account disconnected on permanent refresh failure.
7. Erase local credentials during disconnect and revoke them when the provider supports
   revocation.
8. Never store provider passwords after onboarding.

A headless deployment without a keyring may use an encrypted credential file only when a
separate deployment secret is available. File permissions alone are useful defence in
depth but are not encryption.

Logs must redact:

- `Authorization` and cookie headers;
- passwords, OAuth codes, access tokens, and refresh tokens;
- email addresses and raw provider account IDs;
- calendar event bodies and Listonic request bodies;
- provider errors that may echo request data.

Logs may contain the local operation ID, provider name, status class, duration, and
retry count.

### Google Calendar access

Use the authenticated local `gws` CLI as the sole Calendar runtime prerequisite. The
voicebot passes only bounded, read-only Calendar requests to `gws`; it does not create
an OAuth application, perform provider-owned authorisation, store refresh tokens, or provide
Calendar onboarding commands. Configure the Calendar feature flag, explicit
`profile_bindings`, and local profile/calendar aliases. Profiles absent from those
bindings, including stale aliases, fail closed without invoking `gws`.

The same authenticated `gws` session also serves Gmail. Gmail is independently feature
gated for the `dan` profile and exposes only bounded search/latest/read summaries and
unsent draft creation. It has no send route; opaque message handles are process-local,
device-scoped, expiring, and capped, and attachment content is never downloaded.

### Spotify authentication

Use Spotify Authorization Code with PKCE. Request only:

```text
user-read-playback-state
user-modify-playback-state
```

Spotify playback control requires Premium. Each controlled Spotify account needs its own
authorization and may expose a different set of devices.

The adapter refreshes hourly access tokens using the stored refresh token. Device IDs
are fetched when needed and bound to configured aliases only for a short period because
Spotify device IDs are not guaranteed to remain stable.

If server-side grant revocation is unavailable, disconnect erases local credentials and
instructs the user to remove the app from Spotify's account permissions.

### Listonic authentication

Listonic has no confirmed public production API or third-party OAuth registration. This
integration uses the private production API with the user's explicit approval.

Onboarding should:

1. Open an isolated local browser at Listonic's normal login page.
2. Let the user enter credentials directly into Listonic.
3. Import only the resulting access token, refresh token, and expiry metadata.
4. Destroy the isolated browser profile.
5. Store the refresh credential in the normal credential backend.

The voicebot never receives or stores the Listonic password.

The following production behaviour was verified during a disposable-list test:

```text
GET    /api/lists                              -> 200
POST   /api/lists                              -> 201
GET    /api/lists/{list_id}                    -> 200
GET    /api/lists/{list_id}/items              -> 200
POST   /api/lists/{list_id}/items              -> 201
PATCH  /api/lists/{list_id}/items/{item_id}    -> 200
```

Observed request fields are inconsistent:

- Creating a list uses `Name` and `SortMode`.
- Creating an item uses `Amount`, `Unit`, and lowercase `name`.
- Checking an item uses lowercase `checked` and `itemId`.

This casing remains isolated inside the Listonic adapter. Domain models and tool schemas
use consistent Python naming. The item DELETE endpoint remains unverified and is gated
by `allow_unverified_item_removal: false`; enabling it requires a live check against a
disposable list and is not an assertion that this plan verified it.

Refresh and revocation behaviour must be verified before unattended rollout. If refresh
fails or the observed contract changes, the adapter opens a Listonic-only circuit
breaker and asks for re-onboarding. It must not probe guessed endpoints.

## Profile and alias model

SQLite stores local routing metadata, never credentials:

- `profiles`: household people or account groupings.
- `profile_aliases`: normalised spoken aliases.
- `provider_accounts`: provider, local profile, credential reference, scopes, and state.
- `calendar_bindings`: calendar alias to provider calendar ID.
- `spotify_device_aliases`: profile-scoped local aliases with short-lived provider
  resolution.
- `shopping_list_bindings`: list alias to Listonic list ID.

Resolution algorithm:

1. Unicode-normalise and case-fold the spoken value.
2. Trim punctuation and collapse whitespace.
3. Match exact configured aliases.
4. Use the device default only if no target was spoken.
5. Return `not_found` for zero matches.
6. Return `needs_clarification` for multiple matches.
7. Never fuzzy-select a target for a mutation.

The model cannot create or persist aliases. Alias management belongs to local setup.

## Reminder persistence and scheduling

The current process-per-timer design is not sufficient for reminders. Reminders use a
SQLite-backed scheduler owned by the bot runtime or a single dedicated scheduler
process.

Suggested records:

```text
reminders
notification_queue
scheduler_leases
operations
pending_confirmations
audit_events
```

Each reminder stores:

- a stable local ID;
- profile and optional name;
- message;
- due time in UTC;
- original timezone;
- status and timestamps;
- delivery attempt metadata;
- a deduplication key.

The scheduler:

1. Acquires a renewable single-instance lease.
2. Loads and transactionally claims due reminders.
3. Places them on a durable notification queue.
4. Lets the main bot runtime own TTS and audio arbitration.
5. Acknowledges delivery after playback or explicit dismissal.

A crash after playback may repeat a reminder once, but must not lose it silently.

Current missed-reminder policy:

- Deliver reminders up to 15 minutes late after downtime.
- Mark older reminders as missed.
- Keep missed reminders available through `list_reminders`; they are not automatically
  mentioned on the next interaction.

## Cancellation and barge-in

Every provider adapter receives the active turn's cancellation event.

- Check cancellation before network work.
- Check again immediately before a mutation.
- Cancel read-only network requests when practical.
- Provider adapters never synthesize speech directly.
- Tool results are stored before the text engine starts speaking.
- Once a remote mutation may have committed, do not report it as cancelled or retry it.
- If speech is interrupted after a committed operation, include the result in the next
  turn's context.

Due reminders and completed timers have separate alarm cancellation state. “Stop”
while an alarm is sounding silences that alarm. Outside alarm playback, the current
silent end-of-conversation behaviour remains unchanged.

## Configuration

Configuration contains aliases and feature flags but no secrets:

```yaml
mutations_enabled: true
locale: da-DK
timezone: Europe/Copenhagen

device:
    id: kitchen-speaker
    default_profile: dan

storage:
    database_path: .local/state/voicebot.sqlite
    credential_backend: keyring
    master_key_env: VOICEBOT_MASTER_KEY

profiles:
    dan:
        aliases: [Dan, mig]

integrations:
    google_calendar:
        enabled: false
        default_profile: dan
        profile_bindings: {}
        calendar_aliases: {}
    spotify:
        enabled: false
        default_profile: dan
        device_aliases:
            dan: {}
    listonic:
        enabled: false
        allow_unofficial: false
        allow_unverified_item_removal: false
        list_aliases:
            dan: {}
        default_lists: {}

scheduler:
    poll_seconds: 1
    lease_seconds: 30
    missed_grace_seconds: 900
```

Provider API hosts and OAuth scopes are code constants rather than configurable strings.
This prevents configuration or model output from redirecting credentials to another
host.

## Proposed module layout

```text
src/voicebot/
  auth/
    credentials.py
    listonic.py
    spotify.py
  providers/
    calendar_domain.py
    gws_calendar.py
    listonic.py
    spotify.py
  storage/
    database.py
    migrations.py
    models.py
  tools/
    calendar.py
    reminders.py
    shopping.py
    spotify.py
    timer.py
  notifications.py
  resolution.py
  scheduler.py
  tool_runtime.py
```

The provider layer handles HTTP and provider-specific models. The tool layer exposes
stable domain operations. `tool_runtime.py` handles validation, resolution,
confirmations, cancellation, idempotency, and redaction.

## Implementation phases

### Phase 1: Tool runtime foundation

- Replace dynamic dispatch with a typed tool registry.
- Remove the `TypeError` fallback invocation.
- Add strict runtime argument validation.
- Introduce common result types, operation IDs, cancellation checks, and log redaction.
- Add profile and alias resolution.

### Phase 2: Named timers

- Add names to timer creation, listing, completion, and cancellation.
- Use monotonic deadlines.
- Remove expired timers from active state.
- Reject duplicate names.
- Never stop another timer as a fallback.

### Phase 3: Persistent reminders

- Add SQLite migrations and repository methods.
- Add the scheduler lease and notification queue.
- Support relative and absolute one-off reminders.
- Define restart, missed-reminder, duplicate-delivery, and alarm-dismissal behaviour.

### Phase 4: Read-only Google Calendar

- Require an authenticated local `gws` CLI session; do not add Calendar OAuth
  onboarding or Google client configuration.
- Add profile and calendar aliases.
- Implement event listing and free/busy queries only through `gws`.
- Enforce field filtering and private-event behaviour.
- Roll out first in text-only or shadow mode before spoken results.

### Phase 5: Spotify

- Add OAuth onboarding and refresh.
- Implement device discovery and safe aliases.
- Add playback, control, volume, and now-playing tools.
- Handle Premium, inactive-device, token-expiry, and rate-limit errors explicitly.

### Phase 6: Listonic

- Add explicit unofficial-integration warning and feature flags.
- Implement isolated browser onboarding.
- Verify token refresh using sanitized fixtures.
- Implement list, add, check, and remove-item tools behind an adapter.
- Enable first against a disposable list, then bind selected real list aliases.

### Phase 7: Integrated rollout

- Add end-to-end Danish utterance tests.
- Test cancellation before, during, and after remote mutations.
- Test credential revocation and feature kill switches.
- Document setup, backup, recovery, and provider disconnect procedures.

## Test strategy

### Unit and contract tests

- Snapshot every strict model-visible JSON schema.
- Table-test aliases, ambiguity, defaults, and confirmation policy.
- Inject clocks for relative time, daylight-saving transitions, and clock jumps.
- Use temporary SQLite databases for migrations, leases, crashes, and recovery.
- Mock Spotify OAuth and provider HTTP servers.
- Pin sanitized Listonic request verbs, paths, field casing, and expected statuses.
- Verify Spotify search-to-result binding and current-device resolution.
- Verify calendar field filtering and profile separation.

### Security tests

Place canary values in credentials and private provider payloads. Assert they never
appear in:

- logs;
- exceptions;
- model messages;
- spoken responses;
- SQLite rows not explicitly designated for encrypted credentials.

### Integration tests

Live provider tests are opt-in and excluded from normal CI. They require explicit
environment gates and disposable resources. Listonic tests must create and remove a
uniquely named disposable list.

### Audio and concurrency tests

- Barge-in before a provider call prevents the operation.
- Barge-in after provider commit does not duplicate or misreport the operation.
- Reminder playback can be silenced without ending unrelated timers.
- Concurrent token refresh performs one refresh per account.
- Simultaneous reminders remain ordered and recover after restart.

## Rollout and rollback

Each integration has an independent feature flag. Mutations have a separate global kill
switch.

Rollout order:

1. Ship storage, resolver, onboarding, and integration status while disabled.
2. Enable named timers.
3. Enable reminders for one profile.
4. Enable Google Calendar in shadow mode, then spoken read-only results.
5. Enable Spotify one account at a time.
6. Enable Listonic last, first against a disposable list.

Rollback rules:

- Disabling an integration stops new calls without deleting credentials or reminders.
- Stop the scheduler before restoring a database backup.
- Back up the database before migrations.
- Revoke and erase credentials if an adapter or token path is compromised.
- A Listonic contract failure disables only Listonic.
- Never fall back to browser automation during normal bot operation.

## Implemented defaults

The end-to-end assembly uses these defaults (and the configuration file is the source
of truth for local routing). The setup CLI and bot resolve the same SQLite path (the
`VOICEBOT_DATABASE_PATH` environment override is available for non-Hydra smoke runs),
and both use the OS keyring credential backend:

- Login persists profiles, provider-account metadata, aliases, scopes, status, and
  opaque credential references. A new process rehydrates that metadata before serving
  tools.
- Tool calls use their provider call ID as a stable idempotency key. A completed result
  is reused; an uncertain started operation is never replayed automatically.
- Destructive Listonic removal and Spotify volume above 80 percent use a two-minute,
  device-bound confirmation. The resolved arguments, origin device, and expiry are
  persisted; changed, expired, or cross-device confirmations are rejected.
- Reminder playback is acknowledged only after successful TTS or explicit dismissal;
  playback, network, and cancellation failures remain recoverable.
- Listonic onboarding requires an operator-installed isolated browser helper. The helper
  never receives a password from the bot and is destroyed after token import. Listonic
  refresh has no verified contract and therefore fails closed rather than guessing an
  endpoint.

- [x] `Europe/Copenhagen` is the default timezone.
- [x] Cooking timers are non-persistent and limited to 24 hours.
- [x] Reminders are delivered up to 15 minutes late after downtime.
- [x] Older reminders are marked missed and remain queryable; automatic next-turn
      reporting is not implemented.
- [x] One default profile is used per physical voicebot installation.
- [x] Other people's calendars require an explicit profile name.
- [x] Calendar uses the authenticated local `gws` session; no Google account is
      onboarded by the voicebot.
- [x] Private calendar events are reduced to busy intervals.
- [x] Calendar locations are not exposed by the tool contract.
- [x] Spotify uses the configured profile and exact device aliases.
- [x] Volume is bounded to 0-100 and provider policy errors are surfaced safely.
- [x] Listonic has an explicit default list and remains disabled by default.
- [x] List creation, sharing, and whole-list deletion remain unavailable.
- [x] Removing an individual Listonic item requires local confirmation.
- [x] Listonic's unofficial API requires explicit operator acknowledgement.
- [x] Headless deployments use the OS keyring when available; encrypted-file
      credential storage is not silently enabled. Disposable tests may explicitly
      select memory-only credentials.
- [x] Existing `get_weather`, `get_news`, `search_web`, and `meow` tools remain
      registered with strict schemas alongside the integration tools.

## Definition of done

- Every model-visible tool has a strict schema and deterministic result contract.
- No credential or provider identifier is visible to the model.
- No credential, private event, or shopping-list payload appears in logs.
- Calendar access is technically read-only and uses the authenticated local `gws`
  session.
- Provider accounts and aliases cannot leak across profiles.
- Reminder delivery survives restart and has documented missed-delivery behaviour.
- Barge-in cannot duplicate provider mutations.
- Every integration has a tested feature flag, disconnect procedure, and rollback path.
- Listonic contract drift fails closed without affecting other tools.
