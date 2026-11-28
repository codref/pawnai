# Voice protocol

The contract between pawn-server and the voice assistants: the Linux
`pawnai-assistant` (Python) and the Android `pawnai-assistant-android`
(Kotlin). Both clients implement it by hand, so this page and the golden
files in [`contracts/`](../contracts) are the source of truth.

- `contracts/voice/*.sse`: golden SSE transcripts.
- `contracts/voice/cases.json`: what a client must make of each transcript.
- `contracts/utterances/shared.fixtures`: phrases both clients route to the
  same local intent. `linux-aliases.txt` maps the Android intent names to the
  Linux ones.

`tests/test_voice_contract.py` checks that the server still produces the
golden files. Each client replays them in its own tests. After changing the
protocol, update `contracts/`, then run:

```bash
scripts/sync-contracts.sh          # copy into ../pawnai-assistant and ../pawnai-assistant-android
scripts/sync-contracts.sh --check  # fail when a copy is stale
```

The copies land in `tests/contracts/` (Linux) and
`skill-testkit/src/main/resources/contracts/` (Android). Never edit them
there.

Device registration, presence, commands and wake-word training are in
[ASSISTANTS.md](ASSISTANTS.md).

## Voice turn: `POST /v1/pawn/voice`

Auth: `Authorization: Bearer <api token>`.
Request: `Accept: text/event-stream`, a JSON body:

| field | type | notes |
|---|---|---|
| `conversation` | string | Required and must start with `voice:`. The client makes it (`voice:` + a random id) when a conversation starts. |
| `message` | string | Required. `/reset` clears the conversation and ends its thread. |
| `access` | `chat` \| `full` | `chat` exposes no tools. `full` is the stock pawn surface. The client picks the level after an unlock. |
| `lang` | string | `it` (default) or `en`. Shapes the spoken style. |
| `device_id` | string? | The registered assistant sending the turn. Links the conversation to it for hand-off and the portal mirror. |

Errors before the stream starts:
- `400`: the conversation does not start with `voice:`.
- `422`: the message is empty.
- `401`: the token is wrong.

### Events

Events arrive in order, as `event: <name>` plus one `data:` JSON line.
Lines starting with `:` are keep-alives, sent while the agent works; ignore
them.

| event | data | client action |
|---|---|---|
| `progress` | `{kind, text}` | Optional status line ("Ran planner_day"). Never speak it. |
| `answer` | `{content, access, controls[], display?}` | `content` is spoken (at most about 3 sentences). `display` is Markdown for the screen only, never spoken. |
| `error` | `{message}` | Say a short failure line. The turn has no answer. |
| `done` | `{conversation, run_id?, access?}` | Always last. `run_id` matches `agent_runs`. |

`controls` lists actions the agent asked the client to take:
- `end`: close the conversation and release the floor.
- `lock`: drop back to `chat` access now.

Ignore unknown controls and unknown fields; the server may add new ones.

## Hand-off: "continua qui"

A conversation follows the user from the laptop to the phone and back.

1. `GET /v1/pawn/voice/active?exclude_device=<my id>` returns
   `{thread: {...} | null}`: the newest thread that is `active` and had a
   turn within `devices.handoff_window_s` (default 600 s). Threads held by
   the asking device are skipped.
2. `POST /v1/pawn/voice/{conversation}/claim {device_id}` moves the thread.
   - Returns the thread plus `from_device`.
   - `access` is always `chat` again: an unlock never travels with a
     conversation.
   - `404`: unknown conversation. `409`: the conversation ended or is too old.
3. The client keeps that `conversation` for its next turns. It should say a
   short "Riprendo." and may read back the start of `last_answer`.

## Full access approved on the phone

The laptop cannot do a biometric check, so it asks the phone:

1. Laptop: `POST /v1/pawn/voice/access-requests {device_id, conversation?, minutes}`
   returns `202` with `{id, status: "pending", approver_name, expires_at, …}`.
   - `minutes` is capped at `devices.access_max_minutes` (30).
   - `409`: no phone offers `access_approve`.
2. The server queues the `access_approve` command (an action) on the phone
   and sends it a UnifiedPush wake-up.
3. Phone: shows "Allow *laptop* full access for N min?" and asks for a
   fingerprint or screen lock, then posts `ok` or `denied` as the command
   result. After the prompt it checks `expires_at` again.
4. Laptop: polls `GET /v1/pawn/voice/access-requests/{id}` about every
   second. The `status` becomes `approved` (with `access_until`), `denied`,
   `expired` or `error`.
5. On `approved`, the laptop sends `access: "full"` until `access_until`.

The keypad unlock on the laptop still works as a fallback.

## Device command stream (desktops)

Phones are woken over UnifiedPush and then pull commands. A desktop holds
`GET /v1/devices/{id}/commands/stream` open instead, authenticated with the
device token.

- `event: ready` with `{device_id}`: sent once.
- `event: cmd` with `{commands: [...]}`: the same shape as `…/commands/pending`.
  These rows are now `delivered`.
- `: keep-alive` comments.

For each command, the client runs it and then posts
`POST /v1/devices/{id}/commands/{cid}/result {status: ok|denied|error, result?, message?}`.

- Skip any command whose `expires_at` has passed.
- If the stream drops, reconnect with a backoff, then drain
  `GET …/commands/pending` once.

Tools an assistant may declare in `capabilities`:

| tool | args | notes |
|---|---|---|
| `speak` | `text` | Say it aloud (queue it behind current speech). |
| `notify` | `text`, `title?` | Show it (desktop: a HUD window plus speech). |
| `timer_set` | `seconds`, `label?` | Start a local countdown. All args are strings. |
| `access_approve` | `requester`, `minutes`, `request_id?` | Phone only. Biometric confirmation first. |
| `location_now`, `status_now`, `sms_send` | see PHONE_BRIDGE.md | Phone only. |
