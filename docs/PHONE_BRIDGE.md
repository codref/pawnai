# Phone bridge

pawn-assistant (the Android app) can share opt-in context from the phone with
pawn-server, and pawn-agent can read it through tools. Every category is off
by default and is switched on per phone in the app (Settings → Phone bridge).

Status: push half (registration, ingest, purge, read tools) and the server
side of live phone tools (command queue + UnifiedPush wake-up) are
implemented. Migrations `0024` and `0025`.

## Data flow

```
phone collectors → outbox → POST /v1/devices/{id}/events → phone_events
                                                        └→ knowledge_chunks (phone_<kind>)
pawn-agent ← phone_events / phone_devices / knowledge_search --kind phone_sms
```

## Setup

1. `alembic upgrade head` (migrations `0024`, `0025`).
2. Optional `devices:` section in `pawnai.yaml` (see `pawnai.example.yaml`).
3. In the app: set the server URL + API token, open **Phone bridge**,
   **Register this phone**, then enable the categories to share.

## API

| Method | Path | Auth | Notes |
|--------|------|------|-------|
| POST | `/v1/devices/register` | shared API token | Body `{name}` → `{device_id, device_token, name}`. The token is shown once; only its SHA-256 is stored. |
| GET | `/v1/devices` | shared API token | List devices. |
| GET | `/v1/devices/{id}` | device token or shared | Device view. |
| PUT | `/v1/devices/{id}` | device token or shared | `{name?, unifiedpush_endpoint?, enabled_categories?, capabilities?}`. Removing a category **purges** the server copy of it (`purged_categories` in the reply). Unknown capabilities are dropped. |
| DELETE | `/v1/devices/{id}` | device token or shared | Revokes the device and deletes its events and knowledge chunks. |
| POST | `/v1/devices/{id}/events` | device token or shared | `{events:[…]}` (≤ `devices.max_batch`) → `{accepted, duplicates, rejected}`. |
| POST | `/v1/devices/{id}/commands` | **shared token only** | `{tool, args, timeout_s?}`: queue a live tool and wait for the phone. Returns the command view plus `text`. Handy for curl testing; the agent's CLI tools do the same in-process. |
| GET | `/v1/devices/{id}/commands/pending` | device token or shared | Unexpired pending commands; they become `delivered`. |
| POST | `/v1/devices/{id}/commands/{cid}/result` | device token or shared | `{status: ok\|denied\|error, result?, message?}`. `409` once the command is final (e.g. `expired`). |

## Live tools (command queue)

```
phone_* CliTool ─ insert device_commands(pending) ─ POST {"type":"cmd"} → device's UnifiedPush endpoint (ntfy)
                                                                         ↓ wakes the phone
phone: GET …/commands/pending → run (actions: biometric confirm first) → POST …/commands/{cid}/result
CliTool polls the row → ok | denied | error | expired | delivered (answered too late)
```

- The push body never carries data, because ntfy topics may live on a shared
  server. The phone always pulls the command over the authenticated API.
- The phone declares what it will run in `capabilities`, and the server
  refuses anything else. Declare a capability only when its toggle and
  permission are on.
- `expires_at` = creation + `devices.command_timeout_s` (45 s). The phone must
  skip a command whose `expires_at` has passed, **including after a biometric
  prompt**, so a stale SMS is never sent.
- The phone should also drain `…/commands/pending` when the app starts and
  when the push distributor reconnects.

| tool | args | action (biometric) | result fields |
|------|------|--------------------|---------------|
| `location_now` | — | no | `lat`, `lon`, `accuracy_m`, `ts?`, `place?` |
| `status_now` | — | no | `battery_pct`, `charging`, `network`, `ringer` |
| `sms_send` | `to`, `body` | **yes**: show recipient and exact text | — |
| `notify` | `text`, `title?` | no | — |

A declined confirmation is `status: "denied"`. A failure is `status: "error"`
with a `message`.

### Event shape

```json
{"kind": "sms", "ts": 1764000000000, "dedup_key": "sms:4711",
 "payload": {"address": "+39…", "name": "Mario", "body": "…", "direction": "received"},
 "text": "optional searchable override"}
```

- `ts`: epoch millis, epoch seconds or ISO-8601.
- `dedup_key`: unique per device; re-sending is safe (counted as `duplicates`).
- Events whose kind is not in the device's `enabled_categories` are rejected,
  so a stale phone outbox cannot leak a category the user just turned off.

| kind | payload fields |
|------|----------------|
| `call` | `number`, `name`, `direction` (incoming/outgoing/missed/rejected), `duration_s` |
| `sms` | `address`, `name`, `body`, `direction` (received/sent) |
| `notification` | `package`, `app`, `title`, `text` |
| `location` | `lat`, `lon`, `accuracy_m`, `place?` |
| `status` | `battery_pct`, `charging`, `network`, `ringer` |

## Agent tools

- `phone_events --kind … --since 24h|7d|ISO --contact NAME_OR_NUMBER --query TEXT --limit N --device NAME`
- `phone_devices`
- `phone_location_now`, `phone_status_now`, `phone_notify --text … [--title …]`,
  `phone_sms_send --to NUMBER --body TEXT` (live; `--device NAME` when several phones)
- `knowledge_search --kind phone_sms|phone_notification|phone_call` for older content.

They are grouped in the `phone` skill. Voice turns with `access="chat"` get no
tools at all, so phone data is only reachable after the phone-side unlock.

## Coworker follow-ups (opt-in)

With `devices.coworker_kinds` set (for example `[sms, call]`) and
`coworker.enabled: true`, each **newly ingested** received SMS
(`payload.direction == "received"`) and missed call (`direction == "missed"`)
is passed to the coworker pipeline in a background task after the ingest
reply. Commitments and follow-ups land in `Pawn/Items` and `Pawn/Today.md`
like those from transcripts (`source_kind` `phone_sms` / `phone_call`,
`source_ref` `phone:<device>:<event>`). Duplicates re-sent by the outbox are
not forwarded again; sent SMS, answered or outgoing calls, and other kinds
never are. Default is `[]` (off).

## Retention and privacy

- Events older than `devices.retention_days` are pruned on ingest, together
  with their knowledge chunks (`source_ref = phone:<device>:<event>`).
- Turning a category off, or deleting the device, removes the server copy.
- Device tokens are separate from the shared API token and revocable per phone.
- Coworker Items extracted from phone events (see above) are vault notes and
  item rows of their own; pruning or purging the events does not delete them.
