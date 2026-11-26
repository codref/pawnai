# Phone bridge

pawn-assistant (the Android app) can share opt-in context from the phone with
pawn-server, and pawn-agent can read it through tools. Every category is off
by default and is switched on per phone in the app (Settings → Phone bridge).

Status: **push half implemented** (registration, ingest, purge, read tools).
Live phone tools over UnifiedPush/ntfy (`phone_location_now`, `phone_sms_send`,
…) are the next step; the `device_commands` table already exists for them.

## Data flow

```
phone collectors → outbox → POST /v1/devices/{id}/events → phone_events
                                                        └→ knowledge_chunks (phone_<kind>)
pawn-agent ← phone_events / phone_devices / knowledge_search --kind phone_sms
```

## Setup

1. `alembic upgrade head` (migration `0024`).
2. Optional `devices:` section in `pawnai.yaml` (see `pawnai.example.yaml`).
3. In the app: set the server URL + API token, open **Phone bridge**,
   **Register this phone**, then enable the categories to share.

## API

| Method | Path | Auth | Notes |
|--------|------|------|-------|
| POST | `/v1/devices/register` | shared API token | Body `{name}` → `{device_id, device_token, name}`. The token is shown once; only its SHA-256 is stored. |
| GET | `/v1/devices` | shared API token | List devices. |
| GET | `/v1/devices/{id}` | device token or shared | Device view. |
| PUT | `/v1/devices/{id}` | device token or shared | `{name?, unifiedpush_endpoint?, enabled_categories?}`. Removing a category **purges** the server copy of it (`purged_categories` in the reply). |
| DELETE | `/v1/devices/{id}` | device token or shared | Revokes the device and deletes its events and knowledge chunks. |
| POST | `/v1/devices/{id}/events` | device token or shared | `{events:[…]}` (≤ `devices.max_batch`) → `{accepted, duplicates, rejected}`. |

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
- `knowledge_search --kind phone_sms|phone_notification|phone_call` for older content.

They are grouped in the `phone` skill. Voice turns with `access="chat"` get no
tools at all, so phone data is only reachable after the phone-side unlock.

## Retention and privacy

- Events older than `devices.retention_days` are pruned on ingest, together
  with their knowledge chunks (`source_ref = phone:<device>:<event>`).
- Turning a category off, or deleting the device, removes the server copy.
- Device tokens are separate from the shared API token and revocable per phone.
