# Voice assistants

pawn has two voice clients that work as one assistant:

- `pawnai-assistant` (Linux laptop): wake word, HUD, Piper/Kokoro voice.
- `pawnai-assistant-android` (phone): Speaker and Headphones modes, plus the
  phone bridge.

Both register in the same device registry (`devices`, see
[PHONE_BRIDGE.md](PHONE_BRIDGE.md)) and speak the same
[voice protocol](VOICE_PROTOCOL.md). This page covers what they share
through pawn-server. Diarization and recording sessions are out of scope here.

| Feature | What it does | Server pieces |
|---|---|---|
| Registry | Each assistant is a device with `kind: phone\|desktop`, a per-device token and `capabilities` | `devices`, `/v1/devices/*` |
| Presence | Each assistant reports `active / idle / away` plus flags (headset, on a call, in a session) | `POST /v1/devices/{id}/presence` |
| Commands | The agent (or the portal) runs `speak`, `notify` or `timer_set` on the assistant the user is near | `device_commands`, `…/commands/stream` |
| Hand-off | "continua qui" picks up the other device's conversation | `voice_threads`, `/v1/pawn/voice/active`, `…/claim` |
| Phone unlock | The laptop asks the phone for a biometric approval to get full access | `access_requests`, `/v1/pawn/voice/access-requests` |
| Wake word | Both assistants upload labelled wake clips; the laptop trains; every assistant downloads the active model | `wake_clips`, `wake_models`, `/v1/assistants/wake/*` |
| Portal | The Assistants tab manages all of this, and voice turns are mirrored as chats in the `Voice` folder | `/v1/portal/assistants` |

Migration `0030` adds the tables and the device columns.

## Setup

1. `alembic upgrade head`.
2. **Laptop:** in the portal, go to **Assistants → Register → Laptop**. Copy
   the snippet, or run `pawn-assistant device register --name laptop` on the
   laptop, which uses the shared token once. The laptop then keeps
   `/commands/stream` open while `pawn-assistant serve` runs.
3. **Phone:** in the app, go to **Settings → Pawn server → Register phone**.
   For live tools, connect UnifiedPush (ntfy). To let the phone approve the
   laptop, turn on **Approve other assistants**. For wake training, turn on
   **Share wake clips** (uploads only on Wi-Fi).

Optional `devices:` keys in `pawnai.yaml`:

```yaml
devices:
  presence_fresh_s: 600          # a presence report counts for --device auto
  handoff_window_s: 600          # how long an idle conversation can be taken over
  access_request_timeout_s: 60   # time the phone has to approve
  access_max_minutes: 30         # longest full-access grant
  wake_dir: ""                   # default {state_dir}/wake
  wake_clip_retention_days: 90
  wake_clip_max_bytes: 1000000
  wake_model_max_bytes: 20000000
  mirror_voice_to_portal: true
```

## Routing: `--device auto`

`resolve_device(…, "auto", tool=…)` (`pawn_server/core/device_commands.py`)
picks among the devices that offer the tool, in this order:

1. A fresh (`presence_fresh_s`) `active` presence.
2. A fresh `idle` presence, newest first.
3. The most recently seen device.

The tools pick a device like this:
- `assistant_speak` and `assistant_timer` default to `auto` over phones and
  desktops.
- `phone_*` tools only look at phones. With a single phone, no `--device` is
  needed.

## Agent tools

| tool | effect |
|---|---|
| `assistant_speak --text T [--device NAME\|auto]` | Says one sentence on the assistant |
| `assistant_timer --seconds N [--label L] [--device …]` | Starts a countdown that rings there |
| `phone_notify` | A notification on the phone (unchanged) |
| `phone_devices` | Lists every assistant with its kind, tools and presence |

They are part of the `phone` skill.

## Wake-word training loop

```
assistant wake → 2 s clip labelled confirmed / false_wake / barge / unsure / near_miss
  → POST /v1/assistants/wake/clips (device token, multipart: file, label, device_id, score, ts)
laptop: pawn-assistant wake-train pull     → clips into wakewords/hey_pawn/inbox/<label>/
        wake-train review / accept / train / calibrate (unchanged)
        wake-train publish --model … --notes … → POST /v1/assistants/wake/models (shared token)
assistants: GET /v1/assistants/wake/models/current → download when the sha256 changed
```

- Clips are WAV files up to 1 MB, stored under `wake_dir/clips/<label>/`.
  They are pruned after `wake_clip_retention_days`.
- Revoking a device keeps its clips (with `device_id` cleared), because they
  are training data. Delete them one by one with
  `DELETE /v1/assistants/wake/clips/{id}`.
- Models are versioned, and exactly one is `active`. The portal can publish
  or activate one.
- `GET /v1/assistants/wake/models/{version}` and `…/current` accept the
  device token (`?device_id=`).

## Portal mirror

Each voice turn appends the user message and the answer (with any `display`
text) to a portal chat whose `conversation_key` is the `voice:` key. The
chat sits in the `Voice` folder and is tagged `voice` plus the device name.
After a hand-off the same chat continues. Set
`devices.mirror_voice_to_portal: false` to turn this off. If the mirror
fails, the error is logged and the voice turn is unaffected.

## Portal: Assistants tab

Open it from **More → Assistants**, or press `g a`.

- **List:** each assistant with a presence dot (active, idle, away, offline)
  and when it was last seen. The **Wake word** row opens the clips and
  models.
- **Detail:**
  - Kind, platform, presence and its flags, and the command channel.
  - Tools it can run, and the categories it shares.
  - The active voice conversation, with **Open in chat**.
  - Full-access requests and recent commands.
  - Wake clips it uploaded.
  - **Send test** (speak, notify, timer, status, location), rename and
    revoke.
- **Register:** creates a laptop or phone and shows the token once, with a
  ready-to-paste Linux snippet.
