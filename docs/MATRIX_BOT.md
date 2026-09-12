# Matrix bot

Inbound Matrix chatbot as a `pawn-server serve` worker. Agent turns run **in-process**
(same CliTools / skills / sallm memory as `pawn-agent chat`) — the bot does not call
the HTTP API.

## Install

```bash
uv sync --extra matrix
```

Optional E2EE support comes from `matrix-nio[e2e]`.

## Config

Add a `matrix_bot:` block to `pawnai.yaml` (see `pawnai.example.yaml`):

```yaml
matrix_bot:
  enabled: true
  homeserver_url: https://matrix.example.com
  user_id: "@pawn:example.com"
  user_token: "syt_..."       # prefer token; or set user_password
  device_id: PAWNBOT01        # keep stable with store_path
  device_name: pawn-matrix
  store_path: .matrix-store
  command_prefix: "!pawn"
  inviters:
    - "@you:example.com"
```

Env vars use `PAWN_MATRIX_BOT__*` (e.g. `PAWN_MATRIX_BOT__ENABLED=true`).

`queue_producers.matrix` is a separate outbound notification path for `queue_push`.
It is not used by this inbound worker.

## Run

```bash
# With API / queue / scheduler as configured
pawn-server serve

# Bot only (no HTTP)
pawn-server serve --matrix-only

# Force off even if enabled in YAML
pawn-server serve --no-matrix
```

`--matrix-only` and `--scheduler-only` cannot be combined.

## Chat behaviour

| Context | Behaviour |
|---------|-----------|
| DM (≤2 members) | Free-speak; every text message is a turn |
| Group room | Message must start with `command_prefix` |
| `/reset` | Clears sallm conversation for that room |
| Session key | `matrix:{room_id}` (chat memory, not diarization id) |

Discover diarization sessions with the `sessions_list` tool, same as CLI/API chat.

## Notes

- Reuse the same `device_id` and `store_path` across restarts; a new device id in
  encrypted rooms often causes silent drops until verification.
- Invites are accepted only from `inviters` when that list is non-empty.
- Runs are persisted in `agent_runs` with `source=matrix`.

## Device verification (Element red shield)

matrix-nio 0.26 has Element interop bugs (hex commitment, MAC v1/v2 mismatch).
The bot patches these on startup.

1. Dismiss any stuck Element verify dialog.
2. Restart `pawn-server serve --matrix-only` — log must include
   `Patched matrix-nio SAS for Element (commitment + MAC v2)`.
3. Element (as the bot) → Sessions → `pawn-matrix` → Verify.
4. When emojis appear (same as bot log), click **They match**.
5. Success log: `Marked device verified` and ideally
   `Device verification MAC exchanged`.
