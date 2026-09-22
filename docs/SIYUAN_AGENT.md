# SiYuan @pawn agent loop

Plugin-first workflow: type `@pawn` in SiYuan, the **Pawn** plugin wraps the
block in a TIP callout, you press **Send to Pawn**, and pawn-server runs the
agent. A reviewable draft is appended under the parent; Matrix can alert; an
Approve checkbox indexes the result into sallm memory.

The SiYuan **document** is the agent session (`siyuan:{root_id}`). The callout
body (all child blocks) is the instruction for that turn; parent excerpt,
explicit `((block refs))`, and `siyuan_read` / session tools supply extra
context — same model as the old SQL watcher path.

## Enable

In `pawnai.yaml`:

```yaml
api:
  token: "…"          # plugin Bearer token
  host: "0.0.0.0"
  port: 8000

siyuan:
  url: http://127.0.0.1:6806
  token: "…"
  notebook: "YOUR_NOTEBOOK_ID"

siyuan_watcher:
  enabled: true
  # Plugin/API enqueues; watcher claims leftovers + polls approvals.
  discover_mentions: false
  poll_interval_seconds: 10
  settle_seconds: 45          # only used when discover_mentions: true
  mention_token: "@pawn"
  notebook_allowlist: []      # empty → only siyuan.notebook
  max_ref_depth: 1
  max_context_blocks: 40
  matrix_target: matrix
  max_claims_per_tick: 3

matrix_bot:
  enabled: true
  notify_room_id: "!room:server"
  # … homeserver / token / device as usual …

queue_producers:
  matrix:
    topic: matrix-bot-notifications
    bucket_name: …
```

Apply migrations:

```bash
alembic upgrade head
```

Run with the API + watcher (and Matrix for alerts):

```bash
pawn-server serve
# or
pawn-server serve --siyuan-watcher-only   # approvals only; no HTTP trigger
```

Flags: `--no-siyuan-watcher`, `--siyuan-watcher-only`.

Set `discover_mentions: true` only if you want the legacy SQL poll that
auto-claims `@pawn` after `settle_seconds` (races the plugin button).

## Install the plugin

```bash
ln -s /path/to/parakeet/siyuan-plugin/pawn \
  ~/SiYuan/data/plugins/pawn
```

Enable **Pawn** in SiYuan, then set:

| Setting | Typical value |
|---------|----------------|
| Server URL | `http://127.0.0.1:8000` (reachable from the **kernel**) |
| API token | same as `api.token` |
| Mention token | `@pawn` (must match `siyuan_watcher.mention_token`) |

See [siyuan-plugin/pawn/README.md](../siyuan-plugin/pawn/README.md).

## Human UX

1. Under any block in an allowlisted notebook, write an instruction (plain
   text is fine). Press **Send to Pawn** from the floating toolbar, block
   gutter menu, or ⌥⌘P. With wrap-on-send the paragraph becomes a **TIP**
   callout (SiYuan 3.5+). Add more child blocks inside it afterward — the
   **entire callout** is the request.

   Optional: type a leading `@pawn …` instead. That is only required for the
   legacy watcher SQL scan (`discover_mentions: true`); the plugin button does
   not need the mention.

2. Send posts `{ "block_id": "<callout>" }` to `POST /v1/siyuan/triggers`
   (via SiYuan `forwardProxy`).

3. A **Pawn result — ready for review** section is appended under the parent,
   including:

   - `[ ] Approve for Pawn memory`
   - `[ ] Request changes (reply with @pawn …)`

4. Matrix receives a short alert with a `siyuan://blocks/…` deep link
   (no full note body).

5. Check **Approve for Pawn memory** in SiYuan. On the next watcher tick the
   request is marked `done` and `Agent.remember` indexes the approved content.

6. To refine later, edit the callout (new instruction hash) and Send again, or
   add another `@pawn …` under the same thread (watcher path) / Send another
   block.

## API

```http
POST /v1/siyuan/triggers
Authorization: Bearer <api.token>
Content-Type: application/json

{ "block_id": "20260922120000-xxxxxxx" }
```

Response `202`:

```json
{
  "request_id": "…",
  "status": "claimed",
  "conversation_id": "siyuan:<root_id>",
  "trigger_block_id": "…",
  "started": true
}
```

The server resolves `block_id` to the nearest TIP callout (or plain mention),
reads full callout kramdown as `instruction_text`, upserts
`siyuan_agent_requests`, and runs the agent in the background. Same
trigger + hash while `queued` / `claimed` / `running` / `review` is idempotent
(`started: false`).

## Attributes

| Attr | Purpose |
|------|---------|
| `custom-agent-status` | `draft\|queued\|claimed\|running\|review\|done\|blocked\|cancelled` |
| `custom-agent-request-id` | UUID |
| `custom-agent-output-id` | Result block id |
| `custom-agent-source-hash` | Instruction hash (idempotency) |
| `custom-agent-run-id` | `agent_runs.id` |

Attrs live on the **callout root**.

## Agent tools / skill

CliTools: `siyuan_read`, `siyuan_append`, `siyuan_set_status` (plus session tools
via the `siyuan_tasks` skill). Append-only — do not use `siyuan_save`
delete-recreate for this path.

## Later stages

- Autonomous proposals on `session.completed` / cron scans
- Email and other imported docs as allowlisted context
