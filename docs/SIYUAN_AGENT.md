# SiYuan @pawn agent loop

Plugin-first workflow: insert a **Pawn prompt** block in SiYuan, press **Send
to Pawn**, and pawn-server runs the agent. A reviewable draft is appended under
the parent; Matrix can alert; an Approve checkbox indexes the result into sallm
memory.

The SiYuan **document** is the agent session (`siyuan:{root_id}`). The prompt
body is the instruction for that turn; parent excerpt,
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

`siyuan_watcher.mention_token` still applies to the optional SQL discovery
path (`discover_mentions`). The plugin does not require `@pawn` inside a
prompt block.

See [siyuan-plugin/pawn/README.md](../siyuan-plugin/pawn/README.md).

## Human UX

1. In an allowlisted notebook, type `/prompt` (or `/pawn`) and choose **Pawn
   prompt**. That inserts an empty `;;;pawn/prompt` block and does not send.
   Write the instruction in the block:

   ```markdown
   Deep-analyze the linked notes. Highlight tasks and decisions.
   Use ((20260920113000-abc1234 "context")).
   ```

2. **Send to Pawn** (floating toolbar paper-plane, block gutter menu, or ⌥⌘P).
   If the selection is already a Pawn prompt, that block is posted. Otherwise
   Send replaces the selected blocks with one prompt containing their text,
   then posts it. The prompt is one text region (a left border, no title or
   icon), not a container of child blocks.

3. Send posts `{ "block_id": "<prompt>" }` as JSON to `POST /v1/siyuan/triggers`
   (via SiYuan `forwardProxy`, `payloadEncoding: "json"`).

4. A **Pawn result — ready for review** section is appended under the parent,
   including:

   - `[ ] Approve for Pawn memory`
   - `[ ] Request changes (reply with @pawn …)`

5. Matrix receives a short alert with a `siyuan://blocks/…` deep link
   (no full note body).

6. Check **Approve for Pawn memory** in SiYuan. On the next watcher tick the
   request is marked `done` and `Agent.remember` indexes the approved content.

7. To refine later, edit the prompt (new instruction hash) and Send again, or
   insert another prompt under the same document.

Existing TIP callouts that contain `@pawn`, and paragraphs that start with
`@pawn`, still resolve. The plugin no longer creates TIP callouts.

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

The server resolves `block_id` to the nearest `;;;pawn/prompt` block and uses
the fence body as `instruction_text`. If none is found it falls back to a TIP
callout that contains `@pawn`, then to a block that starts with `@pawn`. It
upserts `siyuan_agent_requests` and runs the agent in the background. Same
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

Attrs live on the **prompt block** (or the legacy callout / mention block).

## Agent tools / skill

CliTools: `siyuan_read`, `siyuan_append`, `siyuan_set_status` (plus session tools
via the `siyuan_tasks` skill). Append-only — do not use `siyuan_save`
delete-recreate for this path.

## Later stages

- Autonomous proposals on `session.completed` / cron scans
- Email and other imported docs as allowlisted context
