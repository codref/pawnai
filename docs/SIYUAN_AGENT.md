# SiYuan @pawn agent loop

Pull-only workflow: Pawn polls SiYuan for `@pawn` instructions, runs the
agent, appends a reviewable draft under the parent block, alerts Matrix, and
indexes approved results into sallm memory.

SiYuan **cannot** reach pawn-server in the current deployment. There is no
plugin webhook in this phase — discovery is SQL polling only.

## Enable

In `pawnai.yaml`:

```yaml
siyuan:
  url: http://127.0.0.1:6806
  token: "…"
  notebook: "YOUR_NOTEBOOK_ID"

siyuan_watcher:
  enabled: true
  poll_interval_seconds: 10
  settle_seconds: 45
  mention_token: "@pawn"
  # empty → only siyuan.notebook
  notebook_allowlist: []
  max_ref_depth: 1
  max_context_blocks: 40
  matrix_target: matrix
  max_claims_per_tick: 3

matrix_bot:
  enabled: true
  notify_room_id: "!room:server"   # outbound ready-for-review alerts
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

Run with the watcher (and Matrix for alerts):

```bash
pawn-server serve
# or
pawn-server serve --siyuan-watcher-only
```

Flags: `--no-siyuan-watcher`, `--siyuan-watcher-only`.

## Human UX

1. Under any block in an allowlisted notebook, write a child paragraph:

   ```markdown
   @pawn Deep-analyze the linked notes. Highlight tasks and decisions.
   Use ((20260920113000-abc1234 "context")).
   ```

2. Finish typing, then leave the block alone for ``settle_seconds`` (default
   **45s**). SiYuan autosaves mid-edit; the watcher only claims after the
   instruction text has stopped changing for that long.

3. On claim, the trigger is rewritten as a SiYuan **TIP** callout (needs 3.5+):
   title from a short excerpt of the prompt, icon 🤖, original `@pawn` body kept.

4. A **Pawn result — ready for review** section is appended under the parent,
   including:

   - `[ ] Approve for Pawn memory`
   - `[ ] Request changes (reply with @pawn …)`

5. Matrix receives a short alert with a `siyuan://blocks/…` deep link
   (no full note body).

6. Check **Approve for Pawn memory** in SiYuan. On the next tick the watcher
   marks `done` and calls `Agent.remember` on the approved content.

7. To refine later, add another `@pawn …` under the same thread (new request).

First watcher start **bootstraps** a poll watermark so historical `@pawn`
strings are not claimed. Only blocks updated after bootstrap are processed.

## Attributes

| Attr | Purpose |
|------|---------|
| `custom-agent-status` | `queued\|claimed\|running\|review\|done\|blocked\|cancelled` |
| `custom-agent-request-id` | UUID |
| `custom-agent-output-id` | Result block id |
| `custom-agent-source-hash` | Instruction hash (idempotency) |
| `custom-agent-run-id` | `agent_runs.id` |

## Agent tools / skill

CliTools: `siyuan_read`, `siyuan_append`, `siyuan_set_status` (plus session tools
via the `siyuan_tasks` skill). Append-only — do not use `siyuan_save`
delete-recreate for this path.

## Later stages

- SiYuan plugin + `POST /v1/siyuan/triggers` when SiYuan can reach Pawn
- Autonomous proposals on `session.completed` / cron scans
- Email and other imported docs as allowlisted context
