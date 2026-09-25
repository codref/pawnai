# Obsidian vault agent loop

Pawn treats notes as plain Markdown under an S3 prefix. Obsidian (desktop +
mobile) syncs that prefix with **Sync Engine**. The Obsidian **Pawn** plugin
creates task notes and optionally calls `pawn-server` over HTTP. Pawn never
imports Obsidian APIs — only the plugin does.

Session key for a note: `note:{path}` (e.g. `note:Projects/Roadmap.md`).

## Vault layout

| Path | Owner | Purpose |
|------|-------|---------|
| `Pawn/Transcripts/{date} {session_id}.md` | Pawn | Speakers + Transcript rewritten on push; Annotations preserved |
| `Pawn/Analyses/{session_id}.md` | Pawn | `session_analyze --save` |
| `Pawn/Tasks/{id}.md` | Shared | One agent task per note (status decides who may write) |
| `Pawn/Notes/…` | Pawn | Free-form agent notes / scheduled output |
| Everything else | You | Editable by Pawn only if frontmatter has `pawn: editable` |
| `.obsidian/` | You | Never touched by Pawn |

## Enable

In `pawnai.yaml`:

```yaml
api:
  token: "…"
  host: "0.0.0.0"
  port: 8000

vault:
  s3:
    bucket: my-obsidian-bucket
    endpoint_url: https://fsn1.your-objectstorage.com
    access_key: "…"
    secret_key: "…"
    region: fsn1
    prefix: ""                 # same prefix Sync Engine uses (bucket root OK)
    path_style: true
  agent_root: Pawn
  auto_push_transcript: true
  obsidian_vault_name: "MyVault"   # for Matrix obsidian:// links

vault_watcher:
  enabled: true
  poll_interval_seconds: 15
  max_claims_per_tick: 3
  matrix_target: matrix

matrix_bot:
  enabled: true
  notify_room_id: "!room:server"
```

Apply migrations:

```bash
alembic upgrade head
```

Run:

```bash
pawn-server serve
# or
pawn-server serve --vault-watcher-only
```

Flags: `--no-vault-watcher`, `--vault-watcher-only`.

## Sync Engine (Obsidian)

Required settings on every device:

| Setting | Value |
|---------|-------|
| Asymmetric storage | **Off** |
| Client-side encryption | **Off** |
| Prefix | Same as `vault.s3.prefix` |
| Sync strategy | Bidirectional |
| Conflict strategy | Smart merge or keep both |
| Interval / startup sync | **On** (this is how devices see Pawn's S3 writes) |

Turn on **bucket versioning** on the vault bucket as your undo.

## Install the plugin

```bash
cd obsidian-plugin/pawn
npm install && npm run build
ln -s /path/to/parakeet/obsidian-plugin/pawn \
  /path/to/vault/.obsidian/plugins/pawn
```

Enable **Pawn** in Obsidian. Set Server URL + API token to match `api.*`.

## Interaction

1. **Ask Pawn** — creates `Pawn/Tasks/<uuid>.md` (`status: todo`), inserts a
   `> [!pawn]` callout, then tries `POST /v1/vault/tasks`.
   - Fast path (server reachable): plugin writes `## Result` and
     `status: review` locally; server only records the DB row.
   - Slow path: task stays `todo`, syncs to S3, vault watcher runs it and
     writes the result to S3; devices pick it up on the next interval sync.
2. **Pawn panel** — Tasks for the active note (Insert / Replace / Reply /
   Approve / Open) and Chat (`/v1/chat/completions` with `user=note:{path}`).
3. **Approve** — sets `approved: true`. HTTP `POST /v1/vault/tasks/{id}/approve`
   when online; otherwise the watcher indexes into sallm memory (`remember`).
4. **Transcripts** — `pawn-diarize push-vault` (or `vault.auto_push_transcript`)
   writes Speakers + Transcript notes under `Pawn/Transcripts/`.

### Task note format

```markdown
---
pawn: task
id: 7f3c2a10-…
status: todo            # todo | running | review | done | blocked
note: "[[Projects/Roadmap]]"
conversation: note:Projects/Roadmap.md
approved: false
---
## Instruction
…
## Context
…
## Result
…
```

While `todo` / `running`, the server owns the file. At `review`, you own it.

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/vault/tasks` | Fast-path run (200) or 202 accepted |
| GET | `/v1/vault/tasks/{id}` | Status |
| POST | `/v1/vault/tasks/{id}/approve` | Index into memory |

All require the same Bearer token as chat completions.

## Agent tools

`note_read`, `note_search`, `note_write`, `note_append`, `task_update`.
Skill `vault_tasks` for watcher/HTTP vault runs; `notes` for save/analyze.

## Diarize CLI

```bash
pawn-diarize push-vault --latest
pawn-diarize push-vault --session myconv
pawn-diarize session-relabel --session myconv -F SPEAKER_00 -T Davide --yes --push-vault
```
