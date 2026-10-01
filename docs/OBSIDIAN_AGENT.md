# Obsidian vault agent loop

Pawn treats notes as plain Markdown under an S3 prefix. Obsidian (desktop +
mobile) syncs that prefix with **Sync Engine**. Pawn never imports Obsidian
APIs — only the plugin does.

Three ways to use Pawn from Obsidian:

1. **Pawn plugin — chat** (`obsidian-plugin/pawn/`): a Copilot-style side
   pane. Streams replies over `POST /v1/pawn/chat` with the active note,
   selection and `@`-mentioned notes as structured context.
2. **Pawn plugin — background jobs**: long work (agent tasks, note pushes,
   uploads) goes to `POST /v1/jobs`. It is accepted at once and reports back
   in the chat thread / Jobs tab. `Pawn/Tasks/*.md` notes mirror `ask` jobs
   for offline and mobile devices.
3. **Stock [obsidian-copilot](https://github.com/logancyang/obsidian-copilot)**
   pointed at Pawn's OpenAI-compatible endpoint (see below).

Session key for a note: `note:{path}` (e.g. `note:Projects/Roadmap.md`).
Free-standing plugin chats use `chat:{uuid}`.

## Vault layout

| Path | Owner | Purpose |
|------|-------|---------|
| `Pawn/Transcripts/{date} {session_id}.md` | Pawn | Speakers + Transcript rewritten on push; Annotations preserved |
| `Pawn/Analyses/{session_id}.md` | Pawn | `session_analyze --save` |
| `Pawn/Tasks/{id}.md` | Shared | Mirror of one `ask` job (status decides who may write) |
| `Pawn/Notes/…` | Pawn | Free-form agent notes / scheduled output / "Save as new note" |
| `Pawn/Items/{id}.md` | Pawn | One coworker inbox item. Set `action:` to file, task, later, or ignore |
| `Pawn/Threads/…` | Pawn | Filed items and open loops for one goal thread |
| `Pawn/Today.md` | Pawn | What needs a tap, then what was filed quietly |
| `Pawn/Daily/{date}.md` | Pawn | Morning briefing |
| `Pawn/Ideas/…` | Pawn | Companion notes for ideas you captured. Your original note is not edited |
| `Pawn/Reviews/{week}.md` | Pawn | Weekly review, including a proposed Goals.md. Chat drafts land in `Pawn/Reviews/goal-proposal.md` |
| `Ideas/…` | You | Idea notes. `/idea` asks Pawn to write the first skeleton |
| `Goals.md` | You | Active threads and attention rules. `/goal`, `/park`, and `/goal apply` are the chat commands that write it |
| `Pawn/Inbox/…` | Pawn | Non-audio files uploaded from the plugin |
| `Pawn/Commands/*.md` | You | Prompt commands for the plugin (`/`, palette, editor menu) |
| Everything else | You | Editable by Pawn only if frontmatter has `pawn: editable` |
| `.obsidian/` | You | Never touched by Pawn |

How to turn the coworker loop on, write `Goals.md`, and triage from the Inbox is in [COWORKER.md](COWORKER.md).

## Enable

In `pawnai.yaml`:

```yaml
api:
  token: "…"
  host: "0.0.0.0"
  port: 8000
  # Defaults shown; Obsidian desktop / mobile origins for CORS.
  cors_origins: ["app://obsidian.md", "capacitor://localhost", "http://localhost"]
  include_system_prompt: false    # prepend obsidian-copilot's system prompt
  stream_progress: true           # tool steps as reasoning_content deltas
  stream_keepalive_seconds: 10
  upload_audio_target: diarize    # queue_producers target for audio uploads
  upload_s3_prefix: uploads/obsidian

# Audio uploads: staged in the main s3: bucket, then published to the
# diarize listener's topic (transcribe-diarize).
queue_producers:
  diarize:
    topic: audio-chunks
    bucket_name: my-bucket

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
| Asymmetric storage | **Off** (mandatory for Pawn) |
| Client-side encryption | **Off** (mandatory for Pawn) |
| Prefix | Same as `vault.s3.prefix` |
| Sync strategy | Bidirectional |
| Conflict strategy | Smart merge or keep both |
| Interval / startup sync | **On** (fallback when the plugin is closed) |

When the Pawn plugin is open and **Resync when the agent writes** is on
(the default), it long-polls `GET /v1/vault/events`. After an agent turn
writes notes (`note_write`, `note_append`, `task_update`, or
`session_analyze --save`), or an ask job updates its task note, the plugin
runs Sync Engine's **Start non-interactive sync** command so this device
pulls those keys without waiting for the interval. Interval and startup sync
stay on: they cover a closed plugin, Sync Engine disabled, and writes that
are not part of an agent turn (diarize `push-vault`, coworker notes written
outside a tool). The long-poll is process-local, same as job events.

Pawn writes ordinary vault paths (`Pawn/Tasks/…`, diary notes, analyses). Sync
Engine **asymmetric storage** flattens remotes to keys like `00000~Welcome.md`
and rejects mixed layouts (error along the lines of *files at remote that don't
adopt asymmetric storage naming*). **Encryption** likewise rewrites keys/bodies
so Pawn cannot read or co-author. Both must stay off on every device that shares
the vault bucket.

If asymmetric was already on: turn it off in Sync Engine settings (let it
migrate), confirm the bucket no longer contains `NNNNN~…` flat keys, then sync.
`VaultStore` refuses writes while root keys like `00000~…` exist (logged as an
error; tools report it instead of claiming a save that Obsidian never sees).
Only after that should `pawn-diarize push-vault` / agent `note_write` run.

Turn on **bucket versioning** on the vault bucket as your undo.

## Install the plugin

Download `pawn.zip` from the GitHub release tagged `obsidian-plugin-v…`
and unzip it into `<vault>/.obsidian/plugins/`. The archive contains a
`pawn/` folder. Enable **Pawn** in Obsidian (Restricted mode off). Set
Server URL + API token to match `api.*`.

From a checkout, `make obsidian-plugin-dist` writes the same zip.
Symlinking the source tree still works after `npm run build`:

```bash
cd obsidian-plugin/pawn
npm install && npm run build
ln -s /path/to/parakeet/obsidian-plugin/pawn \
  /path/to/vault/.obsidian/plugins/pawn
```

See [obsidian-plugin/pawn/README.md](../obsidian-plugin/pawn/README.md) for
the release tag, BRAT, and the full feature list.

## Chat (plugin)

- Right-side pane (ribbon icon or **Pawn: Open chat**). The conversation
  follows the active note (`note:{path}`) by default; pick another note, a
  recent chat, or **New chat** (`chat:{uuid}`) from the header.
- Context chips above the composer: the open note (× unpins that file; a
  dashed pin chip attaches it again), current selection, and extra notes via
  `@`, the `+` chip, or drag-and-drop from the file explorer. With **Send
  local note content** on, the plugin sends note bodies (including unsynced
  edits); otherwise the server reads the vault.
- When the selected model is flagged `vision: true`, dropping or pasting a
  png, jpeg, gif, or webp into the composer attaches it to the next message.
  Images embedded in attached
  notes are sent too (**Include images from attached notes**, on by default),
  including an Excalidraw drawing (`![[name.excalidraw]]`) and an open
  Excalidraw note. The plugin exports the current drawing as a PNG on this
  device, so the diagram is included even when note text is read from the
  vault bucket. Each picture is captioned once in that conversation. A
  changed drawing is exported and captioned again. The refresh button
  beside the paperclip (or `/vision refresh`) reads the same bytes again
  on the next message. Other files still go to the Inbox
  upload job. A page with several images captions up to four per message, in
  document order, and later messages pick up the ones not captioned yet. The
  model sees pixels for one of them; the others stay searchable by caption.
  Drawings are sent after the other pictures so the model sees the diagram.
- Replies stream: tool steps show live ("Ran note_read"), then the answer
  renders as Markdown with **Copy / Insert at cursor / Replace selection
  (diff preview) / Append to note / Save as new note**.
- `/` opens prompt commands (built-ins plus `Pawn/Commands/*.md`) and
  `/reset`, `/new`, `/bg`, `/jobs`. The same commands are in the palette and
  editor menu.
- sallm owns conversation memory; the plugin keeps only a local transcript
  cache. `/reset` clears both.

## Background jobs

Every job is accepted immediately (202) and runs as an asyncio task inside
`pawn-server`. The row lives in `vault_tasks` (`kind` = `ask` | `push_note` |
`upload`). There is no timeout race and no split between a fast and a slow path.

| Kind | Started from | What happens |
|------|--------------|--------------|
| `ask` | "Background" toggle in chat, **Send to Pawn (background)** | Agent turn with note/selection/context; result → job + `Pawn/Tasks/{id}.md` |
| `push_note` | API (`POST /v1/jobs`) | Write/append a vault note (vault guards apply) |
| `upload` | Paperclip or drag of non-image files, file menu **Upload to Pawn** | Audio → staged in `s3:` + `transcribe-diarize` on `api.upload_audio_target`; other files → `Pawn/Inbox/` (text files indexed into memory). png/jpeg/gif/webp dropped or pasted on a vision model stay on the chat message |

The plugin follows `GET /v1/jobs/events` (SSE) on desktop and polls
`GET /v1/jobs` on mobile. Finished jobs raise a notice, update their card in
the chat thread and in the **Jobs** tab, and offer Insert / Replace /
**Approve** (index into sallm memory) / **Dismiss** (close without indexing) /
Cancel / Open task note.

**Offline / mobile fallback:** if the server is unreachable, the plugin writes
`Pawn/Tasks/{id}.md` with `status: todo`. Sync Engine uploads it, the vault
watcher runs it, writes `## Result` + `status: review` back to S3, and the
plugin picks up the synced note. Approving an offline job sets
`approved: true` in the note; the watcher indexes it. Dismissing (or setting
`status: done` with `approved: false`) closes the job without indexing.

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

## Use with obsidian-copilot

[obsidian-copilot](https://github.com/logancyang/obsidian-copilot) (AGPL-3.0)
can talk to Pawn as a custom model with no code changes. Pawn only serves
HTTP, so this does not affect Pawn's license. The Pawn plugin is a separate,
independently written MIT implementation.

1. Copilot settings → **Model** → **Add custom model**:
   - Provider: the OpenAI-compatible option (labelled *OpenAI Format* /
     *3rd party (openai-format)* depending on the Copilot version)
   - Base URL: `http://<pawn-host>:8000/v1`
   - API key: `api.token` from `pawnai.yaml` (any string if the server is open)
   - Model name: `pawn-agent`
   - Enable **CORS** in Copilot if the request is blocked (Pawn also allows
     the Obsidian origins in `api.cors_origins`).
2. Verify the model, then select it for chat.
3. Keep Copilot's embeddings / vault QA on another provider (or off). Pawn
   does not serve `/v1/embeddings`; its own memory and `note_search` cover
   retrieval.

Behaviour notes:

- sallm keeps history per session, so Pawn uses only the last user message
  (Copilot inlines note context into it). Set `api.include_system_prompt:
  true` to also pass Copilot's system prompt.
- Copilot sends no `user` field, so each Copilot chat maps to a session keyed by a
  hash of its first message. Other clients can set `user` or the
  `X-Pawn-Conversation` header.
- Streaming sends keep-alive comments while the agent works and tool steps as
  `reasoning_content`, then the answer. Set `api.stream_progress: false` if a
  client renders reasoning oddly.

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat (Copilot, any client) |
| GET | `/v1/models` | Lists `pawn-agent` |
| POST | `/v1/pawn/chat` | Plugin chat, SSE events `progress` / `answer` / `job` / `error` / `done` |
| POST | `/v1/jobs` | Start `ask` or `push_note` job (202) |
| POST | `/v1/jobs/upload` | Multipart upload job (202) |
| GET | `/v1/jobs?conversation=&status=&limit=` | List jobs, newest first |
| GET | `/v1/jobs/{id}` | One job |
| POST | `/v1/jobs/{id}/approve` | Index an `ask` result into memory |
| POST | `/v1/jobs/{id}/cancel` | Cancel a running job |
| POST | `/v1/jobs/{id}/dismiss` | Close a review `ask` job without indexing |
| GET | `/v1/jobs/events` | SSE `job` events (this server process only) |
| GET | `/v1/vault/events?since=&timeout=` | Long-poll vault writes from agent turns (this process only) |
| POST/GET | `/v1/vault/tasks…` | Deprecated aliases (always 202 now) |

All require the same Bearer token.

`/v1/pawn/chat` body:

```json
{
  "conversation": "note:Projects/Roadmap.md",
  "message": "What changed since last week?",
  "active_note": {"path": "Projects/Roadmap.md", "content": "…optional…"},
  "selection": "…optional…",
  "context": [{"path": "Meetings/2026-09-20.md"}],
  "background": false
}
```

Jobs run inside the server process. A restart abandons running jobs (they
stay `running`); the task note keeps the instruction, so resubmit if needed.

## Agent tools

`note_read`, `note_search`, `note_write`, `note_append`, `task_update`.
Skill `vault_tasks` for watcher/HTTP vault runs; `notes` for save/analyze.
Skill `tasknotes` for TaskNotes tasks, pick-lists, and boards.

## TaskNotes

Ask Pawn to pull action items out of a chat or the latest diarization
sessions. Two ways:

- "List the action items and let me pick." Pawn writes
  `Pawn/TaskNotes/Proposals/{stamp}.md` and stops. Uncheck a line, or edit
  the assignee, due, scheduled, or project on that line, then ask it to
  create the checked tasks. You can also reply with the numbers.
- "Turn this into tasks." Pawn creates the notes immediately.

Each task is `Pawn/TaskNotes/Tasks/{title}.md` with the `task` tag, so it
shows on the TaskNotes kanban and calendar after Sync Engine delivers it.
"Edo's board" is that person's task notes plus `Pawn/TaskNotes/Views/Edo.base`,
not a checklist page at `Pawn/Boards/Edo's Board.md`. A heading in the source
becomes the project. The short action is the task title.
`due` is a date. `scheduled` is that date, or a local time such as
`2026-09-30T09:00` with no timezone suffix. Empty dates stay off the calendar.

A person board is `Pawn/TaskNotes/Views/{Name}.base` (kanban and calendar).
A project board is `{Project} project.base`. Open the `.base` file in
Obsidian. Bases has to be enabled (Obsidian 1.10.1+), and TaskNotes has to
be installed with task identification left on the `task` tag. Add a user
field named `assignee` if you want that column in the task modal.

Tasks you create from the TaskNotes UI stay in `TaskNotes/Tasks/`. Pawn
lists them and will not rewrite them. The same title and assignee are
treated as one task, so a later extraction does not clone the card.

Google or Outlook updates only when TaskNotes export is already enabled,
with the sync trigger set to `scheduled`. Pawn does not hold those tokens.

`tasknotes.display_name` is how "me" is written. When it is empty, the
first name in `tasknotes.me` or `coworker.me` is used. `tasknotes.timezone`
empty means `coworker.timezone`.

## Diarize CLI

```bash
pawn-diarize push-vault --latest
pawn-diarize push-vault --session myconv
pawn-diarize session-relabel --session myconv -F SPEAKER_00 -T Davide --yes --push-vault
```
