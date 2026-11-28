# Pawn Portal

The Pawn Portal is the chat-first web UI for Pawn. It replaces the Obsidian plugin as the main way to talk to the agent, browse the vault and triage coworker output. The Obsidian plugin is frozen: it gets no new features and will be removed once nobody depends on it.

- **Frontend:** `portal/` (Vue 3 + Quasar + TypeScript, Pinia stores, Vite build).
- **Backend:** routers under `/v1/portal/*` in `pawn_server/core/portal_api.py`, `inbox_api.py` and `settings_api.py`, mounted by `pawn_server/core/api_server.py`.
- **Storage:** PostgreSQL `portal_*` tables (migrations `0026_portal`, `0027_portal_index`; models in `pawn_server/core/portal_models.py`). The agent's own memory stays in sallm.

## Running it

```bash
alembic upgrade head               # creates the portal_* tables
make portal                        # vue-tsc + Vite build into portal/dist
pawn-server serve                  # open http://localhost:8000/portal
```

pawn-server serves the build at `/portal` from `api.portal_dir` (default `portal/dist`). Until you build it, `/portal` is simply not mounted. The portal signs in with the same `api.token` bearer token as the rest of the API, and asks for it on first load.

For development:

```bash
make portal-dev                    # Vite on :5173/portal/, proxies /v1 to :8000
VITE_PAWN_MOCK=1 make portal-dev   # in-browser mock API (portal/src/api/mock.ts), no backend
make portal-test                   # vitest + jsdom (portal/src/**/*.test.ts)
```

## Layout

- **Left sidebar: chats.** Saved chats in nested folders, with hierarchical tags (`work/acme/q4`), pin, archive and sorting by updated, created, name or tag. Search covers chats, messages, clips, annotations, vault files and diarization sessions (title and summary). `#tag` in a query filters by tag. A session hit opens the session tab.
- **Right sidebar: files.** The whole vault bucket, with ages and tags. The scratchpad (`portal.scratch_dir`, default `{agent_root}/Scratch`) is always writable. Upload, new note, move and delete follow the vault write guards (free under `Pawn/`, elsewhere only with `pawn: editable`, never `.obsidian/`).
- **Center: workspace.** VS Code-style tabs for chats, files, clips, Inbox, Settings, Sessions, one session, Day, Speakers and Assistants. Groups can be split side by side or floated as windows. A single-click opens an italic preview tab, which is pinned when you send, edit or double-click. The layout is saved through `GET/PUT /v1/portal/workspace`. An empty group shows Today: the next events, sessions that still need a speaker name, and Plan my day.
- **Header.** Day, Sessions (badge = sessions that still need a speaker) and Inbox. Speakers, Assistants and Settings sit in the overflow menu. `g` then `d`, `s`, `i`, `p` or `a` opens Day, Sessions, Inbox, Speakers or Assistants.
- **Save to.** The chip under the files header names the folder that New note and Upload write to. Clicking a folder in the tree targets it (marked with a bar); the chip's menu resets to the scratchpad or picks a recent folder. Remembered per browser.
- **Drag and drop.**
  - Chats: drop a chat or folder on any row of a folder (a folder header or one of its chats) to move it there. Empty space below the tree means the top level (Unfiled for chats). Drop a chat on a tag in the Tags view to add that tag. Drop a vault file on a folder to start a chat there with the file attached. Collapsed folders open after a short hover.
  - Files: drop a vault file on a folder to move it; it asks before replacing a file of the same name. Folders can't be dragged, because the move API handles single files only. Drop files from your desktop on a folder to upload them there, or anywhere else in the sidebar to upload to the Save to folder.
  - Workspace: drop a chat or vault file on a tab group to open it there; drop on an edge to split.

## Chats

Each portal chat (`portal_chats`) owns one sallm conversation key, `chat:{id}` for chats created in the portal. Messages are posted to `POST /v1/portal/chats/{id}/messages`, which streams the same SSE events as `/v1/pawn/chat` (`progress`, `answer`, `job`, `error`, `done`, plus `chat` when the chat is renamed).

The portal keeps its own transcript mirror, `portal_messages`. It gives every message a stable id for clips, annotations and forks, and it stores the raw user text with the context (notes, selection, images, origin, written files) in a separate `context` field instead of inside the prompt. The agent never reads the mirror; sallm remains its memory.

- **Auto-naming.** After the first answer, and again every `portal.auto_name_every` messages, the background model (`agent.default`) titles and tags the chat. It stops once you edit the title or tags (`title_auto` / `tags_auto`). Turn it off with `portal.auto_name: false`.
- **Forks.** "Fork from here" creates a new chat seeded with the transcript up to that message (`pawn_agent/core/sallm_fork.py`). Only user and assistant turns are copied. Skill frames reset and Lance memory chunks are not copied.
- **Side chats.** "Ask about this" on any text opens a new chat whose `origin` records what it was spawned from (`{kind, ref, quote, anchor}`). Its first prompt is grounded in that quote, file or item.
- **Clips.** A few messages, or a selection across them, saved as a `portal_clips` row. A clip can be saved to the scratchpad as a Markdown note.

## Right-click layer

Any element marked `v-target` gets a shared context menu (`portal/src/lib/actions.ts`): Ask about this, Quote in chat or Add to chat context, Open in a new tab, to the side or in a floating window, Fork from here, Save as clip, Annotate, and Copy text, link or path. Panels add their own actions, for example the triage verbs on Inbox items.

## Annotations

A highlight plus a note on any message, file, clip or item (`portal_annotations`, `/v1/portal/annotations`). The anchor is a text-quote selector `{exact, prefix, suffix}` with a `{start, end}` offset hint, computed in `lib/targets.ts`. On every render `lib/anchoring.ts` re-anchors it: exact match, then whitespace-normalized, then a prefix/suffix bridge, then the offset hint. If all of those fail the annotation shows as **orphaned** in the margin. It is never dropped. `lib/highlights.ts` paints highlights with the CSS Custom Highlight API, so the rendered DOM is never wrapped. An annotation can link to a side chat (`side_chat_id`, with `origin.annotation_id` on the chat).

## File index and tags

`pawn_server/core/file_index.py` extracts searchable text into `portal_file_index`: Markdown and plain text, PDF (`pypdf`), DOCX (`python-docx`), and a one-line caption for images when the background model is vision-enabled (`portal.index_images`). Files over `portal.index_max_bytes` are skipped. Portal writes index the file immediately. `pawn-server serve` also reconciles the whole bucket every `portal.index_interval_seconds` (0 disables the loop), to catch Obsidian Sync and agent writes. `POST /v1/portal/files/reindex` forces a full pass.

File tags are written to the note's frontmatter `tags:` when the note is writable Markdown, and to `portal_file_tags` otherwise. Renaming a tag (`POST /v1/portal/tags/rename`) rewrites chats, clips, annotations, file-tag rows and writable frontmatter.

## Inbox

The Inbox tab replaces the plugin's Inbox and has three chips:

- **Items.** Coworker items from `/v1/items`: filter by scope, kind and text, group by day or by session, and triage with TODO, File, Task, Later, Ignore and Delete, or Approve and Reject for proposals. Multi-select delete and "Delete all open" are available. Items can be annotated, and "Ask about this" grounds the side chat on the item's text, quote, reason and note.
- **Ideas.** `Ideas/*.md` notes with `status: inbox` (`GET/POST /v1/portal/inbox/ideas`). Keep sets `status: later`. Goal parks a thread in `Goals.md`. Task writes a TaskNotes note. Drop sets `status: dropped`. A quick-capture box writes a new idea.
- **Captures.** Research inbox atoms under `capture.inbox_dir` (`GET /v1/portal/inbox/captures`), filed or ignored through `POST /v1/captures/file`.

`GET /v1/portal/inbox/counts` feeds the header badge, which refreshes live from vault events.

An item whose `source_kind` is `session` has **Open session**. That opens the session tab on its Items section (`#/session/{id}?section=items`).

## Sessions

The Sessions tab lists recordings (`GET /v1/sessions`): search, date range, speaker facets, and a needs-speakers filter. The header badge is the unfiltered needs-speakers count. Opening a row shows the session tab: a range-capable player (signed audio URLs), the merged transcript, the analysis, open items, and the speakers rail (enroll, relabel, ignore, reidentify, reprocess). Right-click a speaker facet to open that person in the gallery.

Audio: `/v1/sessions/{id}/audio/{index}` streams each recording file with HTTP Range (FLAC, WAV, Ogg/Opus, WebM, MP3, M4A; files without an extension are sniffed). Chunked recordings play as one timeline: each file's global offset comes from `session_state.file_offsets`, filled at transcribe time, or on first use by reading the FLAC STREAMINFO / WAV header from S3. When a header can't be read, the offset is estimated from segment times and the player shows `≈ timing`. Backfill old sessions with `pawn-server portal repair-file-offsets --all` (add `--force` to re-probe). Signed audio links expire after 6 h. An expired link gets `403`, not `401`, so it never counts toward the auth-fail auto-ban, and the player re-signs it and resumes where it was.

## Day

The Day tab is the planner (`GET /v1/planner/day` and `GET /v1/planner/range`). Day and week share one time grid. Drag moves or resizes an event; a linked session is a badge on the event, not a second block. The editor saves title, time, attendees, focus, the linked session and notes. **Plan my day** opens a side chat for that date (`origin.kind = day`). A vault note with `pawn: event` shows **Open in Day** in the file tab. The Obsidian Day tab is the legacy surface; see [PLANNER.md](PLANNER.md).

## Speakers

The Speakers tab is one gallery (`?id=` selects a person). The list searches name, alias and id, and can include inactive people. The detail renames, edits aliases and notes, plays an enrollment from the session audio, removes a voiceprint, merges into another person, and deactivates. **People note** opens `People/{id}.md` or creates it (`POST /v1/speakers/{id}/people`). A rename can queue reidentify for the sessions that still use the previous name. See [SPEAKERS.md](SPEAKERS.md).

## Brainmap

A brainmap is a visualization type: a dark-cork board where cards are pinned up and red string shows how they connect. It always uses the cyberpunk look, whatever skin is set. Open it from **More → Brainmap** (`g b`), or with **Show on brainmap** on any Inbox item, which flies to that card. Each board is one tab (`#/brainmap/{board}?focus={node}`). The first board, `items`, pins up open Items and the emergence web around them:
- the threads, questions, decisions, commitments, predictions and hunches that an Item's observation feeds
- the people involved
- the session tape the Item came from

Every gesture is a verdict that Pawn learns from (`em_labels`, surface `brainmap`):

| Gesture | Effect |
|---|---|
| Drag a card | Pins it where you put it. Double-click lets it drift again. Positions and the viewport are saved per board on the server. |
| Drag a card's push-pin onto another card | Ties a string. Item ↔ subject links the observation (you pick the role: supports, against, mentions, updates, resolves). Subject ↔ thread files the subject under the thread. Item ↔ item starts a lasso. |
| Click a string | Shows what it means and who tied it. **Cut** removes it, and routing will not tie that pair again. With scissors (`X`) one click cuts. |
| Ghost string (dashed cyan) | Pawn's suggestion (nearest subject by embedding). ✓ ties it, ✂ snips it. |
| Queue (`Q`) | Walks every ghost string and proposed card: `Y` ties or accepts, `N` snips or burns, `→` skips. |
| Shift-drag on bare cork | Lasso: tie the cards into a new thread, or file them under an existing one. |
| Card verbs | Items keep their triage verbs. Proposed threads and hunches get **Accept** / **Burn**. Threads can **pull the thread**, which spills evidence that is not on the board as scraps. |

Clicking a card opens its dossier. It shows the text, quote, owner and due date, assumption health, the subject's state, an evidence timeline and its history. **Jump** opens the session and plays from that second. `G` focuses the selected card's web (1 or 2 hops), `F` fits the board and `/` searches.

The code is in two parts:
- **Engine** (`portal/src/lib/brainmap/`, with tests): force layout, string geometry, graph helpers and tie rules. The board registry in `boards.ts` maps a board id to its legend and tie rules.
- **UI** (`portal/src/components/brainmap/`): the board, cards, strings, dossier and queue.

API, under `/v1/portal/brainmaps/{board}`:
- `GET` returns the graph (`statuses`, `kinds`, `focus`, `limit`)
- `GET …/spill/{subject_id}`
- `POST …/strings` `{from, to, role}`
- `DELETE …/strings/{edge_id}`
- `POST …/suggestions/{id}` `{verdict: confirm|snip}`
- `POST …/lasso` `{title | into, node_ids}`
- `GET`/`PUT …/layout`

Card verbs reuse `/v1/items/{id}/action` and `/v1/emergence/subjects/{id}/status|council`. The status call sends `surface: "brainmap"`, so accept and burn are labelled as board verdicts. Layout lives in `portal_boards` (migration `0033`). When `emergence.enabled` is off, the board still shows Items, goal threads, people and tapes, but strings are read-only.

### Boards

You can make your own boards, like chats. They live in the chat sidebar's **Boards** section (the `+` makes one) and can be pinned, tagged, filed in a folder (they then show inside it), archived and deleted. A board is *seed + pins − hidden*: links, parents and subjects stay in the emergence tables, and the board only stores what is on it, note cards, layout and its chat (`portal_boards`, migration `0034`).

| Kind | What is on it |
|---|---|
| **Blank** (`manual`) | Only what you pin, plus one hop of what those cards feed |
| **Lens** (`lens`) | A saved query (`spec`: `q`, `people`, `subject_ids`, `subject_kinds`, `item_kinds`, `sources`, `threads`, `days`), re-run on every load |
| **Question** (`question`) | An `em_subjects` question the night shift works on (below) |

Other ways to start one:
- **Lasso** → *spin them off into a new board*, or *select them for the chat*.
- **Dossier** → *Pin to a board…*, *Open as board* (a lens on that subject and its children), *Investigate overnight* (a question board on it).
- **Search on a board** → More → *Board from search results*.
- **Chat header** → *Board from this chat*: the chat becomes the board's chat and its first turn asks Pawn to propose the cards it is about.

On a user board, `P` (or the selection bar) pins the selection to another board, **Take off the board** hides a card, and note cards can be tied to any card (a string to a note only says what it is about; it needs no emergence).

**Board chat.** A user board docks its chat on the right (`C` toggles it; the width is remembered). Click or shift-click cards to select them: the selection strip shows what the next message asks about. Every turn, not only the first, carries a snapshot of the board (selected cards first, then the rest by degree, capped at about 8,000 characters) and runs with the `board` skill under `BoardProfile`. Pawn never changes the board: `board_propose` writes ghost **strings**, **note cards**, **groups** (a new thread over some cards) and **pins**, which appear dashed on the board and in the queue. Accepting goes through the same tie / lasso / pin code as a gesture, so it becomes a training label with surface `board_chat`; burning a string is a `link` reject. The answer lists how many ghosts it left (`context.board_proposals`). `board_read --node ID` gives the model one card's evidence.

**Question boards** (needs `emergence.enabled`). Making one creates (or reuses) a question subject with `fields.board`, and writes its `Pawn/Questions/` note. Meetings that touch it reach it through routing and the directed pass, which always offers board questions (`emergence.board_directed_slots`). Each night the `boards` job looks back `board_backfill_days` for evidence, judges the nearest `board_max_candidates` observations in one call and links the relevant ones (strings you cut are never re-made), dreams among the board's own evidence (hunches hang off the question), posts **While you slept** into the board chat, and raises a `board_update` provocation when the board moved. Cards and strings added since you last looked carry a gold **NEW** stamp; the banner shows them and **Mark seen** clears them (`seen_at`).

API, under `/v1/portal/boards`:
- `GET` (`?archived=`) lists boards, `items` first
- `POST` `{title, kind, spec?, nodes?, folder_id?, tags?, chat_id?, question?, subject_id?}`
- `GET`/`PATCH`/`DELETE …/{board}`
- `POST …/{board}/pins` and `…/remove` `{node_ids}`
- `POST …/{board}/seen`
- `POST …/{board}/chat` (the board chat, created on first use)
- `GET …/{board}/proposals`, `POST …/{board}/proposals/{id}` `{verdict: accept|reject}`

The graph (`GET /v1/portal/brainmaps/{board}`) of a user board adds `meta` (title, kind, chat, subject, `seen_at`) and `proposals`. Ghost strings are suggestions with `prop:` ids (the suggestion route accepts them), ghost cards are nodes with `meta.proposal_id`, and note cards are `note:{id}`. The board's animation loop stops while its tab is hidden or the window is in the background.

## Settings

The Settings tab edits an allowlisted subset of config without touching `pawnai.yaml` and, for most fields, without a restart. The fields and their groups are defined in `pawn_server/core/settings_registry.py`.

- **Override file.** Changes are written atomically to `{state_dir}/settings.override.yaml` (next to `{state_dir}/background_model`). `load_config` merges it on top of everything else, so CliTool subprocesses see the same values.
- **Precedence.** With an auto-discovered `pawnai.yaml`: overrides > env vars > YAML > defaults. With an explicit config path (`--config` or `PAWN_AGENT_CONFIG`): overrides > YAML > env vars > defaults.
- **API.** `GET /v1/portal/settings` returns fields with their value, source (`override`, `env`, `yaml` or `default`), YAML value and the next three fire times for cron fields. Secrets are only reported as set or not set. `PATCH` validates against a candidate config (422 per field), writes the file, applies live fields in place and lists fields that need a restart. `DELETE /v1/portal/settings/{path}` removes one override. `GET /v1/portal/settings/status` reports the version, DB and Alembic state, queue stats, the loaded config file and the state dir.
- **`agent.default`** goes through the same path as the `/model` chat command, so both stay in sync.
- **Emergence group.** The main `emergence:` knobs: the enable flag, the per-window model, observation types, triage and quote thresholds, windows, few-shot and ranker, the directed pass, the night cron, jobs and budget, provocations per morning and hunches per night. `emergence.enabled` and `emergence.night_cron` need a restart. The night cron uses `coworker.timezone`.
- **Search.** The box at the top of the nav, or `/`, searches every group by label, path, help text and choices. Results from all groups show in one list, and the nav shows a match count per group. A result's group chip, or Enter when only one field matches, opens the field in its group.
- **Diagnostics → Scheduler activity** (`GET /v1/portal/settings/scheduler`, `pawn_server/core/diagnostics.py`): heartbeats of the scheduler and coworker loops, the built-in crons (briefing, weekly review, emergence night shift) with next fires and last result, durable schedules, pending proposals, recent fires, recent background `agent_runs` and the `agent_scheduler:` config. It refreshes every 15 s while shown. Heartbeats come from `pawn_agent/core/worker_status.py` and cover this process only. A worker started with `--scheduler-only` shows as "not in this process", but its fires and runs still appear because they come from the DB.
- **Diagnostics → Configuration** (`GET /v1/portal/settings/config`): the running `AgentConfig` flattened to path = value and filterable. Keys that look like credentials, and passwords inside URLs, are masked. Paths saved as portal overrides are badged, and editable paths link to their field.

## Importing existing chats

Conversations that predate the portal (Obsidian plugin chats, Matrix rooms, the OpenAI-compatible API) live only in sallm. Import them once with:

```bash
pawn-server portal import --dry-run        # list what would be imported
pawn-server portal import                  # chat: and matrix: sessions → folder "Imported"
pawn-server portal import -p matrix: --folder Matrix --tag matrix
pawn-server portal import --all --limit 20 # every session with chat turns, 20 most recent
```

- Each imported chat keeps its sallm session id as its conversation key, so replying from the portal continues the same memory.
- Only user and assistant chat turns are mirrored. Tool calls, tool results and ingest rows stay in sallm.
- The vault context that plugin chat injected into user prompts is split off. The message shows as typed, and the context is kept in the message's `context` (`prompt_context`, `active_note`).
- By default only `chat:` and `matrix:` keys are imported. `note:` tasks, `voice:`, `coworker:` and diarization-session keys are automation; include them with `--prefix` or `--all`.
- Sessions already in the portal are skipped, so the command is safe to re-run. `--folder ''` imports without a folder.

The logic lives in `pawn_server/core/portal_import.py`.

## Config

```yaml
api:
  portal_dir: portal/dist          # built SPA served at /portal
portal:
  scratch_dir: "{agent_root}/Scratch"
  recent_limit: 50
  default_chat_sort: updated
  index_interval_seconds: 300      # 0 disables the reconcile loop
  index_images: true
  index_max_bytes: 25000000
  auto_name: true
  auto_name_every: 6
```

## Assistants

The Assistants tab manages the voice assistants (Linux laptop, Android phone): presence, the active voice conversation, recent commands, phone-approved access requests, a test command, register (token shown once), rename and revoke, plus the shared wake-word clips and models. Voice turns are mirrored as chats in the `Voice` folder. API `GET /v1/portal/assistants` plus `/v1/devices/*` and `/v1/assistants/wake/*`. See [ASSISTANTS.md](ASSISTANTS.md).

## Tests

- Backend: `tests/test_portal_api.py`, `tests/test_portal_inbox.py`, `tests/test_portal_settings.py`, `tests/test_portal_import.py`, `tests/test_file_index.py`, `tests/test_sessions_api.py`, `tests/test_planner.py`, `tests/test_speakers_api.py`, `tests/test_assistants.py`, `tests/test_brainmap.py`.
- Frontend: `make portal` (type check and build) and `make portal-test` (includes the brainmap engine tests in `portal/src/lib/brainmap/*.test.ts` and the mock board in `portal/src/api/mockBrainmap.test.ts`).
