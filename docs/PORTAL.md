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

- **Left sidebar: chats.** Saved chats in nested folders, with hierarchical tags (`work/acme/q4`), pin, archive and sorting by updated, created, name or tag. Search covers chats, messages, clips, annotations and vault files. `#tag` in a query filters by tag.
- **Right sidebar: files.** The whole vault bucket, with ages and tags. The scratchpad (`portal.scratch_dir`, default `{agent_root}/Scratch`) is always writable. Upload, new note, move and delete follow the vault write guards (free under `Pawn/`, elsewhere only with `pawn: editable`, never `.obsidian/`).
- **Center: workspace.** VS Code-style tabs for chats, files, clips, the Inbox and Settings. Groups can be split side by side or floated as windows. A single-click opens an italic preview tab, which is pinned when you send, edit or double-click. The layout is saved through `GET/PUT /v1/portal/workspace`.
- **Header.** An Inbox button with open counts, and a gear for Settings.

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

## Settings

The Settings tab edits an allowlisted subset of config without touching `pawnai.yaml` and, for most fields, without a restart. The fields and their groups are defined in `pawn_server/core/settings_registry.py`.

- **Override file.** Changes are written atomically to `{state_dir}/settings.override.yaml` (next to `{state_dir}/background_model`). `load_config` merges it on top of everything else, so CliTool subprocesses see the same values.
- **Precedence.** With an auto-discovered `pawnai.yaml`: overrides > env vars > YAML > defaults. With an explicit config path (`--config` or `PAWN_AGENT_CONFIG`): overrides > YAML > env vars > defaults.
- **API.** `GET /v1/portal/settings` returns fields with their value, source (`override`, `env`, `yaml` or `default`), YAML value and the next three fire times for cron fields. Secrets are only reported as set or not set. `PATCH` validates against a candidate config (422 per field), writes the file, applies live fields in place and lists fields that need a restart. `DELETE /v1/portal/settings/{path}` removes one override. `GET /v1/portal/settings/status` reports the version, DB and Alembic state, queue stats, the loaded config file and the state dir.
- **`agent.default`** goes through the same path as the `/model` chat command, so both stay in sync.

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

## Tests

- Backend: `tests/test_portal_api.py`, `tests/test_portal_inbox.py`, `tests/test_portal_settings.py`, `tests/test_portal_import.py`, `tests/test_file_index.py`.
- Frontend: `make portal` (type check and build) and `make portal-test`.
