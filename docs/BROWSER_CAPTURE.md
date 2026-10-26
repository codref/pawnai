# Browser snippet capture

The **Pawn Capture** extension (`browser-extension/pawn/`) is a Chrome / Edge /
Firefox side panel with two tabs:

- **Capture** — append ordered text and image snippets to a vault note (or
  Research inbox / session Annotations).
- **Chat** — talk to Pawn about the current page via `POST /v1/pawn/chat`
  (same SSE agent path as the Obsidian plugin). Attach selections or
  screenshots as context; use suggested action chips for common tasks.

Vault writes still sync into Obsidian through Sync Engine. Chat memory lives
in sallm under a `chat:{uuid}` conversation key (New chat starts a fresh id).

## Flow

### Capture tab

1. Open the side panel. The target defaults to **New page**.
2. The **destination** row shows the sticky note (title + full path). When
   `vault.obsidian_vault_name` is set, the title is an `obsidian://open` link.
3. Capture a **selection**, or use **Clip** (region drag on a frozen screenshot,
   or whole **visible area**). Items stack in the tray (reorder / delete before
   save). Only one clip runs at a time. Each tray card can **Attach to chat**.
4. **Save** appends unsaved items, in tray order, to the current target. The
   target stays selected so the next batch continues the same note (except
   Research, which writes one inbox atom per snippet).
5. **Start new note** (destination row) clears the sticky path and starts a
   fresh capture note (in Research mode it only clears the tray).

The panel does not guess which meeting you are in. Pick a diarization session
from the list when you want snippets on that transcript.

### Chat tab

Layout mirrors the Obsidian plugin chat pane (no Background toggle):

1. Toolbar: **conversation picker** (searchable history) · **New chat** ·
   attach selection / Clip.
2. Context chips: page title/URL; sticky capture/session note as
   `active_note` (`inline: false`); attached text/images (max 4 images).
3. Thread + suggested action chips (**Create TODO**, **Capture idea**,
   **Summarize**, **Use session** when Capture target is a session).
4. Composer: message · **model** (bottom; OpenRouter **Effort** / **Route**
   when the catalog exposes them) · attach selection/Clip · Send / Stop.
5. Conversations are local (`chat:{uuid}` keys, up to 40 non-empty); New chat
   starts a fresh id. Image turns prefer a `vision: true` model from
   `GET /v1/pawn/models`.

## Targets

| Target | What happens |
|--------|----------------|
| New page (default) | Creates `Pawn/Captures/{YYYY-MM-DD} {title}.md` with `pawn: capture`. Later saves append to the same path. |
| Research… | One inbox note per snippet under `capture.inbox_dir` (default `Pawn/Captures/Inbox/`). Optional collection + hint. When `capture.auto_enrich` is true, enqueues a `capture_enrich` job per new note. |
| Recent capture | Append to an existing note under `Pawn/Captures/`. |
| Session | If `vault_notes` maps the session to a transcript, snippets are spliced into that note’s `## Annotations` section (preserved on transcript push). Otherwise a capture note is created with `session_id` in frontmatter. |
| Other note | Append to a vault path. Allowed under `Pawn/`; outside that root the note needs `pawn: editable`. |

Images are stored at `Pawn/Captures/assets/{snippet-id}.png` (or jpg/gif/webp)
and embedded with Obsidian wikilinks. Each block is wrapped in
`<!-- pawn-snippet:{id} -->` … `<!-- /pawn-snippet:{id} -->` so retries are
idempotent and the panel can delete a saved row.

Coworker does not extract dump capture notes. Research enrich is a separate
`capture_enrich` job (sallm ReAct with the `research_capture` skill).

## Research enrich

Config (`capture:` in `pawnai.yaml`):

```yaml
capture:
  inbox_dir: "{agent_root}/Captures/Inbox"
  enriched_dir: "{agent_root}/Research"
  entity_path_template: "{enriched_dir}/{collection}/{entity}.md"
  model: ""            # catalog id; empty → background / agent.default
  auto_enrich: true
  auto_file: false     # true → agent may set status: filed when confident
  instructions: ""
```

Collections are **not** declared in config. They emerge as folders under
`enriched_dir`. The extension may leave collection blank (agent proposes) or
pick an existing folder from `GET /v1/captures/collections`.

After enrich, inbox notes have `status: proposed` (or `filed` when auto_file
succeeds). Triage in the Obsidian Inbox **Captures** chip (Items-style cards):
**File** / **Ignore** / **Open** (`POST /v1/captures/file`). Filing links the
capture under the entity note’s `## Captures` section without moving the inbox
file.

Enrich updates capture frontmatter only via the `capture_update` CliTool
(parse + YAML dump), not `note_write`, so captions that contain `:` stay valid
Obsidian properties. Job results for `capture_enrich` store the answer body
only (`strip_tool_trail`); the Jobs tab strips any legacy `[tool]` prefix
before markdown render.

## HTTP API

Bearer token (`api.token`), same as the Obsidian plugin.

### `GET /v1/meta`

Returns `{ "obsidian_vault_name": "…" }` from `vault.obsidian_vault_name` so the
browser panel can build Obsidian open links. Empty string when unset.

### `GET /v1/sessions?limit=&q=`

Recent diarization sessions from the database, plus `transcript_path` when
mapped in `vault_notes`.

### `GET /v1/captures?limit=`

Recent markdown notes under `{agent_root}/Captures/` (assets excluded).

### `GET /v1/captures/collections`

Existing collection folder names under `capture.enriched_dir` (discovered from
the vault, not from config).

### `POST /v1/pawn/chat`

Same native streaming chat as the Obsidian plugin (Bearer token, SSE
`progress` / `answer` / `job` / `error` / `done`). The extension sends
`conversation`, `message`, optional `selection` (attached text + page URL),
optional `active_note` (sticky capture/transcript path, `inline: false`),
optional `images[]` (`role: "question"`), and optional `model`.

### `GET /v1/pawn/models`

Catalog used by the Chat model picker (`vision` flag gates screenshot turns).

### `POST /v1/captures`

```json
{
  "target": {
    "kind": "new",
    "title": "Teams agenda",
    "source_url": "https://…",
    "path": null,
    "session_id": null,
    "collection": null,
    "hint": null
  },
  "snippets": [
    {
      "id": "a1b2c3d4e5f6a7b8",
      "kind": "text",
      "text": "…",
      "source_url": "https://…",
      "captured_at": "2026-10-07T10:00:00Z"
    },
    {
      "id": "c0ffee00c0ffee00",
      "kind": "image",
      "data_base64": "…",
      "media_type": "image/png",
      "captured_at": "2026-10-07T10:01:00Z"
    }
  ]
}
```

`kind` on the target: `new` | `capture` | `note` | `session` | `research`.
Response includes `path`, `mode` (`append`, `annotations`, or `research`),
`written`, `skipped`, and `images`. Research also returns `paths`,
`enrich_paths`, `enrich`, and optionally `jobs`.

Image payloads are capped at 4MB (same as chat vision).

### `POST /v1/captures/file`

File or ignore a research inbox note:

```json
{ "path": "Pawn/Captures/Inbox/…md", "collection": "movies", "entity": "blade-runner" }
```

Or `{ "path": "…", "ignore": true }`.

### `DELETE /v1/captures/snippets/{id}?path=`

Removes one marked block from the note and deletes its asset under
`Pawn/Captures/assets/` when present. The Obsidian plugin replaces
`<!-- pawn-snippet:… -->` markers in the source editor with a chip that
calls this endpoint (local edit first; asset cleanup still runs if the
markers are already gone).

## Build and deploy

```bash
# From the repo root
make browser-extension          # verify runtime files + icons
make browser-extension-dist     # Chrome + Firefox zips under browser-extension/pawn/
```

Or `cd browser-extension/pawn && make dist`.

| Zip | Browser | Notes |
|-----|---------|--------|
| `pawn-capture.zip` | Chrome / Edge | `background.service_worker` + side panel |
| `pawn-capture-firefox.zip` | Firefox | `background.scripts` + sidebar (required for temporary install) |

Both unwrap to `pawn-capture/` with `manifest.json` at the root.

**Chrome / Edge:** Developer mode → **Load unpacked** → select the
`pawn-capture/` folder from the Chrome zip (or `browser-extension/pawn` while
developing — that tree’s `manifest.json` is Chrome). Open the side panel from
the toolbar action; set server URL and `api.token` under Settings.

**Firefox:** unzip `pawn-capture-firefox.zip`, then `about:debugging` →
**Load Temporary Add-on** → that folder’s `manifest.json`. Do not load the
Chrome `manifest.json` (service workers are disabled for temporary add-ons).
Open **View → Sidebar → Pawn Capture**. Temporary until the browser quits
(Firefox 121+).

Full install notes: [browser-extension/pawn/README.md](../browser-extension/pawn/README.md).

## Implementation

- Server: [`pawn_server/core/captures.py`](../pawn_server/core/captures.py),
  [`pawn_server/core/capture_config.py`](../pawn_server/core/capture_config.py),
  [`pawn_server/core/capture_enrich.py`](../pawn_server/core/capture_enrich.py)
- Config: `CaptureConfig` on `AgentConfig` (`pawn_agent/utils/config.py`)
- Skill: `research_capture` in [`pawn_agent/core/sallm_skills.py`](../pawn_agent/core/sallm_skills.py)
- Routes: [`pawn_server/core/api_server.py`](../pawn_server/core/api_server.py)
- Tests: [`tests/test_captures_api.py`](../tests/test_captures_api.py)
- Extension: [`browser-extension/pawn/`](../browser-extension/pawn/)
- Thunderbird counterpart: [THUNDERBIRD_CAPTURE.md](THUNDERBIRD_CAPTURE.md)
- Obsidian Captures chip: [`obsidian-plugin/pawn/src/inbox/`](../obsidian-plugin/pawn/src/inbox/)
