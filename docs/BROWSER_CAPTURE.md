# Browser snippet capture

The **Pawn Capture** extension (`browser-extension/pawn/`) is a Chrome / Edge /
Firefox side panel that appends ordered text and image snippets to a vault note.
There is no chat UI in the extension — continue the conversation in Obsidian or
Matrix after Sync Engine pulls the note.

## Flow

1. Open the side panel. The target defaults to **New page**.
2. Capture a **selection** or drag a **region** on the active tab. Items stack in
   the tray (reorder / delete before save).
3. **Save** appends unsaved items, in tray order, to the current target. The
   target stays selected so the next batch continues the same note.
4. **New page** clears the sticky path and starts a fresh capture note.

The panel does not guess which meeting you are in. Pick a diarization session
from the list when you want snippets on that transcript.

## Targets

| Target | What happens |
|--------|----------------|
| New page (default) | Creates `Pawn/Captures/{YYYY-MM-DD} {title}.md` with `pawn: capture`. Later saves append to the same path. |
| Recent capture | Append to an existing note under `Pawn/Captures/`. |
| Session | If `vault_notes` maps the session to a transcript, snippets are spliced into that note’s `## Annotations` section (preserved on transcript push). Otherwise a capture note is created with `session_id` in frontmatter. |
| Other note | Append to a vault path. Allowed under `Pawn/`; outside that root the note needs `pawn: editable`. |

Images are stored at `Pawn/Captures/assets/{snippet-id}.png` (or jpg/gif/webp)
and embedded with Obsidian wikilinks. Each block is wrapped in
`<!-- pawn-snippet:{id} -->` … `<!-- /pawn-snippet:{id} -->` so retries are
idempotent and the panel can delete a saved row.

Coworker does not extract these notes. The vault scanner skips the agent root,
and capture does not call `process_source`.

## HTTP API

Bearer token (`api.token`), same as the Obsidian plugin.

### `GET /v1/sessions?limit=&q=`

Recent diarization sessions from the database, plus `transcript_path` when
mapped in `vault_notes`.

### `GET /v1/captures?limit=`

Recent markdown notes under `{agent_root}/Captures/` (assets excluded).

### `POST /v1/captures`

```json
{
  "target": {
    "kind": "new",
    "title": "Teams agenda",
    "source_url": "https://…",
    "path": null,
    "session_id": null
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

`kind` on the target: `new` | `capture` | `note` | `session`. Response includes
`path`, `mode` (`append` or `annotations`), `written`, `skipped`, and `images`.

Image payloads are capped at 4MB (same as chat vision).

### `DELETE /v1/captures/snippets/{id}?path=`

Removes one marked block from the note and deletes its asset under
`Pawn/Captures/assets/` when present.

## Build and deploy

```bash
# From the repo root
make browser-extension          # verify runtime files + icons
make browser-extension-dist     # browser-extension/pawn/pawn-capture.zip
```

Or `cd browser-extension/pawn && make dist`. The zip unwraps to
`pawn-capture/` with `manifest.json` at the root.

**Chrome / Edge:** Developer mode → **Load unpacked** → select the
`pawn-capture/` folder (or `browser-extension/pawn` while developing). Open
the side panel from the toolbar action; set server URL and `api.token` under
Settings. Reload the extension on `chrome://extensions` after updating files.

**Firefox:** `about:debugging` → **Load Temporary Add-on** → pick
`manifest.json` inside `pawn-capture/`. Temporary until the browser quits
(Firefox 121+ for side panel).

Full install notes: [browser-extension/pawn/README.md](../browser-extension/pawn/README.md).

## Implementation

- Server: [`pawn_server/core/captures.py`](../pawn_server/core/captures.py)
- Routes: [`pawn_server/core/api_server.py`](../pawn_server/core/api_server.py)
- Tests: [`tests/test_captures_api.py`](../tests/test_captures_api.py)
- Extension: [`browser-extension/pawn/`](../browser-extension/pawn/)
