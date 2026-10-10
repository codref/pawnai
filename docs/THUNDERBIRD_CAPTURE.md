# Thunderbird email capture

The **Pawn Capture** Thunderbird add-on (`thunderbird-extension/pawn/`) is an MV3
MailExtension (Thunderbird 128+) that appends ordered message and image snippets
to a vault note. There is no chat UI — continue in Obsidian or Matrix after Sync
Engine pulls the note.

This client posts to the same `POST /v1/captures` API as the browser extension.
Targets, research enrich, and vault layout are documented in
[BROWSER_CAPTURE.md](BROWSER_CAPTURE.md).

## Companion window (not a docked sidebar)

Thunderbird does not expose `sidebar_action` for extensions. An `action` popup
closes when it loses focus, so it cannot hold a tray. The add-on opens a narrow
**companion window** (`windows.create` type `popup`) instead. Dock it beside the
3-pane view; bounds are restored from the last session.

A true docked pane would need an Experiment API and is out of scope for v1.

## Flow

1. Open the companion window from the mail toolbar action or the message-display
   action.
2. Queue items:
   - **Add displayed message** / **Add selected messages** in the panel
   - Message list context menu → **Send to Pawn tray**
   - Selection in the message pane → **Send selection to Pawn**
   - Attachment context menu → **Send image to Pawn**
3. Items stack in the tray (reorder / delete before save).
4. **Save** appends unsaved items, in tray order, to the current target.

## Snippet format

Each whole-message text snippet is:

```text
Subject: …
From: …
To: …
Date: …
Message-ID: …

<body>
```

- `source_url` is `mid:<Message-ID>` when the header is present (RFC 2392).
- Bodies prefer `text/plain` from `messages.listInlineTextParts`; otherwise HTML
  is converted to plain text (`messengerUtilities.convertToPlainText` on TB 137+,
  simple tag strip on TB 128–136). Trailing quoted-reply blocks are stripped
  best-effort.
- Body text is capped at **20 000** characters (truncation marker appended).
- Image attachments use the same base64 path as browser region captures; files
  over **4 MB** are skipped (server image cap). Non-image attachments are skipped.

No server-side email frontmatter in v1 — structured `from` / participants fields
can wait until research enrich needs them.

## Build and deploy

```bash
# From the repo root
make thunderbird-extension          # verify runtime files + icons
make thunderbird-extension-dist     # pawn-capture-thunderbird.xpi
```

Or `cd thunderbird-extension/pawn && make dist`.

| Artifact | Notes |
|----------|--------|
| `pawn-capture-thunderbird.xpi` | Zip with `manifest.json` at the root |

**Temporary install:** Add-ons Manager → gear → **Debug Add-ons** →
**Load Temporary Add-on** → `thunderbird-extension/pawn/manifest.json` (or the
unzipped xpi). Cleared when Thunderbird quits.

**From file:** gear → **Install Add-on From File** → the `.xpi`. Unsigned local
installs may need `xpinstall.signatures.required` disabled for development.

Set **Server URL** and **API token** (`api.token` from `pawnai.yaml`) in the
panel settings — same values as the Obsidian plugin and browser extension.
Click **Save settings** once so Thunderbird can prompt for host access; API
calls are proxied through the background script (companion-window `fetch` is
CORS-blocked otherwise).

### NetworkError / “failed to fetch”

Host permissions alone are not enough — Firefox/Thunderbird use the same
`NetworkError` string for CORS failures **and** connection refused.

1. Confirm `pawn-server` is up: `curl -sS -i http://127.0.0.1:8000/health`
   (use the same host/port as Settings). If curl fails, fix the server first.
2. **Restart pawn-server** on a build that allows `moz-extension://` via CORS
   (`allow_origin_regex` in `create_app`). Older servers only listed Obsidian
   origins, so the companion window was blocked even with Permissions toggles on.
3. Reload the add-on (v0.1.2+), Settings → **Save settings**.
4. Permissions tab should show **Access your data for all websites** (and/or
   `127.0.0.1` / `localhost`) enabled — Gecko match patterns must not include a
   port; the add-on requests `http://127.0.0.1/*`, which covers `:8000`.
5. If curl works but the add-on still fails, check for an HTTP proxy that
   hijacks localhost (Thunderbird uses the browser network stack).

## Implementation

- Extension: [`thunderbird-extension/pawn/`](../thunderbird-extension/pawn/)
- Server (shared): [`pawn_server/core/captures.py`](../pawn_server/core/captures.py),
  routes in [`pawn_server/core/api_server.py`](../pawn_server/core/api_server.py)
- Browser counterpart: [BROWSER_CAPTURE.md](BROWSER_CAPTURE.md)
