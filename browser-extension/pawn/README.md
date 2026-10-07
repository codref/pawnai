# Pawn Capture (browser extension)

MV3 side panel for Chrome, Edge, and Firefox. Captures ordered text selections
and image regions into the Pawn vault. There is no chat UI — Obsidian and Matrix
remain the places you talk to the agent.

## Build / package

From the repo root:

```bash
make browser-extension          # verify files + icons
make browser-extension-dist     # writes browser-extension/pawn/pawn-capture.zip
```

Or from this directory:

```bash
make build
make dist                       # pawn-capture.zip
make icons                      # regenerate PNG icons
```

The zip contains `pawn-capture/` with `manifest.json` at its root (same layout
as this source tree). There is no TypeScript/npm compile step.

## Deploy (Chrome / Edge)

### From a zip (recommended)

1. `make browser-extension-dist` (or unzip a shared `pawn-capture.zip`).
2. Unzip so you have a folder `pawn-capture/` with `manifest.json` inside.
3. Open `chrome://extensions` (or `edge://extensions`).
4. Enable **Developer mode**.
5. **Load unpacked** → select the `pawn-capture/` folder (not the zip, not the
   parent that only contains the zip).
6. Pin the action if you want a toolbar button; click it to open the side panel.
7. In the panel → **Settings**: server URL (e.g. `http://127.0.0.1:8000`) and
   `api.token` from `pawnai.yaml`.

Updating: rebuild the zip, replace the folder contents (or load the new
folder), then click the extension’s **Reload** on `chrome://extensions`.

### From the git checkout (dev)

**Load unpacked** → this directory (`browser-extension/pawn`) directly. Reload
the extension after editing JS/CSS/HTML.

## Deploy (Firefox)

1. Build/unzip as above so `pawn-capture/manifest.json` exists.
2. Open `about:debugging#/runtime/this-firefox`.
3. **Load Temporary Add-on** → pick `manifest.json` inside `pawn-capture/`.
4. Temporary add-ons are cleared when Firefox quits; reload after each restart
   until you sign/publish via AMO.
5. Firefox 121+ is required for the side panel API used here.

## Server requirements

- `pawn-server` running with the capture routes (`GET/POST /v1/captures`,
  `GET /v1/sessions`). Same Bearer token as the Obsidian plugin.
- Vault Sync Engine (or equivalent) so `Pawn/Captures/` appears in Obsidian.
- For a remote host, use HTTPS or accept that Chromium will warn on mixed
  content; the extension’s host permissions allow `http://` and `https://`.

## Use

1. Target defaults to **New page** (creates `Pawn/Captures/{date} {title}.md`).
2. Or pick a recent capture, a diarization **session** (writes into transcript
   Annotations when mapped), or another vault note path.
3. **Selection** / right-click **Send selection to Pawn**, or **Region** /
   right-click **Capture region to Pawn**.
4. Reorder or drop items in the tray, then **Save**. The target stays sticky so
   the next batch appends to the same note. **New page** starts a fresh note.

## API

Bearer auth against pawn-server:

| Method | Path | Role |
|--------|------|------|
| GET | `/v1/sessions` | Session picker |
| GET | `/v1/captures` | Recent capture notes |
| POST | `/v1/captures` | Append ordered snippets |
| DELETE | `/v1/captures/snippets/{id}?path=` | Remove one block |

See [docs/BROWSER_CAPTURE.md](../../docs/BROWSER_CAPTURE.md).
