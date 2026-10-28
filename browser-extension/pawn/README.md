# Pawn Capture (browser extension)

MV3 side panel for Chrome, Edge, and Firefox. Two tabs:

- **Capture** — ordered text selections and image regions into the Pawn vault.
- **Chat** — ask Pawn about the page / attached clips via `POST /v1/pawn/chat`
  (same agent as the Obsidian plugin), with suggested action chips.

## Build / package

From the repo root:

```bash
make browser-extension          # verify files + icons
make browser-extension-dist     # Chrome + Firefox zips under browser-extension/pawn/
```

Or from this directory:

```bash
make build                      # patch-bumps version + verify files
make dist                       # pawn-capture.zip + pawn-capture-firefox.zip
make package-chrome             # Chrome/Edge only
make package-firefox            # Firefox only
make icons                      # regenerate PNG icons
```

`make build` bumps the patch in both manifests (skipped when `CI=true`).
After **Reload** on `chrome://extensions` / `edge://extensions`, Settings shows
`Extension vX.Y.Z` from `chrome.runtime.getManifest()` — that is the build
actually loaded, not a cached label.

Chrome and Firefox need different manifests: Chrome uses
`background.service_worker`; Firefox temporary installs still require
`background.scripts` (and use `sidebar_action` instead of `side_panel`).
`manifest.json` is Chrome; `manifest.firefox.json` is the Firefox source.
The Firefox zip copies that file to `manifest.json` inside the package.

There is no TypeScript/npm compile step.

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

**Load unpacked** → this directory (`browser-extension/pawn`) directly — it
ships the Chrome `manifest.json`. Reload the extension after editing
JS/CSS/HTML.

## Deploy (Firefox)

Do **not** load this source tree’s `manifest.json` into Firefox — it uses a
service worker and will fail with *background.service_worker is currently
disabled*.

1. `make package-firefox` (or `make browser-extension-dist`).
2. Unzip `pawn-capture-firefox.zip` → `pawn-capture/`.
3. Open `about:debugging#/runtime/this-firefox`.
4. **Load Temporary Add-on** → pick `pawn-capture/manifest.json` from that
   unzipped folder (it already has `background.scripts` and `<all_urls>`).
5. If Selection/Region says **Missing host permission for the tab**:
   - Remove the temporary add-on and load the new zip again (permission
     changes need a fresh install), **or**
   - `about:addons` → Pawn Capture → Permissions → enable **Access your data
     for all websites**.
6. Open an ordinary `https://` tab (not `about:`), select text, then use the
   sidebar **Selection** button or the context menu.
7. Open the UI via the toolbar action, or **View → Sidebar → Pawn Capture**.
8. Temporary add-ons are cleared when Firefox quits; reload after each restart
   until you sign/publish via AMO.
9. Firefox 121+.

## Server requirements

- `pawn-server` running with capture routes (`GET/POST /v1/captures`,
  `GET /v1/sessions`) and chat (`POST /v1/pawn/chat`, `GET /v1/pawn/models`).
  Same Bearer token as the Obsidian plugin.
- Vault Sync Engine (or equivalent) so `Pawn/Captures/` appears in Obsidian.
- For a remote host, use HTTPS or accept that Chromium will warn on mixed
  content; the extension’s host permissions allow `http://` and `https://`.
- Ensure `api.cors_origins` allows the extension origin if you hit CORS errors
  on chat streaming.

## Use

### Capture

1. Target defaults to **New page** (creates `Pawn/Captures/{date} {title}.md`).
2. Or pick a recent capture, a diarization **session** (writes into transcript
   Annotations when mapped), or another vault note path. The destination row
   shows the sticky path (and opens Obsidian when `vault.obsidian_vault_name`
   is configured on the server).
3. **Selection** / right-click **Send selection to Pawn**, or **Clip**
   (region on a frozen screenshot, or whole visible area) / right-click
   **Capture region to Pawn**.
4. Reorder or drop items in the tray, then **Save**. Use the message icon on a
   tray card to **Attach to chat** without saving first.

### Chat

1. Open the **Chat** tab. The top bar has a **conversation picker** (past
   chats) and **New chat**.
2. Attach page clips (selection/region) or vault context via the **+** chip /
   typing **@** — searchable overlay of sessions and recent captures (or type
   a vault path).
3. Ask freely, or use chips that inject prompts (`Create TODO`, `/idea`, …).
4. Model picker sits next to **Send** (bottom). Uses `/v1/pawn/models`.

## API

Bearer auth against pawn-server:

| Method | Path | Role |
|--------|------|------|
| GET | `/v1/meta` | Vault name for Obsidian links |
| GET | `/v1/sessions` | Session picker |
| GET | `/v1/captures` | Recent capture notes |
| POST | `/v1/captures` | Append ordered snippets |
| DELETE | `/v1/captures/snippets/{id}?path=` | Remove one block |
| GET | `/v1/pawn/models` | Chat model catalog |
| POST | `/v1/pawn/chat` | Streaming chat (SSE) |

See [docs/BROWSER_CAPTURE.md](../../docs/BROWSER_CAPTURE.md).
