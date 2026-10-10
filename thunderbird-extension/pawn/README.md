# Pawn Capture (Thunderbird)

MV3 MailExtension for Thunderbird 128+. Captures whole messages, selections, and
image attachments into the Pawn vault via a companion tray window. There is no
chat UI — Obsidian and Matrix remain the places you talk to the agent.

Thunderbird has no extension sidebar API, so the tray lives in a narrow popup
window you can dock beside the 3-pane view.

## Build / package

From the repo root:

```bash
make thunderbird-extension          # verify files + icons
make thunderbird-extension-dist     # pawn-capture-thunderbird.xpi
```

Or from this directory:

```bash
make build
make dist                       # pawn-capture-thunderbird.xpi
```

There is no TypeScript/npm compile step. The `.xpi` is a zip with `manifest.json`
at the root.

## Deploy

### Temporary (dev)

1. Open Thunderbird → **Add-ons and Themes** (or `about:addons`).
2. Gear menu → **Debug Add-ons**.
3. **Load Temporary Add-on** → pick this directory’s `manifest.json`
   (`thunderbird-extension/pawn/manifest.json`), or the unzipped `.xpi` contents.
4. Click the **Pawn Capture** toolbar button (or the message-display action) to
   open the companion window.
5. In the panel → **Settings**: server URL (e.g. `http://127.0.0.1:8000`) and
   `api.token` from `pawnai.yaml`.

Temporary add-ons are cleared when Thunderbird quits; reload after each restart.

### From the `.xpi`

1. `make thunderbird-extension-dist`.
2. Add-ons Manager → gear → **Install Add-on From File** →
   `pawn-capture-thunderbird.xpi`.
3. Unsigned local installs may need
   `xpinstall.signatures.required` set to `false` in the Config Editor for
   development; ATN signing is out of scope for this tree.

## Server requirements

- `pawn-server` running with the capture routes (`GET/POST /v1/captures`,
  `GET /v1/sessions`). Same Bearer token as the Obsidian plugin.
- Vault Sync Engine (or equivalent) so `Pawn/Captures/` appears in Obsidian.

## Use

1. Target defaults to **New page** (creates `Pawn/Captures/{date} {title}.md`).
2. Or pick **Research…**, a recent capture, a diarization **session**, or another
   vault note path.
3. Queue items via:
   - Companion window: **Add displayed message** / **Add selected messages**
   - Message list context menu: **Send to Pawn tray**
   - Selection in the message pane: **Send selection to Pawn**
   - Attachment context menu: **Send image to Pawn** (images ≤ 4MB)
4. Reorder or drop items in the tray, then **Save**. The target stays sticky so
   the next batch appends to the same note. **New page** starts a fresh note.

Bodies include a Subject/From/To/Date/Message-ID header block. `source_url` is
`mid:<Message-ID>` (RFC 2392). Bodies are capped at ~20k characters.

## API

Bearer auth against pawn-server — same endpoints as the browser extension.
See [docs/THUNDERBIRD_CAPTURE.md](../../docs/THUNDERBIRD_CAPTURE.md) and
[docs/BROWSER_CAPTURE.md](../../docs/BROWSER_CAPTURE.md).
