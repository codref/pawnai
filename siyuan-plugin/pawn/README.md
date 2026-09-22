# Pawn (SiYuan plugin)

Wraps `@pawn` paragraphs into a TIP callout and sends the **full callout**
to pawn-server via `POST /v1/siyuan/triggers`.

## Install

**From the UI:** zip this folder and use Marketplace → Install Bazaar package:

```bash
cd /path/to/parakeet/siyuan-plugin/pawn
zip -r ../pawn-0.1.0.zip plugin.json index.js README.md i18n
```

Then enable **Pawn** under Marketplace → Downloaded.

**Dev symlink:**

```bash
ln -s /path/to/parakeet/siyuan-plugin/pawn \
  ~/SiYuan/data/plugins/pawn
```

After updating plugin files, disable/enable the plugin (or restart SiYuan) so
the kernel reloads `index.js`.

## Settings

| Key | Default | Notes |
|-----|---------|-------|
| Server URL | `http://127.0.0.1:8000` | Must be reachable from the SiYuan **kernel** |
| API token | _(empty)_ | Same as `api.token` in `pawnai.yaml` |
| Mention token | `@pawn` | Must match `siyuan_watcher.mention_token` |
| Auto-wrap callout | on | Rewrite leading mention → TIP callout |
| Wrap debounce (ms) | `400` | Delay before auto-wrap |
| Callout icon | 🤖 | Embedded in TIP title |
| Send button label | `Send to Pawn` | Block menu / action label |
| Request timeout (ms) | `15000` | `forwardProxy` timeout (202 returns quickly) |

## Usage

1. Type `@pawn ` at the start of a paragraph, then your instruction.
2. After a short pause the block becomes a TIP callout titled **Pawn**; keep
   editing the body (the caret should stay at the end of the instruction).
3. **Send** when ready — there is no in-callout button. Use either:
   - Block gutter icon (left of the block) → **Send to Pawn**
   - Command palette / hotkey **⌥⌘P** (Configurable under Settings → Keymap → Pawn)
4. pawn-server runs the agent; a review draft appears under the callout parent.
5. Check **Approve for Pawn memory** when ready (watcher indexes it).

The document root is the agent session (`siyuan:{root_id}`) — same as the
previous SQL watcher path.
