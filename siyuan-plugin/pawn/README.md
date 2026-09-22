# Pawn (SiYuan plugin)

Wraps `@pawn` paragraphs into a TIP callout (on send) and posts the **full
callout** to pawn-server via `POST /v1/siyuan/triggers`.

## Install

**From the UI:** zip this folder and use Marketplace → Install Bazaar package:

```bash
cd /path/to/parakeet/siyuan-plugin/pawn
zip -r ../pawn-0.1.2.zip plugin.json index.js README.md i18n
```

Then enable **Pawn** under Marketplace → Downloaded.

**Dev symlink:**

```bash
ln -s /path/to/parakeet/siyuan-plugin/pawn \
  ~/SiYuan/data/plugins/pawn
```

After updating plugin files, disable/enable the plugin (or restart SiYuan) so
the kernel reloads `index.js`. Open editors may need a tab refresh before the
floating-toolbar **Send** button appears.

## Settings

| Key | Default | Notes |
|-----|---------|-------|
| Server URL | `http://127.0.0.1:8000` | Must be reachable from the SiYuan **kernel** |
| API token | _(empty)_ | Same as `api.token` in `pawnai.yaml` |
| Mention token | `@pawn` | Must match `siyuan_watcher.mention_token` |
| Wrap as callout on send | on | TIP callout created when you send (not while typing) |
| Callout icon | 🤖 | Embedded in TIP title |
| Send button label | `Send to Pawn` | Toolbar / menu / command label |
| Request timeout (ms) | `15000` | `forwardProxy` timeout (202 returns quickly) |

## Usage

1. Type `@pawn ` at the start of a paragraph, then your instruction. Keep
   editing normally — nothing rewrites the block while you type.
2. **Send** when ready (any of these):
   - Select text in the block → floating toolbar → **Send to Pawn** (paper-plane icon)
   - Block gutter icon (left of the block) → **Send to Pawn**
   - Hotkey **⌥⌘P** (Alt+Ctrl+P on Linux; remap under Settings → Keymap → Pawn)
3. With wrap-on-send enabled, the paragraph becomes a TIP callout and
   pawn-server runs the agent; a review draft appears under the callout parent.
4. Check **Approve for Pawn memory** when ready (watcher indexes it).

The document root is the agent session (`siyuan:{root_id}`) — same as the
previous SQL watcher path.
