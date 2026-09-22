# Pawn (SiYuan plugin)

Sends the current block (as a TIP callout) to pawn-server via
`POST /v1/siyuan/triggers`. **No `@pawn` prefix required** for Send — that
token is only for the optional server watcher SQL scan.

## Install

**From the UI:** zip this folder and use Marketplace → Install Bazaar package:

```bash
cd /path/to/parakeet/siyuan-plugin/pawn
zip -r ../pawn-0.1.12.zip plugin.json index.js README.md i18n
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
| Mention token (watcher) | `@pawn` | Server watcher only; Send ignores it |
| Wrap as callout on send | on | TIP callout created when you send |
| Callout icon | 🤖 | Embedded in TIP title |
| Send button label | `Send to Pawn` | Toolbar / menu / command label |
| Request timeout (ms) | `15000` | `forwardProxy` timeout (202 returns quickly) |

## Usage

### Plugin Send (primary)

1. Write any instruction in a paragraph (no `@pawn` needed).
2. **Send** when ready:
   - Select text → floating toolbar → **Send to Pawn**
   - Block gutter icon → **Send to Pawn**
   - Hotkey **⌥⌘P** (Alt+Ctrl+P on Linux; remap under Settings → Keymap → Pawn)
3. With wrap-on-send enabled, the paragraph becomes a TIP callout and
   pawn-server runs the agent.

### Watcher mention (optional)

If `siyuan_watcher.discover_mentions: true`, typing a leading `@pawn …` still
lets the server SQL poll claim the block without pressing Send. Keep the
plugin mention token aligned with `siyuan_watcher.mention_token`.

The document root is the agent session (`siyuan:{root_id}`).
