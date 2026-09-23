# Pawn (SiYuan plugin)

Inserts a **Pawn prompt** custom block and posts that block id to pawn-server
via `POST /v1/siyuan/triggers`.

The block is a SiYuan custom block (`;;;pawn/prompt`). It is a single text
region with a left border — no title and no icon. `/prompt` only inserts it.
**Send** turns the selection into that block and sends it.

Requires SiYuan **3.8.3** or newer (custom blocks).

## Install

**From the UI:** zip this folder and use Marketplace → Install Bazaar package:

```bash
cd /path/to/parakeet/siyuan-plugin/pawn
zip -r ../pawn-0.2.1.zip plugin.json index.js index.css README.md i18n
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
| Send button label | `Send to Pawn` | Toolbar / menu / command label |
| Request timeout (ms) | `15000` | `forwardProxy` timeout (202 returns quickly) |

## Usage

1. Type `/prompt` (or `/pawn`) and choose **Pawn prompt**. That paragraph
   becomes a prompt. Text written before the slash is kept. Nothing is sent.
2. Write the instruction in the block. It is one text region (markdown text,
   not nested blocks).
3. **Send** when ready (any of these):
   - Floating toolbar → **Send to Pawn** (paper-plane icon)
   - Block gutter icon → **Send to Pawn**
   - Hotkey **⌥⌘P** (Alt+Ctrl+P on Linux; remap under Settings → Keymap → Pawn)
4. If the selection is already a Pawn prompt, Send posts that block. Otherwise
   Send replaces the selected blocks with one prompt containing their text,
   then posts `{ "block_id" }` as JSON through `forwardProxy`.
5. A review draft appears under the prompt's parent. Check **Approve for Pawn
   memory** when ready (watcher indexes it).

The document root is the agent session (`siyuan:{root_id}`).

Existing TIP callouts that still contain `@pawn` can be sent; the plugin no
longer creates them.
