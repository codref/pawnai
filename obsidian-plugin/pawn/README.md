# Pawn (Obsidian plugin)

Co-author with Pawn AI from Obsidian (desktop and mobile). Creates vault task
notes under `Pawn/Tasks/`, talks to `pawn-server` when reachable, and falls
back to S3 sync for the vault watcher.

## Install

```bash
cd obsidian-plugin/pawn
npm install
npm run build
```

Symlink or copy this folder into your vault:

```bash
ln -s /path/to/parakeet/obsidian-plugin/pawn \
  /path/to/vault/.obsidian/plugins/pawn
```

Enable **Pawn** under Settings → Community plugins (or turn off Safe mode and
enable the local plugin).

## Sync Engine settings (required)

Pawn writes plain Markdown into the same S3 prefix Sync Engine uses. Configure:

| Setting | Value |
|---------|-------|
| Asymmetric storage | **Off** |
| Client-side encryption | **Off** |
| Prefix | Same as `vault.prefix` in `pawnai.yaml` |
| Sync strategy | Bidirectional |
| Conflict strategy | Smart merge or keep both |
| Interval / startup sync | **On** |

## Settings

| Key | Default | Notes |
|-----|---------|-------|
| Server URL | `http://127.0.0.1:8000` | Must be reachable for chat / fast path |
| API token | _(empty)_ | Same as `api.token` in `pawnai.yaml` |
| Agent root | `Pawn` | Folder Pawn owns |
| Fast-path timeout (ms) | `60000` | Wait for HTTP reply before sync fallback |
| Always queue | off | Skip HTTP; leave tasks as `todo` |

## Usage

1. **Ask Pawn** (command palette, editor menu, or mobile toolbar via commands):
   creates `Pawn/Tasks/<uuid>.md`, inserts a `> [!pawn]` callout, and tries
   the HTTP fast path. If the server is down, the task stays `todo` and the
   vault watcher runs it after sync.
2. **Pawn panel** (ribbon / command): Tasks tab for the active note (Insert,
   Replace, Reply, Approve, Open) and Chat tab (online only, `user=note:…`).
3. **Approve** indexes the result into Pawn memory (HTTP when online, else
   watcher after sync sees `approved: true`).

See `docs/OBSIDIAN_AGENT.md` in the PawnAI repo for the full architecture.
