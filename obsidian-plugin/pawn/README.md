# Pawn (Obsidian)

A Copilot-style side pane for **Pawn AI**, plus background jobs that report
back. Works on **desktop and mobile** (`isDesktopOnly: false`).

This plugin was written from scratch (MIT). Its UX takes ideas from
[obsidian-copilot](https://github.com/logancyang/obsidian-copilot) (AGPL-3.0),
but it contains none of that project's code. If you prefer Copilot itself,
point it at Pawn's OpenAI-compatible endpoint; see
[docs/OBSIDIAN_AGENT.md](../../docs/OBSIDIAN_AGENT.md#use-with-obsidian-copilot).

## Install

```bash
npm install
npm run build          # type-check + bundle to main.js
```

Copy only the three runtime files (`main.js`, `manifest.json`, `styles.css`)
into `<vault>/.obsidian/plugins/pawn/` (or symlink this folder on desktop),
then enable **Pawn** under **Settings → Community plugins**. Use `npm run
dev` to rebuild on change.

### Android (ADB)

When vault sync skips `.obsidian/plugins`, push the bundle to a USB-connected
phone:

```bash
make adb-list-vaults                          # discover vault paths
make adb-push ADB_VAULT=/sdcard/Documents/MyVault
```

From the repo root: `make obsidian-plugin-adb ADB_VAULT=…`. Obsidian is
force-stopped after push so the next open reloads the plugin.

## Chat

- Open with the ribbon icon, **Pawn: Open chat**, or **Ask Pawn** in the
  editor menu.
- The conversation follows the active note (`note:<path>`) unless you pick
  another one in the header (pinned) or start a **New chat** (`chat:<uuid>`).
- **Context chips**: the open note (× unpins it; the dashed pin chip attaches
  it again), current selection, and extra notes added with `@`, the `+` chip,
  the file menu (**Add to Pawn chat context**), or by dragging notes from the
  file explorer.
- Replies stream: tool steps appear live, then the answer renders as Markdown.
  Actions on each reply: **Copy**, **Insert at cursor**, **Replace
  selection** (diff preview; targets the selection you asked about),
  **Append to note**, **Save as new note** (`<agent root>/Notes`).
- **Stop** aborts the stream. The server may still finish the turn, and it
  stays in Pawn's memory.
- `/reset` clears the conversation on the server and locally.

## Prompt commands

Type `/` in the composer, use the palette (**Pawn: Prompt: …**, **Pawn: Run
prompt command…**), or the editor menu (**Pawn: …**). Built-ins: Summarize,
Rewrite for clarity, Fix grammar, Translate to English, Extract action items.

Add your own as Markdown files in the commands folder (default
`Pawn/Commands`; **Settings → Create defaults** writes the built-ins there to
edit):

```markdown
---
name: Meeting recap
description: Decisions and owners
slash: recap            # /recap (defaults to the file name)
context_menu: true      # show in the editor menu
background: false       # true = run as a background job
---
Write a recap of {selection}: decisions, owners, open questions.
```

Placeholders: `{selection}` (the selection if any, otherwise the active note),
`{note}` (active note link), `{date}`. The selection and active note are
always attached as structured context.

## Background jobs

- Tick **Background** in the composer, or use **Send to Pawn (background)**
  from the editor menu or palette. The job card appears in the thread right away
  and updates live. The **Jobs** tab lists all jobs (All / Running / This
  conversation).
- **Uploads**: paperclip button, drag files from disk into the pane, or
  **Upload to Pawn** in the file menu. Audio goes to transcription
  (`transcribe-diarize`); other files are saved to `Pawn/Inbox/` (text files
  are also indexed into Pawn's memory).
- Finished jobs raise a notice (click to open). Results offer Insert /
  Replace / Append / Save plus **Approve** (index into Pawn memory),
  **Cancel**, and **Open task note**.
- **Offline**: if the server is unreachable, the job is saved as
  `Pawn/Tasks/<id>.md` (`status: todo`). The server's vault watcher runs it
  after Sync Engine uploads it; the result syncs back and shows in the Jobs
  tab.
- The status bar shows server reachability, running jobs, results
  awaiting review, and inbox items that need a tap. Click it to open the Jobs tab.
- **Inbox** tab lists coworker items (File / Task / Later / Ignore, or Approve / Reject).
- **Quick capture** saves an `#idea` note under `Ideas/`. **Apply goals proposal**
  writes `Goals.md` from the `goals` fence in the active weekly review, after a diff.

## Settings

| Setting | Default | Description |
|--------|---------|-------------|
| Server URL | `http://127.0.0.1:8000` | `pawn-server` base URL |
| API token | *(empty)* | `api.token` from `pawnai.yaml` |
| Default conversation | Per note | Per note (`note:<path>`) or one global chat |
| Include active note | on | Attach the active note by default |
| Send local note content | on | Send note bodies from this device (unsynced edits included) |
| Prompt commands folder | `Pawn/Commands` | Markdown prompt commands |
| Agent root | `Pawn` | Folder Pawn owns (`Tasks/`, `Notes/`, `Inbox/`) |
| Notify when a job finishes | on | Notice on completion |
| Insert callout for background jobs | off | Adds `> [!pawn] [[task note]]` at the cursor |

## Server endpoints used

| Purpose | Endpoint |
|---------|----------|
| Health | `GET /health` |
| Chat (SSE) | `POST /v1/pawn/chat` |
| Jobs | `POST /v1/jobs`, `POST /v1/jobs/upload`, `GET /v1/jobs`, `GET /v1/jobs/{id}` |
| Job actions | `POST /v1/jobs/{id}/approve`, `POST /v1/jobs/{id}/cancel` |
| Live job updates | `GET /v1/jobs/events` (desktop; mobile polls) |

Desktop streams with `fetch`, so the server must allow the `app://obsidian.md`
origin (`api.cors_origins`, on by default). Mobile uses Obsidian's
`requestUrl`, which does not stream: the reply arrives when the turn finishes.
