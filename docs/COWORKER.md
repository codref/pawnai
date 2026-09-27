# Using the coworker loop

The coworker loop keeps a short list of what you are trying to move, files what comes out of meetings and notes, and interrupts you only when an active thread is affected. You triage from the vault, from Matrix, or from the plugin Inbox. The loop stays off until `coworker.enabled` is true.

## Turn it on

Apply the coworker migrations, then point a finished transcript at the loop:

```bash
alembic upgrade head
```

That applies `0016` through `0019` (inbox items, the knowledge index, thread state, and autonomy).

In `pawnai.yaml`:

```yaml
diarize_queue:
  chain_agent:
    enabled: true
    command: session_completed

coworker:
  enabled: true
  goals_path: Goals.md
  timezone: Europe/Rome
  briefing_cron: "0 8 * * *"
  weekly_cron: "0 17 * * 5"
  me:
    - Davide
```

`chain_agent.command: session_completed` replaces the free-form “analyze this conversation” prompt. `command: run` still does that older path. Both need `chain_agent.enabled: true`.

`coworker.me` is the names the morning briefing treats as yours when it looks for overdue commitments. `timezone` is used for quiet hours, the daily cap, and the cron clocks when `Goals.md` does not set its own timezone.

Start the server:

```bash
pawn-server serve
```

`--no-coworker` forces the loop off. `--coworker-only` runs the briefing, the weekly review, and the vault scan without the HTTP API. Check the switch with:

```bash
pawn-server coworker status
```

## Write Goals.md

`coworker.goals_path` (default `Goals.md`) lives at the vault root, outside `Pawn/`. Pawn reads it and does not write it. Put three to seven threads under `## Active`. Everything else stays quiet.

A missing note means “file everything, notify nothing.” Any `pawn:` value other than `goals` is treated the same way.

```markdown
---
pawn: goals
reviewed: 2026-09-28
timezone: Europe/Rome
notify:
  max_per_day: 5
  quiet_hours: "21:00-08:00"
ignore:
  - status meetings with no decisions
  - tooling chatter
autonomy: off
---

## Active

### Project X storage
- why: The storage decision is blocking the Friday deploy.
- movement: A chosen option, or an owner and a date.
- interrupt: A new commitment, a decision with no owner, or a contradiction with keeping it on our S3.

### Hiring backend
- why: Two conversations this month and still no next step.
- movement: A named next conversation, or a written no.
- interrupt: A new name, a stalled loop, or a commitment I made and did not capture.

## Parked

### Mobile inbox
- note: [[Ideas/Mobile inbox]]
- do: Link related meetings and develop the companion note. Do not notify.

## Done recently

### Vault transcripts auto-push
- closed: 2026-09-20
```

Each active thread is a test, not a wish. `why` says why it is on the list. `movement` is what “done enough” looks like. `interrupt` is the only reason Pawn may ping you. A topic mention is not enough; the item has to be a commitment, a decision, an open question, a block, or a contradiction tied to that thread.

`## Parked` is for ideas you want linked and developed with no notification. A heading whose title starts with `Done` is closed history.

`autonomy` in this note overrides `coworker.autonomy.mode`. `off` means Pawn will not start follow-up work on its own.

Attention rules:

| Field | Effect |
|-------|--------|
| `notify.max_per_day` | Cap on interrupts. Default 5. Further items are still filed. |
| `notify.quiet_hours` | `HH:MM-HH:MM`, and the window may wrap midnight. Items are filed and the ping waits. |
| `ignore` | Phrases that match the item are filed and never notified. |
| `timezone` | Local clock for quiet hours and the daily cap. Falls back to `coworker.timezone`, then UTC. |

## What you see after a meeting

When a `transcribe-diarize` job finishes and the chain command is `session_completed`, the server:

1. Extracts decisions, commitments, open questions, blocks, and contradictions.
2. Scores them against the active threads. A hit has to name one of those threads.
3. Writes `Pawn/Items/{short_id}.md` and refreshes `Pawn/Today.md`.
4. Pushes Matrix (and optional ntfy) only for interrupts.

Duplicates you already triaged, ignored fingerprints, quiet hours, and the daily cap still create the item note. They do not notify. `Pawn/Today.md` lists what needs a tap under “Needs you”, then what was filed quietly.

Run one session by hand:

```bash
pawn-server coworker process --session <diarization-session-id>
```

An item note looks like this. The eight-character `short_id` is what you reply with.

```markdown
---
pawn: item
id: 3f2a1c0e-7b44-4d11-9a20-0c5e8b1d6f77
short_id: a1b2c3d4
status: notified
kind: decision
thread: Project X storage
action:
---

Keep the recordings on our S3 through Friday.

> We should not move the bucket before the deploy.

Source: [[Pawn/Transcripts/2026-09-28 session-id.md]]

## Why you were notified

A decision with no owner on Project X storage.
```

## Triage

Use any of the three surfaces. They call the same actions.

| Action | Effect |
|--------|--------|
| `file <id>` | Appends the item to `Pawn/Threads/{slug}.md` |
| `task <id>` | Records an open loop on that thread and opens an ask job |
| `later <id> [YYYY-MM-DD]` | Snoozes until that date, or until tomorrow when the date is omitted |
| `ignore <id>` | Dismisses it and suppresses the same fingerprint later |
| `approve <id>` / `reject <id>` | A schedule proposal, or a research follow-up |

`<id>` is the `short_id` or the full UUID.

**Obsidian.** Open the Pawn pane on the Inbox tab, or run **Show inbox**. Each card has File, Task, Later, and Ignore. Schedule proposals and research proposals show Approve and Reject. Later from the plugin snoozes until tomorrow. The status bar counts items that still need a tap.

**The item note.** Set `action:` to `file`, `task`, `later`, or `ignore` and let it sync. The vault watcher applies it on its next poll (default 15 seconds). `later` from the note is also tomorrow. A note already filed, tasked, or dismissed is left alone.

**Matrix.** In a DM, send the command as the whole message. In a room, put it after `command_prefix` (default `!pawn`):

```text
file a1b2c3d4
later a1b2c3d4 2026-10-03
ignore a1b2c3d4
```

Pawn replies with a one-line receipt and does not start a chat turn.

**HTTP.** With the API token:

```bash
curl -X POST "$PAWN_URL/v1/items/a1b2c3d4/action" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action":"later","arg":"2026-10-03"}'
```

`GET /v1/items?status=notified` lists the inbox.

## Capture an idea

Ideas are notes you write. Pawn does not edit them. A note is picked up when it lives under `coworker.watch_folders` (default `Ideas/`) or carries `coworker.watch_tags` (default `idea`), and only after it has been unchanged for `note_quiet_seconds` (default 120).

Save this as `Ideas/Mobile inbox.md`. It matches the parked thread above.

```markdown
---
tags: [idea]
---

# Mobile inbox

The phone should be a short list of judgments, not a chat transcript.

When a meeting ends, Pawn files decisions and commitments into the vault. On the phone I only want the ones that need a tap: file this, remind me tomorrow, or ignore it.

Open question: does this live as `Pawn/Today.md`, or as a separate inbox note I can pin?
```

The plugin command **Quick capture** writes `Ideas/{date} {title}.md` with `tags: [idea]` for you.

After the quiet window, Pawn writes a companion at `Pawn/Ideas/mobile-inbox.md` that links back to your note, and it indexes the text. Because the matching thread is Parked, this does not notify. Promote the thread under `## Active` when you want interrupts.

`knowledge_search --query "mobile inbox"` searches notes, transcripts, and items. Backfill with:

```bash
pawn-server coworker reindex
pawn-server coworker reindex --notes
pawn-server coworker reindex --sessions
```

With no flags, both notes and transcripts are indexed.

## Phone audio

Set `coworker.capture_audio_dir` to a vault folder your phone recorder syncs into, for example `Pawn/Capture/audio`. New `m4a`, `webm`, `wav`, `mp3`, `ogg`, and `flac` files are copied onto the diarize queue with `chain_agent.command: session_completed`. Pawn leaves a stub at `Pawn/Capture/{session}.md`. An empty `capture_audio_dir` disables this.

The same chain is attached when the plugin uploads audio and the coworker loop is enabled.

## Morning and Friday

`coworker.briefing_cron` (default `0 8 * * *`) unsnoozes items whose date has arrived, writes `Pawn/Daily/{date}.md`, and sends one summary. The summary calls out:

- commitments owned by a name in `coworker.me` that are older than `coworker.commitment_days` (default 7)
- decisions with no owner
- questions that have come up at least three times
- active threads with no movement for `coworker.stale_days` (default 14)

`coworker.weekly_cron` (default `0 17 * * 5`) writes `Pawn/Reviews/{week}.md`. Inside it, a fenced `goals` block is a proposed `Goals.md`. Stale active threads are moved under Parked in that proposal. Open the review, then run **Apply goals proposal**. The plugin shows a diff and writes `Goals.md` only after you confirm.

## How far it may act while you are away

```yaml
coworker:
  autonomy:
    mode: suggest_only   # off | suggest_only | approve_writes | limited_act
    auto_actions: [research]
    max_depth: 2
    max_self_jobs_per_event: 3
    max_self_jobs_per_day: 20
```

| Mode | What Pawn does unattended |
|------|---------------------------|
| `off` | Files and notifies. No follow-up jobs. |
| `suggest_only` | Default. Research shows up as a proposal you approve. |
| `approve_writes` | Same suggestions. Writes outside `Pawn/` still wait for you. |
| `limited_act` | May run a research follow-up into the thread note when `research` is in `auto_actions`. |

`Goals.md` frontmatter `autonomy:` overrides this mode. Writes outside `Pawn/` are never automatic. Self-enqueued work carries a parent run, a depth, and an event id. The queue drops a child that exceeds `max_depth`, the per-event cap, the per-day cap, or that repeats a prompt from the last 24 hours. Those drops are recorded and the message is acknowledged.

Stop unattended follow-ups without disabling the inbox:

```bash
pawn-server coworker pause
pawn-server coworker resume
```

Pause writes `{agent.sallm.state_dir}/coworker.paused` (default `.sallm/coworker.paused`).

Schedule proposals also arrive as inbox items. `approve` and `reject` call the scheduler. A schedule may set `output_note` so each fire is appended to a vault note. Its `session_id` is any conversation key (`note:Path`, `coworker:daily`, or a diarization session id).

## Notify

Matrix uses `coworker.matrix_target` (default `matrix`, the `queue_producers` entry) and `matrix_bot.notify_room_id`. Optional phone push:

```yaml
coworker:
  notify:
    ntfy_url: https://ntfy.example
    topic: pawn
    token: ""
```

Env vars use `PAWN_COWORKER__*` (`PAWN_COWORKER__ENABLED=true`, `PAWN_COWORKER__TIMEZONE=Europe/Rome`).

## Commands

| Command | What it does |
|---------|----------------|
| `pawn-server serve` | Runs the loop when `coworker.enabled` is true |
| `pawn-server serve --no-coworker` | Inbox, briefing, and scan stay off |
| `pawn-server serve --coworker-only` | Loop only, no HTTP API |
| `pawn-server coworker status` | Enabled, paused, and autonomy mode |
| `pawn-server coworker pause` / `resume` | Kill switch for unattended follow-ups |
| `pawn-server coworker process --session ID` | Run the loop for one finished session |
| `pawn-server coworker reindex` | Backfill the knowledge index |
