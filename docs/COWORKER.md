# Coworker loop

Pawn can keep a short list of what you are trying to move, file what comes out of meetings and notes, and interrupt you only when an active thread is affected. You triage from the vault, from Matrix, or from the plugin Inbox.

The loop stays off until `coworker.enabled` is true.

## Goals note

`coworker.goals_path` (default `Goals.md`) lives outside `Pawn/`. Pawn reads it and does not write it. A missing note means "file everything, notify nothing."

```markdown
---
pawn: goals
timezone: Europe/Rome
notify:
  max_per_day: 5
  quiet_hours: "21:00-08:00"
ignore:
  - status meetings
autonomy: off
---

## Active

### Project X storage
- why: Decision is blocking Friday.
- movement: A chosen option, or an owner and a date.
- interrupt: A new commitment or a decision with no owner.

## Parked

### Mobile inbox
- note: [[Ideas/Mobile inbox]]
- do: Link related meetings. Do not notify.
```

`autonomy: off` in the note overrides `coworker.autonomy.mode` from the phone.

## What a finished meeting does

With `diarize_queue.chain_agent.command: session_completed`, a successful `transcribe-diarize` publishes `session_completed` instead of a free-form agent prompt. The server then:

1. Extracts decisions, commitments, open questions, blocks, and contradictions.
2. Scores them against the active threads.
3. Drops interrupts that are duplicates, ignored, inside quiet hours, or over the daily cap. Those items are still filed.
4. Writes `Pawn/Items/{short_id}.md` and refreshes `Pawn/Today.md`.
5. Pushes Matrix (and optional ntfy) only for interrupts.

Manual backfill: `pawn-server coworker process --session <id>`.

## Triage

Reply in Matrix, set `action:` on the item note, or use the plugin Inbox:

| Action | Effect |
|--------|--------|
| `file <id>` | Appends the item to `Pawn/Threads/{slug}.md` |
| `task <id>` | Opens an ask job and records an open loop |
| `later <id> [YYYY-MM-DD]` | Snoozes until that time, or the next day |
| `ignore <id>` | Dismisses it and suppresses the same fingerprint later |
| `approve <id>` / `reject <id>` | Schedule proposals, or a research follow-up |

The morning cron (`coworker.briefing_cron`, default 08:00 in `coworker.timezone`) un-snoozes due items, writes `Pawn/Daily/{date}.md`, and sends one summary. Open loops called out there: your overdue commitments (`coworker.me`), decisions with no owner, questions that have come up at least three times, and active threads with no movement for `coworker.stale_days`.

## Notes, ideas, and the phone

`coworker.watch_folders` (default `Ideas/`) and `coworker.watch_tags` (default `idea`) are scanned by the coworker worker. A note is processed only after it has been unchanged for `note_quiet_seconds`. Idea notes get a companion under `Pawn/Ideas/` that links back; your original note is not edited.

`knowledge_search --query "..."` searches notes, transcripts, and items. Backfill with `pawn-server coworker reindex`.

Audio dropped in `coworker.capture_audio_dir` is copied to the diarize queue with `chain_agent.command: session_completed`. The plugin command **Quick capture** writes `Ideas/{date} {title}.md`.

## Weekly review

`coworker.weekly_cron` (default Friday 17:00) writes `Pawn/Reviews/{week}.md` with a proposed `Goals.md` inside a `goals` fence. The plugin command **Apply goals proposal** shows a diff and writes `Goals.md` only after you confirm.

## Autonomy

```yaml
coworker:
  enabled: true
  autonomy:
    mode: suggest_only   # off | suggest_only | approve_writes | limited_act
    max_self_jobs_per_event: 3
    max_self_jobs_per_day: 20
    max_depth: 2
    auto_actions: [research]
```

`limited_act` may run a research follow-up into the thread note. Writes outside `Pawn/` are never automatic. Self-enqueued runs carry `parent_run_id`, `depth`, and `event_id`; the queue drops anything over the caps or a duplicate prompt. `pawn-server coworker pause` (or `resume`) is the kill switch. Decisions land in `coworker_decisions`.

Schedule proposals also show up as inbox items (`approve` / `reject`). A schedule may set `output_note` so each fire is appended to a vault note. `session_id` is any conversation key (`note:Path`, `coworker:daily`, or a diarization session).

## Notify

Matrix uses `coworker.matrix_target` (default `matrix`) and `matrix_bot.notify_room_id`. Optional phone push:

```yaml
coworker:
  notify:
    ntfy_url: https://ntfy.example
    topic: pawn
    token: ""
```

## Run

`pawn-server serve` starts the loop when `coworker.enabled` is true. `--no-coworker` forces it off. `--coworker-only` runs the loop without the HTTP API.
