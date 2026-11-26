# Agent Scheduler

`pawn-server` can run durable, database-backed `pawn-agent` prompts on a
schedule. Schedules live in PostgreSQL and are claimed by the long-running
server process.

## Configuration

```yaml
agent_scheduler:
  enabled: true
  poll_interval_seconds: 30
  max_due_per_tick: 5
  default_timezone: UTC
  stale_fire_after_seconds: 3600
  allow_agent_auto_apply: false
```

`allow_agent_auto_apply` is intentionally conservative. Agent tools create
proposals; application-owned CLI/API code applies mutations.

## Authority Model

The LLM-facing tool `propose_schedule_change` writes rows to
`agent_schedule_proposals` only. It does not create, update, pause, resume, or
cancel schedules directly.

Use application controls to approve or reject proposals:

```bash
pawn-server schedules proposals
pawn-server schedules approve <proposal-id>
pawn-server schedules reject <proposal-id>
```

## Schedule Controls

```bash
pawn-server schedules list
pawn-server schedules show <schedule-id>
pawn-server schedules pause <schedule-id>
pawn-server schedules resume <schedule-id>
pawn-server schedules cancel <schedule-id>
```

Run the scheduler with the server:

```bash
pawn-server serve
```

Operational flags:

```bash
pawn-server serve --disable-scheduler
pawn-server serve --scheduler-only
```

`session_id` is the conversation key the turn resumes: a diarization session name, `note:{path}`, or something like `coworker:daily`. Optional `output_note` is a vault path; each fire's reply is appended there.

Proposals are still approved with `pawn-server schedules approve`. When `coworker.enabled` is true, the same proposal also appears as an inbox item you can `approve` or `reject` from Matrix or the Obsidian plugin.

## Schedule Shapes

One-shot:

```json
{
  "name": "Tomorrow summary",
  "prompt": "Summarize session abc123",
  "session_id": "abc123",
  "schedule": {"schedule_kind": "once", "run_at": "2026-05-01T09:00:00+00:00"}
}
```

Interval:

```json
{
  "name": "Hourly check",
  "prompt": "Check session abc123 for follow-ups",
  "session_id": "abc123",
  "schedule": {"schedule_kind": "interval", "interval_seconds": 3600}
}
```

Cron schedules require `croniter` to be installed:

```json
{
  "name": "Morning recap",
  "prompt": "Write the daily recap for session abc123",
  "session_id": "abc123",
  "timezone": "Europe/Rome",
  "schedule": {"schedule_kind": "cron", "cron_expression": "0 9 * * *"}
}
```

### Morning day-plan check-in

Interactive planning uses the `day_plan` skill (see [PLANNER.md](PLANNER.md)).
Ask chat to propose a weekday morning schedule, then approve it. Example payload:

```json
{
  "name": "Morning day plan check-in",
  "prompt": "Morning day-plan check-in. Load planner_day and Goals.md. Ask what is going on today (1–3 short questions). After I answer in this conversation, write ## Plan via day_plan_update. On this first unattended turn, queue_push a short Matrix notify with the check-in question; do not invent a completed plan.",
  "session_id": "coworker:daily",
  "timezone": "Europe/Rome",
  "schedule": {"schedule_kind": "cron", "cron_expression": "0 8 * * 1-5"}
}
```

Continue the dialogue on conversation key `coworker:daily` (Obsidian / CLI chat).
This is separate from `coworker.briefing_cron`, which stays a one-shot daily summary.
