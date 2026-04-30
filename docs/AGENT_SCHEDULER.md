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
