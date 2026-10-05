# Agent Scheduler

## Summary

Add a durable scheduler for pawn-agent runs. Scheduled prompts live in
PostgreSQL, a `pawn-server` background worker claims due schedules and runs the
LangGraph agent, and agent-facing tools can propose schedule changes without
being the final authority that applies them.

## Key Changes

- Add a DB-backed scheduler domain in `pawn_agent/core/scheduler.py`.
- Add ORM models in `pawn_agent/utils/db.py` for schedules, schedule proposals,
  and individual fire attempts.
- Add Alembic migration `0010_agent_schedules.py`.
- Add a config section in `pawn_agent.utils.config`:
  `agent_scheduler.enabled`, `poll_interval_seconds`, `max_due_per_tick`,
  `default_timezone`, `stale_fire_after_seconds`, and `allow_agent_auto_apply`.
- Start the scheduler as a third background task in `pawn-server` alongside the
  HTTP API and queue listener.
- Add `pawn-server --disable-scheduler` and, if useful for operations,
  `pawn-server scheduler` or `pawn-server --scheduler-only`.
- Reuse `LangGraphSessionRegistry` for scheduled runs so scheduled prompts keep
  the same conversation/session behavior as API and queue-initiated turns.
- Extract queue run execution into a shared agent-run helper so the queue
  listener and scheduler both create/update `agent_runs` consistently.
- Add `source`, `schedule_id`, and `scheduled_fire_id` fields to `agent_runs`
  so queue/API/scheduled execution history can be distinguished.
- Add agent tool support for schedule management proposals.
- Wire schedule proposal tools into LangGraph routing with conservative planner
  instructions.
- Add API or CLI endpoints/commands for application-owned approval, rejection,
  listing, pausing, resuming, updating, and cancelling.
- Document the schedule contract and approval model in `docs/`.

## Data Model

- `agent_schedules`
  - `id`, `name`, `prompt`, `session_id`, `model`
  - `status`: `active`, `paused`, `cancelled`, `completed`
  - `schedule_kind`: `once`, `interval`, `cron`
  - `timezone`, `run_at`, `interval_seconds`, `cron_expression`
  - `next_run_at`, `last_run_at`, `created_at`, `updated_at`
  - `created_by`: `user`, `agent_proposal`, `system`
  - `created_from_proposal_id`, `revision`, `metadata`
- `agent_schedule_proposals`
  - `id`, `action`: `create`, `update`, `pause`, `resume`, `cancel`
  - `schedule_id`, `proposed_payload`, `rationale`
  - `status`: `proposed`, `approved`, `rejected`, `applied`, `expired`
  - `proposed_by_run_id`, `proposed_by_session_id`
  - `reviewed_by`, `reviewed_at`, `created_at`
- `agent_schedule_fires`
  - `id`, `schedule_id`, `scheduled_for`, `status`
  - `agent_run_id`, `attempt`, `claimed_at`, `started_at`, `completed_at`
  - `error`, `created_at`
  - Unique key on `(schedule_id, scheduled_for)` to prevent duplicate runs.

## Scheduler Flow

- On startup, `pawn-server` creates an async scheduler task when
  `agent_scheduler.enabled` is true.
- Each tick selects due active schedules with `next_run_at <= now()` using a
  Postgres transaction and row locking (`FOR UPDATE SKIP LOCKED`).
- The scheduler creates or claims an `agent_schedule_fires` row for each due
  schedule and advances `next_run_at` before executing the agent.
- Runs happen outside the DB transaction.
- Each fire calls the shared agent runner with:
  - `prompt` from the schedule row
  - `session_id` from the schedule row
  - optional per-schedule model override
  - `source="schedule"` plus schedule/fire ids
- Completion updates both `agent_runs` and `agent_schedule_fires`.
- Failures are recorded durably and do not delete or silently disable the
  schedule.
- Stale `running` or `claimed` fires older than
  `stale_fire_after_seconds` can be retried or marked failed by a later tick.

## Authority Model

- The LLM does not directly create, update, pause, resume, or cancel schedules.
- Agent-facing tools write `agent_schedule_proposals` only.
- The application owns all schedule mutations through deterministic service
  methods:
  - validate schedule shape
  - normalize timezones and next run times
  - check schedule ownership/session scope
  - enforce status transition rules
  - decide whether a proposal is approved, rejected, or left for review
- Default behavior should require explicit user confirmation before applying an
  agent-created proposal.
- If `allow_agent_auto_apply` is ever enabled, it should only apply proposals
  that match a narrow, deterministic policy, such as an explicit user request in
  the current turn plus a valid single schedule mutation.
- Tool responses should say "proposal created" or "proposal approved/applied";
  never imply that the model alone performed an irreversible action.

## Agent Tool Surface

- Add `pawn_agent/tools/propose_schedule.py`.
- Tool name: `propose_schedule_change`.
- Tool args:
  - `action`
  - `schedule_id`
  - `name`
  - `prompt`
  - `schedule`
  - `timezone`
  - `model`
  - `rationale`
- The tool validates only basic payload shape, then stores a proposal through
  the scheduler service.
- The planner may use this tool when the user explicitly asks to schedule,
  reschedule, pause, resume, or cancel recurring agent work.
- The planner should not infer long-running schedules from casual wording.
- Add a reply step after proposal creation so the user can confirm the exact
  schedule details.

## Application Controls

- Add CLI commands under `pawn-server` or `pawn-agent`:
  - `schedules list`
  - `schedules show <id>`
  - `schedules approve <proposal-id>`
  - `schedules reject <proposal-id>`
  - `schedules pause <id>`
  - `schedules resume <id>`
  - `schedules cancel <id>`
- Optionally add FastAPI routes after the CLI path is stable:
  - `GET /v1/schedules`
  - `GET /v1/schedule-proposals`
  - `POST /v1/schedule-proposals/{id}/approve`
  - `POST /v1/schedule-proposals/{id}/reject`
  - `POST /v1/schedules/{id}/pause`
  - `POST /v1/schedules/{id}/resume`
  - `DELETE /v1/schedules/{id}`
- Use the same auth expectations as `/v1/chat/completions` for any HTTP
  management route.

## Test Plan

- Unit-test schedule validation for one-shot, interval, and cron schedules.
- Unit-test timezone normalization and `next_run_at` calculation.
- Unit-test proposal creation for create/update/pause/resume/cancel.
- Unit-test proposal approval and rejection paths.
- Unit-test that agent-facing tools cannot directly mutate
  `agent_schedules`.
- Unit-test due schedule claiming with duplicate-fire protection.
- Unit-test stale fire handling.
- Unit-test scheduler execution with a mocked `LangGraphSessionRegistry`.
- Unit-test shared agent runner persistence for queue and scheduled runs.
- Update queue listener tests after extracting shared run execution.
- Update LangGraph tests so the graph registers `tool_propose_schedule_change`.
- Add CLI/API tests for listing, approval, pause, resume, and cancel controls.
- Run targeted tests:

```bash
pytest tests/test_agent_scheduler.py tests/test_agent_queue_listener.py tests/test_langgraph_chat.py --no-cov
```

## Assumptions

- V1 stores schedules in PostgreSQL rather than in memory or only in
  `pawnai.yaml`.
- `pawn-server` is the right long-running process to own schedule ticks.
- V1 can run scheduled prompts inline through `LangGraphSessionRegistry`;
  publishing due work to `pawn-queue` can be added later if separate workers are
  needed.
- Cron support should use a small dependency such as `croniter`; if avoiding a
  new dependency is preferred, start with `once` and `interval` schedules only.
- User confirmation is the default authority boundary for LLM-proposed schedule
  changes.
