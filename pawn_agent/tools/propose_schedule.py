"""Tool: propose_schedule_change - create schedule-management proposals."""

from __future__ import annotations

from typing import Any, Optional

from pydantic_ai import Tool

from pawn_agent.core.scheduler import AgentSchedulerService
from pawn_agent.utils.config import AgentConfig

NAME = "propose_schedule_change"
DESCRIPTION = (
    "Propose creating, updating, pausing, resuming, or cancelling scheduled "
    "agent work. The proposal must be approved by the application before it applies."
)


def _session_var(session_vars: Any, key: str) -> Optional[str]:
    if not session_vars:
        return None
    if isinstance(session_vars, dict):
        value = session_vars.get(key)
    else:
        value = getattr(session_vars, key, None)
    return str(value) if value else None


async def propose_schedule_change_impl(
    cfg: AgentConfig,
    *,
    action: str,
    schedule_id: Optional[str] = None,
    name: Optional[str] = None,
    prompt: Optional[str] = None,
    schedule: Optional[dict[str, Any]] = None,
    timezone: Optional[str] = None,
    model: Optional[str] = None,
    rationale: Optional[str] = None,
    proposed_by_session_id: Optional[str] = None,
    proposed_by_run_id: Optional[str] = None,
) -> str:
    """Store a schedule proposal and return a user-facing receipt."""
    payload: dict[str, Any] = {}
    if name:
        payload["name"] = name
    if prompt:
        payload["prompt"] = prompt
    if schedule is not None:
        if not isinstance(schedule, dict):
            return "Error: schedule must be a JSON object."
        payload["schedule"] = schedule
        if "session_id" in schedule:
            payload["session_id"] = schedule["session_id"]
    if timezone:
        payload["timezone"] = timezone
    if model:
        payload["model"] = model

    normalized_action = action.strip().lower()
    if normalized_action in {"create", "update"} and not payload:
        return "Error: schedule details are required for create/update proposals."

    try:
        service = AgentSchedulerService(
            cfg.db_dsn,
            default_timezone=cfg.agent_scheduler.default_timezone,
        )
        proposal_id = service.create_proposal(
            action=normalized_action,
            payload=payload,
            schedule_id=schedule_id,
            rationale=rationale,
            proposed_by_run_id=proposed_by_run_id,
            proposed_by_session_id=proposed_by_session_id,
        )
    except Exception as exc:
        return f"Error creating schedule proposal: {exc}"

    return (
        "Schedule proposal created. "
        f"proposal_id={proposal_id}. "
        "It has not been applied yet; the application must approve it first."
    )


def build(cfg: AgentConfig, session_vars=None) -> Tool:
    async def propose_schedule_change(
        action: str,
        schedule_id: Optional[str] = None,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
        schedule: Optional[dict[str, Any]] = None,
        timezone: Optional[str] = None,
        model: Optional[str] = None,
        rationale: Optional[str] = None,
    ) -> str:
        """Propose a durable schedule mutation for application approval.

        Use only when the user explicitly asks to schedule, reschedule, pause,
        resume, or cancel future agent work. This tool creates a proposal only;
        it does not directly mutate schedules.

        Args:
            action: create, update, pause, resume, or cancel.
            schedule_id: Required for update, pause, resume, and cancel.
            name: Human-readable schedule name for create/update.
            prompt: The future prompt the agent should run for create/update.
            schedule: JSON object describing the schedule. Use
                {"schedule_kind": "once", "run_at": "...", "session_id": "..."},
                {"schedule_kind": "interval", "interval_seconds": 3600, ...},
                or {"schedule_kind": "cron", "cron_expression": "0 9 * * *", ...}.
            timezone: IANA timezone name, such as "UTC" or "Europe/Rome".
            model: Optional per-schedule model override.
            rationale: Why the proposal matches the user's request.
        """
        return await propose_schedule_change_impl(
            cfg,
            action=action,
            schedule_id=schedule_id,
            name=name,
            prompt=prompt,
            schedule=schedule,
            timezone=timezone,
            model=model,
            rationale=rationale,
            proposed_by_session_id=_session_var(session_vars, "session_id"),
            proposed_by_run_id=_session_var(session_vars, "run_id"),
        )

    return Tool(propose_schedule_change)
