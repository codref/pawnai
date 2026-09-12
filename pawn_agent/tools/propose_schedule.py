"""Schedule proposals (``schedule_propose`` CliTool)."""

from __future__ import annotations

from typing import Any, Optional

from pawn_agent.core.scheduler import AgentSchedulerService
from pawn_agent.utils.config import AgentConfig


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
