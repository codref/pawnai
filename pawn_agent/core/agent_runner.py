"""Shared LangGraph execution helper for queue and scheduled agent runs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

from pawn_agent.utils.db import create_agent_run, update_agent_run
from pawn_agent.utils.model_utils import _apply_model_override


@dataclass(frozen=True)
class AgentRunResult:
    """Result metadata for one persisted agent execution."""

    run_id: str
    response: str


async def run_agent_turn(
    *,
    cfg: Any,
    registry: Any,
    prompt: Optional[str],
    session_id: Optional[str],
    model: Optional[str] = None,
    message_id: Optional[str] = None,
    source: str,
    command: str = "run",
    schedule_id: Optional[str] = None,
    scheduled_fire_id: Optional[str] = None,
) -> AgentRunResult:
    """Persist and execute one LangGraph turn.

    The caller owns acknowledgement semantics (queue ack/nack, schedule-fire
    status). This helper owns the common ``agent_runs`` lifecycle.
    """
    effective_cfg = cfg
    if model:
        effective_cfg = copy.copy(cfg)
        _apply_model_override(effective_cfg, model)

    run_id = create_agent_run(
        cfg.db_dsn,
        message_id=message_id,
        source=source,
        schedule_id=schedule_id,
        scheduled_fire_id=scheduled_fire_id,
        command=command,
        prompt=prompt,
        session_id=session_id,
        model=effective_cfg.pydantic_model,
    )
    update_agent_run(cfg.db_dsn, run_id, "running")

    try:
        if not prompt:
            raise ValueError("'prompt' is required for the 'run' command")
        if not session_id:
            raise ValueError(
                "'session_id' is required for the 'run' command - "
                "it must be the diarization session name used by agent tools"
            )

        reply = await registry.handle_turn(session_id, prompt, effective_cfg, cfg.db_dsn)
        update_agent_run(cfg.db_dsn, run_id, "completed", response=reply)
        return AgentRunResult(run_id=run_id, response=reply)
    except Exception as exc:
        update_agent_run(cfg.db_dsn, run_id, "failed", error=str(exc))
        raise
