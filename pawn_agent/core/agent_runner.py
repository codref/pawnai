"""Shared turn runner for API, queue, and scheduled agent runs.

Owns: ``agent_runs`` row lifecycle around one registry turn.
Does not own: queue ack/nack or schedule-fire status (callers handle that).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pawn_agent.utils.db import create_agent_run, update_agent_run
from pawn_agent.utils.model_utils import _apply_model_override

logger = logging.getLogger(__name__)


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
    on_progress: Optional[Callable[[str, dict[str, Any]], None]] = None,
    parent_run_id: Optional[str] = None,
    depth: int = 0,
    event_id: Optional[str] = None,
) -> AgentRunResult:
    """Persist and execute one sallm agent turn.

    ``session_id`` is the conversation key. For queue/scheduler sources it is
    typically also the diarization session name tools should prefer. For
    ``source=vault`` / ``command=vault_run`` it is ``note:<vault-path>``.

    ``on_progress`` is an optional sync callback ``(kind, attrs)`` invoked from
    the ask() worker thread (e.g. Matrix status edits).
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
        parent_run_id=parent_run_id,
        depth=depth,
        event_id=event_id,
    )
    update_agent_run(cfg.db_dsn, run_id, "running")

    from pawn_agent.core.coworker.lineage import lineage_env  # noqa: PLC0415

    try:
        if not prompt:
            raise ValueError(f"'prompt' is required for the '{command}' command")
        if not session_id:
            if source == "vault" or command == "vault_run":
                raise ValueError(
                    "'session_id' is required for vault_run — "
                    "use conversation key note:<vault-path>"
                )
            raise ValueError(
                "'session_id' is required for the 'run' command - "
                "it must be the diarization session name used by agent tools"
            )

        vault_paths: list[str] = []
        with lineage_env(run_id, depth=depth, event_id=event_id or run_id):
            reply = await registry.handle_turn(
                session_id,
                prompt,
                effective_cfg,
                cfg.db_dsn,
                on_progress=on_progress,
                vault_paths_out=vault_paths,
            )
        update_agent_run(cfg.db_dsn, run_id, "completed", response=reply)
        if vault_paths:
            _publish_vault_writes(vault_paths, source=source, run_id=run_id)
        return AgentRunResult(run_id=run_id, response=reply)
    except Exception as exc:
        update_agent_run(cfg.db_dsn, run_id, "failed", error=str(exc))
        raise


def _publish_vault_writes(paths: list[str], *, source: str, run_id: str) -> None:
    """Tell connected Obsidian clients to resync. Never fails the turn."""
    try:
        from pawn_server.core.vault_events import publish_vault_event  # noqa: PLC0415

        publish_vault_event(paths, source=source, run_id=run_id)
    except Exception:
        logger.exception("vault resync publish failed run_id=%s", run_id)
