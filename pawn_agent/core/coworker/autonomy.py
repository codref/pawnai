"""Autonomy policy and the coworker pause flag."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pawn_agent.utils.config import AgentConfig
from pawn_core.goals import Goals

MODES = frozenset({"off", "suggest_only", "approve_writes", "limited_act"})


@dataclass
class AutonomyDecision:
    decision: str  # allow | needs_approval | deny
    reason: str


def pause_file(cfg: AgentConfig) -> Path:
    state = Path(getattr(cfg.sallm, "state_dir", ".sallm"))
    return state / "coworker.paused"


def is_paused(cfg: AgentConfig) -> bool:
    return pause_file(cfg).is_file()


def set_paused(cfg: AgentConfig, paused: bool) -> Path:
    path = pause_file(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if paused:
        path.write_text("paused\n", encoding="utf-8")
    elif path.exists():
        path.unlink()
    return path


def effective_mode(cfg: AgentConfig, goals: Optional[Goals] = None) -> str:
    if goals is not None and (goals.autonomy or "").strip().lower() == "off":
        return "off"
    mode = (cfg.coworker.autonomy.mode or "suggest_only").strip().lower()
    if mode not in MODES:
        return "suggest_only"
    return mode


def decide(
    cfg: AgentConfig,
    action_kind: str,
    *,
    goals: Optional[Goals] = None,
    writes_outside_pawn: bool = False,
) -> AutonomyDecision:
    """Decide whether a follow-up may run, needs approval, or is denied."""
    if is_paused(cfg):
        return AutonomyDecision("deny", "paused")
    mode = effective_mode(cfg, goals)
    if mode == "off":
        return AutonomyDecision("deny", "autonomy off")
    if writes_outside_pawn:
        return AutonomyDecision("deny", "writes outside Pawn/ are never automatic")
    if mode in {"suggest_only", "approve_writes"}:
        return AutonomyDecision("needs_approval", mode)
    allowed = {name.strip().lower() for name in cfg.coworker.autonomy.auto_actions}
    if action_kind.strip().lower() in allowed:
        return AutonomyDecision("allow", "limited_act")
    return AutonomyDecision("needs_approval", "not in auto_actions")


def reject_reason(
    *,
    depth: int,
    event_count: int,
    day_count: int,
    max_depth: int,
    max_per_event: int,
    max_per_day: int,
    duplicate: bool,
) -> Optional[str]:
    """Return why a self-enqueued run must be dropped, or None to allow it."""
    if duplicate:
        return "duplicate"
    if depth > max_depth:
        return "max_depth"
    if event_count >= max_per_event:
        return "max_self_jobs_per_event"
    if day_count >= max_per_day:
        return "max_self_jobs_per_day"
    return None
