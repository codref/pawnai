"""Autonomy decisions, lineage, and job recovery classification."""

from pawn_agent.core.coworker.autonomy import AutonomyDecision, decide, reject_reason
from pawn_agent.core.coworker.lineage import child_lineage
from pawn_agent.core.coworker.recovery import recovery_plan
from pawn_agent.utils.config import AgentConfig
from pawn_core.goals import Goals


def test_decide_modes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = AgentConfig()
    cfg.coworker.autonomy.mode = "off"
    assert decide(cfg, "research").decision == "deny"
    cfg.coworker.autonomy.mode = "suggest_only"
    assert decide(cfg, "research").decision == "needs_approval"
    cfg.coworker.autonomy.mode = "limited_act"
    cfg.coworker.autonomy.auto_actions = ["research"]
    assert decide(cfg, "research").decision == "allow"
    assert decide(cfg, "email").decision == "needs_approval"
    assert decide(cfg, "research", writes_outside_pawn=True).decision == "deny"
    goals = Goals(autonomy="off")
    assert decide(cfg, "research", goals=goals) == AutonomyDecision("deny", "autonomy off")


def test_reject_reason():
    assert (
        reject_reason(
            depth=3,
            event_count=0,
            day_count=0,
            max_depth=2,
            max_per_event=3,
            max_per_day=20,
            duplicate=False,
        )
        == "max_depth"
    )
    assert (
        reject_reason(
            depth=1,
            event_count=0,
            day_count=0,
            max_depth=2,
            max_per_event=3,
            max_per_day=20,
            duplicate=True,
        )
        == "duplicate"
    )
    assert (
        reject_reason(
            depth=1,
            event_count=0,
            day_count=0,
            max_depth=2,
            max_per_event=3,
            max_per_day=20,
            duplicate=False,
        )
        is None
    )


def test_child_lineage():
    assert child_lineage({}) == {}
    fields = child_lineage(
        {"PAWN_PARENT_RUN_ID": "run-1", "PAWN_RUN_DEPTH": "1", "PAWN_EVENT_ID": "evt"}
    )
    assert fields == {"parent_run_id": "run-1", "depth": 2, "event_id": "evt"}


def test_recovery_plan():
    assert recovery_plan("ask", "do the thing", {}) == "respawn"
    assert recovery_plan("ask", "do the thing", {"recovered": True}) == "abandon"
    assert recovery_plan("upload", "x", {}) == "abandon"
    assert recovery_plan("ask", "  ", {}) == "abandon"
