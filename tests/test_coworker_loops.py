"""Open-loop detectors and the weekly review renderer."""

from datetime import datetime, timedelta, timezone

from pawn_agent.core.coworker.loops import (
    my_commitments,
    ownerless_decisions,
    recurring_questions,
    stale_threads,
)
from pawn_agent.core.coworker.review import extract_goals_block, render_review
from pawn_core.goals import GoalThread, Goals


def test_loops():
    now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    old = (now - timedelta(days=10)).isoformat()
    items = [
        {
            "kind": "commitment",
            "owner": "Davide",
            "status": "new",
            "created_at": old,
            "text": "Send the doc",
        },
        {"kind": "decision", "owner": "", "status": "new", "text": "Use S3"},
        {"kind": "open_question", "recurrence": 3, "status": "new", "text": "Who owns it?"},
    ]
    assert my_commitments(items, aliases=["Davide"], now=now)[0]["text"] == "Send the doc"
    assert ownerless_decisions(items)[0]["text"] == "Use S3"
    assert recurring_questions(items)[0]["text"] == "Who owns it?"
    threads = [{"name": "X", "status": "active", "last_movement_at": now - timedelta(days=20)}]
    assert stale_threads(threads, now=now)[0]["name"] == "X"


def test_review_contains_goals_block():
    goals = Goals(
        threads=[GoalThread(name="Project X", status="active", why="Friday")],
        max_per_day=5,
    )
    body = render_review(
        week="2026-W39",
        goals=goals,
        items=[],
        stale=[{"name": "Project X"}],
        commitments=[],
        ownerless=[],
        themes=["hiring"],
    )
    proposed = extract_goals_block(body)
    assert "pawn: goals" in proposed
    assert "### Project X" in proposed
    assert "## Parked" in proposed
