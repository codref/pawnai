from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import sqlalchemy as sa

from pawn_agent.core.scheduler import (
    AgentSchedulerService,
    calculate_next_run_at,
    run_scheduler_tick,
    validate_schedule_payload,
)
from pawn_agent.utils.db import (
    AgentRun,
    AgentSchedule,
    AgentScheduleFire,
    AgentScheduleProposal,
)
from pawn_core.database import Base


def _sqlite_dsn(tmp_path: Path) -> str:
    db_path = tmp_path / "scheduler.db"
    dsn = f"sqlite:///{db_path}"
    engine = sa.create_engine(dsn)
    Base.metadata.create_all(
        engine,
        tables=[
            AgentRun.__table__,
            AgentSchedule.__table__,
            AgentScheduleProposal.__table__,
            AgentScheduleFire.__table__,
        ],
    )
    engine.dispose()
    return dsn


def _payload(run_at: datetime | None = None) -> dict:
    return {
        "name": "Daily recap",
        "prompt": "Summarize session s1",
        "session_id": "s1",
        "schedule": {
            "schedule_kind": "once",
            "run_at": (run_at or datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
    }


def test_validate_once_schedule_normalizes_next_run_at() -> None:
    run_at = datetime.now(timezone.utc) + timedelta(hours=2)

    details = validate_schedule_payload(_payload(run_at), default_timezone="UTC")

    assert details.schedule_kind == "once"
    assert details.session_id == "s1"
    assert details.next_run_at == run_at


def test_calculate_interval_next_run_at_uses_interval() -> None:
    now = datetime(2026, 4, 30, 10, 0, tzinfo=timezone.utc)

    next_run = calculate_next_run_at(
        schedule_kind="interval",
        timezone_name="UTC",
        after=now,
        interval_seconds=300,
    )

    assert next_run == now + timedelta(seconds=300)


def test_proposal_approval_creates_schedule(tmp_path: Path) -> None:
    dsn = _sqlite_dsn(tmp_path)
    service = AgentSchedulerService(dsn)

    proposal_id = service.create_proposal(
        action="create",
        payload=_payload(),
        rationale="User asked for it.",
        proposed_by_session_id="chat-1",
    )
    created_id = service.approve_proposal(proposal_id, reviewed_by="tester")

    schedules = service.list_schedules()
    proposals = service.list_proposals(include_resolved=True)
    assert created_id == schedules[0]["id"]
    assert schedules[0]["created_by"] == "agent_proposal"
    assert proposals[0]["status"] == "applied"
    assert proposals[0]["schedule_id"] == created_id


def test_reject_proposal_does_not_create_schedule(tmp_path: Path) -> None:
    dsn = _sqlite_dsn(tmp_path)
    service = AgentSchedulerService(dsn)

    proposal_id = service.create_proposal(action="create", payload=_payload())
    service.reject_proposal(proposal_id, reviewed_by="tester")

    assert service.list_schedules(include_inactive=True) == []
    proposals = service.list_proposals(include_resolved=True)
    assert proposals[0]["status"] == "rejected"


def test_claim_due_schedule_creates_one_fire_and_advances(tmp_path: Path) -> None:
    dsn = _sqlite_dsn(tmp_path)
    service = AgentSchedulerService(dsn)
    now = datetime.now(timezone.utc)
    due = now - timedelta(minutes=1)

    with sa.orm.Session(sa.create_engine(dsn)) as db:
        db.add(
            AgentSchedule(
                id="sched-1",
                name="Interval",
                prompt="Run it",
                session_id="s1",
                status="active",
                schedule_kind="interval",
                timezone="UTC",
                interval_seconds=60,
                next_run_at=due,
                created_at=due,
                updated_at=due,
                created_by="user",
                revision=1,
                metadata_json={},
            )
        )
        db.commit()

    claimed = service.claim_due_schedules(limit=5, now=now)

    assert len(claimed) == 1
    assert claimed[0].schedule_id == "sched-1"
    with sa.orm.Session(sa.create_engine(dsn)) as db:
        fires = db.scalars(sa.select(AgentScheduleFire)).all()
        schedule = db.get(AgentSchedule, "sched-1")
        assert len(fires) == 1
        assert fires[0].status == "claimed"
        assert schedule is not None
        assert schedule.next_run_at == (due + timedelta(seconds=60)).replace(tzinfo=None)


def test_scheduler_tick_executes_claimed_fire(tmp_path: Path) -> None:
    dsn = _sqlite_dsn(tmp_path)
    service = AgentSchedulerService(dsn)
    now = datetime.now(timezone.utc) - timedelta(minutes=1)
    service.create_schedule(
        {
            "name": "Interval",
            "prompt": "Run it",
            "session_id": "s1",
            "schedule": {"schedule_kind": "interval", "interval_seconds": 60},
        }
    )
    with sa.orm.Session(sa.create_engine(dsn)) as db:
        schedule = db.scalars(sa.select(AgentSchedule)).one()
        schedule.next_run_at = now
        db.commit()

    cfg = SimpleNamespace(
        db_dsn=dsn,
        agent_scheduler=SimpleNamespace(
            default_timezone="UTC",
            stale_fire_after_seconds=3600,
            max_due_per_tick=5,
        ),
    )

    with patch(
        "pawn_agent.core.scheduler.run_agent_turn",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(run_id="run-1", response="ok"),
    ) as mock_run:
        count = asyncio.run(run_scheduler_tick(cfg, registry=SimpleNamespace()))

    assert count == 1
    mock_run.assert_awaited_once()
    with sa.orm.Session(sa.create_engine(dsn)) as db:
        fire = db.scalars(sa.select(AgentScheduleFire)).one()
        assert fire.status == "completed"
        assert fire.agent_run_id == "run-1"
