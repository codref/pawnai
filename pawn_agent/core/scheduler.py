"""Durable scheduler service for pawn-agent runs."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import sqlalchemy as sa

from pawn_agent.core.agent_runner import run_agent_turn
from pawn_agent.core.langgraph_registry import LangGraphSessionRegistry
from pawn_agent.utils.db import AgentSchedule, AgentScheduleFire, AgentScheduleProposal
from pawn_core.database import _get_session

logger = logging.getLogger(__name__)

SCHEDULE_STATUSES = {"active", "paused", "cancelled", "completed"}
SCHEDULE_KINDS = {"once", "interval", "cron"}
PROPOSAL_ACTIONS = {"create", "update", "pause", "resume", "cancel"}
PROPOSAL_STATUSES = {"proposed", "approved", "rejected", "applied", "expired"}
FIRE_STATUSES = {"claimed", "running", "completed", "failed"}


@dataclass(frozen=True)
class SchedulePayload:
    """Validated schedule details used to create or update a schedule."""

    name: str
    prompt: str
    session_id: str
    model: Optional[str]
    schedule_kind: str
    timezone_name: str
    run_at: Optional[datetime]
    interval_seconds: Optional[int]
    cron_expression: Optional[str]
    next_run_at: Optional[datetime]
    metadata: Optional[dict[str, Any]]


@dataclass(frozen=True)
class ClaimedScheduleFire:
    """Detached data needed to run a due schedule outside the DB transaction."""

    fire_id: str
    schedule_id: str
    scheduled_for: datetime
    prompt: str
    session_id: str
    model: Optional[str]


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: Any, *, timezone_name: str) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid datetime {value!r}; use ISO 8601") from exc
    else:
        raise ValueError("run_at must be an ISO 8601 string or datetime")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(timezone_name))
    return dt.astimezone(timezone.utc)


def _timezone(name: Optional[str], default_timezone: str) -> str:
    tz_name = (name or default_timezone or "UTC").strip()
    try:
        ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {tz_name}") from exc
    return tz_name


def _next_cron_after(expression: str, after: datetime, timezone_name: str) -> datetime:
    try:
        from croniter import croniter  # type: ignore
    except ImportError as exc:
        raise ValueError(
            "cron schedules require the optional 'croniter' dependency to be installed"
        ) from exc

    local_after = _coerce_utc(after).astimezone(ZoneInfo(timezone_name))
    return croniter(expression, local_after).get_next(datetime).astimezone(timezone.utc)


def calculate_next_run_at(
    *,
    schedule_kind: str,
    timezone_name: str,
    after: Optional[datetime] = None,
    run_at: Optional[datetime] = None,
    interval_seconds: Optional[int] = None,
    cron_expression: Optional[str] = None,
) -> Optional[datetime]:
    """Return the next UTC fire time for a normalized schedule."""
    baseline = _coerce_utc(after or utc_now())
    if schedule_kind == "once":
        if run_at is None:
            raise ValueError("once schedules require run_at")
        return _coerce_utc(run_at) if _coerce_utc(run_at) > baseline else None
    if schedule_kind == "interval":
        if interval_seconds is None or interval_seconds <= 0:
            raise ValueError("interval schedules require interval_seconds > 0")
        if run_at is not None and _coerce_utc(run_at) > baseline:
            return _coerce_utc(run_at)
        return baseline + timedelta(seconds=interval_seconds)
    if schedule_kind == "cron":
        if not cron_expression:
            raise ValueError("cron schedules require cron_expression")
        return _next_cron_after(cron_expression, baseline, timezone_name)
    raise ValueError(f"Unsupported schedule kind: {schedule_kind!r}")


def _extract_schedule_fields(payload: dict[str, Any]) -> dict[str, Any]:
    schedule = payload.get("schedule")
    if isinstance(schedule, dict):
        merged = dict(schedule)
        for key in (
            "schedule_kind",
            "kind",
            "run_at",
            "interval_seconds",
            "cron_expression",
            "timezone",
        ):
            if key in payload and payload[key] not in (None, ""):
                merged[key] = payload[key]
        return merged
    return payload


def validate_schedule_payload(
    payload: dict[str, Any],
    *,
    default_timezone: str = "UTC",
    now: Optional[datetime] = None,
    existing: Optional[AgentSchedule] = None,
    partial: bool = False,
) -> SchedulePayload:
    """Validate and normalize a create/update payload."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    schedule_fields = _extract_schedule_fields(payload)
    source = existing

    def _field(name: str, default: Any = None) -> Any:
        if name in payload and payload[name] not in (None, ""):
            return payload[name]
        if name in schedule_fields and schedule_fields[name] not in (None, ""):
            return schedule_fields[name]
        if source is not None:
            if name == "metadata":
                return source.metadata_json
            if name == "timezone_name":
                return source.timezone
            return getattr(source, name, default)
        return default

    timezone_name = _timezone(
        _field("timezone") or _field("timezone_name"),
        default_timezone,
    )
    schedule_kind = str(_field("schedule_kind") or _field("kind") or "").strip().lower()
    if not schedule_kind and source is not None:
        schedule_kind = source.schedule_kind
    if schedule_kind not in SCHEDULE_KINDS:
        raise ValueError("schedule_kind must be one of: once, interval, cron")

    name = str(_field("name") or "").strip()
    prompt = str(_field("prompt") or "").strip()
    session_id = str(_field("session_id") or "").strip()
    model = _field("model")
    metadata = _field("metadata") or {}

    if not partial or source is None:
        if not name:
            raise ValueError("name is required")
        if not prompt:
            raise ValueError("prompt is required")
        if not session_id:
            raise ValueError("session_id is required")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")

    run_at = _parse_datetime(_field("run_at"), timezone_name=timezone_name)
    interval_raw = _field("interval_seconds")
    interval_seconds = int(interval_raw) if interval_raw not in (None, "") else None
    cron_expression = _field("cron_expression")
    cron_expression = str(cron_expression).strip() if cron_expression else None

    next_run_at = calculate_next_run_at(
        schedule_kind=schedule_kind,
        timezone_name=timezone_name,
        after=now or utc_now(),
        run_at=run_at,
        interval_seconds=interval_seconds,
        cron_expression=cron_expression,
    )
    if next_run_at is None and schedule_kind == "once":
        raise ValueError("run_at must be in the future for a new one-shot schedule")

    return SchedulePayload(
        name=name,
        prompt=prompt,
        session_id=session_id,
        model=str(model).strip() if model else None,
        schedule_kind=schedule_kind,
        timezone_name=timezone_name,
        run_at=run_at,
        interval_seconds=interval_seconds,
        cron_expression=cron_expression,
        next_run_at=next_run_at,
        metadata=metadata,
    )


def serialize_schedule(row: AgentSchedule) -> dict[str, Any]:
    """Return an API/CLI-safe schedule dictionary."""
    return {
        "id": row.id,
        "name": row.name,
        "prompt": row.prompt,
        "session_id": row.session_id,
        "model": row.model,
        "status": row.status,
        "schedule_kind": row.schedule_kind,
        "timezone": row.timezone,
        "run_at": row.run_at.isoformat() if row.run_at else None,
        "interval_seconds": row.interval_seconds,
        "cron_expression": row.cron_expression,
        "next_run_at": row.next_run_at.isoformat() if row.next_run_at else None,
        "last_run_at": row.last_run_at.isoformat() if row.last_run_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "created_by": row.created_by,
        "created_from_proposal_id": row.created_from_proposal_id,
        "revision": row.revision,
        "metadata": row.metadata_json or {},
    }


def serialize_proposal(row: AgentScheduleProposal) -> dict[str, Any]:
    """Return an API/CLI-safe proposal dictionary."""
    return {
        "id": row.id,
        "action": row.action,
        "schedule_id": row.schedule_id,
        "proposed_payload": row.proposed_payload,
        "rationale": row.rationale,
        "status": row.status,
        "proposed_by_run_id": row.proposed_by_run_id,
        "proposed_by_session_id": row.proposed_by_session_id,
        "reviewed_by": row.reviewed_by,
        "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


class AgentSchedulerService:
    """Deterministic schedule/proposal mutation service."""

    def __init__(self, dsn: str, *, default_timezone: str = "UTC") -> None:
        self.dsn = dsn
        self.default_timezone = default_timezone

    def create_proposal(
        self,
        *,
        action: str,
        payload: dict[str, Any],
        schedule_id: Optional[str] = None,
        rationale: Optional[str] = None,
        proposed_by_run_id: Optional[str] = None,
        proposed_by_session_id: Optional[str] = None,
    ) -> str:
        """Store a proposed schedule mutation without applying it."""
        normalized_action = action.strip().lower()
        if normalized_action not in PROPOSAL_ACTIONS:
            raise ValueError("action must be one of: create, update, pause, resume, cancel")
        if normalized_action in {"update", "pause", "resume", "cancel"} and not schedule_id:
            raise ValueError(f"{normalized_action} proposals require schedule_id")
        if normalized_action == "create":
            validate_schedule_payload(payload, default_timezone=self.default_timezone)
        elif normalized_action == "update":
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")

        proposal_id = str(uuid.uuid4())
        row = AgentScheduleProposal(
            id=proposal_id,
            action=normalized_action,
            schedule_id=schedule_id,
            proposed_payload=payload,
            rationale=rationale,
            status="proposed",
            proposed_by_run_id=proposed_by_run_id,
            proposed_by_session_id=proposed_by_session_id,
            created_at=utc_now(),
        )
        with _get_session(self.dsn) as db:
            db.add(row)
        return proposal_id

    def create_schedule(
        self,
        payload: dict[str, Any],
        *,
        created_by: str = "user",
        created_from_proposal_id: Optional[str] = None,
    ) -> str:
        """Create an active schedule from an application-owned payload."""
        details = validate_schedule_payload(payload, default_timezone=self.default_timezone)
        now = utc_now()
        schedule_id = str(uuid.uuid4())
        row = AgentSchedule(
            id=schedule_id,
            name=details.name,
            prompt=details.prompt,
            session_id=details.session_id,
            model=details.model,
            status="active",
            schedule_kind=details.schedule_kind,
            timezone=details.timezone_name,
            run_at=details.run_at,
            interval_seconds=details.interval_seconds,
            cron_expression=details.cron_expression,
            next_run_at=details.next_run_at,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            created_from_proposal_id=created_from_proposal_id,
            revision=1,
            metadata_json=details.metadata,
        )
        with _get_session(self.dsn) as db:
            db.add(row)
        return schedule_id

    def update_schedule(self, schedule_id: str, payload: dict[str, Any]) -> None:
        """Update an existing schedule and recalculate its next fire time."""
        with _get_session(self.dsn) as db:
            row = db.get(AgentSchedule, schedule_id)
            if row is None:
                raise ValueError(f"Unknown schedule: {schedule_id}")
            if row.status in {"cancelled", "completed"}:
                raise ValueError(f"Cannot update a {row.status} schedule")
            details = validate_schedule_payload(
                payload,
                default_timezone=self.default_timezone,
                existing=row,
                partial=True,
            )
            row.name = details.name
            row.prompt = details.prompt
            row.session_id = details.session_id
            row.model = details.model
            row.schedule_kind = details.schedule_kind
            row.timezone = details.timezone_name
            row.run_at = details.run_at
            row.interval_seconds = details.interval_seconds
            row.cron_expression = details.cron_expression
            row.next_run_at = details.next_run_at
            row.updated_at = utc_now()
            row.revision += 1
            row.metadata_json = details.metadata

    def approve_proposal(self, proposal_id: str, *, reviewed_by: str = "user") -> Optional[str]:
        """Approve and apply a proposed mutation. Returns created schedule id if any."""
        with _get_session(self.dsn) as db:
            proposal = db.get(AgentScheduleProposal, proposal_id)
            if proposal is None:
                raise ValueError(f"Unknown proposal: {proposal_id}")
            if proposal.status != "proposed":
                raise ValueError(f"Proposal is already {proposal.status}")
            action = proposal.action
            schedule_id = proposal.schedule_id
            proposed_payload = dict(proposal.proposed_payload or {})

        created_schedule_id: Optional[str] = None
        if action == "create":
            created_schedule_id = self.create_schedule(
                proposed_payload,
                created_by="agent_proposal",
                created_from_proposal_id=proposal_id,
            )
        elif action == "update":
            if not schedule_id:
                raise ValueError("update proposal has no schedule_id")
            self.update_schedule(schedule_id, proposed_payload)
        elif action == "pause":
            self.pause_schedule(str(schedule_id))
        elif action == "resume":
            self.resume_schedule(str(schedule_id))
        elif action == "cancel":
            self.cancel_schedule(str(schedule_id))

        with _get_session(self.dsn) as db:
            applied = db.get(AgentScheduleProposal, proposal_id)
            if applied is not None:
                applied.status = "applied"
                applied.reviewed_by = reviewed_by
                applied.reviewed_at = utc_now()
                if created_schedule_id:
                    applied.schedule_id = created_schedule_id
        return created_schedule_id

    def reject_proposal(self, proposal_id: str, *, reviewed_by: str = "user") -> None:
        """Reject a proposed mutation without applying it."""
        with _get_session(self.dsn) as db:
            row = db.get(AgentScheduleProposal, proposal_id)
            if row is None:
                raise ValueError(f"Unknown proposal: {proposal_id}")
            if row.status != "proposed":
                raise ValueError(f"Proposal is already {row.status}")
            row.status = "rejected"
            row.reviewed_by = reviewed_by
            row.reviewed_at = utc_now()

    def pause_schedule(self, schedule_id: str) -> None:
        self._set_schedule_status(schedule_id, "paused", allowed={"active"})

    def resume_schedule(self, schedule_id: str) -> None:
        with _get_session(self.dsn) as db:
            row = db.get(AgentSchedule, schedule_id)
            if row is None:
                raise ValueError(f"Unknown schedule: {schedule_id}")
            if row.status != "paused":
                raise ValueError(f"Cannot resume a {row.status} schedule")
            row.next_run_at = calculate_next_run_at(
                schedule_kind=row.schedule_kind,
                timezone_name=row.timezone,
                after=utc_now(),
                run_at=row.run_at,
                interval_seconds=row.interval_seconds,
                cron_expression=row.cron_expression,
            )
            row.status = "active" if row.next_run_at is not None else "completed"
            row.updated_at = utc_now()
            row.revision += 1

    def cancel_schedule(self, schedule_id: str) -> None:
        self._set_schedule_status(schedule_id, "cancelled", allowed={"active", "paused"})

    def _set_schedule_status(self, schedule_id: str, status: str, *, allowed: set[str]) -> None:
        with _get_session(self.dsn) as db:
            row = db.get(AgentSchedule, schedule_id)
            if row is None:
                raise ValueError(f"Unknown schedule: {schedule_id}")
            if row.status not in allowed:
                raise ValueError(f"Cannot set {row.status} schedule to {status}")
            row.status = status
            row.updated_at = utc_now()
            row.revision += 1

    def list_schedules(self, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        with _get_session(self.dsn) as db:
            stmt = sa.select(AgentSchedule).order_by(AgentSchedule.created_at.desc())
            if not include_inactive:
                stmt = stmt.where(AgentSchedule.status.in_(["active", "paused"]))
            return [serialize_schedule(row) for row in db.scalars(stmt).all()]

    def get_schedule(self, schedule_id: str) -> dict[str, Any]:
        with _get_session(self.dsn) as db:
            row = db.get(AgentSchedule, schedule_id)
            if row is None:
                raise ValueError(f"Unknown schedule: {schedule_id}")
            return serialize_schedule(row)

    def list_proposals(self, *, include_resolved: bool = False) -> list[dict[str, Any]]:
        with _get_session(self.dsn) as db:
            stmt = sa.select(AgentScheduleProposal).order_by(
                AgentScheduleProposal.created_at.desc()
            )
            if not include_resolved:
                stmt = stmt.where(AgentScheduleProposal.status == "proposed")
            return [serialize_proposal(row) for row in db.scalars(stmt).all()]

    def claim_due_schedules(
        self, *, limit: int, now: Optional[datetime] = None
    ) -> list[ClaimedScheduleFire]:
        """Claim active due schedules and advance their next run timestamps."""
        claim_time = _coerce_utc(now or utc_now())
        claimed: list[ClaimedScheduleFire] = []
        with _get_session(self.dsn) as db:
            stmt = (
                sa.select(AgentSchedule)
                .where(
                    AgentSchedule.status == "active",
                    AgentSchedule.next_run_at.is_not(None),
                    AgentSchedule.next_run_at <= claim_time,
                )
                .order_by(AgentSchedule.next_run_at.asc())
                .limit(limit)
            )
            if db.bind and db.bind.dialect.name == "postgresql":
                stmt = stmt.with_for_update(skip_locked=True)
            schedules = list(db.scalars(stmt).all())

            for schedule in schedules:
                scheduled_for = _coerce_utc(schedule.next_run_at or claim_time)
                fire = AgentScheduleFire(
                    id=str(uuid.uuid4()),
                    schedule_id=schedule.id,
                    scheduled_for=scheduled_for,
                    status="claimed",
                    attempt=1,
                    claimed_at=claim_time,
                    created_at=claim_time,
                )
                existing_fire = db.scalars(
                    sa.select(AgentScheduleFire)
                    .where(
                        AgentScheduleFire.schedule_id == schedule.id,
                        AgentScheduleFire.scheduled_for == scheduled_for,
                    )
                    .limit(1)
                ).first()
                if existing_fire is not None:
                    continue
                db.add(fire)
                db.flush()

                schedule.last_run_at = scheduled_for
                next_run = calculate_next_run_at(
                    schedule_kind=schedule.schedule_kind,
                    timezone_name=schedule.timezone,
                    after=scheduled_for,
                    run_at=schedule.run_at,
                    interval_seconds=schedule.interval_seconds,
                    cron_expression=schedule.cron_expression,
                )
                schedule.next_run_at = next_run
                if schedule.schedule_kind == "once":
                    schedule.status = "completed"
                schedule.updated_at = claim_time
                schedule.revision += 1

                claimed.append(
                    ClaimedScheduleFire(
                        fire_id=fire.id,
                        schedule_id=schedule.id,
                        scheduled_for=scheduled_for,
                        prompt=schedule.prompt,
                        session_id=schedule.session_id,
                        model=schedule.model,
                    )
                )
        return claimed

    def mark_fire_running(self, fire_id: str, *, agent_run_id: Optional[str] = None) -> None:
        with _get_session(self.dsn) as db:
            row = db.get(AgentScheduleFire, fire_id)
            if row is None:
                raise ValueError(f"Unknown schedule fire: {fire_id}")
            row.status = "running"
            row.started_at = utc_now()
            if agent_run_id:
                row.agent_run_id = agent_run_id

    def mark_fire_completed(self, fire_id: str, *, agent_run_id: Optional[str] = None) -> None:
        with _get_session(self.dsn) as db:
            row = db.get(AgentScheduleFire, fire_id)
            if row is None:
                raise ValueError(f"Unknown schedule fire: {fire_id}")
            row.status = "completed"
            row.completed_at = utc_now()
            if agent_run_id:
                row.agent_run_id = agent_run_id

    def mark_fire_failed(
        self,
        fire_id: str,
        *,
        error: str,
        agent_run_id: Optional[str] = None,
    ) -> None:
        with _get_session(self.dsn) as db:
            row = db.get(AgentScheduleFire, fire_id)
            if row is None:
                raise ValueError(f"Unknown schedule fire: {fire_id}")
            row.status = "failed"
            row.completed_at = utc_now()
            row.error = error
            if agent_run_id:
                row.agent_run_id = agent_run_id

    def mark_stale_fires_failed(self, *, stale_after_seconds: int) -> int:
        cutoff = utc_now() - timedelta(seconds=stale_after_seconds)
        with _get_session(self.dsn) as db:
            rows = list(
                db.scalars(
                    sa.select(AgentScheduleFire).where(
                        AgentScheduleFire.status.in_(["claimed", "running"]),
                        sa.or_(
                            AgentScheduleFire.started_at <= cutoff,
                            sa.and_(
                                AgentScheduleFire.started_at.is_(None),
                                AgentScheduleFire.claimed_at <= cutoff,
                            ),
                        ),
                    )
                ).all()
            )
            for row in rows:
                row.status = "failed"
                row.completed_at = utc_now()
                row.error = "Marked failed after stale scheduler claim"
            return len(rows)


async def run_scheduler_tick(
    cfg: Any,
    *,
    registry: Optional[LangGraphSessionRegistry] = None,
) -> int:
    """Claim and execute one batch of due schedule fires."""
    service = AgentSchedulerService(
        cfg.db_dsn,
        default_timezone=cfg.agent_scheduler.default_timezone,
    )
    service.mark_stale_fires_failed(
        stale_after_seconds=cfg.agent_scheduler.stale_fire_after_seconds
    )
    claimed = service.claim_due_schedules(limit=cfg.agent_scheduler.max_due_per_tick)
    if not claimed:
        return 0

    active_registry = registry or LangGraphSessionRegistry()
    for fire in claimed:
        run_id: Optional[str] = None
        try:
            result = await run_agent_turn(
                cfg=cfg,
                registry=active_registry,
                prompt=fire.prompt,
                session_id=fire.session_id,
                model=fire.model,
                source="schedule",
                schedule_id=fire.schedule_id,
                scheduled_fire_id=fire.fire_id,
            )
            run_id = result.run_id
            service.mark_fire_completed(fire.fire_id, agent_run_id=run_id)
        except Exception as exc:
            logger.error("Scheduled fire %s failed: %s", fire.fire_id, exc, exc_info=True)
            service.mark_fire_failed(fire.fire_id, error=str(exc), agent_run_id=run_id)
    return len(claimed)


async def start_scheduler(
    cfg: Any,
    *,
    registry: Optional[LangGraphSessionRegistry] = None,
) -> None:
    """Run the durable scheduler loop until cancelled."""
    logger.info(
        "Starting pawn-agent scheduler | interval=%ss max_due=%s",
        cfg.agent_scheduler.poll_interval_seconds,
        cfg.agent_scheduler.max_due_per_tick,
    )
    try:
        while True:
            try:
                await run_scheduler_tick(cfg, registry=registry)
            except Exception as exc:
                logger.error("Scheduler tick failed: %s", exc, exc_info=True)
            await asyncio.sleep(cfg.agent_scheduler.poll_interval_seconds)
    except asyncio.CancelledError:
        logger.info("Scheduler cancelled - shutting down cleanly")
        raise
