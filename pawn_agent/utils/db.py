"""ORM models and DB session factory for pawn-agent."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, Session, mapped_column

from pawn_core.database import Base as _Base  # noqa: F401
from pawn_core.database import (
    GraphTriple,
    SessionAnalysis,
    SpeakerName,
    TranscriptionSegment,
    _get_session,
    get_engine,
    make_db_session,
)


class AgentRun(_Base):
    """Persists every agent execution for history / auditability."""

    __tablename__ = "agent_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    message_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False, default="queue", index=True)
    schedule_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    scheduled_fire_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    command: Mapped[str] = mapped_column(String, nullable=False)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    model: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class AgentSchedule(_Base):
    """A durable schedule for future pawn-agent turns."""

    __tablename__ = "agent_schedules"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active", index=True)
    schedule_kind: Mapped[str] = mapped_column(String, nullable=False)
    timezone: Mapped[str] = mapped_column(String, nullable=False)
    run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    interval_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cron_expression: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    next_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[str] = mapped_column(String, nullable=False, default="user")
    created_from_proposal_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    metadata_json: Mapped[Optional[Any]] = mapped_column("metadata", JSON, nullable=True)


class AgentScheduleProposal(_Base):
    """A model- or user-created proposal for a schedule mutation."""

    __tablename__ = "agent_schedule_proposals"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    action: Mapped[str] = mapped_column(String, nullable=False, index=True)
    schedule_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("agent_schedules.id"), nullable=True, index=True
    )
    proposed_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="proposed", index=True)
    proposed_by_run_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("agent_runs.id"), nullable=True
    )
    proposed_by_session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class AgentScheduleFire(_Base):
    """One due execution attempt for an agent schedule."""

    __tablename__ = "agent_schedule_fires"
    __table_args__ = (
        UniqueConstraint("schedule_id", "scheduled_for", name="uq_agent_schedule_fire_once"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    schedule_id: Mapped[str] = mapped_column(
        String, ForeignKey("agent_schedules.id"), nullable=False, index=True
    )
    scheduled_for: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="claimed", index=True)
    agent_run_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("agent_runs.id"), nullable=True, index=True
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class SiyuanAgentRequest(_Base):
    """Durable lifecycle for one ``@pawn`` instruction discovered in SiYuan."""

    __tablename__ = "siyuan_agent_requests"
    __table_args__ = (
        UniqueConstraint(
            "trigger_block_id",
            "instruction_hash",
            name="uq_siyuan_agent_request_trigger_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    trigger_block_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    parent_block_id: Mapped[str] = mapped_column(String, nullable=False)
    root_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    notebook_id: Mapped[str] = mapped_column(String, nullable=False)
    instruction_hash: Mapped[str] = mapped_column(String, nullable=False)
    source_updated: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued", index=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    output_block_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    agent_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    matrix_notify_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    instruction_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class SiyuanWatcherState(_Base):
    """Single-row poll cursor for the SiYuan @pawn watcher."""

    __tablename__ = "siyuan_watcher_state"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    poll_watermark: Mapped[str] = mapped_column(String, nullable=False, default="")
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


def get_session_analysis(session_id: str, dsn: str) -> Optional[SessionAnalysis]:
    """Return the most recent SessionAnalysis row for *session_id*, or None."""
    from sqlalchemy import select

    with Session(get_engine(dsn)) as db:
        row = db.scalars(
            select(SessionAnalysis)
            .where(SessionAnalysis.session_id == session_id)
            .order_by(SessionAnalysis.analyzed_at.desc())
            .limit(1)
        ).first()
        if row is None:
            return None
        db.expunge(row)
        from sqlalchemy.orm import make_transient

        make_transient(row)
        return row


def save_session_analysis(
    session_id: Optional[str],
    source: str,
    model: str,
    title: Optional[str],
    summary: Optional[str],
    key_topics: Optional[str],
    speaker_highlights: Optional[str],
    sentiment: Optional[str],
    sentiment_tags: Optional[List[str]],
    tags: Optional[List[str]],
    dsn: str,
) -> str:
    """Insert a new row into ``session_analysis`` and return its UUID."""
    row_id = str(uuid.uuid4())
    row = SessionAnalysis(
        id=row_id,
        session_id=session_id,
        source=source,
        model=model,
        title=title,
        summary=summary,
        key_topics=key_topics,
        speaker_highlights=speaker_highlights,
        sentiment=sentiment,
        sentiment_tags=sentiment_tags,
        tags=tags,
        analyzed_at=datetime.now(timezone.utc),
    )
    with _get_session(dsn) as db:
        db.add(row)
    return row_id


def save_graph_triples(
    session_id: str,
    triples: List[Tuple[str, str, str]],
    model: str,
    dsn: str,
) -> int:
    """Delete existing triples for *session_id* and bulk-insert new ones.

    Returns the number of triples inserted.
    """
    from sqlalchemy import delete

    now = datetime.now(timezone.utc)
    rows = [
        GraphTriple(
            id=str(uuid.uuid4()),
            session_id=session_id,
            subject=s,
            relation=r,
            object=o,
            model=model,
            extracted_at=now,
        )
        for s, r, o in triples
    ]
    with _get_session(dsn) as db:
        db.execute(delete(GraphTriple).where(GraphTriple.session_id == session_id))
        db.add_all(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# AgentRun helpers
# ---------------------------------------------------------------------------


def create_agent_run(
    dsn: str,
    *,
    message_id: Optional[str] = None,
    source: str = "queue",
    schedule_id: Optional[str] = None,
    scheduled_fire_id: Optional[str] = None,
    command: str,
    prompt: Optional[str] = None,
    session_id: Optional[str] = None,
    model: str,
) -> str:
    """Insert a new ``agent_runs`` row with status ``pending``. Returns the UUID."""
    row_id = str(uuid.uuid4())
    row = AgentRun(
        id=row_id,
        message_id=message_id,
        source=source,
        schedule_id=schedule_id,
        scheduled_fire_id=scheduled_fire_id,
        command=command,
        prompt=prompt,
        session_id=session_id,
        model=model,
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    with _get_session(dsn) as db:
        db.add(row)
    return row_id


def update_agent_run(
    dsn: str,
    run_id: str,
    status: str,
    *,
    response: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Update an ``agent_runs`` row's status and optional response/error."""
    now = datetime.now(timezone.utc)
    with _get_session(dsn) as db:
        row = db.get(AgentRun, run_id)
        if row is None:
            return
        row.status = status
        if status == "running":
            row.started_at = now
        if status in ("completed", "failed"):
            row.completed_at = now
        if response is not None:
            row.response = response
        if error is not None:
            row.error = error


# ---------------------------------------------------------------------------
# SiYuan agent request helpers
# ---------------------------------------------------------------------------

_WATCHER_STATE_ID = "default"


def upsert_siyuan_agent_request(
    dsn: str,
    *,
    request_id: str,
    trigger_block_id: str,
    parent_block_id: str,
    root_id: str,
    notebook_id: str,
    instruction_hash: str,
    conversation_id: str,
    instruction_text: Optional[str] = None,
    source_updated: Optional[str] = None,
    status: str = "queued",
) -> str:
    """Insert or refresh a request for *trigger_block_id* + *instruction_hash*.

    - Same trigger+hash → return existing id (idempotent).
    - Same trigger still ``queued`` with a different hash (user kept typing) →
      rewrite the row in place so we do not collide on the reused SiYuan
      ``custom-agent-request-id``.
    - Otherwise insert; if *request_id* is already taken, allocate a new UUID.
    """
    now = datetime.now(timezone.utc)
    with _get_session(dsn) as db:
        by_hash = (
            db.query(SiyuanAgentRequest)
            .filter_by(
                trigger_block_id=trigger_block_id,
                instruction_hash=instruction_hash,
            )
            .one_or_none()
        )
        if by_hash is not None:
            return by_hash.id

        # Absorb mid-edit revisions while still queued (typing / autosave race).
        open_queued = (
            db.query(SiyuanAgentRequest)
            .filter_by(trigger_block_id=trigger_block_id, status="queued")
            .order_by(SiyuanAgentRequest.created_at.desc())
            .first()
        )
        if open_queued is not None:
            open_queued.instruction_hash = instruction_hash
            open_queued.instruction_text = instruction_text
            open_queued.source_updated = source_updated
            open_queued.parent_block_id = parent_block_id
            open_queued.root_id = root_id
            open_queued.notebook_id = notebook_id
            open_queued.conversation_id = conversation_id
            open_queued.updated_at = now
            return open_queued.id

        effective_id = request_id
        if db.get(SiyuanAgentRequest, effective_id) is not None:
            effective_id = str(uuid.uuid4())

        row = SiyuanAgentRequest(
            id=effective_id,
            trigger_block_id=trigger_block_id,
            parent_block_id=parent_block_id,
            root_id=root_id,
            notebook_id=notebook_id,
            instruction_hash=instruction_hash,
            source_updated=source_updated,
            status=status,
            conversation_id=conversation_id,
            instruction_text=instruction_text,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
    return effective_id


def get_siyuan_agent_request(dsn: str, request_id: str) -> Optional[SiyuanAgentRequest]:
    """Return a detached request row or None."""
    with Session(get_engine(dsn)) as db:
        row = db.get(SiyuanAgentRequest, request_id)
        if row is None:
            return None
        db.expunge(row)
        from sqlalchemy.orm import make_transient

        make_transient(row)
        return row


def list_siyuan_agent_requests(
    dsn: str,
    *,
    status: Optional[str] = None,
    statuses: Optional[List[str]] = None,
    root_id: Optional[str] = None,
    newest_first: bool = False,
    limit: int = 50,
) -> List[SiyuanAgentRequest]:
    """List request rows, optionally filtered by status and document root."""
    with Session(get_engine(dsn)) as db:
        q = db.query(SiyuanAgentRequest)
        if status:
            q = q.filter(SiyuanAgentRequest.status == status)
        if statuses:
            q = q.filter(SiyuanAgentRequest.status.in_(statuses))
        if root_id:
            q = q.filter(SiyuanAgentRequest.root_id == root_id)
        order = (
            SiyuanAgentRequest.created_at.desc()
            if newest_first
            else SiyuanAgentRequest.created_at.asc()
        )
        rows = q.order_by(order).limit(limit).all()
        out: List[SiyuanAgentRequest] = []
        from sqlalchemy.orm import make_transient

        for row in rows:
            db.expunge(row)
            make_transient(row)
            out.append(row)
        return out


def update_siyuan_agent_request(
    dsn: str,
    request_id: str,
    *,
    status: Optional[str] = None,
    output_block_id: Optional[str] = None,
    agent_run_id: Optional[str] = None,
    matrix_notify_id: Optional[str] = None,
    error_code: Optional[str] = None,
    indexed_at: Optional[datetime] = None,
) -> None:
    """Patch mutable fields on a SiYuan agent request."""
    now = datetime.now(timezone.utc)
    with _get_session(dsn) as db:
        row = db.get(SiyuanAgentRequest, request_id)
        if row is None:
            return
        if status is not None:
            row.status = status
        if output_block_id is not None:
            row.output_block_id = output_block_id
        if agent_run_id is not None:
            row.agent_run_id = agent_run_id
        if matrix_notify_id is not None:
            row.matrix_notify_id = matrix_notify_id
        if error_code is not None:
            row.error_code = error_code
        if indexed_at is not None:
            row.indexed_at = indexed_at
        row.updated_at = now


def claim_siyuan_agent_request(dsn: str, request_id: str) -> bool:
    """Atomically move ``queued`` → ``claimed``. Returns False if not claimable."""
    now = datetime.now(timezone.utc)
    with _get_session(dsn) as db:
        row = db.get(SiyuanAgentRequest, request_id)
        if row is None or row.status != "queued":
            return False
        row.status = "claimed"
        row.updated_at = now
        return True


def get_siyuan_poll_watermark(dsn: str) -> str:
    """Return the watcher poll watermark (empty string if unset)."""
    with Session(get_engine(dsn)) as db:
        row = db.get(SiyuanWatcherState, _WATCHER_STATE_ID)
        return row.poll_watermark if row else ""


def set_siyuan_poll_watermark(dsn: str, watermark: str) -> None:
    """Persist the watcher poll watermark."""
    now = datetime.now(timezone.utc)
    with _get_session(dsn) as db:
        row = db.get(SiyuanWatcherState, _WATCHER_STATE_ID)
        if row is None:
            db.add(
                SiyuanWatcherState(
                    id=_WATCHER_STATE_ID,
                    poll_watermark=watermark,
                    updated_at=now,
                )
            )
        else:
            row.poll_watermark = watermark
            row.updated_at = now
