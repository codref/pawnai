"""ORM models and DB session factory for pawn-agent."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, Session, mapped_column

from pawn_core.database import (  # noqa: F401
    Base as _Base,
    GraphTriple,
    SessionAnalysis,
    SpeakerName,
    TranscriptionSegment,
    _get_session,
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


class GraphTopology(_Base):
    """Static topology snapshot for a graph version."""

    __tablename__ = "graph_topologies"
    __table_args__ = (
        UniqueConstraint("graph_name", "graph_version", name="uq_graph_topologies_name_version"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    graph_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    graph_version: Mapped[str] = mapped_column(String, nullable=False)
    topology: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class GraphRunEvent(_Base):
    """Append-only graph execution event for an ``agent_runs`` row."""

    __tablename__ = "graph_run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_graph_run_events_run_sequence"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String, ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    thread_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    trace_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    graph_name: Mapped[str] = mapped_column(String, nullable=False)
    graph_version: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    node_name: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    from_node: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    to_node: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    router_choice: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)


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


def get_session_analysis(session_id: str, dsn: str) -> Optional[SessionAnalysis]:
    """Return the most recent SessionAnalysis row for *session_id*, or None."""
    from sqlalchemy import select

    engine = create_engine(dsn)
    with Session(engine) as db:
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
# Graph run event helpers
# ---------------------------------------------------------------------------


def upsert_graph_topology(
    dsn: str,
    *,
    graph_name: str,
    graph_version: str,
    topology: dict[str, Any],
) -> str:
    """Insert or update one static graph topology snapshot."""
    row_id = f"{graph_name}:{graph_version}"
    with _get_session(dsn) as db:
        row = db.get(GraphTopology, row_id)
        if row is None:
            db.add(
                GraphTopology(
                    id=row_id,
                    graph_name=graph_name,
                    graph_version=graph_version,
                    topology=topology,
                    created_at=datetime.now(timezone.utc),
                )
            )
        else:
            row.topology = topology
    return row_id


def save_graph_run_event(
    dsn: str,
    *,
    run_id: str,
    sequence: int,
    thread_id: Optional[str],
    trace_id: Optional[str],
    graph_name: str,
    graph_version: str,
    event_type: str,
    timestamp: datetime,
    node_name: Optional[str] = None,
    from_node: Optional[str] = None,
    to_node: Optional[str] = None,
    router_choice: Optional[str] = None,
    duration_ms: Optional[int] = None,
    status: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
) -> str:
    """Append one graph execution event and return its UUID."""
    row_id = str(uuid.uuid4())
    row = GraphRunEvent(
        id=row_id,
        run_id=run_id,
        sequence=sequence,
        thread_id=thread_id,
        trace_id=trace_id,
        graph_name=graph_name,
        graph_version=graph_version,
        event_type=event_type,
        node_name=node_name,
        from_node=from_node,
        to_node=to_node,
        router_choice=router_choice,
        timestamp=timestamp,
        duration_ms=duration_ms,
        status=status,
        payload=payload,
    )
    with _get_session(dsn) as db:
        db.add(row)
    return row_id
