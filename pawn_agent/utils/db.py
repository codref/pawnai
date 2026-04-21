"""ORM models and DB session factory for pawn-agent."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from sqlalchemy import DateTime, String, Text, create_engine
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
    """Persists every queue-initiated agent execution for history / auditability."""

    __tablename__ = "agent_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    message_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
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
