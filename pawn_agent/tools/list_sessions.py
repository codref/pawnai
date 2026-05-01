from __future__ import annotations

from pawn_agent.core.session_candidates import SessionCandidate, render_session_catalog
from pawn_agent.utils.config import AgentConfig

NAME = "list_sessions"
DESCRIPTION = (
    "List available conversation sessions from the database, "
    "with segment counts, duration, and last-updated timestamps. "
    "Use this before query_conversation when you need to discover session IDs."
)


def list_session_candidates_impl(
    cfg: AgentConfig,
    name_filter: str = "",
    limit: int = 10,
) -> list[SessionCandidate]:
    """Return structured conversation session candidates from the database."""
    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import Session

    from pawn_agent.utils.db import SessionAnalysis, TranscriptionSegment

    name_filter_clean = name_filter.strip()

    stmt = (
        select(
            TranscriptionSegment.session_id,
            func.count(TranscriptionSegment.id).label("segments"),
            func.min(TranscriptionSegment.start_time).label("first_start"),
            func.max(TranscriptionSegment.end_time).label("last_end"),
            func.min(TranscriptionSegment.created_at).label("created_at"),
            func.max(TranscriptionSegment.created_at).label("updated_at"),
        )
        .where(TranscriptionSegment.session_id.is_not(None))
        .group_by(TranscriptionSegment.session_id)
        .order_by(func.max(TranscriptionSegment.created_at).desc())
    )

    if name_filter_clean:
        stmt = stmt.where(TranscriptionSegment.session_id.ilike(f"%{name_filter_clean}%"))

    stmt = stmt.limit(limit)

    engine = create_engine(cfg.db_dsn)
    try:
        with Session(engine) as db:
            rows = db.execute(stmt).all()
            session_ids = [str(row.session_id) for row in rows if row.session_id]
            analyses_by_session: dict[str, SessionAnalysis] = {}
            if session_ids:
                analysis_rows = db.scalars(
                    select(SessionAnalysis)
                    .where(SessionAnalysis.session_id.in_(session_ids))
                    .order_by(SessionAnalysis.session_id, SessionAnalysis.analyzed_at.desc())
                ).all()
                for analysis in analysis_rows:
                    if analysis.session_id and analysis.session_id not in analyses_by_session:
                        analyses_by_session[str(analysis.session_id)] = analysis
    finally:
        if hasattr(engine, "dispose"):
            engine.dispose()

    candidates: list[SessionCandidate] = []
    for row in rows:
        if not row.session_id:
            continue
        session_id = str(row.session_id)
        duration_s = (row.last_end or 0.0) - (row.first_start or 0.0)
        analysis = analyses_by_session.get(session_id)
        updated_at = getattr(row, "updated_at", None) or getattr(row, "last_updated", None)
        candidates.append(
            SessionCandidate(
                session_id=session_id,
                title=analysis.title if analysis else None,
                updated_at=updated_at,
                created_at=getattr(row, "created_at", None),
                summary=analysis.summary if analysis else None,
                segments=int(row.segments) if row.segments is not None else None,
                duration_seconds=duration_s,
            )
        )
    return candidates


def list_sessions_impl(cfg: AgentConfig, name_filter: str = "", limit: int = 10) -> str:
    """List conversation sessions stored in the database.

    Returns session IDs with segment counts, duration, and last-updated
    time, most recent first. Use this to discover which sessions exist
    before calling query_conversation or get_analysis.

    Args:
        cfg: Agent configuration containing the DB DSN.
        name_filter: Optional substring to match against session_id
            (case-insensitive). Empty string returns all sessions.
        limit: Maximum number of sessions to return (default 10).
    """
    try:
        candidates = list_session_candidates_impl(cfg, name_filter=name_filter, limit=limit)
        if not candidates:
            if name_filter.strip():
                return f"No sessions found matching '{name_filter.strip()}'."
            return "No sessions found in the database."
        return render_session_catalog(candidates)

    except Exception as exc:
        return f"Error listing sessions: {exc}"


def build(cfg: AgentConfig):
    from pydantic_ai import Tool

    def list_sessions(name_filter: str = "", limit: int = 10) -> str:
        """List conversation sessions stored in the database.

        Returns session IDs with segment counts, duration, and last-updated
        time, most recent first. Use this to discover which sessions exist
        before calling query_conversation or get_analysis.

        Args:
            name_filter: Optional substring to match against session_id
                (case-insensitive). Empty string returns all sessions.
            limit: Maximum number of sessions to return (default 10).
        """
        return list_sessions_impl(cfg, name_filter=name_filter, limit=limit)

    return Tool(list_sessions)
