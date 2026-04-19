from __future__ import annotations

from pawn_agent.utils.config import AgentConfig

NAME = "list_sessions"
DESCRIPTION = (
    "List available conversation sessions from the database, "
    "with segment counts, duration, and last-updated timestamps. "
    "Use this before query_conversation when you need to discover session IDs."
)


def _fmt_duration(seconds: float) -> str:
    total = int(max(seconds, 0))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


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
        from sqlalchemy import create_engine, func, select
        from sqlalchemy.orm import Session

        from pawn_agent.utils.db import TranscriptionSegment

        name_filter_clean = name_filter.strip()

        stmt = (
            select(
                TranscriptionSegment.session_id,
                func.count(TranscriptionSegment.id).label("segments"),
                func.min(TranscriptionSegment.start_time).label("first_start"),
                func.max(TranscriptionSegment.end_time).label("last_end"),
                func.max(TranscriptionSegment.created_at).label("last_updated"),
            )
            .group_by(TranscriptionSegment.session_id)
            .order_by(func.max(TranscriptionSegment.created_at).desc())
        )

        if name_filter_clean:
            stmt = stmt.where(
                TranscriptionSegment.session_id.ilike(f"%{name_filter_clean}%")
            )

        stmt = stmt.limit(limit)

        engine = create_engine(cfg.db_dsn)
        with Session(engine) as db:
            rows = db.execute(stmt).all()

        if not rows:
            if name_filter_clean:
                return f"No sessions found matching '{name_filter_clean}'."
            return "No sessions found in the database."

        lines = [f"Found {len(rows)} session(s):"]
        for row in rows:
            duration_s = (row.last_end or 0.0) - (row.first_start or 0.0)
            updated_str = (
                row.last_updated.strftime("%Y-%m-%d %H:%M")
                if row.last_updated
                else "unknown"
            )
            lines.append(
                f"  {row.session_id}"
                f"  |  segments: {row.segments}"
                f"  |  duration: {_fmt_duration(duration_s)}"
                f"  |  updated: {updated_str}"
            )
        return "\n".join(lines)

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
