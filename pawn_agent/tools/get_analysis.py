"""Tool: get_analysis — retrieve a stored session analysis from the database."""

from __future__ import annotations

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig

NAME = "get_analysis"
DESCRIPTION = (
    "Retrieve the most recent stored analysis for a session from the database. "
    "Use this to READ an existing analysis (title, summary, topics, sentiment, tags) "
    "without re-running it. Never call analyze_summary just to read what is already stored."
)


def build(cfg: AgentConfig) -> Tool:
    def get_analysis(session_id: str) -> str:
        """Return the most recent stored analysis for a session.

        Reads directly from the session_analysis table — no LLM call,
        no transcript fetch. Use this whenever the user asks to see, retrieve,
        or reference an existing analysis rather than produce a new one.

        Args:
            session_id: Unique session identifier stored in the database.
        """
        from pawn_agent.utils.db import get_session_analysis

        row = get_session_analysis(session_id, cfg.db_dsn)
        if row is None:
            return f"No analysis found for session '{session_id}'. Use analyze_summary to create one."

        parts = []
        if row.title:
            parts.append(f"## Title\n{row.title}")
        if row.summary:
            parts.append(f"## Summary\n{row.summary}")
        if row.key_topics:
            parts.append(f"## Key Topics / Keywords\n{row.key_topics}")
        if row.speaker_highlights:
            parts.append(f"## Speaker Highlights\n{row.speaker_highlights}")
        if row.sentiment:
            parts.append(f"## Sentiment\n{row.sentiment}")
        if row.sentiment_tags:
            parts.append(f"## Sentiment Tags\n{', '.join(row.sentiment_tags)}")
        if row.tags:
            parts.append(f"## Tags\n{', '.join(row.tags)}")

        analyzed_at = row.analyzed_at.strftime("%Y-%m-%d %H:%M UTC") if row.analyzed_at else "unknown"
        header = f"*(session: {session_id} · analyzed: {analyzed_at} · model: {row.model})*"
        return header + "\n\n" + "\n\n".join(parts)

    return Tool(get_analysis)
