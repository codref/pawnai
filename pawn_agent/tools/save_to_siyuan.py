"""Save Markdown / stored analysis to SiYuan (``siyuan_save`` CliTool)."""

from __future__ import annotations

from typing import Optional

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.siyuan import do_save_to_siyuan


def format_analysis_markdown(row) -> str:
    """Render a SessionAnalysis ORM row as Markdown suitable for SiYuan."""
    parts: list[str] = []
    if row.title:
        parts.append(f"# {row.title}")
    if row.summary:
        parts.append(f"## Summary\n\n{row.summary}")
    if row.key_topics:
        parts.append(f"## Key Topics / Keywords\n\n{row.key_topics}")
    if row.speaker_highlights:
        parts.append(f"## Speaker Highlights\n\n{row.speaker_highlights}")
    if row.sentiment:
        parts.append(f"## Sentiment\n\n{row.sentiment}")
    if row.sentiment_tags:
        parts.append(f"## Sentiment Tags\n\n{', '.join(row.sentiment_tags)}")
    if row.tags:
        parts.append(f"## Tags\n\n{', '.join(row.tags)}")
    return "\n\n".join(parts).strip()


def save_to_siyuan_impl(
    cfg: AgentConfig,
    session_id: str,
    content: str,
    title: Optional[str] = None,
    path: Optional[str] = None,
    tags: Optional[list] = None,
) -> str:
    """Save already-generated content to SiYuan using the shared helper path."""
    try:
        return do_save_to_siyuan(cfg, session_id, title, content, path, tags=tags)
    except Exception as exc:
        return f"Error saving to SiYuan: {exc}"


def save_analysis_to_siyuan_impl(
    cfg: AgentConfig,
    session_id: str,
    *,
    title: Optional[str] = None,
    path: Optional[str] = None,
) -> str:
    """Load the latest DB analysis for *session_id* and save it to SiYuan."""
    from pawn_agent.utils.db import get_session_analysis  # noqa: PLC0415

    row = get_session_analysis(session_id, cfg.db_dsn)
    if row is None:
        return (
            f"Error: no stored analysis for session {session_id!r}. "
            "Run session_analyze first, or session_analyze --save to write SiYuan directly."
        )
    content = format_analysis_markdown(row)
    if not content:
        return f"Error: analysis for {session_id!r} has no content fields to save."
    doc_title = title or row.title or session_id
    all_tags = list(row.tags or []) + list(row.sentiment_tags or [])
    return save_to_siyuan_impl(
        cfg,
        session_id=session_id,
        content=content,
        title=doc_title,
        path=path,
        tags=all_tags or None,
    )
