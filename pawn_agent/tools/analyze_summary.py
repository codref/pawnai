"""Structured session analysis (``session_analyze`` CliTool)."""

from __future__ import annotations

from typing import Optional

from pawn_agent.utils.analysis import run_analysis
from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.db import get_session_analysis
from pawn_agent.utils.siyuan import do_save_to_siyuan


async def analyze_summary_impl(
    cfg: AgentConfig,
    session_id: str,
    *,
    save: bool = False,
    title: Optional[str] = None,
) -> str:
    """Run the standard structured analysis and optionally save it to SiYuan."""
    try:
        content = await run_analysis(cfg, session_id)

        if save:
            analysis = get_session_analysis(session_id, cfg.db_dsn)
            doc_title = title or (analysis.title if analysis else None) or session_id
            all_tags = (
                list(analysis.tags or []) + list(analysis.sentiment_tags or [])
                if analysis
                else []
            )
            do_save_to_siyuan(
                cfg,
                session_id,
                doc_title,
                content,
                path=None,
                tags=all_tags or None,
            )
            return f"Analysis saved to database and SiYuan (title: {doc_title!r}).\n\n{content}"

        return content
    except Exception as exc:
        return f"Error performing structured analysis: {exc}"
