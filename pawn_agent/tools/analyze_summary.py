"""Tool: analyze_summary — standard structured analysis, saved to the database."""

from __future__ import annotations

from typing import Optional

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.siyuan import do_save_to_siyuan
from pawn_agent.utils.db import get_session_analysis
from pawn_agent.utils.analysis import run_analysis, parse_sections

NAME = "analyze_summary"
DESCRIPTION = (
    "Run the standard structured analysis on a session transcript "
    "(Title / Summary / Key Topics / Speaker Highlights / Sentiment / Tags) "
    "and persist the result to the database. "
    "Use this for the canonical session summary, not for bespoke tasks."
)


def build(cfg: AgentConfig) -> Tool:
    async def analyze_summary(
        session_id: str,
        save: bool = False,
        title: Optional[str] = None,
    ) -> str:
        """Run the standard structured analysis on a conversation session.

        Produces a fixed-format report with Title, Summary, Key Topics,
        Speaker Highlights, Sentiment, Sentiment Tags, and Tags sections,
        then persists the result to the session_analysis database table.

        Set save=true (and optionally a title) to also store the full report
        as a SiYuan document.

        Args:
            session_id: Unique session identifier stored in the database.
            save: Set to true to also save the report to SiYuan.
            title: Override the auto-generated document title when save=true.
        """
        try:
            content = await run_analysis(cfg, session_id)

            if save:
                analysis = get_session_analysis(session_id, cfg.db_dsn)
                doc_title = title or (analysis.title if analysis else None) or session_id
                all_tags = (
                    list(analysis.tags or []) + list(analysis.sentiment_tags or [])
                    if analysis else []
                )
                do_save_to_siyuan(
                    cfg, session_id, doc_title, content,
                    path=None, tags=all_tags or None,
                )
                return f"Analysis saved to database and SiYuan (title: {doc_title!r}).\n\n{content}"

            return content
        except Exception as exc:
            return f"Error performing structured analysis: {exc}"

    return Tool(analyze_summary)
