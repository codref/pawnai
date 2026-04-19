"""Tool: query_conversation — fetch and return a session transcript."""

from __future__ import annotations

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.transcript import fetch_transcript


NAME = "query_conversation"
DESCRIPTION = (
    "Fetch and return the full transcript for a session from the database. "
    "Use this when the user explicitly wants to inspect or quote the transcript itself. "
    "Do not call this before analyze_custom or analyze_summary, because those tools "
    "already load the transcript internally."
)


def query_conversation_impl(cfg: AgentConfig, session_id: str) -> str:
    """Retrieve and return the full transcript for a session from the database."""
    return fetch_transcript(cfg, session_id)


def build(cfg: AgentConfig):
    from pydantic_ai import Tool

    def query_conversation(session_id: str) -> str:
        """Retrieve and return the full transcript for a session from the database.

        Use this only when the user asks to inspect, quote, or review the raw
        transcript itself. Do not call this before ``analyze_custom`` or
        ``analyze_summary`` — those tools already fetch the transcript and
        calling this first wastes context and can duplicate database reads.

        Args:
            session_id: Unique session identifier stored in the database.
        """
        return query_conversation_impl(cfg, session_id)

    return Tool(query_conversation)
