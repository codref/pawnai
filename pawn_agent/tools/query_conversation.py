"""Tool: query_conversation — fetch and return a session transcript."""

from __future__ import annotations

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.transcript import fetch_transcript


NAME = "query_conversation"
DESCRIPTION = "Fetch and return the full transcript for a session from the database."


def build(cfg: AgentConfig) -> Tool:
    def query_conversation(session_id: str) -> str:
        """Retrieve and return the full transcript for a session from the database.

        Use this first when the user asks about the content of a conversation
        or wants to analyse a specific session.

        Args:
            session_id: Unique session identifier stored in the database.
        """
        return fetch_transcript(cfg, session_id)

    return Tool(query_conversation)
