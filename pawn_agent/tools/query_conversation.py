"""Fetch a session transcript (``session_transcript`` CliTool)."""

from __future__ import annotations

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.transcript import fetch_transcript


def query_conversation_impl(cfg: AgentConfig, session_id: str) -> str:
    """Retrieve and return the full transcript for a session from the database."""
    return fetch_transcript(cfg, session_id)
