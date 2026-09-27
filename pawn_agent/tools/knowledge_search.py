"""Semantic search across notes, transcripts, and coworker items."""

from __future__ import annotations

from pawn_agent.core.coworker.link import format_hits
from pawn_agent.utils.config import AgentConfig
from pawn_core.knowledge_index import search_chunks


def knowledge_search_impl(
    cfg: AgentConfig,
    query: str,
    *,
    kind: str = "",
    limit: int = 8,
) -> str:
    """Return a short list of matching chunks."""
    try:
        hits = search_chunks(cfg, query, limit=limit, kind=kind)
    except Exception as exc:
        return f"Error searching knowledge: {exc}"
    return format_hits(hits)
