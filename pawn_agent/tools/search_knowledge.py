"""Tool: search_knowledge — semantic vector search over sessions and SiYuan notes."""

from __future__ import annotations

import asyncio
from typing import Optional

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig


NAME = "search_knowledge"
DESCRIPTION = (
    "Perform a semantic similarity search over stored transcript chunks and "
    "SiYuan note content."
)


async def search_knowledge_impl(
    cfg: AgentConfig,
    query: str,
    source_type: Optional[str] = None,
    session_id: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Search the knowledge store and return formatted results."""
    from pawn_agent.core.store import NS_SESSIONS, NS_SIYUAN, get_store  # noqa: PLC0415

    top_k = max(1, min(20, top_k))
    store = await get_store(cfg)

    # Determine which namespaces to search.
    search_sessions = source_type in (None, "transcript", "session")
    search_siyuan = source_type in (None, "siyuan")

    tasks = []
    labels = []
    if search_sessions:
        tasks.append(store.asearch(NS_SESSIONS, query=query, limit=top_k))
        labels.append("session")
    if search_siyuan:
        tasks.append(store.asearch(NS_SIYUAN, query=query, limit=top_k))
        labels.append("siyuan")

    if not tasks:
        return "No relevant chunks found in the knowledge store."

    result_groups = await asyncio.gather(*tasks)

    # Merge and sort by score descending; higher score = more similar.
    merged: list[tuple[float, str, object]] = []
    for label, results in zip(labels, result_groups):
        for item in results:
            score = item.score if item.score is not None else 0.0
            merged.append((score, label, item))
    merged.sort(key=lambda x: x[0], reverse=True)
    merged = merged[:top_k]

    if not merged:
        return "No relevant chunks found in the knowledge store."

    # Filter by session_id if requested.
    if session_id:
        merged = [
            (s, lbl, item)
            for s, lbl, item in merged
            if (item.value or {}).get("session_id") == session_id
            or str(item.key).startswith(session_id)
        ]
        if not merged:
            return f"No relevant chunks found for session '{session_id}'."

    parts: list[str] = []
    for i, (score, label, item) in enumerate(merged, 1):
        value = item.value or {}
        text = value.get("text", "").strip()
        speaker = value.get("speaker_name") or value.get("speaker", "")

        if label == "session":
            sid = value.get("session_id", item.key)
            source_label = f"Session: {sid}"
            title = value.get("title", "")
            if title and title != sid:
                source_label += f" ({title})"
        else:
            page_title = value.get("page_title") or value.get("display_name", item.key)
            source_label = f"SiYuan: {page_title}"

        speaker_part = f"Speaker: {speaker}  " if speaker else ""
        header = (
            f"[{i}] {source_label}  {speaker_part}(similarity: {score:.3f})"
        )
        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


def build(cfg: AgentConfig) -> Tool:
    async def search_knowledge(
        query: str,
        source_type: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """Perform a semantic similarity search over the knowledge store of stored
        transcript chunks and SiYuan note blocks. Use this when the user
        asks a question about past conversations or notes that may span
        multiple sessions, or when you need context beyond a single transcript.
        Returns the most relevant text chunks with source, speaker, and
        similarity scores.

        Args:
            query: Natural language search query.
            source_type: Filter by source type: 'transcript' for session chunks,
                'siyuan' for note page chunks, or omit to search both.
            session_id: Restrict the search to a specific session.
                Omit to search across all sessions.
            top_k: Number of top results to return (1-20).
        """
        return await search_knowledge_impl(cfg, query, source_type, session_id, top_k)

    return Tool(search_knowledge)

