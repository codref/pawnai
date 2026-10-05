"""Link a new item to related knowledge-index hits."""

from __future__ import annotations

import logging
from typing import Any

from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)


def related_lines(
    cfg: AgentConfig, text: str, source_ref: str, *, limit: int = 5
) -> tuple[list[str], int]:
    """Return wiki-link lines and a recurrence count of similar prior items."""
    from pawn_core.knowledge_index import search_chunks  # noqa: PLC0415

    hits = search_chunks(cfg, text, limit=limit)
    lines: list[str] = []
    recurrence = 0
    for hit in hits:
        if hit.get("source_ref") == source_ref:
            continue
        ref = hit.get("source_ref") or ""
        heading = hit.get("heading") or ""
        if hit.get("source_kind") == "item":
            recurrence += 1
        label = f"[[{ref}]]" if ref.endswith(".md") or "/" in ref else ref
        if heading:
            label = f"{label} — {heading}"
        lines.append(label)
    return lines, recurrence


def format_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "(no matches)"
    rows = []
    for hit in hits:
        ref = hit.get("source_ref") or ""
        heading = hit.get("heading") or ""
        snippet = (hit.get("text") or "").replace("\n", " ")[:180]
        rows.append(f"- {hit.get('source_kind')} {ref} {heading}: {snippet}")
    return "\n".join(rows) + "\n"
