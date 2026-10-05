"""Shared pgvector index over notes, transcripts, and coworker items.

Embeddings use the same Ollama model as sallm (``agent.sallm.embedding_*``).
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional, cast

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.db import KnowledgeChunk, _get_session
from pawn_core.database import get_engine

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]


def chunk_markdown(text: str, *, max_chars: int = 1200) -> list[tuple[str, str]]:
    """Split Markdown into ``(heading, text)`` chunks."""
    body = text or ""
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return _window("", body, max_chars)
    chunks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        heading = match.group(2).strip()
        section = body[start:end].strip()
        piece = f"{match.group(0).strip()}\n{section}".strip()
        chunks.extend(_window(heading, piece, max_chars))
    return [chunk for chunk in chunks if chunk[1].strip()]


def chunk_transcript(text: str, *, max_chars: int = 1200) -> list[tuple[str, str]]:
    """Split a speaker transcript into windows."""
    return _window("", text or "", max_chars)


def _window(heading: str, text: str, max_chars: int) -> list[tuple[str, str]]:
    body = (text or "").strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [(heading, body)]
    out: list[tuple[str, str]] = []
    step = max(200, max_chars - 100)
    for start in range(0, len(body), step):
        out.append((heading, body[start : start + max_chars].strip()))
    return out


def substantial_change(old: str, new: str, *, min_chars: int = 40) -> bool:
    """True when a note changed enough to be worth another coworker pass."""
    if not (old or "").strip():
        return bool((new or "").strip())
    if abs(len(new or "") - len(old or "")) >= min_chars:
        return True
    old_heads = set(_HEADING_RE.findall(old or ""))
    new_heads = set(_HEADING_RE.findall(new or ""))
    return bool(new_heads - old_heads)


def _embedder(cfg: AgentConfig) -> Callable[[str], list[float]]:
    from sallm.memory.embedding import make_embed_fn  # noqa: PLC0415

    model = cfg.sallm.embedding_model
    api_base = cfg.sallm.embedding_api_base
    dimensions = int(getattr(cfg.coworker, "embed_dim", 1024) or 1024)
    return cast(Callable[[str], list[float]], make_embed_fn(model, api_base, dimensions))


def index_text(
    cfg: AgentConfig,
    *,
    source_kind: str,
    source_ref: str,
    text: str,
    embed: Optional[Callable[[str], list[float]]] = None,
) -> int:
    """Replace chunks for *source_ref*. Returns how many chunks were stored."""
    if source_kind in {"note", "analysis"}:
        chunks = chunk_markdown(text)
    elif source_kind in {"transcript", "session"}:
        chunks = chunk_transcript(text)
    else:
        chunks = _window("", text, 1200)
    if not chunks:
        delete_source(cfg.db_dsn, source_ref)
        return 0
    embed_fn = embed or _embedder(cfg)
    now = datetime.now(timezone.utc)
    rows = []
    for heading, piece in chunks:
        vector = embed_fn(piece)
        rows.append(
            KnowledgeChunk(
                id=str(uuid.uuid4()),
                source_kind=source_kind,
                source_ref=source_ref,
                heading=heading or None,
                text=piece,
                embedding=vector,
                content_hash=content_hash(piece),
                updated_at=now,
            )
        )
    with _get_session(cfg.db_dsn) as db:
        db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source_ref == source_ref))
        db.add_all(rows)
    return len(rows)


def delete_source(dsn: str, source_ref: str) -> None:
    with _get_session(dsn) as db:
        db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source_ref == source_ref))


def search_chunks(
    cfg: AgentConfig, query: str, *, limit: int = 8, kind: str = ""
) -> list[dict[str, Any]]:
    """Nearest chunks for *query*. Returns plain dicts."""
    text = (query or "").strip()
    if not text:
        return []
    vector = _embedder(cfg)(text)
    with Session(get_engine(cfg.db_dsn)) as db:
        stmt = select(KnowledgeChunk).order_by(KnowledgeChunk.embedding.cosine_distance(vector))
        if kind:
            stmt = stmt.where(KnowledgeChunk.source_kind == kind)
        rows = db.scalars(stmt.limit(max(1, limit))).all()
        return [
            {
                "source_kind": row.source_kind,
                "source_ref": row.source_ref,
                "heading": row.heading or "",
                "text": row.text,
            }
            for row in rows
        ]
