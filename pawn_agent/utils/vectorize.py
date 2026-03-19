"""Text vectorization for RAG: transcripts and SiYuan pages.

Moved from pawn_diarize.core.vectorize so that pawn_agent owns the full
RAG-ingestion pipeline alongside its analysis and search tools.

- :func:`vectorize_session` — embeds all transcript segments for a named
  session using speaker-turn chunking and stores them in ``rag_sources`` +
  ``text_chunks``.

- :func:`vectorize_siyuan_page` — fetches a SiYuan page's content blocks,
  embeds them, and stores them in ``rag_sources`` + ``text_chunks``.

Both functions are idempotent: re-running replaces existing chunks for the
same source.

The embedding model is loaded lazily and cached at module level so that
multiple calls within the same process share a single model instance.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple  # Tuple kept for vectorize_session return

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from pawn_agent.utils.db import (
    RagSource,
    TextChunk,
    get_session_analysis,
)
from pawn_agent.utils.siyuan import siyuan_post


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Minimum character length for a chunk to be indexed.
_MIN_CHUNK_CHARS = 20

# SiYuan block types that contain meaningful prose content.
_CONTENT_BLOCK_TYPES = {"p", "h", "h1", "h2", "h3", "h4", "h5", "h6"}

# Strip common Markdown syntax for cleaner embedding text.
_MD_STRIP_RE = re.compile(r"[*_`#\[\]()>~\-]+")


# ──────────────────────────────────────────────────────────────────────────────
# Embedding model cache
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_CACHE: Dict[str, Any] = {}


def load_embedding_model(
    model_name: str,
    device: str = "cpu",
    truncate_dim: Optional[int] = None,
) -> Any:
    """Load a SentenceTransformer model, caching by ``(model_name, device, truncate_dim)``."""
    from sentence_transformers import SentenceTransformer

    key = f"{model_name}:{device}:{truncate_dim}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(
            model_name, device=device, truncate_dim=truncate_dim
        )
    return _MODEL_CACHE[key]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _chunk_id(prefix: str) -> str:
    return hashlib.sha256(prefix.encode()).hexdigest()[:16]


def _source_id(source_type: str, external_id: str) -> str:
    return f"{source_type}:{external_id}"


def _clean_markdown(text: str) -> str:
    return _MD_STRIP_RE.sub(" ", text).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Session vectorization
# ──────────────────────────────────────────────────────────────────────────────


def vectorize_session(
    session_id: str,
    db_dsn: str,
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embed_device: str = "cpu",
    embed_dim: Optional[int] = None,
) -> Tuple[int, str]:
    """Embed a session and store as RAG chunks.

    If a stored analysis exists the analysis fields are used as the primary
    content (overview chunk + speaker-highlights chunk).  When no analysis is
    available the function falls back to speaker-turn chunking of the raw
    transcript segments.

    Idempotent — existing chunks for this session are replaced.

    Returns:
        ``(chunk_count, strategy)`` where strategy is ``"analysis"`` or
        ``"transcript"``.

    Raises:
        ValueError: If neither analysis nor transcript segments exist.
    """
    engine = create_engine(db_dsn)
    analysis = get_session_analysis(session_id, db_dsn)

    src_id = _source_id("transcript", session_id)
    now = datetime.now(timezone.utc)
    embed_model_obj = load_embedding_model(embed_model, embed_device, truncate_dim=embed_dim)

    # ── Analysis path ──────────────────────────────────────────────────────────
    if analysis is not None:
        texts_to_embed: List[Dict[str, Any]] = []

        # Chunk 1: overview (title + summary + key topics)
        overview_parts = []
        if analysis.title:
            overview_parts.append(f"Title: {analysis.title}")
        if analysis.summary:
            overview_parts.append(f"Summary: {analysis.summary}")
        if analysis.key_topics:
            overview_parts.append(f"Key topics:\n{analysis.key_topics}")
        if analysis.tags:
            overview_parts.append(f"Tags: {', '.join(analysis.tags)}")
        if overview_parts:
            texts_to_embed.append({
                "id": _chunk_id(f"transcript:{session_id}:analysis_overview"),
                "text": "\n".join(overview_parts),
                "extra": {"chunk_type": "analysis_overview", "tags": list(analysis.tags or [])},
            })

        # Chunk 2: speaker highlights
        if analysis.speaker_highlights:
            texts_to_embed.append({
                "id": _chunk_id(f"transcript:{session_id}:analysis_speakers"),
                "text": f"Speaker highlights:\n{analysis.speaker_highlights}",
                "extra": {"chunk_type": "analysis_speakers"},
            })

        # Chunk 3: sentiment
        if analysis.sentiment:
            texts_to_embed.append({
                "id": _chunk_id(f"transcript:{session_id}:analysis_sentiment"),
                "text": f"Sentiment: {analysis.sentiment}",
                "extra": {"chunk_type": "analysis_sentiment"},
            })

        if not texts_to_embed:
            raise ValueError(f"Stored analysis for '{session_id}' has no embeddable content.")

        embeddings = embed_model_obj.encode(
            [t["text"] for t in texts_to_embed], batch_size=32, show_progress_bar=False
        )

        with Session(engine) as db:
            for ch in db.scalars(select(TextChunk).where(TextChunk.source_id == src_id)).all():
                db.delete(ch)
            db.merge(RagSource(
                id=src_id,
                source_type="transcript",
                external_id=session_id,
                display_name=analysis.title or session_id,
                extra_data={"indexed_from": "analysis"},
                created_at=now,
            ))
            for item, emb in zip(texts_to_embed, embeddings):
                db.add(TextChunk(
                    id=item["id"],
                    source_id=src_id,
                    text=item["text"],
                    embedding=emb.tolist(),
                    extra_data=item["extra"],
                    created_at=now,
                ))
            db.commit()

        return len(texts_to_embed), "analysis"

    raise ValueError(
        f"No stored analysis found for session '{session_id}'. "
        "Run analyze_summary first or use the vectorize tool which handles this automatically."
    )


# ──────────────────────────────────────────────────────────────────────────────
# SiYuan page vectorization
# ──────────────────────────────────────────────────────────────────────────────


def vectorize_siyuan_page(
    page_id: str,
    siyuan_url: str,
    siyuan_token: str,
    db_dsn: str,
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embed_device: str = "cpu",
    embed_dim: Optional[int] = None,
) -> int:
    """Fetch a SiYuan page's content blocks, embed them, and store as RAG chunks.

    Idempotent — existing chunks for this page are replaced.

    Returns:
        Number of chunks stored.

    Raises:
        ValueError: If no usable content blocks are found for *page_id*.
    """
    try:
        info = siyuan_post(siyuan_url, siyuan_token, "/api/block/getBlockInfo", {"id": page_id})
    except Exception:
        info = {}
    page_title = info.get("rootTitle") or page_id
    page_path = info.get("path", "")

    try:
        raw_blocks = siyuan_post(siyuan_url, siyuan_token, "/api/block/getChildBlocks", {"id": page_id})
        if not isinstance(raw_blocks, list):
            raw_blocks = []
    except Exception:
        raw_blocks = []

    content_blocks = []
    for blk in raw_blocks:
        if blk.get("type", "") not in _CONTENT_BLOCK_TYPES:
            continue
        text = _clean_markdown(blk.get("markdown") or blk.get("content") or "")
        if len(text) < _MIN_CHUNK_CHARS:
            continue
        content_blocks.append((blk.get("id", ""), text))

    if not content_blocks:
        raise ValueError(f"No usable content blocks found for SiYuan page '{page_id}'")

    model = load_embedding_model(embed_model, embed_device, truncate_dim=embed_dim)
    texts = [text for _, text in content_blocks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    src_id = _source_id("siyuan", page_id)
    now = datetime.now(timezone.utc)
    engine = create_engine(db_dsn)

    with Session(engine) as db:
        for ch in db.scalars(select(TextChunk).where(TextChunk.source_id == src_id)).all():
            db.delete(ch)

        db.merge(RagSource(
            id=src_id,
            source_type="siyuan",
            external_id=page_id,
            display_name=page_title,
            extra_data={"path": page_path},
            created_at=now,
        ))

        for i, ((block_id, text), emb) in enumerate(zip(content_blocks, embeddings)):
            db.add(TextChunk(
                id=_chunk_id(f"siyuan:{page_id}:{block_id}"),
                source_id=src_id,
                text=text,
                embedding=emb.tolist(),
                extra_data={"block_id": block_id, "page_title": page_title},
                created_at=now,
            ))
        db.commit()

    return len(content_blocks)
