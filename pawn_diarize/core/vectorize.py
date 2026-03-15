"""Text vectorization for RAG: transcripts and SiYuan pages.

This module handles ingestion into the RAG index:

- :func:`vectorize_session` — embeds all transcript segments for a named
  session using speaker-turn chunking and stores them in ``rag_sources`` +
  ``text_chunks``.

- :func:`vectorize_siyuan_page` — fetches a SiYuan page's content blocks,
  embeds them, and stores them in ``rag_sources`` + ``text_chunks``.

Both functions are idempotent: re-running replaces the existing chunks for
the same source.

The embedding model is loaded lazily and cached at module level so that
multiple calls within the same process share a single model instance.

Usage::

    from pawn_diarize.core.vectorize import vectorize_session

    n = vectorize_session(
        session_id="standup-2026-03-10",
        db_dsn="postgresql+psycopg://...",
        embed_model="Qwen/Qwen3-Embedding-0.6B",
        embed_device="cpu",
    )
    print(f"Indexed {n} chunks")
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import (
    RagSource,
    SessionAnalysis,
    SpeakerName,
    TextChunk,
    TranscriptionSegment,
    get_engine,
    get_session_analysis,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model cache
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_CACHE: Dict[str, Any] = {}


def load_embedding_model(
    model_name: str,
    device: str = "cpu",
    truncate_dim: Optional[int] = None,
) -> Any:
    """Load a SentenceTransformer model, caching by ``(model_name, device, truncate_dim)``.

    Args:
        model_name: HuggingFace model name or local path.
        device: Compute device: ``"cpu"`` or ``"cuda"``.
        truncate_dim: If set, truncate output embeddings to this many dimensions
            using matryoshka representation learning.  Qwen3-Embedding supports
            this natively.  Must be ≤ 2000 to use an HNSW index in pgvector.

    Returns:
        A :class:`sentence_transformers.SentenceTransformer` instance.
    """
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
    """Return a 16-char hex deterministic ID from a string prefix."""
    return hashlib.sha256(prefix.encode()).hexdigest()[:16]


def _source_id(source_type: str, external_id: str) -> str:
    return f"{source_type}:{external_id}"


# ──────────────────────────────────────────────────────────────────────────────
# Transcript chunking
# ──────────────────────────────────────────────────────────────────────────────


def chunk_transcript_by_speaker_turns(
    segments: List[TranscriptionSegment],
    name_lookup: Dict[Tuple[str, str], str],
) -> List[Dict[str, Any]]:
    """Group contiguous same-speaker segments into speaker-turn chunks.

    Each chunk corresponds to one uninterrupted speech turn.  Merging adjacent
    segments from the same speaker keeps chunks semantically coherent and
    preserves speaker attribution.

    Args:
        segments: :class:`TranscriptionSegment` rows ordered by
            ``segment_index`` ascending.
        name_lookup: Mapping of ``(audio_file, original_speaker_label)`` to
            a human-readable display name.

    Returns:
        List of chunk dicts::

            {
                "speaker_name": str,
                "start_time": float,
                "end_time": float,
                "text": str,          # space-joined segment texts
                "audio_file": str,
                "segment_ids": list[str],
            }
    """
    chunks: List[Dict[str, Any]] = []
    buffer_texts: List[str] = []
    buffer_ids: List[str] = []
    current_speaker: Optional[str] = None
    current_start: float = 0.0
    current_end: float = 0.0
    current_audio: str = ""

    for seg in segments:
        resolved = name_lookup.get(
            (seg.audio_file, seg.original_speaker_label),
            seg.original_speaker_label or "Speaker",
        )

        if resolved != current_speaker:
            if current_speaker is not None and buffer_texts:
                text = " ".join(buffer_texts)
                if len(text) >= _MIN_CHUNK_CHARS:
                    chunks.append(
                        {
                            "speaker_name": current_speaker,
                            "start_time": current_start,
                            "end_time": current_end,
                            "text": text,
                            "audio_file": current_audio,
                            "segment_ids": list(buffer_ids),
                        }
                    )
            current_speaker = resolved
            buffer_texts = []
            buffer_ids = []
            current_start = seg.start_time
            current_audio = seg.audio_file

        stripped = (seg.text or "").strip()
        if stripped:
            buffer_texts.append(stripped)
        buffer_ids.append(seg.id)
        current_end = seg.end_time

    # Flush last buffer
    if current_speaker is not None and buffer_texts:
        text = " ".join(buffer_texts)
        if len(text) >= _MIN_CHUNK_CHARS:
            chunks.append(
                {
                    "speaker_name": current_speaker,
                    "start_time": current_start,
                    "end_time": current_end,
                    "text": text,
                    "audio_file": current_audio,
                    "segment_ids": list(buffer_ids),
                }
            )

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Session summary chunk
# ──────────────────────────────────────────────────────────────────────────────


def _build_summary_chunk_text(
    session_id: str,
    segments: List[TranscriptionSegment],
    name_lookup: Dict[Tuple[str, str], str],
    analysis: Optional[Any],
) -> Tuple[str, List[str]]:
    """Compose the text and tags for the session-level summary chunk.

    The returned text is embedded alongside the per-speaker-turn chunks so
    that semantic queries like "standup with Tom" can surface the right
    session even without SQL tag filters.

    Args:
        session_id: Session identifier.
        segments: All transcript segments for the session, ordered by index.
        name_lookup: Mapping of ``(audio_file, label)`` to display name.
        analysis: Most recent :class:`SessionAnalysis` row, or ``None``.

    Returns:
        ``(text, tags)`` — the text to embed and the tags list for
        ``extra_data`` (empty list when no analysis is available).
    """
    # Participants: unique resolved names in first-seen order
    seen: Dict[str, None] = {}
    for seg in segments:
        name = name_lookup.get(
            (seg.audio_file, seg.original_speaker_label),
            seg.original_speaker_label or "Speaker",
        )
        seen[name] = None
    participants = list(seen.keys())

    duration_min = max(0, round((segments[-1].end_time - segments[0].start_time) / 60))

    lines = [f"Session: {session_id}"]

    if analysis and analysis.title:
        lines.append(f"Title: {analysis.title}")

    lines.append(f"Participants: {', '.join(participants)}")
    lines.append(f"Duration: {duration_min} minutes")

    if analysis and analysis.summary:
        lines.append(f"Summary: {analysis.summary}")

    if analysis and analysis.key_topics:
        lines.append(f"Key topics: {analysis.key_topics}")

    tags: List[str] = []
    if analysis and analysis.tags:
        tags = list(analysis.tags)
        lines.append(f"Tags: {', '.join(tags)}")

    return "\n".join(lines), tags


# ──────────────────────────────────────────────────────────────────────────────
# Session vectorization
# ──────────────────────────────────────────────────────────────────────────────


def vectorize_session(
    session_id: str,
    db_dsn: str,
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embed_device: str = "cpu",
    embed_dim: Optional[int] = None,
) -> int:
    """Embed all transcript segments for *session_id* and store as RAG chunks.

    Chunking strategy: contiguous segments from the same speaker are merged
    into a single chunk.  This preserves speaker attribution and produces
    semantically coherent units.

    The operation is idempotent — existing chunks for this session are deleted
    before inserting fresh ones.

    Args:
        session_id: Session identifier matching ``TranscriptionSegment.session_id``.
        db_dsn: PostgreSQL DSN.
        embed_model: HuggingFace model name or local path.
        embed_device: Compute device for the embedding model.

    Returns:
        Number of chunks stored.

    Raises:
        ValueError: If no segments are found for *session_id*.
    """
    engine = get_engine(db_dsn)

    with Session(engine) as db:
        # Fetch all segments for this session, ordered
        segments = list(
            db.scalars(
                select(TranscriptionSegment)
                .where(TranscriptionSegment.session_id == session_id)
                .order_by(TranscriptionSegment.segment_index)
            )
        )
        if not segments:
            raise ValueError(f"No transcription segments found for session '{session_id}'")

        # Build name lookup: (audio_file, label) → display name
        audio_files = list({s.audio_file for s in segments})
        speaker_rows = list(
            db.scalars(
                select(SpeakerName).where(SpeakerName.audio_file.in_(audio_files))
            )
        )
        name_lookup: Dict[Tuple[str, str], str] = {
            (r.audio_file, r.local_speaker_label): r.speaker_name
            for r in speaker_rows
        }

    chunks = chunk_transcript_by_speaker_turns(segments, name_lookup)
    if not chunks:
        return 0

    # Fetch existing analysis to enrich the session summary chunk (may be None)
    analysis = get_session_analysis(session_id, engine)
    summary_text, summary_tags = _build_summary_chunk_text(
        session_id, segments, name_lookup, analysis
    )

    # Load model and embed summary + all turn chunks in a single batch
    model = load_embedding_model(embed_model, embed_device, truncate_dim=embed_dim)
    all_texts = [summary_text] + [c["text"] for c in chunks]
    all_embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=False)
    summary_embedding = all_embeddings[0]
    chunk_embeddings = all_embeddings[1:]

    src_id = _source_id("transcript", session_id)
    now = datetime.now(timezone.utc)
    audio_files_list = sorted({c["audio_file"] for c in chunks})

    with Session(engine) as db:
        # Delete existing chunks for this source (idempotency)
        existing_chunks = db.scalars(
            select(TextChunk).where(TextChunk.source_id == src_id)
        ).all()
        for ch in existing_chunks:
            db.delete(ch)

        # Upsert rag_sources row
        source_row = RagSource(
            id=src_id,
            source_type="transcript",
            external_id=session_id,
            display_name=session_id,
            extra_data={"audio_files": audio_files_list},
            created_at=now,
        )
        db.merge(source_row)

        # Insert session summary chunk
        db.add(
            TextChunk(
                id=_chunk_id(f"transcript:{session_id}:summary"),
                source_id=src_id,
                speaker_name=None,
                start_time=None,
                end_time=None,
                text=summary_text,
                embedding=summary_embedding.tolist(),
                extra_data={"chunk_type": "session_summary", "tags": summary_tags},
                created_at=now,
            )
        )

        # Insert per-speaker-turn chunks
        for i, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings)):
            db.add(
                TextChunk(
                    id=_chunk_id(f"transcript:{session_id}:{i}"),
                    source_id=src_id,
                    speaker_name=chunk["speaker_name"],
                    start_time=chunk["start_time"],
                    end_time=chunk["end_time"],
                    text=chunk["text"],
                    embedding=emb.tolist(),
                    extra_data={
                        "segment_ids": chunk["segment_ids"],
                        "audio_file": chunk["audio_file"],
                    },
                    created_at=now,
                )
            )
        db.commit()

    return len(chunks) + 1  # +1 for the session summary chunk


# ──────────────────────────────────────────────────────────────────────────────
# SiYuan page vectorization
# ──────────────────────────────────────────────────────────────────────────────

# Minimum character length for any chunk to be indexed.
# Filters out ASR noise (single words, fillers) and empty SiYuan blocks.
_MIN_CHUNK_CHARS = 20

# Block types in SiYuan that contain meaningful prose content.
_CONTENT_BLOCK_TYPES = {"p", "h", "h1", "h2", "h3", "h4", "h5", "h6"}
_MIN_BLOCK_CHARS = _MIN_CHUNK_CHARS

# Strip common Markdown syntax to get a cleaner text for embedding.
_MD_STRIP_RE = re.compile(r"[*_`#\[\]()>~\-]+")


def _clean_markdown(text: str) -> str:
    return _MD_STRIP_RE.sub(" ", text).strip()


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

    Only paragraph and heading blocks are indexed.  Blocks shorter than
    ``_MIN_BLOCK_CHARS`` characters after stripping Markdown syntax are skipped.

    The operation is idempotent — existing chunks for this page are replaced.

    Args:
        page_id: SiYuan document block ID.
        siyuan_url: Base URL of the SiYuan instance.
        siyuan_token: SiYuan API token.
        db_dsn: PostgreSQL DSN.
        embed_model: HuggingFace model name or local path.
        embed_device: Compute device for the embedding model.

    Returns:
        Number of chunks stored.

    Raises:
        ValueError: If no usable content blocks are found for *page_id*.
    """
    from .siyuan import SiyuanClient

    client = SiyuanClient(url=siyuan_url, token=siyuan_token)

    # Fetch page metadata (title, path)
    info = client.get_block_info(page_id)
    page_title = info.get("rootTitle") or page_id
    page_path = info.get("path", "")

    # Fetch child blocks
    raw_blocks = client.get_child_blocks(page_id)

    # Filter to meaningful content blocks
    content_blocks = []
    for blk in raw_blocks:
        btype = blk.get("type", "")
        if btype not in _CONTENT_BLOCK_TYPES:
            continue
        text = _clean_markdown(blk.get("markdown") or blk.get("content") or "")
        if len(text) < _MIN_BLOCK_CHARS:
            continue
        content_blocks.append((blk.get("id", ""), text))

    if not content_blocks:
        raise ValueError(f"No usable content blocks found for SiYuan page '{page_id}'")

    # Load model and embed
    model = load_embedding_model(embed_model, embed_device, truncate_dim=embed_dim)
    texts = [text for _, text in content_blocks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    src_id = _source_id("siyuan", page_id)
    now = datetime.now(timezone.utc)
    engine = get_engine(db_dsn)

    with Session(engine) as db:
        # Delete existing chunks for this source (idempotency)
        existing_chunks = db.scalars(
            select(TextChunk).where(TextChunk.source_id == src_id)
        ).all()
        for ch in existing_chunks:
            db.delete(ch)

        # Upsert rag_sources row
        source_row = RagSource(
            id=src_id,
            source_type="siyuan",
            external_id=page_id,
            display_name=page_title,
            extra_data={"path": page_path},
            created_at=now,
        )
        db.merge(source_row)

        # Insert fresh chunks
        for i, ((block_id, text), emb) in enumerate(zip(content_blocks, embeddings)):
            db.add(
                TextChunk(
                    id=_chunk_id(f"siyuan:{page_id}:{block_id}"),
                    source_id=src_id,
                    text=text,
                    embedding=emb.tolist(),
                    extra_data={"block_id": block_id, "page_title": page_title},
                    created_at=now,
                )
            )
        db.commit()

    return len(content_blocks)
