"""Speaker embedding management using PostgreSQL + pgvector."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
from sqlalchemy import select, text

from .config import DEFAULT_DB_DSN
from .database import Embedding, SpeakerName, get_engine, get_session, init_db


class EmbeddingManager:
    """Manage speaker embeddings backed by PostgreSQL + pgvector.

    Args:
        db_dsn: PostgreSQL DSN, e.g.
                ``"postgresql+psycopg://user:pass@host:5432/db"``.
                Falls back to :data:`~pawnai.core.config.DEFAULT_DB_DSN`
                (which respects the ``DATABASE_URL`` environment variable).
    """

    def __init__(self, db_dsn: str = DEFAULT_DB_DSN) -> None:
        self.db_dsn = db_dsn
        self._engine = get_engine(db_dsn)
        init_db(self._engine)

    # ──────────────────────────────────────────────────────────────────────────
    # Write operations
    # ──────────────────────────────────────────────────────────────────────────

    def add_embedding(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        audio_path: str,
    ) -> None:
        """Store (or replace) an embedding for *speaker_id*.

        Uses ``speaker_id`` as both the record ``id`` and the
        ``local_speaker_label`` so the record can be retrieved via
        :meth:`get_all_embeddings`.

        Args:
            speaker_id: Unique identifier for the speaker (user-supplied).
            embedding: Normalised 512-dim float32 embedding array.
            audio_path: Path to the audio source.
        """
        vec = embedding.flatten().tolist()
        record = Embedding(
            id=speaker_id,
            audio_file=str(audio_path),
            local_speaker_label=speaker_id,
            start_time=0.0,
            end_time=0.0,
            embedding=vec,
        )
        with get_session(self._engine) as session:
            session.merge(record)  # upsert on primary key

    def add_speaker_name(
        self,
        speaker_id: str,
        name: str,
        count: int = 1,  # retained for API compat; not persisted
    ) -> None:
        """Add or update a human-readable name for *speaker_id*.

        Args:
            speaker_id: Unique speaker identifier.
            name: Human-readable speaker name.
            count: Unused (kept for backward API compatibility).
        """
        record = SpeakerName(
            id=speaker_id,
            audio_file="",
            local_speaker_label=speaker_id,
            speaker_name=name,
            labeled_at=datetime.now(timezone.utc),
        )
        with get_session(self._engine) as session:
            session.merge(record)

    # ──────────────────────────────────────────────────────────────────────────
    # Read operations
    # ──────────────────────────────────────────────────────────────────────────

    def search_similar_speakers(
        self,
        embedding: np.ndarray,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the *limit* most similar speaker embeddings.

        Similarity is measured by cosine distance using pgvector's ``<=>``
        operator (``distance = 1 - cosine_similarity``).

        Args:
            embedding: Query embedding vector.
            limit: Maximum number of results.

        Returns:
            List of dicts with keys ``id``, ``audio_file``,
            ``local_speaker_label``, ``start_time``, ``end_time``, and
            ``_distance`` (cosine distance, lower is better).
        """
        vec_str = "[" + ",".join(str(v) for v in embedding.flatten().tolist()) + "]"
        sql = text(
            "SELECT id, audio_file, local_speaker_label, start_time, end_time, "
            "(embedding <=> CAST(:vec AS vector)) AS _distance "
            "FROM embeddings "
            "ORDER BY embedding <=> CAST(:vec AS vector) "
            "LIMIT :limit"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql, {"vec": vec_str, "limit": limit}).fetchall()
        return [row._asdict() for row in rows]

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Return all embedding records.

        Returns:
            List of dicts containing all
            :class:`~pawnai.core.database.Embedding` columns plus a
            ``speaker_id`` alias for ``local_speaker_label``.
        """
        with get_session(self._engine) as session:
            rows = session.execute(select(Embedding)).scalars().all()
        results = []
        for row in rows:
            results.append(
                {
                    "speaker_id": row.local_speaker_label,
                    "id": row.id,
                    "audio_file": row.audio_file,
                    "local_speaker_label": row.local_speaker_label,
                    "start_time": row.start_time,
                    "end_time": row.end_time,
                    "embedding": row.embedding,
                }
            )
        return results

    def get_speaker_names(self) -> Dict[str, str]:
        """Return a mapping of speaker identifier to human-readable name.

        Returns:
            ``{local_speaker_label: speaker_name}`` dict.
        """
        with get_session(self._engine) as session:
            rows = session.execute(select(SpeakerName)).scalars().all()
        return {row.local_speaker_label: row.speaker_name for row in rows}
