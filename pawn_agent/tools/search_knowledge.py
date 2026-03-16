"""Tool: search_knowledge — semantic vector search over the RAG index."""

from __future__ import annotations

from typing import Optional

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig


NAME = "search_knowledge"
DESCRIPTION = (
    "Perform a semantic similarity search over stored transcript chunks and "
    "SiYuan note content."
)


def build(cfg: AgentConfig) -> Tool:
    # Load the embedding model once at session startup, shared across all calls.
    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer(
        cfg.embed_model,
        device=cfg.embed_device,
        truncate_dim=cfg.embed_dim if cfg.embed_dim else None,
    )

    def search_knowledge(
        query: str,
        source_type: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """Perform a semantic similarity search over the RAG index of stored
        transcript chunks and SiYuan note blocks. Use this when the user
        asks a question about past conversations or notes that may span
        multiple sessions, or when you need context beyond a single transcript.
        Returns the most relevant text chunks with source, speaker, timestamps,
        and similarity scores.

        Args:
            query: Natural language search query. The query is embedded and matched
                semantically against all indexed transcript chunks and SiYuan page blocks.
            source_type: Filter by source type: 'transcript' for conversation chunks,
                'siyuan' for note page chunks, or omit to search both.
            session_id: Restrict the search to a specific session (transcript chunks only).
                Omit to search across all sessions.
            top_k: Number of top results to return (1-20).
        """
        from sqlalchemy import text as sa_text

        from pawn_agent.utils.db import make_db_session

        top_k = max(1, min(20, top_k))

        query_vec = _model.encode(query, show_progress_bar=False).tolist()
        query_vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

        where_parts: list[str] = []
        bind: dict = {"query_vec": query_vec_str, "top_k": top_k}

        if source_type:
            where_parts.append("rs.source_type = :source_type")
            bind["source_type"] = source_type

        if session_id:
            where_parts.append("rs.external_id = :session_id")
            bind["session_id"] = session_id

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        sql = sa_text(
            f"""
            SELECT
                tc.id,
                rs.source_type,
                rs.external_id,
                rs.display_name,
                tc.speaker_name,
                tc.start_time,
                tc.end_time,
                tc.text,
                tc.metadata,
                tc.embedding <=> CAST(:query_vec AS vector) AS distance
            FROM text_chunks tc
            JOIN rag_sources rs ON rs.id = tc.source_id
            {where_sql}
            ORDER BY distance ASC
            LIMIT :top_k
            """
        )

        db = make_db_session(cfg.db_dsn)
        try:
            rows = db.execute(sql, bind).fetchall()
        finally:
            db.close()

        if not rows:
            return "No relevant chunks found in the RAG index."

        parts: list[str] = []
        for i, row in enumerate(rows, 1):
            if row.source_type == "transcript":
                source_label = f"Session: {row.external_id}"
                if row.display_name and row.display_name != row.external_id:
                    source_label += f" ({row.display_name})"
            else:
                source_label = f"SiYuan: {row.display_name or row.external_id}"

            similarity = 1.0 - float(row.distance)
            chunk_meta = row.metadata or {}
            is_summary = chunk_meta.get("chunk_type") == "session_summary"

            if is_summary:
                tags = chunk_meta.get("tags") or []
                tag_str = f"  [tags: {', '.join(tags)}]" if tags else ""
                header = (
                    f"[{i}] Session Overview — {source_label}{tag_str}  "
                    f"(similarity: {similarity:.3f})"
                )
            else:
                speaker_part = f"Speaker: {row.speaker_name}  " if row.speaker_name else ""
                time_part = ""
                if row.start_time is not None and row.end_time is not None:
                    s_m, s_s = divmod(row.start_time, 60)
                    e_m, e_s = divmod(row.end_time, 60)
                    time_part = f"[{int(s_m):02d}:{s_s:05.2f} → {int(e_m):02d}:{e_s:05.2f}]  "
                header = f"[{i}] {source_label}  {speaker_part}{time_part}(similarity: {similarity:.3f})"

            parts.append(f"{header}\n{(row.text or '').strip()}")

        return "\n\n---\n\n".join(parts)

    return Tool(search_knowledge)
