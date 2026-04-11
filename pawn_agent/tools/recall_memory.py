"""Tool: recall_memory — retrieve facts from persistent vector memory."""

from __future__ import annotations

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig


NAME = "recall_memory"
DESCRIPTION = (
    "Search persistent vector memory for facts, preferences, or information "
    "stored in previous sessions."
)


def build(cfg: AgentConfig) -> Tool:
    # Load the embedding model once at session startup, shared across all calls.
    from pawn_agent.utils.vectorize import load_embedding_model

    _model = load_embedding_model(
        cfg.embed_model,
        cfg.embed_device,
        truncate_dim=cfg.embed_dim if cfg.embed_dim else None,
        local_files_only=cfg.embed_local_files_only,
    )

    def recall_memory(
        query: str,
        top_k: int = 5,
    ) -> str:
        """Search persistent memory for facts and preferences stored in previous sessions.

        Call this proactively whenever the user's question might relate to something
        they previously asked you to remember, or when you need personal context to
        give a better answer. Typical triggers include:

        - Any question about the user's preferences, habits, settings, or identity
        - References to past decisions or context ("as we discussed", "like last time",
          "you know my preference", "as I told you")
        - Personalisation requests ("recommend something for me", "set it to my usual")
        - Questions where a stored fact would change or improve the answer
        - After the user says "do you remember..." or "didn't I tell you..."

        Returns memory entries with their IDs and the date they were saved. Pass a
        memory_id to forget_memory if the user asks to delete or update a specific entry.

        Args:
            query: Natural language description of what you are looking for. Phrase it
                as a statement of what you hope to find, e.g. "user's preferred
                programming language" or "standing meeting time".
            top_k: Maximum number of memories to return (1–10, default 5).
        """
        from sqlalchemy import text as sa_text

        from pawn_agent.utils.db import make_db_session

        top_k = max(1, min(10, top_k))

        query_vec = _model.encode(query, show_progress_bar=False).tolist()
        query_vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

        sql = sa_text(
            """
            SELECT
                rs.external_id   AS memory_id,
                rs.display_name,
                rs.created_at,
                rs.metadata      AS source_meta,
                tc.text,
                tc.embedding <=> CAST(:query_vec AS vector) AS distance
            FROM text_chunks tc
            JOIN rag_sources rs ON rs.id = tc.source_id
            WHERE rs.source_type = 'memory'
            ORDER BY distance ASC
            LIMIT :top_k
            """
        )

        db = make_db_session(cfg.db_dsn)
        try:
            rows = db.execute(sql, {"query_vec": query_vec_str, "top_k": top_k}).fetchall()
        finally:
            db.close()

        if not rows:
            return "No memories stored yet."

        parts: list[str] = []
        for i, row in enumerate(rows, 1):
            similarity = 1.0 - float(row.distance)
            meta = row.source_meta or {}
            tags = meta.get("tags") or []
            tag_str = f"  [tags: {', '.join(tags)}]" if tags else ""

            if row.created_at is not None:
                try:
                    from datetime import timezone as _tz
                    ts = row.created_at
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=_tz.utc)
                    date_str = ts.strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    date_str = str(row.created_at)
            else:
                date_str = "unknown date"

            header = (
                f"[{i}] memory_id={row.memory_id}  saved={date_str}"
                f"{tag_str}  (similarity: {similarity:.3f})"
            )
            parts.append(f"{header}\n{(row.text or '').strip()}")

        return "\n\n---\n\n".join(parts)

    return Tool(recall_memory)
