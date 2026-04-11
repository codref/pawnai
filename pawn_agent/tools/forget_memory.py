"""Tool: forget_memory — delete a specific memory entry by ID."""

from __future__ import annotations

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig


NAME = "forget_memory"
DESCRIPTION = (
    "Delete a specific memory entry by its ID. "
    "Use recall_memory first to find the memory_id, then call this to remove it."
)


def build(cfg: AgentConfig) -> Tool:

    def forget_memory(memory_id: str) -> str:
        """Delete a previously stored memory entry permanently.

        Call this when the user says "forget that", "don't remember that",
        "remove that memory", or asks to delete or update a specific fact.

        The typical flow is:
          1. Call recall_memory to find the relevant memory and its memory_id.
          2. If multiple entries match, confirm with the user which one to delete.
          3. Call forget_memory(memory_id) to permanently remove it.

        To update a fact: forget the old one and memorize the new one.

        Args:
            memory_id: The UUID of the memory to delete, as returned by
                memorize or shown in recall_memory results.
        """
        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import Session

        from pawn_agent.utils.db import RagSource, TextChunk

        src_id = f"memory:{memory_id}"
        engine = create_engine(cfg.db_dsn)

        with Session(engine) as db:
            source = db.get(RagSource, src_id)
            if source is None:
                return f"No memory found with id '{memory_id}'."

            label = source.display_name or memory_id

            # Delete TextChunk rows first — no FK cascade between the tables.
            for ch in db.scalars(
                select(TextChunk).where(TextChunk.source_id == src_id)
            ).all():
                db.delete(ch)

            db.delete(source)
            db.commit()

        return f"Forgotten (id: {memory_id}): {label}"

    return Tool(forget_memory)
