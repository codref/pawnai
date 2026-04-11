"""Tool: memorize — save a fact to persistent vector memory."""

from __future__ import annotations

from typing import List, Optional

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig


NAME = "memorize"
DESCRIPTION = (
    "Save a fact, preference, or piece of information to persistent vector memory "
    "so it can be recalled in future sessions."
)


def build(cfg: AgentConfig) -> Tool:
    # Load the embedding model once at session startup, shared across all calls.
    from pawn_agent.utils.vectorize import load_embedding_model

    load_embedding_model(
        cfg.embed_model,
        cfg.embed_device,
        truncate_dim=cfg.embed_dim if cfg.embed_dim else None,
        local_files_only=cfg.embed_local_files_only,
    )

    def memorize(
        fact: str,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Save a fact or preference to persistent memory so it can be recalled later.

        Call this when the user says "remember that...", "memorize...",
        "keep in mind that...", "note that...", or gives any similar instruction
        to retain information across sessions.

        Write the fact as a self-contained sentence so it is meaningful without
        surrounding context. For example, prefer "The user's name is Davide" over
        just "Davide", or "The user prefers dark mode in all UIs" over "dark mode".

        Each call creates an independent memory entry with a unique ID that can
        be referenced for deletion later.

        Args:
            fact: The exact fact, preference, or information to remember.
            tags: Optional list of short label strings for categorisation
                (e.g. ["preference", "ui"]).
        """
        from pawn_agent.utils.vectorize import vectorize_memory

        try:
            memory_id = vectorize_memory(
                fact=fact,
                db_dsn=cfg.db_dsn,
                embed_model=cfg.embed_model,
                embed_device=cfg.embed_device,
                embed_dim=cfg.embed_dim if cfg.embed_dim else None,
                embed_local_files_only=cfg.embed_local_files_only,
                tags=tags,
            )
        except Exception as exc:
            return f"Error saving memory: {exc}"

        from datetime import datetime, timezone

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        tag_str = f" [tags: {', '.join(tags)}]" if tags else ""
        preview = fact[:80] + ("..." if len(fact) > 80 else "")
        return f"Memorized on {date_str} (id: {memory_id}){tag_str}: {preview}"

    return Tool(memorize)
