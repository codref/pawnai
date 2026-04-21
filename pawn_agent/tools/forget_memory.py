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

    async def forget_memory(memory_id: str) -> str:
        """Delete a previously stored memory entry permanently.

        Call this when the user says "forget that", "don't remember that",
        "remove that memory", or asks to delete or update a specific fact.

        The typical flow is:
          1. Call recall_memory to find the relevant memory and its memory_id.
          2. If multiple entries match, confirm with the user which one to delete.
          3. Call forget_memory(memory_id) to permanently remove it.

        To update a fact: forget the old one and memorize the new one.

        Args:
            memory_id: The key of the memory to delete, as returned by
                memorize or shown in recall_memory results.
        """
        from pawn_agent.core.store import NS_MEMORIES, get_store  # noqa: PLC0415

        store = await get_store(cfg)
        existing = await store.aget(NS_MEMORIES, memory_id)
        if existing is None:
            return f"No memory found with id '{memory_id}'."

        label = (existing.value or {}).get("text", memory_id)[:80]
        await store.adelete(NS_MEMORIES, memory_id)
        return f"Forgotten (id: {memory_id}): {label}"

    return Tool(forget_memory)

