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
    async def recall_memory(
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
        from pawn_agent.core.store import NS_MEMORIES, get_store  # noqa: PLC0415

        top_k = max(1, min(10, top_k))
        store = await get_store(cfg)
        results = await store.asearch(NS_MEMORIES, query=query, limit=top_k)

        if not results:
            return "No memories stored yet."

        parts: list[str] = []
        for i, item in enumerate(results, 1):
            value = item.value or {}
            text = value.get("text", "")
            tags = value.get("tags") or []
            saved_at = value.get("saved_at", "unknown date")
            similarity = item.score if item.score is not None else 0.0
            tag_str = f"  [tags: {', '.join(tags)}]" if tags else ""
            header = (
                f"[{i}] memory_id={item.key}  saved={saved_at}"
                f"{tag_str}  (similarity: {similarity:.3f})"
            )
            parts.append(f"{header}\n{text.strip()}")

        return "\n\n---\n\n".join(parts)

    return Tool(recall_memory)

