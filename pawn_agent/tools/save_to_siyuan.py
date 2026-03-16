"""Tool: save_to_siyuan — save already-generated content to SiYuan Notes."""

from __future__ import annotations

from typing import Optional

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.siyuan import do_save_to_siyuan


NAME = "save_to_siyuan"
DESCRIPTION = "Save already-generated Markdown content to SiYuan Notes as a new document."


def build(cfg: AgentConfig) -> Tool:
    def save_to_siyuan(
        session_id: str,
        title: str,
        content: str,
        path: Optional[str] = None,
    ) -> str:
        """Save already-generated content to SiYuan Notes as a structured document.

        Use this ONLY when you already have the final content as a string (e.g.
        from analyze_conversation). If you still need to run an analysis first,
        use analyze_custom with save=true instead.
        Pass 'path' to specify an explicit document path so a new document is
        created each time; otherwise the configured path template is used.

        Args:
            session_id: Session identifier used to resolve the document path.
            title: Document title used in the path and as the page heading.
            content: Full Markdown content to store.
            path: Optional explicit SiYuan path (e.g. '/Notes/2026-03-13/epics').
                When provided it overrides the configured path template so that each
                call creates a distinct document.
        """
        try:
            return do_save_to_siyuan(cfg, session_id, title, content, path)
        except Exception as exc:
            return f"Error saving to SiYuan: {exc}"

    return Tool(save_to_siyuan)
