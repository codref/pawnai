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
        content: str,
        title: Optional[str] = None,
        path: Optional[str] = None,
    ) -> str:
        """Save already-generated content to SiYuan Notes as a new child document
        under the session page (conversations/{date}/{session_id}/{title}).

        Use this ONLY when you already have the final content as a string (e.g.
        from analyze_conversation). If you still need to run an analysis first,
        use analyze_custom with save=true instead.

        Each call creates a distinct document nested under the session page, so
        multiple focused analyses accumulate rather than overwriting each other.

        IMPORTANT — content encoding: the content string is transmitted as JSON.
        Avoid raw LaTeX-style backslash sequences such as \\( \\) \\[ \\] \\{
        as they are invalid JSON escape codes. Use plain Unicode, dollar-sign
        math notation ($...$), or spell out mathematical expressions in words.

        Args:
            session_id: Session identifier; determines the parent page in the tree.
            content: Full Markdown content to store.
            title: Document title. Inferred from the first ``# Heading`` in
                *content* when omitted.
            path: Optional explicit SiYuan path override. Bypasses the template
                entirely — use sparingly.
        """
        try:
            return do_save_to_siyuan(cfg, session_id, title, content, path)
        except Exception as exc:
            return f"Error saving to SiYuan: {exc}"

    return Tool(save_to_siyuan)
