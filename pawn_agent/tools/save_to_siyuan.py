"""Tool: save_to_siyuan — save already-generated content to SiYuan Notes."""

from __future__ import annotations

from typing import Optional

from copilot import CopilotClient, Tool, define_tool
from pydantic import BaseModel, Field

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.siyuan import do_save_to_siyuan


NAME = "save_to_siyuan"
DESCRIPTION = "Save already-generated Markdown content to SiYuan Notes as a new document."


class SaveToSiyuanParams(BaseModel):
    session_id: str = Field(description="Session identifier used to resolve the document path.")
    title: str = Field(description="Document title used in the path and as the page heading.")
    content: str = Field(description="Full Markdown content to store (e.g. from analyze_conversation).")
    path: Optional[str] = Field(
        default=None,
        description="Optional explicit SiYuan path (e.g. '/Notes/2026-03-13/epics'). "
                    "When provided it overrides the configured path template so that each "
                    "call creates a distinct document. Use this whenever the user asks to "
                    "save to a *new* note or a specific location.",
    )


def build(cfg: AgentConfig, client: CopilotClient) -> Tool:
    @define_tool(
        description=(
            "Save already-generated content to SiYuan Notes as a structured document. "
            "Use this ONLY when you already have the final content as a string (e.g. "
            "from analyze_conversation). If you still need to run an analysis first, "
            "use analyze_and_save_custom instead so the content never passes through "
            "this session. "
            "Pass 'path' to specify an explicit document path so a new document is "
            "created each time; otherwise the configured path template is used."
        )
    )
    def save_to_siyuan(params: SaveToSiyuanParams) -> str:
        try:
            return do_save_to_siyuan(cfg, params.session_id, params.title, params.content, params.path)
        except Exception as exc:
            return f"Error saving to SiYuan: {exc}"

    return save_to_siyuan  # type: ignore[return-value]
