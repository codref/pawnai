"""Tool: query_conversation — fetch and return a session transcript."""

from __future__ import annotations

from copilot import CopilotClient, Tool, define_tool
from pydantic import BaseModel, Field

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.transcript import fetch_transcript


NAME = "query_conversation"
DESCRIPTION = "Fetch and return the full transcript for a session from the database."


class QueryConversationParams(BaseModel):
    session_id: str = Field(description="Unique session identifier stored in the database.")


def build(cfg: AgentConfig, client: CopilotClient) -> Tool:
    @define_tool(
        description=(
            "Retrieve and return the full transcript for a session from the database. "
            "Use this first when the user asks about the content of a conversation "
            "or wants to analyse a specific session."
        )
    )
    def query_conversation(params: QueryConversationParams) -> str:
        return fetch_transcript(cfg, params.session_id)

    return query_conversation  # type: ignore[return-value]
