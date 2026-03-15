"""Tool: analyze_custom — free-form analysis, optionally saved to SiYuan."""

from __future__ import annotations

from typing import Optional  # used for title field

from copilot import CopilotClient, MessageOptions, PermissionHandler, Tool, define_tool
from pydantic import BaseModel, Field

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.transcript import fetch_transcript
from pawn_agent.utils.siyuan import do_save_to_siyuan

NAME = "analyze_custom"
DESCRIPTION = (
    "Perform a free-form analysis of a transcript. "
    "Optionally saves the result to SiYuan when the user asks to."
)


class AnalyzeCustomParams(BaseModel):
    session_id: str = Field(description="Unique session identifier stored in the database.")
    instruction: str = Field(
        description="The analysis task to perform on the transcript, e.g. "
                    "'extract epics and user stories', 'list action items', "
                    "'identify decisions made'."
    )
    title: Optional[str] = Field(
        default=None,
        description="Document title used as the SiYuan page heading. Required when save=true.",
    )
    save: bool = Field(
        default=False,
        description="Set to true whenever the user asks to save, store, or put the result "
                    "somewhere (e.g. 'store it under Notes', 'save it in SiYuan', "
                    "'put it in a note'). Set to false to just return the analysis.",
    )


def build(cfg: AgentConfig, client: CopilotClient) -> Tool:
    @define_tool(
        description=(
            "Perform a custom analysis of a conversation session's transcript. "
            "Use this for ANY request that is not the standard structured analysis "
            "(title/summary/topics/sentiment), such as extracting epics, user stories, "
            "action items, decisions, risks, or any other bespoke task. "
            "The analysis is grounded EXCLUSIVELY on the stored transcript. "
            "Set save=true and a title whenever the user's request includes any intent to "
            "save, store, or write the result (e.g. 'store it under Notes', 'save it in "
            "SiYuan', 'put it in a note'). The document path is resolved automatically."
        )
    )
    async def analyze_custom(params: AnalyzeCustomParams) -> str:
        try:
            transcript = fetch_transcript(cfg, params.session_id)
            if transcript.startswith("Error") or transcript.startswith("No transcript"):
                return transcript

            prompt = (
                "You are a helpful assistant. Your task is to analyse the conversation "
                "transcript below and nothing else. Do NOT use any knowledge outside of "
                "this transcript.\n"
                "\n"
                f"Task: {params.instruction}\n"
                "\n"
                "---\n"
                "TRANSCRIPT:\n"
                f"{transcript}\n"
                "---\n"
                "Respond only with the result of the task above. "
                "Do not repeat the transcript or the instructions."
            )

            analysis_session = await client.create_session(
                {
                    "model": cfg.model,
                    "on_permission_request": PermissionHandler.approve_all,
                    "system_message": {
                        "mode": "replace",
                        "content": (
                            "You are a helpful assistant. "
                            "Analyse only the transcript provided in the user message. "
                            "Do not use any knowledge about the user's codebase, tools, "
                            "or environment."
                        ),
                    },
                }
            )
            try:
                response = await analysis_session.send_and_wait(
                    MessageOptions(prompt=prompt),
                    timeout=120,
                )
                content = response.data.content if (response is not None and response.data.content is not None) else ""
            finally:
                await analysis_session.disconnect()

            if params.save:
                title = params.title or params.instruction[:60]
                doc_id = do_save_to_siyuan(cfg, params.session_id, title, content, path=None)
                return f"Saved to SiYuan (doc id: {doc_id})"

            return content
        except Exception as exc:
            return f"Error performing custom analysis: {exc}"

    return analyze_custom  # type: ignore[return-value]
