"""Tool: analyze_custom — free-form analysis, optionally saved to SiYuan."""

from __future__ import annotations

from typing import Optional

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.transcript import fetch_transcript
from pawn_agent.utils.siyuan import do_save_to_siyuan

NAME = "analyze_custom"
DESCRIPTION = (
    "Perform a free-form analysis of a transcript. "
    "Optionally saves the result to SiYuan when the user asks to."
)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Analyse only the transcript provided in the user message. "
    "Do not use any knowledge about the user's codebase, tools, or environment."
)


def build(cfg: AgentConfig) -> Tool:
    async def analyze_custom(
        session_id: str,
        instruction: str,
        save: bool = False,
        title: Optional[str] = None,
    ) -> str:
        """Perform a custom analysis of a conversation session's transcript.

        Use this for ANY request that is not the standard structured analysis
        (title/summary/topics/sentiment), such as extracting epics, user stories,
        action items, decisions, risks, or any other bespoke task.
        The analysis is grounded EXCLUSIVELY on the stored transcript.
        Set save=true and a title whenever the user's request includes any intent to
        save, store, or write the result (e.g. 'store it under Notes', 'save it in
        SiYuan', 'put it in a note'). The document path is resolved automatically.

        Args:
            session_id: Unique session identifier stored in the database.
            instruction: The analysis task to perform on the transcript, e.g.
                'extract epics and user stories', 'list action items',
                'identify decisions made'.
            save: Set to true whenever the user asks to save, store, or put the result
                somewhere. Set to false to just return the analysis.
            title: Document title used as the SiYuan page heading. Required when save=true.
        """
        from pawn_agent.core.copilot_sub import run as copilot_run

        try:
            transcript = fetch_transcript(cfg, session_id)
            if transcript.startswith("Error") or transcript.startswith("No transcript"):
                return transcript

            prompt = (
                "You are a helpful assistant. Your task is to analyse the conversation "
                "transcript below and nothing else. Do NOT use any knowledge outside of "
                "this transcript.\n"
                "\n"
                f"Task: {instruction}\n"
                "\n"
                "---\n"
                "TRANSCRIPT:\n"
                f"{transcript}\n"
                "---\n"
                "Respond only with the result of the task above. "
                "Do not repeat the transcript or the instructions."
            )

            content = await copilot_run(cfg, prompt, system_prompt=_SYSTEM_PROMPT)

            if save:
                doc_title = title or instruction[:60]
                doc_id = do_save_to_siyuan(cfg, session_id, doc_title, content, path=None)
                return f"Saved to SiYuan (doc id: {doc_id})"

            return content
        except Exception as exc:
            return f"Error performing custom analysis: {exc}"

    return Tool(analyze_custom)
