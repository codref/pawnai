"""Tool: analyze_custom — run a bespoke free-form analysis on a session."""

from __future__ import annotations

from typing import Optional

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig

NAME = "analyze_custom"
DESCRIPTION = (
    "Run a bespoke free-form analysis on a session transcript using a custom instruction. "
    "Use this for extensive reports, action-point analysis, or any task that should go "
    "beyond the standard structured summary. This tool already loads the transcript itself, "
    "so do not call query_conversation first. Set save=true only when a save receipt is enough; "
    "if the user also needs the full analysis in chat, run analyze_custom first and then save_to_siyuan."
)

_SYSTEM_PROMPT = (
    "You are an expert conversation analyst. "
    "Analyse only the transcript provided in the user message. "
    "Follow the user's requested format and depth. "
    "Return plain Markdown only."
)


def build(cfg: AgentConfig) -> Tool:
    async def analyze_custom(
        session_id: str,
        instruction: str,
        save: bool = False,
        title: Optional[str] = None,
        path: Optional[str] = None,
    ) -> str:
        """Run a bespoke analysis on a conversation session transcript.

        Use this when the user wants a custom report or a deeper analysis than
        the standard ``analyze_summary`` format can provide. This tool fetches
        the full transcript, applies the supplied instruction, and optionally
        saves the resulting Markdown to SiYuan.

        This tool already fetches the transcript internally. Do not call
        ``query_conversation`` first unless the user explicitly wants the raw
        transcript text in the reply.

        Args:
            session_id: Unique session identifier stored in the database.
            instruction: Free-form analysis instruction, such as "Create an
                extensive analysis of the action points discussed and explain
                their implications".
            save: Set to true to also save the generated Markdown to SiYuan.
                Use ``save=false`` when the user needs the full analysis in
                chat, then call ``save_to_siyuan`` with that generated content.
            title: Optional SiYuan document title override when ``save=true``.
            path: Optional explicit SiYuan path override when ``save=true``.
        """
        from pawn_agent.core.llm_sub import run as llm_run
        from pawn_agent.utils.siyuan import do_save_to_siyuan
        from pawn_agent.utils.transcript import fetch_transcript

        try:
            transcript = fetch_transcript(cfg, session_id)
            if transcript.startswith("Error") or transcript.startswith("No transcript"):
                return transcript

            prompt = (
                f"Task:\n{instruction.strip()}\n\n"
                "Requirements:\n"
                "- Base the analysis only on the transcript.\n"
                "- Include concrete evidence and detail where useful.\n"
                "- If action items, risks, decisions, blockers, or follow-ups appear, "
                "surface them explicitly.\n"
                "- Return the final answer as polished Markdown.\n\n"
                f"---\nTRANSCRIPT:\n{transcript}\n---"
            )

            content = await llm_run(cfg, prompt, system_prompt=_SYSTEM_PROMPT)

            if save:
                save_result = do_save_to_siyuan(cfg, session_id, title, content, path)
                doc_label = title or session_id
                return (
                    f"Saved custom analysis to SiYuan for session {session_id!r} "
                    f"(title: {doc_label!r}, url: {save_result})."
                )

            return content
        except Exception as exc:
            return f"Error running custom analysis: {exc}"

    return Tool(analyze_custom)
