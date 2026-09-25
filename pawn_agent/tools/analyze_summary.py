"""Structured session analysis (``session_analyze`` CliTool)."""

from __future__ import annotations

from typing import Optional

from pawn_agent.tools.save_to_vault import save_analysis_to_vault_impl
from pawn_agent.utils.analysis import run_analysis
from pawn_agent.utils.config import AgentConfig


async def analyze_summary_impl(
    cfg: AgentConfig,
    session_id: str,
    *,
    save: bool = False,
    title: Optional[str] = None,
) -> str:
    """Run the standard structured analysis and optionally note vault export."""
    try:
        content = await run_analysis(cfg, session_id)

        if save:
            vault_msg = save_analysis_to_vault_impl(cfg, session_id, title=title)
            return f"{vault_msg}\n\n{content}"

        return content
    except Exception as exc:
        return f"Error performing structured analysis: {exc}"
