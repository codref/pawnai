"""Tool executor: LLM analysis of transcripts via pawnai AnalysisEngine.

Registered as: ``analysis.run``
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pawn_agent.config import AgentConfig


async def run(params: Dict[str, Any], cfg: "AgentConfig") -> Dict[str, Any]:
    """Analyse a transcript (from text, file, or DB session) with Copilot.

    Args:
        params: One of:
            - ``transcript``: plain-text transcript string (from a prior step)
            - ``input_path``: path to a transcript file or audio file
            - ``session``: session ID to load from the DB
          Optional: ``analysis_type`` (``"summary"`` | ``"topics"`` | ``"sentiment"``
          | ``"tags"`` | ``"knowledge_graph"`` | ``"all"``), ``db_dsn``, ``model``.
        cfg:    Agent configuration.

    Returns:
        ``{"analysis": str, "session_id": str | None}``
    """

    def _do_analyze() -> Dict[str, Any]:
        from pawnai.core.analysis import AnalysisEngine  # lazy import

        model: str = params.get("model", cfg.llm_model)
        engine = AnalysisEngine(model=model)

        transcript: Any = params.get("transcript")
        input_path: Any = params.get("input_path")
        session: Any = params.get("session")
        analysis_type: str = params.get("analysis_type", "all")
        db_dsn: Any = params.get("db_dsn")

        if transcript:
            output = engine.analyze(transcript)
        elif input_path:
            output = engine.analyze_file(
                input_path,
                analysis_type=analysis_type,
                db_dsn=db_dsn,
            )
        elif session:
            output = engine.analyze_session(
                session_id=session,
                analysis_type=analysis_type,
                db_dsn=db_dsn,
            )
        else:
            raise ValueError(
                "analysis.run: one of 'transcript', 'input_path', or 'session' is required"
            )

        return {"analysis": output, "session_id": session}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_analyze)
