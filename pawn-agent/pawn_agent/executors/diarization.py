"""Tool executor: speaker diarization via pawnai.

Registered as: ``diarization.run``
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pawn_agent.config import AgentConfig


async def run(params: Dict[str, Any], cfg: "AgentConfig") -> Dict[str, Any]:
    """Run speaker diarization on an audio file.

    Args:
        params: Must contain ``local_path``. Optional: ``threshold``,
                ``store_new``, ``db_dsn``, ``output``.
        cfg:    Agent configuration.

    Returns:
        ``{"segments": list, "session_id": str | None, "output_path": str | None}``
    """

    def _do_diarize() -> Dict[str, Any]:
        from pawnai.core.diarization import DiarizationEngine  # lazy import

        engine = DiarizationEngine()

        audio_path: str = params["local_path"]
        threshold: float = float(params.get("threshold", 0.7))
        store_new: bool = bool(params.get("store_new", True))
        db_dsn: Any = params.get("db_dsn")
        output: Any = params.get("output")

        result = engine.diarize(
            audio_path,
            threshold=threshold,
            store_new=store_new,
            db_dsn=db_dsn,
            output_path=output,
        )

        if isinstance(result, dict):
            return {
                "segments": result.get("segments", []),
                "session_id": result.get("session_id"),
                "output_path": result.get("output_path"),
            }
        return {"segments": [], "session_id": None, "output_path": None}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_diarize)
