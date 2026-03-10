"""Tool executor: speech-to-text transcription via pawnai.

Registered as: ``transcription.run``
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pawn_agent.config import AgentConfig


async def run(params: Dict[str, Any], cfg: "AgentConfig") -> Dict[str, Any]:
    """Transcribe audio using pawnai's TranscriptionEngine.

    Args:
        params: Must contain ``local_path``. Optional: ``timestamps`` (bool),
                ``device``, ``backend``, ``chunk_duration``, ``session``,
                ``db_dsn``.
        cfg:    Agent configuration.

    Returns:
        ``{"transcript": str, "session_id": str | None, "output_path": str | None}``
    """

    def _do_transcribe() -> Dict[str, Any]:
        from pawnai.core.transcription import TranscriptionEngine  # lazy import

        engine = TranscriptionEngine(
            device=params.get("device", "cuda"),
            backend=params.get("backend", "nemo"),
        )

        audio_path: str = params["local_path"]
        timestamps: bool = bool(params.get("timestamps", True))
        session: Any = params.get("session")
        db_dsn: Any = params.get("db_dsn")
        output: Any = params.get("output")
        chunk_duration: Any = params.get("chunk_duration")

        result = engine.transcribe(
            audio_path,
            timestamps=timestamps,
            output_path=output,
            session_id=session,
            db_dsn=db_dsn,
            chunk_duration=chunk_duration,
        )

        if isinstance(result, dict):
            return {
                "transcript": result.get("text", ""),
                "session_id": result.get("session_id"),
                "output_path": result.get("output_path"),
            }
        # Plain string result
        return {"transcript": str(result), "session_id": None, "output_path": None}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_transcribe)
