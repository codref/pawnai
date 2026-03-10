"""Tool executor: combined transcription + diarization pipeline via pawnai.

Registered as: ``combined.run``
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pawn_agent.config import AgentConfig


async def run(params: Dict[str, Any], cfg: "AgentConfig") -> Dict[str, Any]:
    """Run the combined transcribe-diarize pipeline.

    Args:
        params: Must contain ``local_path``. Optional: ``threshold``,
                ``store_new``, ``device``, ``backend``, ``chunk_duration``,
                ``cross_file_threshold``, ``no_timestamps``, ``session``,
                ``db_dsn``, ``output``, ``verbose``.
        cfg:    Agent configuration.

    Returns:
        ``{"transcript": str, "session_id": str | None, "segments": list,
           "output_path": str | None}``
    """

    def _do_combined() -> Dict[str, Any]:
        from pawnai.core.combined import transcribe_with_diarization  # lazy import

        audio_paths = [params["local_path"]]

        result = transcribe_with_diarization(
            audio_paths,
            threshold=float(params.get("threshold", 0.7)),
            store_new=bool(params.get("store_new", True)),
            device=params.get("device", "cuda"),
            chunk_duration=params.get("chunk_duration"),
            cross_file_threshold=float(params.get("cross_file_threshold", 0.85)),
            no_timestamps=bool(params.get("no_timestamps", False)),
            session_id=params.get("session"),
            db_dsn=params.get("db_dsn"),
            output_path=params.get("output"),
            verbose=bool(params.get("verbose", False)),
            backend=params.get("backend", "nemo"),
        )

        if isinstance(result, dict):
            return {
                "transcript": result.get("text", ""),
                "session_id": result.get("session_id"),
                "segments": result.get("segments", []),
                "output_path": result.get("output_path"),
            }
        return {
            "transcript": str(result),
            "session_id": None,
            "segments": [],
            "output_path": None,
        }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_combined)
