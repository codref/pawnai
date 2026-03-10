"""Tool executor: push analysis to SiYuan Notes via pawnai.

Registered as: ``siyuan.sync``
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pawn_agent.config import AgentConfig


async def sync(params: Dict[str, Any], cfg: "AgentConfig") -> Dict[str, Any]:
    """Push a session's analysis to SiYuan Notes.

    Args:
        params: Must contain ``session_id`` or ``analysis`` (markdown text).
                Optional: ``db_dsn``.
        cfg:    Agent configuration (provides SiYuan URL + token from
                the ``siyuan:`` section of ``.pawnai.yml``).

    Returns:
        ``{"siyuan_block_id": str | None, "synced": bool}``
    """

    def _do_sync() -> Dict[str, Any]:
        from pawnai.core.siyuan import SiYuanClient  # lazy import

        # SiYuan config lives in the underlying pawnai config file
        # (same YAML, reused by AgentConfig._raw)
        siyuan_cfg: Dict[str, Any] = cfg.get("siyuan") or {}
        if not siyuan_cfg:
            raise RuntimeError(
                "No 'siyuan:' section in .pawnai.yml — cannot sync to SiYuan Notes"
            )

        client = SiYuanClient(
            url=siyuan_cfg.get("url", "http://127.0.0.1:6806"),
            token=siyuan_cfg.get("token", ""),
            notebook=siyuan_cfg.get("notebook", ""),
        )

        session_id: Any = params.get("session_id") or params.get("session")
        analysis: Any = params.get("analysis")
        db_dsn: Any = params.get("db_dsn")

        block_id = client.push_session(
            session_id=session_id,
            analysis_text=analysis,
            db_dsn=db_dsn,
        )

        return {"siyuan_block_id": block_id, "synced": True}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_sync)
