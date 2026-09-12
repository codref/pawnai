"""In-process pool of sallm chat sessions for API / queue / scheduler.

Owns: conversation_id → SallmChatSession mapping, per-session locks, reset,
and idle eviction. Same façade shape as the old LangGraph registry so
``run_agent_turn`` and callers stay small.

Does not own: PostgreSQL langgraph_session_state (unused; sallm SQLite is
the source of truth for chat memory).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pawn_agent.core.sallm_factory import build_optional_tracer
from pawn_agent.core.sallm_session import SallmChatSession

logger = logging.getLogger(__name__)


class SallmSessionRegistry:
    """Map ``session_id`` (conversation key) → :class:`SallmChatSession`.

    Concurrency:
    - ``_registry_lock`` serialises creation.
    - Per-session ``asyncio.Lock`` prevents overlapping turns that would
      interleave durable SQLite writes for the same conversation.
    - Different sessions run concurrently.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SallmChatSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def _build_session(self, session_id: str, cfg: Any) -> SallmChatSession:
        trace = build_optional_tracer(session_id=session_id, cfg=cfg)
        # create() is sync (opens SQLite/Lance); keep the event loop free.
        return await asyncio.to_thread(
            SallmChatSession.create,
            cfg,
            conversation_id=session_id,
            trace=trace,
        )

    async def get_or_create(self, session_id: str, cfg: Any, db_dsn: str = "") -> SallmChatSession:
        """Return a cached session, creating it if needed.

        ``db_dsn`` is accepted for call-site compatibility with the old
        registry signature; durable chat state lives in sallm files, not PG.
        """
        del db_dsn  # unused — kept for API compatibility
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.apply_config(cfg)
            return session
        async with self._registry_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = await self._build_session(session_id, cfg)
            else:
                self._sessions[session_id].apply_config(cfg)
        return self._sessions[session_id]

    async def handle_turn(
        self,
        session_id: str,
        text: str,
        cfg: Any,
        db_dsn: str = "",
        **_kwargs: Any,
    ) -> str:
        """Process one user turn under the per-session lock.

        Extra kwargs (e.g. legacy ``graph_recorder``) are ignored so callers
        can be updated gradually.
        """
        session = await self.get_or_create(session_id, cfg, db_dsn)
        async with self._session_lock(session_id):
            return await session.handle_user_input(text)

    async def reset(self, session_id: str, db_dsn: str = "") -> None:
        """Clear durable memory and drop the in-memory session."""
        del db_dsn
        async with self._registry_lock:
            existing = self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)
        if existing is not None:
            await existing.reset()

    async def stats(self, session_id: str, cfg: Any, db_dsn: str = "") -> str:
        """Format context-aware stats for *session_id* (creates session if needed)."""
        session = await self.get_or_create(session_id, cfg, db_dsn)
        async with self._session_lock(session_id):
            return await asyncio.to_thread(session.format_stats)

    def evict_all(self) -> int:
        """Drop in-memory sessions (idle timeout). Durable files remain.

        Next access rebuilds an Agent pointed at the same state_path/session_id.
        """
        count = len(self._sessions)
        self._sessions.clear()
        self._locks.clear()
        if count:
            logger.info("Sallm registry: evicted %d session(s)", count)
        return count
