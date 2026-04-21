"""LangGraph session registry for the HTTP API server.

Manages a pool of :class:`LangGraphChatSession` instances keyed by
``session_id``.  Sessions are lazily created on first access.  Their
persistent state (durable_facts, artifacts, recent_messages) is loaded from
PostgreSQL on creation and saved back after every turn, so they survive
server restarts.

Transient per-turn fields (incoming_prompt, route_kind, action_plan, …)
are intentionally NOT persisted.

Usage::

    registry = LangGraphSessionRegistry()
    reply = await registry.handle_turn(session_id, prompt, cfg, cfg.db_dsn)
    await registry.reset(session_id, cfg.db_dsn)
    evicted = registry.evict_all()   # called by the idle-timeout handler
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Keys from ARTIFACT_DEFAULTS that are worth persisting.
# tool_output is transient (cleared after every dispatch cycle) — skip it.
_PERSISTENT_ARTIFACT_KEYS = (
    "latest_generated_content",
    "latest_generated_title",
    "session_catalog_output",
    "latest_session_transcript",
)

# Keys from DURABLE_FACT_DEFAULTS.
_DURABLE_FACT_KEYS = ("latest_session_id",)


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers (synchronous — called from async code via run_in_executor or
# directly; each call creates/disposes its own engine to stay thread-safe)
# ──────────────────────────────────────────────────────────────────────────────


def _load_session_state(session_id: str, db_dsn: str) -> Optional[dict]:
    """Return the persisted state dict for *session_id*, or ``None`` if absent."""
    import sqlalchemy as sa

    engine = sa.create_engine(db_dsn)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT durable_facts, artifacts, recent_messages "
                    "FROM langgraph_session_state WHERE session_id = :sid"
                ),
                {"sid": session_id},
            ).fetchone()
    finally:
        engine.dispose()

    if row is None:
        return None
    return {
        "durable_facts": dict(row.durable_facts or {}),
        "artifacts": dict(row.artifacts or {}),
        "recent_messages": list(row.recent_messages or []),
    }


def _save_session_state(session_id: str, state: Any, db_dsn: str) -> None:
    """Upsert the persistent portion of *state* for *session_id*."""
    import sqlalchemy as sa
    from pawn_agent.core.langgraph_state import (  # noqa: PLC0415
        ensure_langgraph_state,
        get_recent_messages,
        get_state_field,
    )

    current = ensure_langgraph_state(state)
    durable_facts = {k: get_state_field(current, k) for k in _DURABLE_FACT_KEYS}
    artifacts = {k: get_state_field(current, k) for k in _PERSISTENT_ARTIFACT_KEYS}
    recent_messages = get_recent_messages(current)

    engine = sa.create_engine(db_dsn)
    try:
        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO langgraph_session_state "
                    "  (session_id, durable_facts, artifacts, recent_messages, updated_at) "
                    "VALUES (:sid, CAST(:df AS jsonb), CAST(:art AS jsonb), CAST(:rm AS jsonb), NOW()) "
                    "ON CONFLICT (session_id) DO UPDATE SET "
                    "  durable_facts    = EXCLUDED.durable_facts, "
                    "  artifacts        = EXCLUDED.artifacts, "
                    "  recent_messages  = EXCLUDED.recent_messages, "
                    "  updated_at       = EXCLUDED.updated_at"
                ),
                {
                    "sid": session_id,
                    "df": json.dumps(durable_facts),
                    "art": json.dumps(artifacts),
                    "rm": json.dumps(recent_messages),
                },
            )
    finally:
        engine.dispose()


def _delete_session_state(session_id: str, db_dsn: str) -> None:
    """Remove the persisted state row for *session_id* (no-op if absent)."""
    import sqlalchemy as sa

    engine = sa.create_engine(db_dsn)
    try:
        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    "DELETE FROM langgraph_session_state WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
    finally:
        engine.dispose()


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────


class LangGraphSessionRegistry:
    """In-process registry mapping ``session_id`` → :class:`LangGraphChatSession`.

    Thread / concurrency safety:
    - ``_registry_lock`` serialises creation of new sessions.
    - Per-session ``asyncio.Lock`` entries in ``_locks`` prevent concurrent
      turns for the *same* session (which would corrupt ``self._state``).
    - Different sessions run fully concurrently.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def _build_session(self, session_id: str, cfg: Any, db_dsn: str) -> Any:
        """Create a new LangGraphChatSession and hydrate it from the DB."""
        from pawn_agent.core.langgraph_chat import LangGraphChatSession  # noqa: PLC0415
        from pawn_agent.core.langgraph_state import (  # noqa: PLC0415
            ensure_langgraph_state,
            set_recent_messages,
            set_state_fields,
        )

        session = await LangGraphChatSession.create(cfg, emit=lambda _: None)

        loop = asyncio.get_running_loop()
        saved = await loop.run_in_executor(
            None, _load_session_state, session_id, db_dsn
        )
        if saved:
            state = set_state_fields(
                dict(session._state),
                **{k: v for k, v in saved["durable_facts"].items() if v is not None},
                **{k: v for k, v in saved["artifacts"].items() if v is not None},
            )
            state = set_recent_messages(state, saved["recent_messages"])
            session._state = ensure_langgraph_state(state)
            logger.debug("Restored LangGraph state for session %r", session_id)

        return session

    # ── public API ────────────────────────────────────────────────────────────

    async def get_or_create(self, session_id: str, cfg: Any, db_dsn: str) -> Any:
        """Return the cached session, creating and hydrating it if needed."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        async with self._registry_lock:
            # Double-check after acquiring the lock
            if session_id not in self._sessions:
                self._sessions[session_id] = await self._build_session(
                    session_id, cfg, db_dsn
                )
        return self._sessions[session_id]

    async def handle_turn(
        self, session_id: str, text: str, cfg: Any, db_dsn: str
    ) -> str:
        """Process one user turn and persist the resulting state.

        Concurrent calls for the same session are serialised by the
        per-session lock; calls for different sessions run in parallel.
        """
        session = await self.get_or_create(session_id, cfg, db_dsn)
        async with self._session_lock(session_id):
            reply = await session.handle_user_input(text)
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None, _save_session_state, session_id, session._state, db_dsn
                )
            except Exception:
                logger.warning(
                    "Failed to persist LangGraph state for session %r",
                    session_id,
                    exc_info=True,
                )
            return reply

    async def reset(self, session_id: str, db_dsn: str) -> None:
        """Clear in-memory and persisted state for *session_id*."""
        async with self._registry_lock:
            if session_id in self._sessions:
                await self._sessions[session_id].reset()
                del self._sessions[session_id]
            self._locks.pop(session_id, None)

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, _delete_session_state, session_id, db_dsn
            )
        except Exception:
            logger.warning(
                "Failed to delete LangGraph state for session %r",
                session_id,
                exc_info=True,
            )

    def evict_all(self) -> int:
        """Evict all sessions from the in-memory cache.

        Called by the idle-timeout handler.  Sessions will be rebuilt from
        the DB on next access, so no state is lost.
        Returns the number of sessions evicted.
        """
        count = len(self._sessions)
        self._sessions.clear()
        self._locks.clear()
        if count:
            logger.info("LangGraph registry: evicted %d session(s)", count)
        return count
