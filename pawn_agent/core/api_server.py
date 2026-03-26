"""FastAPI HTTP interface for pawn-agent.

Exposes the PydanticAI agent over a simple REST API so external clients
(e.g. a Matrix chatbot) can drive conversations without queue access.

Session history is persisted in PostgreSQL via :mod:`pawn_agent.core.session_store`
and is shared with the queue listener — conversations started via the queue
can be continued through the API and vice versa.

Endpoints
---------
POST /chat
    Send a prompt for a session.  History is loaded automatically.

DELETE /sessions/{session_id}
    Clear all stored turns for a session (start fresh).

GET /health
    Liveness probe — no auth required.

GET /docs
    Swagger UI (FastAPI built-in).

GET /openapi.json
    OpenAPI spec (FastAPI built-in, auto-generated from Pydantic models).

Authentication
--------------
All endpoints except ``/health`` require ``Authorization: Bearer <token>``.
If ``api.token`` is not set in ``pawnai.yaml`` the server starts in open
mode with a warning — useful for local development.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level state
# ──────────────────────────────────────────────────────────────────────────────

_cfg: Optional[Any] = None  # AgentConfig, set by create_app()
_idle_handle: Optional[asyncio.TimerHandle] = None

_security = HTTPBearer(auto_error=False)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    session_id: str
    prompt: str


class ChatResponse(BaseModel):
    reply: str


# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────


def _get_cfg() -> Any:
    if _cfg is None:  # pragma: no cover
        raise RuntimeError("Server not initialised — call create_app(cfg) first")
    return _cfg


def _require_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
    cfg: Any = Depends(_get_cfg),
) -> None:
    token = cfg.api_token
    if not token:
        # No token configured — open access (dev mode)
        return
    if credentials is None or credentials.credentials != token:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")


# ──────────────────────────────────────────────────────────────────────────────
# Idle timer — clears the agent cache after inactivity
# ──────────────────────────────────────────────────────────────────────────────


def _clear_agent_cache() -> None:
    from pawn_agent.core.queue_listener import _agent_cache  # noqa: PLC0415

    count = len(_agent_cache)
    _agent_cache.clear()
    if count:
        logger.info("Model idle timeout reached — cleared %d cached agent(s)", count)


def _schedule_idle_reset(loop: asyncio.AbstractEventLoop, timeout_seconds: float) -> None:
    global _idle_handle
    if _idle_handle is not None:
        _idle_handle.cancel()
    _idle_handle = loop.call_later(timeout_seconds, _clear_agent_cache)


# ──────────────────────────────────────────────────────────────────────────────
# Thread-executor helper
# ──────────────────────────────────────────────────────────────────────────────


def _run_turn(agent: Any, prompt: str, history: list) -> Any:
    """Synchronous wrapper — runs in a thread executor from the async endpoint."""
    return agent._agent.run_sync(prompt, message_history=history)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="pawn-agent API",
    description="HTTP interface for the pawn-agent conversational AI.",
    version="1.0.0",
)


@app.get("/health", include_in_schema=True)
async def health() -> dict:
    """Liveness probe. No authentication required."""
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(_require_token)],
)
async def chat(
    req: ChatRequest,
    cfg: Any = Depends(_get_cfg),
) -> ChatResponse:
    """Send a prompt to the agent for the given session.

    The server loads the full conversation history from the database,
    runs the agent, appends the new turn, and returns the reply.
    Use the same ``session_id`` across calls to maintain context.
    """
    from pawn_agent.core.queue_listener import _get_or_create_agent  # noqa: PLC0415
    from pawn_agent.core.session_store import append_turn, load_history  # noqa: PLC0415

    loop = asyncio.get_running_loop()
    _schedule_idle_reset(loop, cfg.api_model_idle_timeout_minutes * 60)

    source_id = str(uuid.uuid4())
    history = load_history(req.session_id, cfg.db_dsn)

    agent = _get_or_create_agent(cfg)

    try:
        result = await loop.run_in_executor(None, _run_turn, agent, req.prompt, history)
    except Exception as exc:
        logger.error("Agent error for session %r: %s", req.session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    append_turn(source_id, req.session_id, list(result.new_messages()), cfg.db_dsn)
    return ChatResponse(reply=result.output)


@app.delete(
    "/sessions/{session_id}",
    status_code=204,
    dependencies=[Depends(_require_token)],
)
async def delete_session(
    session_id: str,
    cfg: Any = Depends(_get_cfg),
) -> Response:
    """Clear all stored turns for a session.

    The next ``/chat`` call with the same ``session_id`` will start fresh.
    Returns 404 if the session does not exist.
    """
    from pawn_agent.core.session_store import delete_session as _delete  # noqa: PLC0415

    deleted = _delete(session_id, cfg.db_dsn)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return Response(status_code=204)


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────


def create_app(cfg: Any) -> FastAPI:
    """Initialise the FastAPI app with the given config and return it."""
    global _cfg
    _cfg = cfg
    return app
