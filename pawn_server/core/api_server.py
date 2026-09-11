"""FastAPI HTTP interface for pawn-server.

Exposes the PydanticAI agent over a REST API with an OpenAI-compatible
``POST /v1/chat/completions`` endpoint so any OpenAI client can drive
conversations directly — no litellm proxy required.

Session history is persisted in PostgreSQL via :mod:`pawn_agent.core.session_store`
and is shared with the queue listener — conversations started via the queue
can be continued through the API and vice versa.

Endpoints
---------
POST /v1/chat/completions
    OpenAI-compatible chat completions.  Handles the ``/reset`` sentinel
    inline.  Supports ``stream=true`` (SSE, word-by-word after full generation).

DELETE /sessions/{session_id}
    Clear all stored turns for a session (start fresh).

POST /knowledge
    Index content into the RAG vector store (inline text, session transcript,
    or SiYuan page).

POST /v1/audio/transcriptions
    OpenAI-compatible audio transcription.  Accepts WAV, FLAC, and any format
    convertible by ffmpeg (OGG Opus, MP3, M4A, WebM, …).  ``response_format``
    controls the response: ``json`` (default), ``verbose_json`` (with
    word-level timestamps), or ``text`` (bare string).

POST /v1/audio/speech
    OpenAI-compatible text-to-speech.  Uses NeMo FastPitch + HiFi-GAN.
    ``response_format`` controls audio encoding: ``wav`` (default), ``mp3``,
    ``opus``, ``aac``, ``flac``, or ``pcm``.  ``speed`` (0.25 – 4.0) maps to
    FastPitch *pace*.  Models are lazily loaded on the first call and
    automatically evicted after ``models.tts_idle_timeout_minutes`` of
    inactivity.

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

Model selection
---------------
The server always routes through the LangGraph agent.  The ``model`` field
is accepted for OpenAI client compatibility but its value is ignored.

Session management
------------------
Set the ``user`` field to your session ID.  If omitted, a stable UUID is
derived from the MD5 of the first user message so the same conversation
always maps to the same session.

Reset
-----
Send ``/reset`` as the last user message to clear the session history.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional, Union

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level state
# ──────────────────────────────────────────────────────────────────────────────

_cfg: Optional[Any] = None  # AgentConfig, set by create_app()
_idle_handle: Optional[asyncio.TimerHandle] = None
_transcription_engine: Optional[Any] = None  # pawn_core.TranscriptionEngine, lazy-loaded
_transcription_lock = threading.Lock()
_tts_engine: Optional[Any] = None  # pawn_core.TTSEngine, lazy-loaded
_tts_lock = threading.Lock()
_tts_idle_handle: Optional[asyncio.TimerHandle] = None

# LangGraph session registry — lazily populated, survives across requests.
from pawn_agent.core.langgraph_registry import LangGraphSessionRegistry  # noqa: E402
from pawn_agent.core.agent_runner import run_agent_turn  # noqa: E402
from pawn_agent.core.graph_events import (  # noqa: E402
    LANGGRAPH_CHAT_GRAPH_NAME,
    LANGGRAPH_CHAT_GRAPH_VERSION,
    LANGGRAPH_CHAT_TOPOLOGY,
)

_langgraph_registry = LangGraphSessionRegistry()

_security = HTTPBearer(auto_error=False)

_RESET_SENTINEL = "/reset"


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas — OpenAI-compatible chat completions
# ──────────────────────────────────────────────────────────────────────────────


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: List[ChatCompletionMessage]
    user: Optional[str] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas — transcription & TTS
# ──────────────────────────────────────────────────────────────────────────────


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response (response_format=json)."""

    text: str


class SpeechRequest(BaseModel):
    """OpenAI-compatible TTS request body for POST /v1/audio/speech."""

    model_config = ConfigDict(extra="ignore")

    model: str
    input: str
    voice: str = "alloy"  # accepted for OpenAI compat; ignored
    response_format: str = "wav"  # wav | mp3 | opus | aac | flac | pcm
    speed: float = 1.0  # 0.25 – 4.0
    language: Optional[str] = None  # BCP-47 code, e.g. "en", "it", "fr"; falls back to config


# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────


def _get_cfg() -> Any:
    if _cfg is None:  # pragma: no cover
        raise RuntimeError("Server not initialised — call create_app(cfg) first")
    return _cfg


def _get_transcription_engine(cfg: Any) -> Any:
    """Return the module-level TranscriptionEngine, creating it on first call."""
    global _transcription_engine
    if _transcription_engine is None:
        with _transcription_lock:
            if _transcription_engine is None:
                from pawn_core.transcription import TranscriptionEngine  # noqa: PLC0415

                _transcription_engine = TranscriptionEngine(
                    device=cfg.transcription_device,
                    backend=cfg.transcription_backend,
                    model_name=cfg.transcription_model,
                )
    return _transcription_engine


def _get_tts_engine(cfg: Any) -> Any:
    """Return the module-level TTSEngine, creating it on first call (no model loaded yet)."""
    global _tts_engine
    if _tts_engine is None:
        with _tts_lock:
            if _tts_engine is None:
                from pawn_core.tts import TTSEngine  # noqa: PLC0415

                _tts_engine = TTSEngine(
                    device=cfg.tts_device,
                    language_id=cfg.tts_language,
                    voice=cfg.tts_voice,
                )
    return _tts_engine


def _require_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
    cfg: Any = Depends(_get_cfg),
) -> None:
    token = cfg.api_token
    if not token:
        return
    if credentials is None or credentials.credentials != token:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")


# ──────────────────────────────────────────────────────────────────────────────
# Idle timer — clears the agent cache after inactivity
# ──────────────────────────────────────────────────────────────────────────────


def _clear_agent_cache() -> None:
    _langgraph_registry.evict_all()
    logger.info("Model idle timeout reached — LangGraph sessions evicted")


def _schedule_idle_reset(loop: asyncio.AbstractEventLoop, timeout_seconds: float) -> None:
    global _idle_handle
    if _idle_handle is not None:
        _idle_handle.cancel()
    _idle_handle = loop.call_later(timeout_seconds, _clear_agent_cache)


def _clear_tts_models() -> None:
    """Unload TTS models from memory after the idle timeout fires.

    Runs on the event loop thread — kept intentionally short.  ``unload()``
    only sets references to None and clears the CUDA cache; it acquires the
    engine's internal lock, which should be uncontended at idle time.
    """
    if _tts_engine is not None:
        try:
            _tts_engine.unload()
            logger.info("TTS idle timeout reached — models unloaded from memory")
        except Exception:
            logger.warning("TTS idle unload failed", exc_info=True)


def _schedule_tts_idle_reset(loop: asyncio.AbstractEventLoop, timeout_seconds: float) -> None:
    global _tts_idle_handle
    if _tts_idle_handle is not None:
        _tts_idle_handle.cancel()
    _tts_idle_handle = loop.call_later(timeout_seconds, _clear_tts_models)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _last_user_message(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return ""


def _is_reset(messages: List[dict]) -> bool:
    return _last_user_message(messages).strip() == _RESET_SENTINEL


def _session_id(messages: List[dict], user: Optional[str]) -> str:
    if user:
        return user
    first = next((m["content"] for m in messages if m.get("role") == "user"), "")
    if first:
        return str(uuid.UUID(hashlib.md5(first.encode()).hexdigest()))
    return str(uuid.uuid4())


def _build_openai_response(reply: str, model: str) -> ChatCompletionResponse:
    completion_tokens = len(reply) // 4
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=reply),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
        ),
    )


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        rendered = value.isoformat()
        return rendered.replace("+00:00", "Z")
    return str(value)


def _duration_ms(started_at: Any, completed_at: Any) -> Optional[int]:
    if started_at is None or completed_at is None:
        return None
    try:
        return int((completed_at - started_at).total_seconds() * 1000)
    except Exception:
        return None


def _prompt_preview(prompt: Optional[str], limit: int = 120) -> str:
    if not prompt:
        return ""
    normalized = " ".join(prompt.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _query_agent_runs(dsn: str, *, limit: int, offset: int) -> dict[str, Any]:
    import sqlalchemy as sa  # noqa: PLC0415

    engine = sa.create_engine(dsn)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                sa.text(
                    "SELECT id, source, session_id, model, status, created_at, started_at, "
                    "completed_at, prompt FROM agent_runs "
                    "ORDER BY created_at DESC NULLS LAST LIMIT :limit OFFSET :offset"
                ),
                {"limit": limit, "offset": offset},
            ).fetchall()
    finally:
        engine.dispose()

    return {
        "runs": [
            {
                "id": row.id,
                "source": row.source,
                "session_id": row.session_id,
                "model": row.model,
                "status": row.status,
                "created_at": _iso(row.created_at),
                "started_at": _iso(row.started_at),
                "completed_at": _iso(row.completed_at),
                "duration_ms": _duration_ms(row.started_at, row.completed_at),
                "prompt_preview": _prompt_preview(row.prompt),
            }
            for row in rows
        ],
        "limit": limit,
        "offset": offset,
    }


def _query_agent_run(dsn: str, run_id: str) -> dict[str, Any] | None:
    import sqlalchemy as sa  # noqa: PLC0415

    engine = sa.create_engine(dsn)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT id, message_id, source, schedule_id, scheduled_fire_id, command, "
                    "prompt, session_id, model, status, response, error, created_at, "
                    "started_at, completed_at FROM agent_runs WHERE id = :run_id"
                ),
                {"run_id": run_id},
            ).fetchone()
    finally:
        engine.dispose()
    if row is None:
        return None
    return {
        "id": row.id,
        "message_id": row.message_id,
        "source": row.source,
        "schedule_id": row.schedule_id,
        "scheduled_fire_id": row.scheduled_fire_id,
        "command": row.command,
        "prompt": row.prompt,
        "session_id": row.session_id,
        "model": row.model,
        "status": row.status,
        "response": row.response,
        "error": row.error,
        "created_at": _iso(row.created_at),
        "started_at": _iso(row.started_at),
        "completed_at": _iso(row.completed_at),
        "duration_ms": _duration_ms(row.started_at, row.completed_at),
    }


def _query_graph_run_events(dsn: str, run_id: str) -> list[dict[str, Any]]:
    import sqlalchemy as sa  # noqa: PLC0415

    engine = sa.create_engine(dsn)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                sa.text(
                    "SELECT id, run_id, sequence, thread_id, trace_id, graph_name, "
                    "graph_version, event_type, node_name, from_node, to_node, "
                    "router_choice, timestamp, duration_ms, status, payload "
                    "FROM graph_run_events WHERE run_id = :run_id ORDER BY sequence ASC"
                ),
                {"run_id": run_id},
            ).fetchall()
    finally:
        engine.dispose()
    return [
        {
            "id": row.id,
            "run_id": row.run_id,
            "sequence": row.sequence,
            "thread_id": row.thread_id,
            "trace_id": row.trace_id,
            "graph_name": row.graph_name,
            "graph_version": row.graph_version,
            "event_type": row.event_type,
            "node_name": row.node_name,
            "from_node": row.from_node,
            "to_node": row.to_node,
            "router_choice": row.router_choice,
            "timestamp": _iso(row.timestamp),
            "duration_ms": row.duration_ms,
            "status": row.status,
            "payload": row.payload,
        }
        for row in rows
    ]


def _query_graph_topology(dsn: str, graph_name: str, graph_version: str) -> dict[str, Any]:
    import sqlalchemy as sa  # noqa: PLC0415

    engine = sa.create_engine(dsn)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT topology FROM graph_topologies "
                    "WHERE graph_name = :graph_name AND graph_version = :graph_version"
                ),
                {"graph_name": graph_name, "graph_version": graph_version},
            ).fetchone()
    finally:
        engine.dispose()
    return dict(row.topology) if row is not None and row.topology else LANGGRAPH_CHAT_TOPOLOGY


def _build_graph_response(dsn: str, run_id: str) -> dict[str, Any] | None:
    run = _query_agent_run(dsn, run_id)
    if run is None:
        return None
    events = _query_graph_run_events(dsn, run_id)
    graph_name = events[0]["graph_name"] if events else LANGGRAPH_CHAT_GRAPH_NAME
    graph_version = events[0]["graph_version"] if events else LANGGRAPH_CHAT_GRAPH_VERSION
    topology = _query_graph_topology(dsn, graph_name, graph_version)
    path: list[dict[str, Any]] = []
    previous_node = "__start__"
    for event in events:
        if event["event_type"] == "node_start" and event.get("node_name"):
            path.append(
                {
                    "from": previous_node,
                    "to": event["node_name"],
                    "sequence": event["sequence"],
                }
            )
            previous_node = event["node_name"]
        elif event["event_type"] == "edge_taken" and event.get("to_node") == "__end__":
            path.append(
                {
                    "from": event["from_node"],
                    "to": "__end__",
                    "sequence": event["sequence"],
                }
            )
    node_status: dict[str, dict[str, Any]] = {}
    router_decisions: list[dict[str, Any]] = []
    node_started_at: dict[str, str] = {}
    for event in events:
        node_name = event.get("node_name")
        if event["event_type"] == "node_start" and node_name:
            node_started_at[node_name] = event["timestamp"]
            node_status.setdefault(node_name, {})["status"] = event["status"] or "running"
            node_status[node_name]["started_at"] = event["timestamp"]
        elif event["event_type"] == "node_end" and node_name:
            status = node_status.setdefault(node_name, {})
            status["status"] = event["status"] or "completed"
            status["started_at"] = status.get("started_at") or node_started_at.get(node_name)
            status["completed_at"] = event["timestamp"]
            status["duration_ms"] = event["duration_ms"]
        elif event["event_type"] == "error" and node_name:
            status = node_status.setdefault(node_name, {})
            status["status"] = "failed"
            status["completed_at"] = event["timestamp"]
            status["duration_ms"] = event["duration_ms"]
            status["error"] = (event.get("payload") or {}).get("error")
        elif event["event_type"] == "router_decision":
            router_decisions.append(
                {
                    "sequence": event["sequence"],
                    "node_name": node_name,
                    "from_node": event["from_node"],
                    "to_node": event["to_node"],
                    "router_choice": event["router_choice"],
                    "timestamp": event["timestamp"],
                    "payload": event["payload"],
                }
            )
    return {
        "run_id": run_id,
        "run": run,
        "graph_name": graph_name,
        "graph_version": graph_version,
        "nodes": topology.get("nodes", []),
        "edges": topology.get("edges", []),
        "path": path,
        "node_status": node_status,
        "router_decisions": router_decisions,
    }


async def _stream_sse(reply: str, model: str) -> AsyncIterator[str]:
    """Yield OpenAI-compatible SSE chunks for *reply*, word by word."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # Opening chunk carries the role
    opening = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(opening)}\n\n"

    # Stream word by word, preserving spacing
    words = reply.split(" ")
    for i, word in enumerate(words):
        content = word if i == 0 else f" {word}"
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="pawn-server API",
    description="HTTP interface for the pawn-server conversational AI.",
    version="1.0.0",
)


@app.get("/health", include_in_schema=True)
async def health() -> dict:
    """Liveness probe. No authentication required."""
    return {"status": "ok"}


@app.get(
    "/api/agent-runs",
    dependencies=[Depends(_require_token)],
)
async def list_agent_runs(
    limit: int = 25,
    offset: int = 0,
    cfg: Any = Depends(_get_cfg),
) -> dict[str, Any]:
    """List recent persisted agent runs for graph visualization."""
    bounded_limit = max(1, min(limit, 100))
    bounded_offset = max(0, offset)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(
            _query_agent_runs,
            cfg.db_dsn,
            limit=bounded_limit,
            offset=bounded_offset,
        ),
    )


@app.get(
    "/api/agent-runs/{run_id}",
    dependencies=[Depends(_require_token)],
)
async def get_agent_run(run_id: str, cfg: Any = Depends(_get_cfg)) -> dict[str, Any]:
    """Return one persisted agent run."""
    loop = asyncio.get_running_loop()
    run = await loop.run_in_executor(None, _query_agent_run, cfg.db_dsn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return {"run": run}


@app.get(
    "/api/agent-runs/{run_id}/events",
    dependencies=[Depends(_require_token)],
)
async def get_agent_run_events(run_id: str, cfg: Any = Depends(_get_cfg)) -> dict[str, Any]:
    """Return ordered append-only graph events for one run."""
    loop = asyncio.get_running_loop()
    run = await loop.run_in_executor(None, _query_agent_run, cfg.db_dsn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    events = await loop.run_in_executor(None, _query_graph_run_events, cfg.db_dsn, run_id)
    return {"events": events}


@app.get(
    "/api/agent-runs/{run_id}/graph",
    dependencies=[Depends(_require_token)],
)
async def get_agent_run_graph(run_id: str, cfg: Any = Depends(_get_cfg)) -> dict[str, Any]:
    """Return topology plus execution path/status for one run."""
    loop = asyncio.get_running_loop()
    graph = await loop.run_in_executor(None, _build_graph_response, cfg.db_dsn, run_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return graph


@app.get("/graph-viewer", response_class=HTMLResponse, dependencies=[Depends(_require_token)])
async def graph_viewer() -> HTMLResponse:
    """Serve the standalone internal graph execution viewer."""
    return HTMLResponse("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PawnAI Graph Runs</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #172026;
      --muted: #66717c;
      --line: #d8dee6;
      --blue: #2563eb;
      --green: #16803c;
      --red: #c24135;
      --amber: #a16207;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 0 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 { font-size: 17px; margin: 0; font-weight: 650; }
    select, button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
    }
    button { cursor: pointer; }
    main {
      height: calc(100vh - 56px);
      display: grid;
      grid-template-columns: 280px minmax(360px, 1fr) 360px;
      grid-template-rows: minmax(0, 1fr) 220px;
    }
    aside, section {
      min-width: 0;
      min-height: 0;
      background: var(--panel);
      border-right: 1px solid var(--line);
    }
    #graphPanel {
      position: relative;
      overflow: auto;
      background: #fbfcfe;
    }
    #details { padding: 16px; overflow: auto; border-right: 0; }
    #timeline {
      grid-column: 1 / 4;
      border-top: 1px solid var(--line);
      border-right: 0;
      overflow: auto;
      padding: 10px 16px;
    }
    #runs { overflow: auto; padding: 10px; }
    .run {
      width: 100%;
      text-align: left;
      height: auto;
      margin-bottom: 8px;
      padding: 10px;
      border-radius: 6px;
    }
    .run.active { border-color: var(--blue); box-shadow: 0 0 0 1px var(--blue); }
    .muted { color: var(--muted); }
    .status { font-weight: 650; }
    .status.completed { color: var(--green); }
    .status.failed { color: var(--red); }
    .status.running { color: var(--blue); }
    svg { display: block; min-width: 960px; min-height: 640px; }
    .edge { stroke: #b8c0cc; stroke-width: 2; fill: none; marker-end: url(#arrow); }
    .edge.taken { stroke: var(--blue); stroke-width: 4; }
    .node rect { fill: #fff; stroke: #aeb8c5; stroke-width: 1.5; rx: 7; }
    .node.completed rect { stroke: var(--green); stroke-width: 2.5; }
    .node.failed rect { stroke: var(--red); stroke-width: 2.5; }
    .node.running rect { stroke: var(--blue); stroke-width: 2.5; }
    .node.router rect { fill: #fff8e6; }
    .node.tool rect { fill: #eef6ff; }
    .node.system rect { fill: #eef0f3; }
    .node text { font-size: 12px; pointer-events: none; }
    .node { cursor: pointer; }
    .event {
      display: grid;
      grid-template-columns: 54px 145px 180px 1fr;
      gap: 8px;
      padding: 6px 0;
      border-bottom: 1px solid #eef1f4;
      cursor: pointer;
    }
    .event:hover { background: #f7f9fc; }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f4f6f8;
      padding: 10px;
      border-radius: 6px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Graph Runs</h1>
    <button id="refresh">Refresh</button>
    <span id="runMeta" class="muted"></span>
  </header>
  <main>
    <aside id="runs"></aside>
    <section id="graphPanel"><svg id="graph"></svg></section>
    <aside id="details"><p class="muted">Select a run, node, or event.</p></aside>
    <section id="timeline"></section>
  </main>
  <script>
    const state = { runs: [], runId: null, graph: null, events: [], selectedNode: null };
    const $ = (id) => document.getElementById(id);
    const statusClass = (value) => `status ${value || ""}`;

    async function fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json();
    }

    async function loadRuns() {
      const data = await fetchJson("/api/agent-runs?limit=50");
      state.runs = data.runs || [];
      if (!state.runId && state.runs[0]) state.runId = state.runs[0].id;
      renderRuns();
      if (state.runId) await loadRun(state.runId);
    }

    async function loadRun(runId) {
      state.runId = runId;
      const [graph, events] = await Promise.all([
        fetchJson(`/api/agent-runs/${runId}/graph`),
        fetchJson(`/api/agent-runs/${runId}/events`),
      ]);
      state.graph = graph;
      state.events = events.events || [];
      state.selectedNode = null;
      renderRuns();
      renderGraph();
      renderTimeline();
      renderDetails();
    }

    function renderRuns() {
      $("runs").innerHTML = state.runs.map((run) => `
        <button class="run ${run.id === state.runId ? "active" : ""}" data-run="${run.id}">
          <div><strong>${run.id.slice(0, 8)}</strong> <span class="${statusClass(run.status)}">${run.status}</span></div>
          <div class="muted">${run.session_id || ""}</div>
          <div>${escapeHtml(run.prompt_preview || "")}</div>
        </button>
      `).join("") || "<p class='muted'>No runs found.</p>";
      document.querySelectorAll("[data-run]").forEach((el) => {
        el.onclick = () => loadRun(el.dataset.run);
      });
    }

    function layout(nodes) {
      const cols = {
        system: 0,
        node: 1,
        router: 2,
        tool: 3,
        response: 4,
      };
      const seen = {};
      const positions = {};
      nodes.forEach((node) => {
        const col = cols[node.kind] ?? 1;
        seen[col] = seen[col] || 0;
        positions[node.id] = { x: 80 + col * 190, y: 60 + seen[col] * 86 };
        seen[col] += 1;
      });
      if (positions.__end__) positions.__end__ = { x: 80 + 5 * 190, y: 60 };
      return positions;
    }

    function renderGraph() {
      const svg = $("graph");
      if (!state.graph) {
        svg.innerHTML = "";
        return;
      }
      const nodes = state.graph.nodes || [];
      const edges = state.graph.edges || [];
      const pos = layout(nodes);
      const taken = new Set((state.graph.path || []).map((e) => `${e.from}->${e.to}`));
      const nodeById = Object.fromEntries(nodes.map((n) => [n.id, n]));
      svg.setAttribute("viewBox", "0 0 1120 680");
      svg.innerHTML = `
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <path d="M0,0 L0,6 L9,3 z" fill="#8b96a5"></path>
          </marker>
        </defs>
        ${edges.map((edge) => edgePath(edge, pos, taken)).join("")}
        ${nodes.map((node) => nodeSvg(node, pos[node.id], state.graph.node_status?.[node.id])).join("")}
      `;
      svg.querySelectorAll("[data-node]").forEach((el) => {
        el.onclick = () => {
          state.selectedNode = el.dataset.node;
          renderDetails();
        };
      });
      const run = state.graph.run || {};
      $("runMeta").textContent = `${run.session_id || ""} ${run.status || ""} ${run.duration_ms || ""}ms`;
    }

    function edgePath(edge, pos, taken) {
      const a = pos[edge.source], b = pos[edge.target];
      if (!a || !b) return "";
      const x1 = a.x + 150, y1 = a.y + 22, x2 = b.x, y2 = b.y + 22;
      const mid = Math.max(20, (x2 - x1) / 2);
      const d = `M${x1},${y1} C${x1 + mid},${y1} ${x2 - mid},${y2} ${x2},${y2}`;
      return `<path class="edge ${taken.has(`${edge.source}->${edge.target}`) ? "taken" : ""}" d="${d}"></path>`;
    }

    function nodeSvg(node, p, status) {
      if (!p) return "";
      const cls = [node.kind || "node", status?.status || ""].join(" ");
      const duration = status?.duration_ms != null ? `${status.duration_ms}ms` : "";
      return `
        <g class="node ${cls}" data-node="${node.id}" transform="translate(${p.x},${p.y})">
          <rect width="150" height="44"></rect>
          <text x="12" y="19">${escapeHtml(node.label || node.id)}</text>
          <text x="12" y="35" fill="#66717c">${escapeHtml(status?.status || node.kind || "")} ${duration}</text>
        </g>
      `;
    }

    function renderTimeline() {
      $("timeline").innerHTML = state.events.map((event) => `
        <div class="event" data-event="${event.sequence}">
          <span>#${event.sequence}</span>
          <span>${event.event_type}</span>
          <span>${event.node_name || event.from_node || ""}</span>
          <span class="${statusClass(event.status)}">${event.router_choice || event.status || ""}</span>
        </div>
      `).join("") || "<p class='muted'>No graph events for this run.</p>";
      document.querySelectorAll("[data-event]").forEach((el) => {
        el.onclick = () => {
          const event = state.events.find((item) => String(item.sequence) === el.dataset.event);
          state.selectedNode = event?.node_name || event?.from_node || null;
          renderDetails(event);
        };
      });
    }

    function renderDetails(event = null) {
      const node = state.selectedNode;
      const status = node ? state.graph?.node_status?.[node] : null;
      const related = node ? state.events.filter((e) => e.node_name === node || e.from_node === node) : [];
      $("details").innerHTML = `
        <h2>${escapeHtml(node || state.runId || "Graph Run")}</h2>
        ${status ? `<p><span class="${statusClass(status.status)}">${status.status}</span> ${status.duration_ms || ""}ms</p>` : ""}
        ${event ? `<h3>Event #${event.sequence}</h3><pre>${escapeHtml(JSON.stringify(event, null, 2))}</pre>` : ""}
        <h3>Related Events</h3>
        <pre>${escapeHtml(JSON.stringify(related.slice(-12), null, 2))}</pre>
      `;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[ch]));
    }

    $("refresh").onclick = loadRuns;
    loadRuns().catch((error) => {
      $("details").innerHTML = `<h2>Load failed</h2><pre>${escapeHtml(error.message)}</pre>`;
    });
  </script>
</body>
</html>
        """)


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(_require_token)],
)
async def chat_completions(
    req: ChatCompletionRequest,
    cfg: Any = Depends(_get_cfg),
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """OpenAI-compatible chat completions endpoint.

    All requests are handled by the LangGraph agent.  The ``model`` field is
    accepted for OpenAI client compatibility but ignored.

    Send ``/reset`` as the last user message to clear the session history.
    """
    logger.debug("chat/completions raw payload: %s", req.model_dump())
    messages = [m.model_dump() for m in req.messages]
    session_id = _session_id(messages, req.user)
    loop = asyncio.get_running_loop()
    _schedule_idle_reset(loop, cfg.api_model_idle_timeout_minutes * 60)

    if req.user:
        logger.info("chat/completions: session_id=%r model=%r", session_id, req.model)
    else:
        logger.warning(
            "chat/completions: session_id=%r model=%r "
            "— 'user' field not set in request payload, session_id derived from message hash",
            session_id,
            req.model,
        )

    if _is_reset(messages):
        await _langgraph_registry.reset(session_id, cfg.db_dsn)
        reply = "Session reset."
        if req.stream:
            return StreamingResponse(_stream_sse(reply, req.model), media_type="text/event-stream")
        return _build_openai_response(reply, req.model)

    prompt = _last_user_message(messages)
    if not prompt:
        raise HTTPException(status_code=422, detail="No user message found in messages")

    try:
        result = await run_agent_turn(
            cfg=cfg,
            registry=_langgraph_registry,
            prompt=prompt,
            session_id=session_id,
            source="api",
            command="run",
        )
        reply = result.response
    except Exception as exc:
        logger.error("LangGraph agent error for session %r: %s", session_id, exc, exc_info=True)
        reply = f"Agent error: {exc}"

    if req.stream:
        return StreamingResponse(_stream_sse(reply, req.model), media_type="text/event-stream")
    return _build_openai_response(reply, req.model)


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

    The next ``/v1/chat/completions`` call with the same ``session_id`` (via
    the ``user`` field) will start fresh.  Returns 404 if the session does not
    exist.
    """
    await _langgraph_registry.reset(session_id, cfg.db_dsn)
    return Response(status_code=204)


@app.post(
    "/v1/audio/transcriptions",
    response_model=None,
    dependencies=[Depends(_require_token)],
)
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),  # accepted for OpenAI compat; always uses parakeet
    response_format: str = Form(default="json"),  # "json" | "text"
    cfg: Any = Depends(_get_cfg),
) -> Union[TranscriptionResponse, PlainTextResponse, dict]:
    """OpenAI-compatible audio transcription endpoint.

    Accepts WAV, FLAC, and any format convertible by ffmpeg (OGG Opus from
    Matrix, MP3, M4A, WebM, …).  Files not natively supported by libsndfile
    are transparently converted to 16 kHz mono WAV before transcription.

    The ``model`` parameter is accepted for protocol compatibility with OpenAI
    clients (e.g. ``whisper-1``, ``pawn-transcribe``) but is ignored — the
    engine is always configured via ``pawnai.yaml`` (``models.transcription_model``).

    ``response_format`` controls the response body:

    * ``json`` (default) — ``{"text": "…"}``
    * ``verbose_json`` — ``{"text": "…", "words": […]}`` with word-level timestamps
    * ``text`` — bare string, Content-Type: text/plain
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    # libsndfile handles WAV/FLAC/AIFF natively; everything else (OGG Opus,
    # MP3, M4A, WebM …) must be re-encoded to WAV via ffmpeg first.
    _NATIVE_FORMATS = {".wav", ".flac", ".aiff", ".aif"}

    verbose = response_format == "verbose_json"

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    # Convert to WAV when the format is not supported by libsndfile (e.g. OGG Opus).
    wav_path: Optional[str] = None
    transcribe_path = tmp_path
    if suffix.lower() not in _NATIVE_FORMATS:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                check=True,
                capture_output=True,
            )
            transcribe_path = wav_path
        except subprocess.CalledProcessError as exc:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
            raise HTTPException(
                status_code=422, detail=f"Audio conversion failed: {stderr}"
            ) from exc

    loop = asyncio.get_running_loop()
    try:

        def _do_transcribe() -> dict:
            engine = _get_transcription_engine(cfg)
            results = engine.transcribe([transcribe_path], include_timestamps=verbose)
            return results[0] if results else {"text": ""}

        result = await loop.run_in_executor(None, _do_transcribe)
    except Exception as exc:
        logger.error("Transcription error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        if wav_path:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    text = result.get("text", "")

    if response_format == "text":
        return PlainTextResponse(content=text)
    if response_format == "verbose_json":
        words = [
            {"word": w["word"], "start": w.get("start"), "end": w.get("end")}
            for w in result.get("word_timestamps", [])
        ]
        return {"text": text, "words": words} if words else {"text": text}
    return TranscriptionResponse(text=text)


_SPEECH_FORMATS: dict = {
    # format → (ffmpeg output args, media_type)
    "mp3": (["-f", "mp3"], "audio/mpeg"),
    "opus": (["-f", "opus"], "audio/ogg"),
    "aac": (["-f", "adts"], "audio/aac"),
    "flac": (["-f", "flac"], "audio/flac"),
    "pcm": (["-f", "s16le", "-ac", "1"], "audio/pcm"),
}


@app.post(
    "/v1/audio/speech",
    response_model=None,
    dependencies=[Depends(_require_token)],
)
async def audio_speech(
    body: SpeechRequest,
    cfg: Any = Depends(_get_cfg),
) -> Response:
    """OpenAI-compatible text-to-speech endpoint.

    Synthesises *body.input* with Kokoro TTS and returns audio.  Pipelines are
    loaded lazily per language and evicted after
    ``models.tts_idle_timeout_minutes`` of inactivity.

    ``language`` defaults to ``models.tts_language`` in ``pawnai.yaml`` and can
    be overridden per-request (BCP-47, e.g. ``"en"``, ``"it"``, ``"fr"``).
    ``voice`` accepts OpenAI aliases (``"alloy"``, ``"echo"`` …) or native
    Kokoro IDs (``"af_heart"``, ``"am_echo"`` …); defaults to
    ``models.tts_voice``.  ``model`` is accepted for OpenAI compat but ignored.

    ``response_format`` controls audio encoding:

    * ``wav`` (default) — uncompressed PCM, returned directly
    * ``mp3`` / ``opus`` / ``aac`` / ``flac`` / ``pcm`` — converted via ffmpeg
    """
    import subprocess  # noqa: PLC0415

    if not (0.25 <= body.speed <= 4.0):
        raise HTTPException(status_code=422, detail="speed must be between 0.25 and 4.0")

    loop = asyncio.get_running_loop()
    _schedule_tts_idle_reset(loop, cfg.tts_idle_timeout_minutes * 60)

    try:
        engine = _get_tts_engine(cfg)
        wav_bytes: bytes = await loop.run_in_executor(
            None,
            lambda: engine.synthesize(
                body.input,
                speed=body.speed,
                language_id=body.language or cfg.tts_language,
                voice=body.voice,
            ),
        )
    except Exception as exc:
        logger.error("TTS synthesis error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {exc}") from exc

    fmt = body.response_format.lower()
    if fmt == "wav":
        return Response(content=wav_bytes, media_type="audio/wav")

    if fmt not in _SPEECH_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported response_format '{fmt}'. Use: wav, mp3, opus, aac, flac, pcm",
        )

    ffmpeg_args, media_type = _SPEECH_FORMATS[fmt]
    cmd = ["ffmpeg", "-y", "-f", "wav", "-i", "pipe:0"] + ffmpeg_args + ["pipe:1"]
    try:
        proc = subprocess.run(
            cmd,
            input=wav_bytes,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {stderr}") from exc

    return Response(content=proc.stdout, media_type=media_type)


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────


def create_app(cfg: Any) -> FastAPI:
    """Initialise the FastAPI app with the given config and return it."""
    global _cfg
    _cfg = cfg
    return app
