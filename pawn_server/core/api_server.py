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

Model selection / session mode
------------------------------
The ``model`` field controls both the backend model and whether history is
loaded from the database (stateful) or taken from the request (stateless):

    ``pawn-agent``                           — stateful,  default model
    ``pawn-agent/openai:gpt-4o``             — stateful,  gpt-4o
    ``pawn-agent/stateless``                 — stateless, default model
    ``pawn-agent/stateless/openai:gpt-4o``   — stateless, override = openai:gpt-4o

Stateless mode uses the ``messages`` array from the request as history
directly, without touching the database.  Useful with clients that manage
conversation history themselves (e.g. Open WebUI, Continue).

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
import hashlib
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
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
    # pydantic_ai ModelSettings passthrough
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None  # maps to stop_sequences
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None


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
# Pydantic schemas — RAG ingestion
# ──────────────────────────────────────────────────────────────────────────────


class KnowledgeIngestRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: Optional[str] = None          # inline plain text
    session_id: Optional[str] = None    # index an existing session transcript
    siyuan_path: Optional[str] = None   # index a SiYuan page by path


class KnowledgeIngestResponse(BaseModel):
    chunks: int
    message: str


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response (response_format=json)."""

    text: str


class SpeechRequest(BaseModel):
    """OpenAI-compatible TTS request body for POST /v1/audio/speech."""

    model_config = ConfigDict(extra="ignore")

    model: str
    input: str
    voice: str = "alloy"                  # accepted for OpenAI compat; ignored
    response_format: str = "wav"          # wav | mp3 | opus | aac | flac | pcm
    speed: float = 1.0                    # 0.25 – 4.0
    language: Optional[str] = None        # BCP-47 code, e.g. "en", "it", "fr"; falls back to config


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
    from pawn_server.core.queue_listener import _agent_cache  # noqa: PLC0415

    count = len(_agent_cache)
    _agent_cache.clear()
    if count:
        logger.info("Model idle timeout reached — cleared %d cached agent(s)", count)


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
    global _tts_engine
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


def _parse_model(model_str: str) -> tuple:
    """Parse the model string into (stateless, model_override).

    Supported forms:
        pawn-agent                      → stateful,  no override
        pawn-agent/openai:gpt-4o        → stateful,  override = openai:gpt-4o
        pawn-agent/stateless            → stateless, no override
        pawn-agent/stateless/openai:gpt-4o → stateless, override = openai:gpt-4o
    """
    # strip the leading component (pawn-agent / pawn_agent / default / …)
    first_slash = model_str.find("/")
    remainder = model_str[first_slash + 1:] if first_slash != -1 else ""

    if remainder.startswith("stateless"):
        after = remainder[len("stateless"):]
        override_str = after.lstrip("/") or None
        return True, override_str or None
    else:
        override_str = remainder or None
        return False, override_str or None


def _openai_messages_to_history(messages: List[dict]) -> list:
    """Convert an OpenAI messages array to pydantic_ai ModelMessage history.

    Skips system messages (handled by the agent's own system prompt).
    """
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart  # noqa: PLC0415

    history = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=content)]))
    return history


def _build_model_settings(req: ChatCompletionRequest) -> dict:
    s: dict = {}
    if req.temperature is not None:
        s["temperature"] = req.temperature
    if req.max_tokens is not None:
        s["max_tokens"] = req.max_tokens
    if req.top_p is not None:
        s["top_p"] = req.top_p
    if req.stop:
        s["stop_sequences"] = req.stop
    if req.frequency_penalty is not None:
        s["frequency_penalty"] = req.frequency_penalty
    if req.logit_bias is not None:
        s["logit_bias"] = req.logit_bias
    return s


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
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
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
# Thread-executor helper
# ──────────────────────────────────────────────────────────────────────────────


def _run_turn(agent: Any, prompt: str, history: list, model_settings: dict) -> Any:
    """Synchronous wrapper — runs in a thread executor from the async endpoint."""
    return agent.run_sync(
        prompt,
        message_history=history,
        model_settings=model_settings or None,
    )


def _history_mode(cfg: Any) -> str:
    return getattr(cfg, "history_mode", "raw")


def _history_kwargs(cfg: Any) -> dict[str, Any]:
    return {
        "strip_thinking": getattr(cfg, "strip_thinking", True),
        "recent_turns": getattr(cfg, "history_recent_turns", 4),
        "replay_max_tokens": getattr(cfg, "history_replay_max_tokens", 8000),
        "max_text_chars": getattr(cfg, "history_max_text_chars", 500),
        "sanitize_leaked_thoughts": getattr(cfg, "history_sanitize_leaked_thoughts", True),
    }


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

    Send ``/reset`` as the last user message to clear the session history
    instead of running the agent.  ``stream=true`` is accepted but ignored —
    the response is always a complete JSON object.
    """
    from pawn_server.core.queue_listener import _get_or_create_agent  # noqa: PLC0415
    from pawn_agent.core.session_store import (  # noqa: PLC0415
        append_turn,
        build_replay_history,
        delete_session as _delete_session,
        load_history,
    )

    logger.debug("chat/completions raw payload: %s", req.model_dump())
    messages = [m.model_dump() for m in req.messages]
    session_id = _session_id(messages, req.user)
    loop = asyncio.get_running_loop()
    _schedule_idle_reset(loop, cfg.api_model_idle_timeout_minutes * 60)

    stateless_check, _ = _parse_model(req.model)
    if req.user:
        logger.info(
            "chat/completions: session_id=%r model=%r stateless=%s",
            session_id, req.model, stateless_check,
        )
    else:
        logger.warning(
            "chat/completions: session_id=%r model=%r stateless=%s "
            "— 'user' field not set in request payload, session_id derived from message hash",
            session_id, req.model, stateless_check,
        )

    if _is_reset(messages):
        deleted = _delete_session(session_id, cfg.db_dsn)
        reply = f"Session reset. ({deleted} turn{'s' if deleted != 1 else ''} deleted)"
        if req.stream:
            return StreamingResponse(_stream_sse(reply, req.model), media_type="text/event-stream")
        return _build_openai_response(reply, req.model)

    prompt = _last_user_message(messages)
    if not prompt:
        raise HTTPException(status_code=422, detail="No user message found in messages")

    stateless, model_override = stateless_check, _parse_model(req.model)[1]
    model_settings = _build_model_settings(req)

    if stateless:
        # Use the client-provided messages as history (all but the last user message)
        history = _openai_messages_to_history(messages[:-1])
    else:
        if _history_mode(cfg) == "raw":
            history = load_history(
                session_id,
                cfg.db_dsn,
                strip_thinking=getattr(cfg, "strip_thinking", True),
            )
        else:
            history = build_replay_history(
                session_id,
                cfg.db_dsn,
                **_history_kwargs(cfg),
            )

    # Bind agents to session_id in stateful mode so SessionVars can persist.
    agent_session_id = None if stateless else session_id
    agent = _get_or_create_agent(
        cfg,
        model_override=model_override,
        session_id=agent_session_id,
    )
    source_id = str(uuid.uuid4())

    try:
        result = await agent.run_async(
            prompt,
            message_history=history,
            model_settings=model_settings or None,
        )
    except Exception as exc:
        # Return all agent errors as a 200 assistant message rather than 500.
        # A 500 causes LiteLLM (and other proxies) to retry — which never helps
        # for model-side failures such as malformed tool-call JSON (e.g. invalid
        # escape sequences like \( \) from LaTeX in generated content).
        logger.error("Agent error for session %r: %s", session_id, exc, exc_info=True)
        reply = f"Agent error: {exc}"
        if req.stream:
            return StreamingResponse(_stream_sse(reply, req.model), media_type="text/event-stream")
        return _build_openai_response(reply, req.model)

    if not stateless:
        append_turn(source_id, session_id, list(result.new_messages()), cfg.db_dsn)

    if req.stream:
        return StreamingResponse(_stream_sse(result.output, req.model), media_type="text/event-stream")
    return _build_openai_response(result.output, req.model)


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
    from pawn_agent.core.session_store import delete_session as _delete  # noqa: PLC0415

    deleted = _delete(session_id, cfg.db_dsn)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return Response(status_code=204)


@app.post(
    "/knowledge",
    response_model=KnowledgeIngestResponse,
    status_code=201,
    dependencies=[Depends(_require_token)],
)
async def ingest_knowledge(
    req: KnowledgeIngestRequest,
    cfg: Any = Depends(_get_cfg),
) -> KnowledgeIngestResponse:
    """Index content into the RAG vector store.

    Provide exactly one source field:

    - **text** — embed inline plain text directly.
    - **session_id** — index an existing session's analysis.  The session must
      have a stored analysis; run ``analyze_summary`` via the agent first if
      needed.
    - **siyuan_path** — index a SiYuan page by its human-readable path
      (e.g. ``/Notes/MyPage``).
    """
    sources = [x for x in (req.text, req.session_id, req.siyuan_path) if x]
    if len(sources) != 1:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of: text, session_id, siyuan_path",
        )

    loop = asyncio.get_running_loop()

    if req.session_id:
        from pawn_agent.utils.vectorize import vectorize_session  # noqa: PLC0415

        session_id = req.session_id
        try:
            n, _ = await loop.run_in_executor(
                None,
                lambda: vectorize_session(
                    session_id=session_id,
                    db_dsn=cfg.db_dsn,
                    embed_model=cfg.embed_model,
                    embed_device=cfg.embed_device,
                    embed_dim=cfg.embed_dim,
                    embed_local_files_only=cfg.embed_local_files_only,
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return KnowledgeIngestResponse(
            chunks=n,
            message=f"Indexed {n} chunks for session '{session_id}'.",
        )

    if req.siyuan_path:
        from pawn_core.siyuan import siyuan_post  # noqa: PLC0415
        from pawn_agent.utils.vectorize import vectorize_siyuan_page  # noqa: PLC0415

        if not cfg.siyuan_notebook:
            raise HTTPException(status_code=400, detail="siyuan.notebook is not configured")

        siyuan_path = req.siyuan_path
        try:
            ids = siyuan_post(
                cfg.siyuan_url,
                cfg.siyuan_token,
                "/api/filetree/getIDsByHPath",
                {"path": siyuan_path, "notebook": cfg.siyuan_notebook},
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"SiYuan error: {exc}"
            ) from exc

        if not isinstance(ids, list) or not ids:
            raise HTTPException(
                status_code=404, detail=f"No SiYuan page found at '{siyuan_path}'"
            )

        page_id = ids[0]
        try:
            n = await loop.run_in_executor(
                None,
                lambda: vectorize_siyuan_page(
                    page_id=page_id,
                    siyuan_url=cfg.siyuan_url,
                    siyuan_token=cfg.siyuan_token,
                    db_dsn=cfg.db_dsn,
                    embed_model=cfg.embed_model,
                    embed_device=cfg.embed_device,
                    embed_dim=cfg.embed_dim,
                    embed_local_files_only=cfg.embed_local_files_only,
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return KnowledgeIngestResponse(
            chunks=n,
            message=f"Indexed {n} chunks for SiYuan page '{siyuan_path}'.",
        )

    # text branch
    from pawn_agent.utils.vectorize import vectorize_text  # noqa: PLC0415

    text_content = req.text
    try:
        n = await loop.run_in_executor(
            None,
            lambda: vectorize_text(
                content=text_content,
                db_dsn=cfg.db_dsn,
                embed_model=cfg.embed_model,
                embed_device=cfg.embed_device,
                embed_dim=cfg.embed_dim,
                embed_local_files_only=cfg.embed_local_files_only,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return KnowledgeIngestResponse(
        chunks=n,
        message=f"Indexed {n} chunk{'s' if n != 1 else ''} from inline text.",
    )


@app.post(
    "/v1/audio/transcriptions",
    response_model=None,
    dependencies=[Depends(_require_token)],
)
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),       # accepted for OpenAI compat; always uses parakeet
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
    "mp3":  (["-f", "mp3"],                           "audio/mpeg"),
    "opus": (["-f", "opus"],                          "audio/ogg"),
    "aac":  (["-f", "adts"],                          "audio/aac"),
    "flac": (["-f", "flac"],                          "audio/flac"),
    "pcm":  (["-f", "s16le", "-ac", "1"], "audio/pcm"),
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
        raise HTTPException(
            status_code=500, detail=f"Audio conversion failed: {stderr}"
        ) from exc

    return Response(content=proc.stdout, media_type=media_type)


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────


def create_app(cfg: Any) -> FastAPI:
    """Initialise the FastAPI app with the given config and return it."""
    global _cfg
    _cfg = cfg
    return app
