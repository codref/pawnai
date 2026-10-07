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
    OpenAI-compatible chat completions (works with obsidian-copilot).  Handles
    the ``/reset`` sentinel inline.  ``stream=true`` sends SSE keep-alives and
    tool progress (``reasoning_content``) while the agent runs, then the answer.

GET /v1/models
    OpenAI-compatible model list (``pawn-agent``).

POST /v1/pawn/chat
    Native SSE chat for the Pawn Obsidian plugin (typed progress/answer/job
    events, structured note context).

POST /v1/jobs, POST /v1/jobs/upload, GET /v1/jobs, GET /v1/jobs/{id},
POST /v1/jobs/{id}/approve, POST /v1/jobs/{id}/cancel, POST /v1/jobs/{id}/dismiss,
GET /v1/jobs/events
    Background jobs (ask / push_note / upload); always accepted with 202.
    ``/v1/vault/tasks*`` remain as deprecated aliases.

DELETE /sessions/{session_id}
    Clear all stored turns for a session (start fresh).

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
    Swagger UI (FastAPI built-in).  Disabled when ``api.enable_docs`` is false.

GET /openapi.json
    OpenAPI spec (FastAPI built-in, auto-generated from Pydantic models).
    Disabled when ``api.enable_docs`` is false.

Authentication
--------------
All endpoints except ``/health`` require ``Authorization: Bearer <token>``.
If ``api.token`` is not set in ``pawnai.yaml`` the server starts in open
mode with a warning — useful for local development.

When the API is exposed beyond localhost, set ``api.enable_docs: false`` and
rely on ``api.whitelist_ips`` / auto-blacklist (see ``pawn-server blacklist``).

Model selection
---------------
The server always routes through the sallm agent.  ``/v1/chat/completions``
honors a catalog id (``provider@model``) and ignores any other ``model``
value, including ``pawn-agent``.  ``POST /v1/pawn/chat`` and ask jobs take
the same id and fall back to the background default when it is omitted.

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
from typing import Any, AsyncIterator, List, Optional, Union

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
_cors_installed = False
_ip_guard_installed = False

_DOC_PATHS = frozenset({"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"})

from pawn_agent.core.agent_runner import run_agent_turn  # noqa: E402

# Sallm session registry — lazily populated, survives across requests.
from pawn_agent.core.sallm_registry import SallmSessionRegistry  # noqa: E402
from pawn_agent.core.sallm_session import strip_tool_trail  # noqa: E402

_sallm_registry = SallmSessionRegistry()

_security = HTTPBearer(auto_error=False)

_RESET_SENTINEL = "/reset"


def get_sallm_registry() -> SallmSessionRegistry:
    """Return the process-wide sallm registry."""
    return _sallm_registry


def set_sallm_registry(registry: SallmSessionRegistry) -> None:
    """Replace the process-wide registry (tests / serve wiring)."""
    global _sallm_registry
    _sallm_registry = registry


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas — OpenAI-compatible chat completions
# ──────────────────────────────────────────────────────────────────────────────


def _flatten_content(value: Any) -> str:
    """Collapse OpenAI content parts (``[{"type": "text", "text": ...}]``) to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in (None, "text", "input_text"):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p)
    return str(value)


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, value: Any) -> str:
        return _flatten_content(value)


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


class JobCreateRequest(BaseModel):
    """POST /v1/jobs — accept a background job (always 202)."""

    model_config = ConfigDict(extra="ignore")

    kind: str = "ask"
    id: Optional[str] = None
    conversation: Optional[str] = None
    note_path: Optional[str] = None
    # ask
    instruction: Optional[str] = None
    selection: Optional[str] = None
    context_paths: List[str] = Field(default_factory=list)
    context: Optional[str] = None
    # Catalog id (provider@model). Omitted turns use the background default.
    model: Optional[str] = None
    # OpenRouter reasoning effort: none, low, medium, high.
    reasoning: Optional[str] = None
    # OpenRouter route: balanced, nitro, floor, exacto.
    route: Optional[str] = None
    # push_note
    path: Optional[str] = None
    content: Optional[str] = None
    mode: str = "replace"


class JobApproveRequest(BaseModel):
    """POST /v1/jobs/{id}/approve — index a result into agent memory."""

    model_config = ConfigDict(extra="ignore")

    result: Optional[str] = None


class ItemActionRequest(BaseModel):
    """POST /v1/items/{id}/action — triage a coworker inbox item."""

    model_config = ConfigDict(extra="ignore")

    action: str
    arg: Optional[str] = None


class ItemsDeleteRequest(BaseModel):
    """POST /v1/items/delete — dismiss one or many items."""

    model_config = ConfigDict(extra="ignore")

    ids: Optional[list[str]] = None
    all_open: bool = False
    # Comma list, or "all" for every status (when all_open).
    statuses: Optional[str] = None
    kind: Optional[str] = None
    q: Optional[str] = None


class ContextNote(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str
    content: Optional[str] = None


class ChatImage(BaseModel):
    """One image on a native chat turn (dropped file or a note embed)."""

    model_config = ConfigDict(extra="ignore")

    filename: str
    media_type: str = ""
    data_base64: str
    role: str = "question"


class PawnChatRequest(BaseModel):
    """POST /v1/pawn/chat — native streaming chat for the Pawn Obsidian plugin."""

    model_config = ConfigDict(extra="ignore")

    conversation: str
    message: str = ""
    active_note: Optional[ContextNote] = None
    selection: Optional[str] = None
    context: List[ContextNote] = Field(default_factory=list)
    background: bool = False
    # Catalog id (provider@model). Omitted turns use the background default.
    model: Optional[str] = None
    # OpenRouter reasoning effort: none, low, medium, high.
    reasoning: Optional[str] = None
    # OpenRouter route: balanced, nitro, floor, exacto.
    route: Optional[str] = None
    images: List[ChatImage] = Field(default_factory=list)
    # Recaption images this session has already seen.
    force_caption: bool = False
    # When false, note embeds are not read from the vault.
    include_note_images: bool = True


class VaultTaskCreateRequest(BaseModel):
    """Deprecated: POST /v1/vault/tasks (alias of an ``ask`` job)."""

    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    instruction: str
    note_path: Optional[str] = None
    context: Optional[str] = None
    conversation: Optional[str] = None
    timeout_seconds: Optional[float] = None


class VaultTaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    agent_run_id: Optional[str] = None
    error_code: Optional[str] = None


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
    _sallm_registry.evict_all()
    logger.info("Model idle timeout reached — sallm sessions evicted")


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


def _session_id(messages: List[dict], user: Optional[str], header: Optional[str] = None) -> str:
    if user:
        return user
    if header and header.strip():
        return header.strip()
    first = next((m["content"] for m in messages if m.get("role") == "user"), "")
    if first:
        return str(uuid.UUID(hashlib.md5(first.encode()).hexdigest()))
    return str(uuid.uuid4())


def _system_prompt(messages: List[dict]) -> str:
    return "\n\n".join(
        m["content"] for m in messages if m.get("role") == "system" and m.get("content")
    )


def _build_prompt(messages: List[dict], include_system: bool) -> str:
    prompt = _last_user_message(messages)
    if not include_system or not prompt:
        return prompt
    system = _system_prompt(messages)
    if not system:
        return prompt
    return f"Client instructions (from the calling app):\n{system}\n\nUser message:\n{prompt}"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


_SSE_KEEPALIVE = ": keep-alive\n\n"


def _chunk(completion_id: str, created: int, model: str, delta: dict, finish: Any = None) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


async def _stream_agent_turn_openai(
    cfg: Any,
    *,
    prompt: str,
    session_id: str,
    model: str,
    agent_model: Optional[str] = None,
) -> AsyncIterator[str]:
    """OpenAI SSE while the agent runs: keep-alives, progress as reasoning, answer."""
    from pawn_server.core.progress import stream_turn  # noqa: PLC0415

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    show_progress = bool(getattr(cfg.api, "stream_progress", True))
    yield _sse(_chunk(completion_id, created, model, {"role": "assistant", "content": ""}))

    async def _run(on_progress: Any) -> Any:
        return await run_agent_turn(
            cfg=cfg,
            registry=_sallm_registry,
            prompt=prompt,
            session_id=session_id,
            source="api",
            command="run",
            model=agent_model,
            on_progress=on_progress,
        )

    reply = ""
    async for event, data in stream_turn(
        _run, keepalive_seconds=float(getattr(cfg.api, "stream_keepalive_seconds", 10.0))
    ):
        if event == "keepalive":
            yield _SSE_KEEPALIVE
        elif event == "progress":
            if show_progress:
                delta = {"reasoning_content": f"{data['text']}\n"}
                yield _sse(_chunk(completion_id, created, model, delta))
        elif event == "error":
            logger.error("sallm agent error for session %r: %s", session_id, data)
            reply = f"Agent error: {data}"
        elif event == "result":
            reply = data.response

    words = reply.split(" ")
    for i, word in enumerate(words):
        content = word if i == 0 else f" {word}"
        yield _sse(_chunk(completion_id, created, model, {"content": content}))
    yield _sse(_chunk(completion_id, created, model, {}, "stop"))
    yield "data: [DONE]\n\n"


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


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(_require_token)],
)
async def chat_completions(
    req: ChatCompletionRequest,
    cfg: Any = Depends(_get_cfg),
    x_pawn_conversation: Optional[str] = Header(default=None),
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """OpenAI-compatible chat completions endpoint.

    All requests are handled by the sallm agent.  The ``model`` field is
    A catalog id (``provider@model``) selects that profile and provider.
    Any other value, including ``pawn-agent``, is ignored and the background
    default is used.  sallm owns the
    conversation history, so only the last user message is sent to the agent
    (plus the client system prompt when ``api.include_system_prompt`` is on).

    Session key: ``user`` field, then ``X-Pawn-Conversation`` header, then a
    hash of the first user message.

    Send ``/reset`` as the last user message to clear the session history.
    """
    logger.debug("chat/completions raw payload: %s", req.model_dump())
    messages = [m.model_dump() for m in req.messages]
    session_id = _session_id(messages, req.user, x_pawn_conversation)
    loop = asyncio.get_running_loop()
    _schedule_idle_reset(loop, cfg.api_model_idle_timeout_minutes * 60)

    if req.user or x_pawn_conversation:
        logger.info("chat/completions: session_id=%r model=%r", session_id, req.model)
    else:
        logger.warning(
            "chat/completions: session_id=%r model=%r "
            "— 'user' field not set in request payload, session_id derived from message hash",
            session_id,
            req.model,
        )

    if _is_reset(messages):
        await _sallm_registry.reset(session_id, cfg.db_dsn)
        reply = "Session reset."
        if req.stream:
            return StreamingResponse(_stream_sse(reply, req.model), media_type="text/event-stream")
        return _build_openai_response(reply, req.model)

    from pawn_agent.core.coworker.slash import resolve_chat_message  # noqa: PLC0415

    resolved = await resolve_chat_message(
        cfg, _last_user_message(messages), registry=_sallm_registry
    )
    if resolved.mode == "reply":
        if req.stream:
            return StreamingResponse(
                _stream_sse(resolved.text, req.model), media_type="text/event-stream"
            )
        return _build_openai_response(resolved.text, req.model)

    prompt = _build_prompt(messages, bool(getattr(cfg.api, "include_system_prompt", False)))
    if resolved.rewritten:
        prompt = resolved.text
    if not prompt:
        raise HTTPException(status_code=422, detail="No user message found in messages")

    from pawn_agent.utils.model_catalog import catalog_model_or_none  # noqa: PLC0415

    agent_model = catalog_model_or_none(cfg, req.model)
    if req.stream:
        return StreamingResponse(
            _stream_agent_turn_openai(
                cfg,
                prompt=prompt,
                session_id=session_id,
                model=req.model,
                agent_model=agent_model,
            ),
            media_type="text/event-stream",
        )

    try:
        result = await run_agent_turn(
            cfg=cfg,
            registry=_sallm_registry,
            prompt=prompt,
            session_id=session_id,
            source="api",
            command="run",
            model=agent_model,
        )
        reply = result.response
    except Exception as exc:
        logger.error("sallm agent error for session %r: %s", session_id, exc, exc_info=True)
        reply = f"Agent error: {exc}"

    return _build_openai_response(reply, req.model)


@app.get("/v1/pawn/models", dependencies=[Depends(_require_token)])
async def list_pawn_models(cfg: Any = Depends(_get_cfg)) -> dict:
    """Selectable catalog ids for the Pawn plugin. No credentials."""
    from pawn_agent.utils.model_catalog import public_catalog  # noqa: PLC0415

    return public_catalog(cfg)


@app.get("/v1/models", dependencies=[Depends(_require_token)])
async def list_models() -> dict:
    """OpenAI-compatible model list (one virtual model: the Pawn agent)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "pawn-agent",
                "object": "model",
                "created": 0,
                "owned_by": "pawnai",
            }
        ],
    }


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
    await _sallm_registry.reset(session_id, cfg.db_dsn)
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


# ── Background jobs ───────────────────────────────────────────────────────────


def _job_error(exc: Exception) -> HTTPException:
    from pawn_server.core.jobs import JobError  # noqa: PLC0415

    if isinstance(exc, JobError):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


async def _create_job(cfg: Any, body: JobCreateRequest) -> dict:
    from pawn_server.core import jobs  # noqa: PLC0415
    from pawn_server.core.vault_tasks import new_task_id  # noqa: PLC0415

    job_id = (body.id or "").strip() or new_task_id()
    if body.kind == "ask":
        row = await jobs.create_ask_job(
            cfg,
            registry=_sallm_registry,
            job_id=job_id,
            instruction=body.instruction or "",
            note_path=body.note_path,
            selection=body.selection,
            context_paths=body.context_paths,
            context=body.context,
            conversation=body.conversation,
            model=(body.model or "").strip() or None,
            reasoning=(body.reasoning or "").strip() or None,
            route=(body.route or "").strip() or None,
        )
    elif body.kind == "push_note":
        row = await jobs.create_push_note_job(
            cfg,
            job_id=job_id,
            path=body.path or body.note_path or "",
            content=body.content or "",
            mode=body.mode,
            conversation=body.conversation,
        )
    else:
        raise jobs.JobError(
            f"unknown job kind {body.kind!r} (use ask or push_note; uploads go to "
            "/v1/jobs/upload)",
            status_code=422,
        )
    return jobs.serialize_job(row)


@app.post("/v1/jobs", status_code=202, dependencies=[Depends(_require_token)])
async def job_create(body: JobCreateRequest, cfg: Any = Depends(_get_cfg)) -> JSONResponse:
    """Accept a background job. Always returns 202 with the job record."""
    try:
        job = await _create_job(cfg, body)
    except Exception as exc:
        raise _job_error(exc) from exc
    return JSONResponse(status_code=202, content=job)


@app.post("/v1/jobs/upload", status_code=202, dependencies=[Depends(_require_token)])
async def job_upload(
    file: UploadFile = File(...),
    conversation: Optional[str] = Form(default=None),
    note_path: Optional[str] = Form(default=None),
    index: bool = Form(default=False),
    job_id: Optional[str] = Form(default=None, alias="id"),
    cfg: Any = Depends(_get_cfg),
) -> JSONResponse:
    """Upload a file: audio is queued for transcribe-diarize, other files land in Pawn/Inbox/."""
    from pawn_server.core import jobs  # noqa: PLC0415
    from pawn_server.core.vault_tasks import new_task_id  # noqa: PLC0415

    data = await file.read()
    try:
        row = await jobs.create_upload_job(
            cfg,
            registry=_sallm_registry,
            job_id=(job_id or "").strip() or new_task_id(),
            filename=file.filename or "upload.bin",
            data=data,
            content_type=file.content_type,
            note_path=note_path,
            conversation=conversation,
            index=index,
        )
    except Exception as exc:
        raise _job_error(exc) from exc
    return JSONResponse(status_code=202, content=jobs.serialize_job(row))


@app.get("/v1/jobs", dependencies=[Depends(_require_token)])
async def job_list(
    conversation: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None, description="Comma-separated statuses"),
    limit: int = Query(default=50),
    cfg: Any = Depends(_get_cfg),
) -> dict:
    """List jobs newest first, optionally filtered by conversation / status."""
    from pawn_server.core import jobs  # noqa: PLC0415

    statuses = [s.strip() for s in (status or "").split(",") if s.strip()]
    return {
        "object": "list",
        "data": jobs.list_jobs(cfg, conversation=conversation, statuses=statuses, limit=limit),
    }


@app.get("/v1/jobs/events", dependencies=[Depends(_require_token)])
async def job_events_stream(request: Request, cfg: Any = Depends(_get_cfg)) -> StreamingResponse:
    """SSE stream of job updates (``event: job`` with the full job record)."""
    from pawn_server.core import jobs  # noqa: PLC0415
    from pawn_server.core.job_events import SHUTDOWN_EVENT, job_events  # noqa: PLC0415

    keepalive = float(getattr(cfg.api, "stream_keepalive_seconds", 10.0))
    queue = job_events.subscribe()

    async def _gen() -> AsyncIterator[str]:
        try:
            yield _sse_event("ready", {"subscribers": job_events.subscriber_count})
            while True:
                if job_events.closed or await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=keepalive)
                except asyncio.TimeoutError:
                    yield _SSE_KEEPALIVE
                    continue
                if event is SHUTDOWN_EVENT:
                    break
                job = await asyncio.to_thread(jobs.get_job, cfg, event["job_id"])
                yield _sse_event("job", job or event)
        finally:
            job_events.unsubscribe(queue)

    return StreamingResponse(_gen(), media_type="text/event-stream")


_VAULT_POLL_MAX_SECONDS = 25.0


@app.get("/v1/vault/events", dependencies=[Depends(_require_token)])
async def vault_events_poll(
    since: int = Query(default=0, ge=0),
    timeout: float = Query(default=_VAULT_POLL_MAX_SECONDS, ge=0, le=_VAULT_POLL_MAX_SECONDS),
) -> dict:
    """Long-poll vault writes from agent turns in this process.

    Returns immediately when an event newer than ``since`` is buffered.
    Otherwise waits up to ``timeout`` seconds (max 25). An empty ``events``
    list means the client should poll again. ``resync`` is set when the ring
    dropped events behind ``since``.
    """
    from pawn_server.core.vault_events import vault_events  # noqa: PLC0415

    return await vault_events.wait(since, timeout)


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(_require_token)])
async def job_get(job_id: str, cfg: Any = Depends(_get_cfg)) -> dict:
    from pawn_server.core import jobs  # noqa: PLC0415

    job = jobs.get_job(cfg, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


async def _approve(cfg: Any, job_id: str, result: Optional[str]) -> Any:
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415
    from pawn_server.core.vault_tasks import approve_vault_task  # noqa: PLC0415

    try:
        store = vault_store_from_config(cfg)
    except Exception:
        store = None
    outcome = await approve_vault_task(
        cfg, job_id, registry=_sallm_registry, store=store, result_text=result
    )
    if outcome.error_code == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    if outcome.error_code == "not_ready":
        raise HTTPException(status_code=409, detail=f"Job is {outcome.status}, not ready")
    if outcome.error_code == "index_failed":
        raise HTTPException(status_code=500, detail="Could not index into agent memory")
    return outcome


@app.post("/v1/jobs/{job_id}/approve", dependencies=[Depends(_require_token)])
async def job_approve(job_id: str, body: JobApproveRequest, cfg: Any = Depends(_get_cfg)) -> dict:
    """Index an ``ask`` job's result into sallm memory and mark it done."""
    from pawn_server.core import jobs  # noqa: PLC0415

    job = jobs.get_job(cfg, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["kind"] != "ask":
        raise HTTPException(status_code=409, detail="Only ask jobs can be approved")
    await _approve(cfg, job_id, body.result)
    return jobs.get_job(cfg, job_id) or job


@app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(_require_token)])
async def job_cancel(job_id: str, cfg: Any = Depends(_get_cfg)) -> dict:
    """Stop tracking a running job and mark it blocked/cancelled."""
    from pawn_server.core import jobs  # noqa: PLC0415

    try:
        return await jobs.cancel_job(cfg, job_id)
    except Exception as exc:
        raise _job_error(exc) from exc


@app.post("/v1/jobs/{job_id}/dismiss", dependencies=[Depends(_require_token)])
async def job_dismiss(job_id: str, cfg: Any = Depends(_get_cfg)) -> dict:
    """Close a review ask job without indexing the result into memory."""
    from pawn_server.core import jobs  # noqa: PLC0415

    try:
        return await jobs.dismiss_job(cfg, job_id)
    except Exception as exc:
        raise _job_error(exc) from exc


@app.get("/v1/items", dependencies=[Depends(_require_token)])
async def items_list(
    status: Optional[str] = None,
    statuses: Optional[str] = None,
    kind: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    cfg: Any = Depends(_get_cfg),
) -> dict:
    """List coworker inbox items, newest first."""
    from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415

    status_list: Optional[list[str]] = None
    single = status or None
    if statuses:
        raw = statuses.strip().lower()
        if raw in {"*", "all"}:
            status_list = None
            single = None
        else:
            status_list = [part.strip() for part in statuses.split(",") if part.strip()]
    elif not status:
        status_list = list(itemdb.OPEN_STATUSES)
    capped = min(max(1, limit), 200)
    start = max(0, offset)

    def _load() -> tuple[list, int]:
        rows = itemdb.list_items(
            cfg.db_dsn,
            status=single,
            statuses=status_list,
            kind=kind or None,
            q=q or None,
            limit=capped,
            offset=start,
        )
        total = itemdb.count_items(
            cfg.db_dsn,
            status=single,
            statuses=status_list,
            kind=kind or None,
            q=q or None,
        )
        return rows, total

    rows, total = await asyncio.to_thread(_load)
    return {"items": rows, "total": total}


@app.post("/v1/items/delete", dependencies=[Depends(_require_token)])
async def items_delete(
    body: ItemsDeleteRequest,
    cfg: Any = Depends(_get_cfg),
) -> dict:
    """Delete selected items or all open items matching optional filters."""
    from pawn_agent.core.coworker.actions import delete_items  # noqa: PLC0415

    if not body.all_open and not body.ids:
        raise HTTPException(status_code=400, detail="Provide ids or all_open=true")
    # None = default open; [] = every status; else explicit list.
    status_filter: Optional[list[str]] = None
    if body.statuses is not None:
        raw = body.statuses.strip().lower()
        if raw in {"*", "all", ""}:
            status_filter = []
        else:
            status_filter = [part.strip() for part in body.statuses.split(",") if part.strip()]
    result = await delete_items(
        cfg,
        ids=body.ids,
        all_open=body.all_open,
        statuses=status_filter,
        kind=body.kind,
        q=body.q,
        registry=_sallm_registry,
    )
    return result


@app.post("/v1/items/{item_id}/action", dependencies=[Depends(_require_token)])
async def item_action(
    item_id: str,
    body: ItemActionRequest,
    cfg: Any = Depends(_get_cfg),
) -> dict:
    """Apply file, task, todo, delete, later, ignore, approve, or reject to one item."""
    from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415
    from pawn_agent.core.coworker.actions import apply_action  # noqa: PLC0415

    receipt = await apply_action(cfg, item_id, body.action, body.arg, registry=_sallm_registry)
    item = await asyncio.to_thread(itemdb.get_item, cfg.db_dsn, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=receipt)
    return {"receipt": receipt, "item": item}


# ── Deprecated vault task aliases (plugin <= 0.1) ─────────────────────────────


def _legacy_status(job: dict) -> VaultTaskStatusResponse:
    return VaultTaskStatusResponse(
        task_id=job["id"],
        status=job["status"],
        result=job.get("result"),
        agent_run_id=job.get("agent_run_id"),
        error_code=job.get("error_code"),
    )


@app.post("/v1/vault/tasks", deprecated=True, dependencies=[Depends(_require_token)])
async def vault_task_create(
    body: VaultTaskCreateRequest, cfg: Any = Depends(_get_cfg)
) -> JSONResponse:
    """Deprecated alias: starts an ``ask`` job and returns 202 immediately."""
    try:
        job = await _create_job(
            cfg,
            JobCreateRequest(
                kind="ask",
                id=body.id,
                instruction=body.instruction,
                note_path=body.note_path,
                context=body.context,
                conversation=body.conversation,
            ),
        )
    except Exception as exc:
        raise _job_error(exc) from exc
    return JSONResponse(
        status_code=202,
        content={"task_id": job["id"], "status": "accepted", "message": "running in background"},
    )


@app.get(
    "/v1/vault/tasks/{task_id}",
    deprecated=True,
    response_model=VaultTaskStatusResponse,
    dependencies=[Depends(_require_token)],
)
async def vault_task_get(task_id: str, cfg: Any = Depends(_get_cfg)) -> VaultTaskStatusResponse:
    return _legacy_status(await job_get(task_id, cfg))


@app.post(
    "/v1/vault/tasks/{task_id}/approve",
    deprecated=True,
    response_model=VaultTaskStatusResponse,
    dependencies=[Depends(_require_token)],
)
async def vault_task_approve(
    task_id: str, body: JobApproveRequest, cfg: Any = Depends(_get_cfg)
) -> VaultTaskStatusResponse:
    return _legacy_status(await job_approve(task_id, body, cfg))


# ── Native streaming chat for the Pawn plugin ─────────────────────────────────


@app.post("/v1/pawn/chat", dependencies=[Depends(_require_token)])
async def pawn_chat(body: PawnChatRequest, cfg: Any = Depends(_get_cfg)) -> StreamingResponse:
    """SSE chat with typed events: ``progress``, ``answer``, ``job``, ``error``, ``done``.

    Context (active note, selection, extra notes) is structured; the server
    builds the agent prompt. ``background: true`` turns the message into an
    ``ask`` job instead of a live turn.
    """
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415
    from pawn_server.core.chat_context import (  # noqa: PLC0415
        NoteRef,
        build_chat_prompt,
        resolve_notes,
    )
    from pawn_server.core.progress import stream_turn  # noqa: PLC0415

    conversation = body.conversation.strip()
    message = body.message.strip()
    if not conversation or (not message and not body.images):
        raise HTTPException(status_code=422, detail="conversation and message are required")
    if not message and body.images:
        message = "Look at this image."
    loop = asyncio.get_running_loop()
    _schedule_idle_reset(loop, cfg.api_model_idle_timeout_minutes * 60)
    keepalive = float(getattr(cfg.api, "stream_keepalive_seconds", 10.0))

    async def _gen() -> AsyncIterator[str]:
        if message == _RESET_SENTINEL:
            await _sallm_registry.reset(conversation, cfg.db_dsn)
            yield _sse_event("answer", {"content": "Session reset."})
            yield _sse_event("done", {"conversation": conversation})
            return

        from pawn_agent.core.coworker.slash import resolve_chat_message  # noqa: PLC0415

        resolved = await resolve_chat_message(cfg, message, registry=_sallm_registry)
        if resolved.mode == "reply":
            yield _sse_event("answer", {"content": resolved.text})
            yield _sse_event("done", {"conversation": conversation})
            return
        agent_message = resolved.text if resolved.rewritten else message

        if body.background:
            context_paths = [n.path for n in body.context]
            try:
                job = await _create_job(
                    cfg,
                    JobCreateRequest(
                        kind="ask",
                        instruction=agent_message,
                        conversation=conversation,
                        note_path=body.active_note.path if body.active_note else None,
                        selection=body.selection,
                        context_paths=context_paths,
                        model=(body.model or "").strip() or None,
                        reasoning=(body.reasoning or "").strip() or None,
                        route=(body.route or "").strip() or None,
                    ),
                )
            except Exception as exc:
                yield _sse_event("error", {"message": str(exc)})
            else:
                yield _sse_event("job", job)
            yield _sse_event("done", {"conversation": conversation})
            return

        try:
            store = vault_store_from_config(cfg)
        except Exception:
            store = None
        active = (
            NoteRef(path=body.active_note.path, content=body.active_note.content)
            if body.active_note
            else None
        )
        if active is not None:
            active = (await resolve_notes(store, [active]))[0]
        extra = await resolve_notes(store, [NoteRef(n.path, n.content) for n in body.context])
        if resolved.rewritten:
            prompt = agent_message
        else:
            prompt = build_chat_prompt(
                message, active_note=active, selection=body.selection, context=extra
            )

        from pawn_agent.core.vision import (  # noqa: PLC0415
            MAX_CANDIDATE_IMAGES,
            ImageRejected,
            decode_chat_image,
            load_note_images,
        )

        try:
            decoded = [
                decode_chat_image(
                    filename=image.filename,
                    media_type=image.media_type,
                    data_base64=image.data_base64,
                    role=image.role,
                )
                for image in body.images[:MAX_CANDIDATE_IMAGES]
            ]
        except ImageRejected as exc:
            yield _sse_event("error", {"message": str(exc)})
            yield _sse_event("done", {"conversation": conversation})
            return
        questions = [image for image in decoded if image.role == "question"]
        supplied_context = [image for image in decoded if image.role == "context"]
        note_images: list = []
        if body.include_note_images and not supplied_context:
            note_pairs = []
            if active is not None:
                note_pairs.append((active.path, active.content))
            note_pairs.extend((note.path, note.content) for note in extra)
            note_images = await asyncio.to_thread(load_note_images, store, note_pairs)
        turn_images = (questions + supplied_context + note_images)[:MAX_CANDIDATE_IMAGES]

        async def _run(on_progress: Any) -> Any:
            return await run_agent_turn(
                cfg=cfg,
                registry=_sallm_registry,
                prompt=prompt,
                session_id=conversation,
                source="obsidian",
                command="run",
                model=(body.model or "").strip() or None,
                reasoning=(body.reasoning or "").strip() or None,
                route=(body.route or "").strip() or None,
                on_progress=on_progress,
                images=turn_images or None,
                force_caption=bool(body.force_caption),
            )

        run_id: Optional[str] = None
        async for event, data in stream_turn(_run, keepalive_seconds=keepalive):
            if event == "keepalive":
                yield _SSE_KEEPALIVE
            elif event == "progress":
                yield _sse_event("progress", data)
            elif event == "error":
                logger.error("pawn chat error for %r: %s", conversation, data)
                yield _sse_event("error", {"message": data})
            elif event == "result":
                run_id = data.run_id
                yield _sse_event("answer", {"content": strip_tool_trail(data.response or "")})
        yield _sse_event("done", {"conversation": conversation, "run_id": run_id})

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────


def _apply_docs_enabled(enabled: bool) -> None:
    """Register or strip FastAPI docs / OpenAPI routes based on *enabled*."""
    app.router.routes = [
        route for route in app.router.routes if getattr(route, "path", None) not in _DOC_PATHS
    ]
    if enabled:
        app.openapi_url = "/openapi.json"
        app.docs_url = "/docs"
        app.redoc_url = "/redoc"
        app.setup()
    else:
        app.openapi_url = None
        app.docs_url = None
        app.redoc_url = None


def create_app(cfg: Any) -> FastAPI:
    """Initialise the FastAPI app with the given config and return it."""
    global _cfg, _cors_installed, _ip_guard_installed
    _cfg = cfg

    from pawn_server.core.ip_guard import (  # noqa: PLC0415
        IpGuardMiddleware,
        reset_counters,
    )

    reset_counters()
    enable_docs = bool(getattr(getattr(cfg, "api", None), "enable_docs", True))
    _apply_docs_enabled(enable_docs)

    origins = list(getattr(getattr(cfg, "api", None), "cors_origins", None) or [])
    if origins and not _cors_installed:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        _cors_installed = True

    if not _ip_guard_installed:
        app.add_middleware(IpGuardMiddleware, get_cfg=lambda: _cfg)
        _ip_guard_installed = True

    return app
