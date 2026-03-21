"""Queue listener for Pawn Diarize.

Consumes messages from a pawn-queue topic and dispatches them to the
appropriate core function.  Each message payload is a structured dict with at
minimum a ``command`` key whose value matches a Pawn Diarize CLI command name
(e.g. ``transcribe-diarize``).  Additional keys are the command's parameters.
Unrecognised keys are silently ignored; omitted parameters fall back to the
per-command defaults defined in :data:`COMMAND_DEFAULTS`.

Concurrency / long-running jobs
---------------------------------
pawn-queue's ``consumer.listen()`` keeps the lease alive via a background
``_lease_refresher`` task, so jobs that take several minutes (e.g. CPU-based
diarization) will not be re-queued while the handler is executing.  Set
``visibility_timeout_seconds`` to at least as long as your longest expected
job in the ``diarize_queue:`` section of ``pawnai.yaml``.

Usage::

    import asyncio
    from pawn_diarize.core.config import AppConfig
    from pawn_diarize.core.queue_listener import start_listener

    asyncio.run(start_listener(AppConfig()))
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model cache — keeps loaded engine instances alive between jobs and releases
# them after a configurable idle period.
# ──────────────────────────────────────────────────────────────────────────────


class ModelCache:
    """Cache loaded ML engine instances between queue jobs.

    Engines are created on first use and reused for subsequent jobs with
    matching parameters.  Call :meth:`release` to free GPU/CPU memory.
    """

    def __init__(self) -> None:
        self._transcription_engine: Optional[Any] = None
        self._transcription_key: Optional[Tuple[str, str]] = None
        self._diarization_engine: Optional[Any] = None
        self._diarization_key: Optional[Optional[str]] = None

    @property
    def has_models(self) -> bool:
        return (
            self._transcription_engine is not None
            or self._diarization_engine is not None
        )

    def get_transcription_engine(self, device: str, backend: str) -> Any:
        """Return a cached (or newly created) :class:`TranscriptionEngine`."""
        key: Tuple[str, str] = (device, backend)
        if self._transcription_engine is None or self._transcription_key != key:
            from .transcription import TranscriptionEngine
            logger.info(
                "ModelCache: loading TranscriptionEngine (device=%s, backend=%s)",
                device, backend,
            )
            self._transcription_engine = TranscriptionEngine(device=device, backend=backend)
            self._transcription_key = key
        return self._transcription_engine

    def get_diarization_engine(self, device: str) -> Any:
        """Return a cached (or newly created) :class:`DiarizationEngine`."""
        diarize_device: Optional[str] = device if device != "cpu" else None
        if self._diarization_engine is None or self._diarization_key != diarize_device:
            from .diarization import DiarizationEngine
            logger.info(
                "ModelCache: loading DiarizationEngine (device=%s)",
                diarize_device,
            )
            self._diarization_engine = DiarizationEngine(device=diarize_device)
            self._diarization_key = diarize_device
        return self._diarization_engine

    def release(self) -> None:
        """Drop all cached engine references and free GPU memory."""
        if not self.has_models:
            return
        logger.info("ModelCache: releasing ML models from memory")
        self._transcription_engine = None
        self._transcription_key = None
        self._diarization_engine = None
        self._diarization_key = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("ModelCache: CUDA memory cache cleared")
        except ImportError:
            pass

# ──────────────────────────────────────────────────────────────────────────────
# Per-command parameter defaults
# These mirror the Typer CLI option defaults so producers only need to supply
# values that deviate from the defaults.
# ──────────────────────────────────────────────────────────────────────────────

COMMAND_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "transcribe-diarize": {
        "audio_paths": [],          # required — producer MUST supply this
        "output": None,
        "session": None,
        "db_dsn": None,             # None → AppConfig default
        "threshold": 0.7,
        "store_new": True,
        "device": "cuda",
        "chunk_duration": None,
        "cross_file_threshold": 0.85,
        "no_timestamps": False,
        "verbose": False,
        "backend": "nemo",
    },
    "transcribe": {
        "audio_paths": [],
        "output": None,
        "session": None,
        "db_dsn": None,
        "timestamps": True,
        "device": "cuda",
        "chunk_duration": None,
        "backend": "nemo",
    },
    "diarize": {
        "audio_paths": [],
        "output": None,
        "db_dsn": None,
        "threshold": 0.7,
        "store_new": True,
    },
    "embed": {
        "audio_paths": [],
        "speaker_id": None,         # required for embed
        "db_dsn": None,
    },
    "sync-siyuan": {
        "session": None,           # required unless all_sessions=True
        "all_sessions": False,
        "notebook": None,          # falls back to siyuan.notebook in config
        "token": None,             # falls back to siyuan.token in config
        "url": None,               # falls back to siyuan.url in config
        "path_template": None,
        "daily_note": True,
        "daily_path_template": None,
        "db_dsn": None,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _merge_params(command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge *payload* over the defaults for *command*.

    Args:
        command: Normalised command name (e.g. ``"transcribe-diarize"``).
        payload: Raw message payload dict (``command`` key already removed).

    Returns:
        Merged parameter dict ready for the dispatch function.
    """
    defaults = dict(COMMAND_DEFAULTS.get(command, {}))
    defaults.update(payload)
    return defaults


def _resolve_db_dsn(params: Dict[str, Any], cfg: Any) -> str:
    """Return the DB DSN from *params* or fall back to the AppConfig default."""
    from .config import DEFAULT_DB_DSN
    return params.get("db_dsn") or cfg.get("db_dsn") or DEFAULT_DB_DSN


# ──────────────────────────────────────────────────────────────────────────────
# Command dispatch
# ──────────────────────────────────────────────────────────────────────────────


async def dispatch(
    command: str,
    params: Dict[str, Any],
    cfg: Any,
    model_cache: Optional[ModelCache] = None,
) -> None:
    """Dispatch a single command to the appropriate core function.

    All execution is synchronous inside the core modules, so the heavy call is
    run in a thread executor to avoid blocking the asyncio event loop.

    Args:
        command: Pawn Diarize command name (e.g. ``"transcribe-diarize"``),
        params: Merged parameter dict (defaults already applied).
        cfg: :class:`~pawn_diarize.core.config.AppConfig` instance.
        model_cache: Optional :class:`ModelCache` for reusing loaded engines
            across jobs and releasing them after an idle period.

    Raises:
        ValueError: If *command* is not supported.
        Exception: Re-raises any exception from the underlying core function.
    """
    loop = asyncio.get_event_loop()

    if command == "transcribe-diarize":
        await loop.run_in_executor(None, _run_transcribe_diarize, params, cfg, model_cache)
    elif command == "transcribe":
        await loop.run_in_executor(None, _run_transcribe, params, cfg, model_cache)
    elif command == "diarize":
        await loop.run_in_executor(None, _run_diarize, params, cfg, model_cache)
    elif command == "embed":
        await loop.run_in_executor(None, _run_embed, params, cfg)
    elif command == "sync-siyuan":
        await loop.run_in_executor(None, _run_sync_siyuan, params, cfg)
    else:
        raise ValueError(f"Unsupported command: {command!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-command runners  (synchronous — called via run_in_executor)
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_audio_paths(paths: List[str], cfg: Any) -> tuple[List[str], List[str]]:
    """Download any s3:// paths and return (resolved_local, temp_dirs)."""
    from .s3 import is_s3_path, S3Client, parse_s3_uri, expand_s3_glob
    import tempfile
    from pathlib import Path

    if not any(is_s3_path(p) for p in paths):
        return paths, []

    s3_cfg = cfg.get_s3_config()
    if s3_cfg is None:
        raise RuntimeError(
            "Message contains s3:// paths but no 's3:' section is configured in .pawn-diarize.yml"
        )

    client = S3Client.from_dict(s3_cfg)
    dl_dir = s3_cfg.get("download_dir")
    base_dir = Path(dl_dir) if dl_dir else None
    if base_dir:
        base_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pawn_diarize_queue_", dir=str(base_dir) if base_dir else None))

    expanded: List[str] = []
    for path in paths:
        if is_s3_path(path) and any(c in path for c in ("*", "?", "[")):
            matches = expand_s3_glob(path, client)
            expanded.extend(matches)
        else:
            expanded.append(path)

    resolved: List[str] = []
    for path in expanded:
        if not is_s3_path(path):
            resolved.append(path)
            continue
        bucket, key = parse_s3_uri(path, configured_bucket=client.bucket)
        local_path = tmp_dir / Path(key).name
        logger.info("Downloading s3://%s/%s → %s", bucket, key, local_path)
        client.download_file(key, str(local_path), bucket=bucket)
        resolved.append(str(local_path))

    return resolved, [str(tmp_dir)]


def _cleanup(temp_dirs: List[str]) -> None:
    import shutil
    for d in temp_dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


def _run_transcribe_diarize(
    params: Dict[str, Any], cfg: Any, model_cache: Optional[ModelCache] = None
) -> None:
    import json as _json
    from pathlib import Path as _Path
    from .combined import transcribe_with_diarization, format_transcript_with_speakers
    from .config import DEFAULT_DB_DSN
    from .database import (
        get_engine, init_db,
        load_session_state, save_session_state, save_transcription_segments,
    )

    audio_paths: List[str] = params.get("audio_paths") or []
    if not audio_paths:
        raise ValueError("transcribe-diarize: 'audio_paths' is required")

    resolved, temps = _resolve_audio_paths(audio_paths, cfg)
    try:
        db_dsn = _resolve_db_dsn(params, cfg)
        session = params.get("session")
        threshold = float(params.get("threshold", 0.7))
        store_new = bool(params.get("store_new", True))
        device = params.get("device", "cuda")
        chunk_duration = params.get("chunk_duration")
        cross_file_threshold = float(params.get("cross_file_threshold", 0.85))
        no_timestamps = bool(params.get("no_timestamps", False))
        verbose = bool(params.get("verbose", False))
        backend = params.get("backend", "nemo")
        output = params.get("output")

        # Load session state if a session name was provided
        engine = get_engine(db_dsn)
        init_db(engine)

        prior_embeddings: Optional[Dict[str, Any]] = None
        time_cursor: float = 0.0
        processed_files: List[str] = []
        prior_segment_count: int = 0

        if session:
            # load_session_state returns (embeddings, time_cursor, processed_files, segment_count)
            prior_embeddings, time_cursor, processed_files, prior_segment_count = load_session_state(session, engine)

        result = transcribe_with_diarization(
            audio_path=resolved if len(resolved) > 1 else resolved[0],
            db_dsn=db_dsn,
            similarity_threshold=threshold,
            store_new_speakers=store_new,
            device=device,
            chunk_duration=chunk_duration,
            cross_file_threshold=cross_file_threshold,
            verbose=verbose,
            backend=backend,
            prior_speaker_embeddings=prior_embeddings,
            time_cursor=time_cursor,
            transcription_engine=(
                model_cache.get_transcription_engine(device, backend)
                if model_cache is not None else None
            ),
            diarization_engine=(
                model_cache.get_diarization_engine(device)
                if model_cache is not None else None
            ),
        )

        # Save session state
        if session and result:
            new_processed = processed_files + (resolved if isinstance(resolved, list) else [resolved])
            save_session_state(
                session,
                result.get("session_speaker_embeddings") or {},
                result.get("new_time_cursor", 0.0),
                new_processed,
                engine,
            )
            if result.get("segments"):
                save_transcription_segments(result["segments"], session, engine, start_index=prior_segment_count)

        # Format and write output
        text = format_transcript_with_speakers(
            result, include_timestamps=not no_timestamps
        )
        if output:
            out_path = _Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if output.endswith(".json"):
                out_path.write_text(_json.dumps(result, indent=2, default=str))
            else:
                out_path.write_text(text)
            logger.info("Output written to %s", output)
        else:
            logger.info("Transcript:\n%s", text)

    finally:
        _cleanup(temps)


def _run_transcribe(
    params: Dict[str, Any], cfg: Any, model_cache: Optional[ModelCache] = None
) -> None:
    import json as _json
    from pathlib import Path as _Path

    audio_paths: List[str] = params.get("audio_paths") or []
    if not audio_paths:
        raise ValueError("transcribe: 'audio_paths' is required")

    resolved, temps = _resolve_audio_paths(audio_paths, cfg)
    try:
        device = params.get("device", "cuda")
        backend = params.get("backend", "nemo")
        timestamps = bool(params.get("timestamps", True))
        chunk_duration = params.get("chunk_duration")
        output = params.get("output")

        if model_cache is not None:
            engine = model_cache.get_transcription_engine(device, backend)
        else:
            from .transcription import TranscriptionEngine
            engine = TranscriptionEngine(device=device, backend=backend)
        results = engine.transcribe(
            resolved,
            include_timestamps=timestamps,
            chunk_duration=chunk_duration,
        )
        combined = " ".join(r.get("text", "") if isinstance(r, dict) else str(r) for r in results)

        if output:
            _Path(output).parent.mkdir(parents=True, exist_ok=True)
            _Path(output).write_text(combined)
            logger.info("Transcription written to %s", output)
        else:
            logger.info("Transcription:\n%s", combined)
    finally:
        _cleanup(temps)


def _run_diarize(
    params: Dict[str, Any], cfg: Any, model_cache: Optional[ModelCache] = None
) -> None:
    import json as _json
    from pathlib import Path as _Path

    audio_paths: List[str] = params.get("audio_paths") or []
    if not audio_paths:
        raise ValueError("diarize: 'audio_paths' is required")

    resolved, temps = _resolve_audio_paths(audio_paths, cfg)
    try:
        db_dsn = _resolve_db_dsn(params, cfg)
        threshold = float(params.get("threshold", 0.7))
        store_new = bool(params.get("store_new", True))
        output = params.get("output")
        device = params.get("device", "cuda")

        if model_cache is not None:
            engine = model_cache.get_diarization_engine(device)
        else:
            from .diarization import DiarizationEngine
            engine = DiarizationEngine()
        result = engine.diarize(
            resolved,
            db_dsn=db_dsn,
            similarity_threshold=threshold,
            store_new_speakers=store_new,
        )

        text = _json.dumps(result, indent=2, default=str)
        if output:
            _Path(output).parent.mkdir(parents=True, exist_ok=True)
            _Path(output).write_text(text)
            logger.info("Diarization written to %s", output)
        else:
            logger.info("Diarization result:\n%s", text)
    finally:
        _cleanup(temps)


def _run_embed(params: Dict[str, Any], cfg: Any) -> None:
    from .embeddings import EmbeddingManager

    audio_paths: List[str] = params.get("audio_paths") or []
    speaker_id: Optional[str] = params.get("speaker_id")
    if not audio_paths:
        raise ValueError("embed: 'audio_paths' is required")
    if not speaker_id:
        raise ValueError("embed: 'speaker_id' is required")

    resolved, temps = _resolve_audio_paths(audio_paths, cfg)
    try:
        db_dsn = _resolve_db_dsn(params, cfg)
        manager = EmbeddingManager(db_dsn=db_dsn)
        for path in resolved:
            manager.add_speaker(speaker_id=speaker_id, audio_path=path)
            logger.info("Embedding stored for speaker '%s' from %s", speaker_id, path)
    finally:
        _cleanup(temps)


def _run_sync_siyuan(params: Dict[str, Any], cfg: Any) -> None:
    from .siyuan import (
        SiyuanClient,
        SiyuanError,
        format_session_markdown,
        resolve_path_template,
        DEFAULT_PATH_TEMPLATE,
        DEFAULT_DAILY_PATH_TEMPLATE,
    )
    from .database import get_engine, init_db, get_session_analysis
    from .config import DEFAULT_DB_DSN

    session: Optional[str] = params.get("session")
    all_sessions: bool = bool(params.get("all_sessions", False))
    if not session and not all_sessions:
        raise ValueError("sync-siyuan: 'session' or 'all_sessions': true is required")

    sy_cfg = cfg.get_siyuan_config() or {}
    resolved_url = params.get("url") or sy_cfg.get("url", "http://127.0.0.1:6806")
    resolved_token = params.get("token") or sy_cfg.get("token", "")
    resolved_notebook = params.get("notebook") or sy_cfg.get("notebook", "")
    resolved_path_tpl = params.get("path_template") or sy_cfg.get("path_template", DEFAULT_PATH_TEMPLATE)
    resolved_daily_tpl = params.get("daily_path_template") or sy_cfg.get("daily_note_path", DEFAULT_DAILY_PATH_TEMPLATE)
    daily_note: bool = bool(params.get("daily_note", True))

    if not resolved_notebook:
        raise ValueError("sync-siyuan: 'notebook' is required (or set siyuan.notebook in .pawn-diarize.yml)")

    db_dsn = _resolve_db_dsn(params, cfg)
    engine = get_engine(db_dsn)
    init_db(engine)

    client = SiyuanClient(url=resolved_url, token=resolved_token, notebook_id=resolved_notebook)

    from sqlalchemy.orm import Session as OrmSession
    from .database import SessionAnalysis
    from sqlalchemy import select

    with OrmSession(engine) as orm_session:
        if all_sessions:
            rows = orm_session.execute(select(SessionAnalysis)).scalars().all()
            session_ids = list({r.session_id for r in rows})
        else:
            session_ids = [session]

    for sid in session_ids:
        analysis = get_session_analysis(sid, engine)
        if analysis is None:
            logger.warning("sync-siyuan: no analysis found for session %r — skipping", sid)
            continue

        # Load transcript text (same approach as CLI sync_siyuan command)
        try:
            from .analysis import AnalysisEngine as _AE
            transcript = _AE()._load_transcript("", db_dsn=db_dsn, session_id=sid)
        except Exception:
            transcript = "_Transcript not available._"

        md = format_session_markdown(
            title=analysis.title,
            summary=analysis.summary,
            key_topics=analysis.key_topics,
            speaker_highlights=analysis.speaker_highlights,
            sentiment=analysis.sentiment,
            sentiment_tags=analysis.sentiment_tags,
            tags=analysis.tags,
            session_id=analysis.session_id,
            source=analysis.source,
            analyzed_at=analysis.analyzed_at,
            transcript=transcript,
            model=getattr(analysis, "model", ""),
        )
        doc_path = resolve_path_template(
            resolved_path_tpl,
            session_id=sid,
            title=analysis.title,
        )
        attrs: Dict[str, str] = {"custom-pawn-diarize-session": sid}
        all_tags = list(analysis.tags or []) + list(analysis.sentiment_tags or [])
        if all_tags:
            attrs["tags"] = ",".join(all_tags)
        if getattr(analysis, "model", None):
            attrs["custom-model"] = analysis.model

        doc_id = client.upsert_session_doc(
            notebook=resolved_notebook,
            path=doc_path,
            markdown=md,
            attrs=attrs,
        )
        logger.info("sync-siyuan: synced session %r → %s (doc_id=%s)", sid, doc_path, doc_id)

        if daily_note and resolved_daily_tpl:
            try:
                from datetime import datetime, timezone
                daily_path = resolve_path_template(
                    resolved_daily_tpl,
                    session_id=sid,
                    title=None,
                    now=datetime.now(timezone.utc),
                )
                client.append_daily_note_link(
                    notebook=resolved_notebook,
                    daily_path=daily_path,
                    doc_id=doc_id,
                    title=analysis.title or sid,
                )
                logger.info("sync-siyuan: backlink added to daily note %s", daily_path)
            except Exception as exc:
                logger.warning("sync-siyuan: could not write daily note backlink: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# Message handler
# ──────────────────────────────────────────────────────────────────────────────


def make_message_handler(
    cfg: Any,
    model_cache: Optional[ModelCache] = None,
    model_idle_timeout_seconds: float = 600.0,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """Return the async message handler coroutine for pawn-queue.

    The handler:
    1. Reads ``payload["command"]`` to determine which function to call.
    2. Merges the remaining payload keys over per-command defaults.
    3. Calls :func:`dispatch` (CPU-bound work runs in a thread executor).
    4. Acks the message on success, nacks (dead-letters) on failure.
    5. After each successful job, resets the model idle timer.  When no job
       arrives within *model_idle_timeout_seconds*, :meth:`ModelCache.release`
       is called to free GPU/CPU memory.

    Args:
        cfg: :class:`~pawn_diarize.core.config.AppConfig` instance.
        model_cache: Optional :class:`ModelCache` for engine reuse and
            idle-based release.  When *None* each job creates its own engines.
        model_idle_timeout_seconds: Seconds of inactivity after which cached
            models are released.  Defaults to 600 (10 minutes).

    Returns:
        Async callable suitable for ``consumer.listen(handler)``.
    """
    _idle_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    async def _idle_timeout() -> None:
        try:
            await asyncio.sleep(model_idle_timeout_seconds)
            if model_cache is not None and model_cache.has_models:
                logger.info(
                    "Model idle timeout (%.0f s) reached — releasing models from memory",
                    model_idle_timeout_seconds,
                )
                model_cache.release()
        except asyncio.CancelledError:
            pass

    def _reset_idle_timer() -> None:
        nonlocal _idle_task
        if model_cache is None:
            return
        if _idle_task is not None and not _idle_task.done():
            _idle_task.cancel()
        _idle_task = asyncio.create_task(_idle_timeout())

    async def handler(msg: Any) -> None:  # msg: pawn_queue.Message
        payload: Dict[str, Any] = dict(msg.payload)
        command: Optional[str] = payload.pop("command", None)

        if not command:
            logger.error(
                "Message %s has no 'command' key — sending to dead-letter", msg.id
            )
            await msg.nack()
            return

        command = command.strip().lower()
        if command not in COMMAND_DEFAULTS:
            logger.error(
                "Message %s: unsupported command %r — sending to dead-letter",
                msg.id,
                command,
            )
            await msg.nack()
            return

        params = _merge_params(command, payload)
        logger.info("Processing message %s: command=%r", msg.id, command)

        try:
            await dispatch(command, params, cfg, model_cache)
            logger.info("Message %s completed successfully — ack", msg.id)
            await msg.ack()
            _reset_idle_timer()
        except Exception as exc:
            logger.error(
                "Message %s failed: %s — sending to dead-letter", msg.id, exc,
                exc_info=True,
            )
            await msg.nack()

    return handler


# ──────────────────────────────────────────────────────────────────────────────
# Listener bootstrap
# ──────────────────────────────────────────────────────────────────────────────

#: Default topic name when none is configured.
DEFAULT_TOPIC = "pawn-diarize-jobs"
#: Default consumer registration name.
DEFAULT_CONSUMER_NAME = "pawn-diarize-listener"


async def start_listener(
    cfg: Any,
    topic_override: Optional[str] = None,
    consumer_name_override: Optional[str] = None,
) -> None:
    """Set up pawn-queue and block until cancelled.

    Reads the ``diarize_queue:`` section of :class:`~pawn_diarize.core.config.AppConfig` to
    build the :class:`pawn_queue.PawnQueue` instance, registers a consumer,
    and calls ``consumer.listen(handler)`` which blocks until the asyncio task
    is cancelled (e.g. via ``KeyboardInterrupt``).

    Args:
        cfg: Active :class:`~pawn_diarize.core.config.AppConfig` instance.
        topic_override: When supplied, overrides the topic name from config.
        consumer_name_override: When supplied, overrides the consumer name.

    Raises:
        RuntimeError: If the ``diarize_queue:`` section is missing from ``pawnai.yaml``.
    """
    try:
        from pawn_queue import PawnQueueBuilder
    except ImportError as exc:
        raise ImportError(
            "pawn-queue is not installed. Run: uv pip install pawn-queue"
        ) from exc

    queue_cfg: Optional[Dict[str, Any]] = cfg.get_queue_config()
    if queue_cfg is None:
        raise RuntimeError(
            "No 'diarize_queue:' section found in pawnai.yaml. "
            "Add a diarize_queue: section with at minimum 'bucket_name'. "
            "S3 credentials are read from the top-level 's3:' section."
        )

    # S3 credentials come from the shared top-level s3: section.
    s3_cfg: Dict[str, Any] = cfg.get_s3_config() or {}
    if not s3_cfg:
        raise RuntimeError(
            "No 's3:' section found in .pawn-diarize.yml. "
            "The queue listener requires S3 credentials in the top-level 's3:' section."
        )

    topic = topic_override or queue_cfg.get("topic", DEFAULT_TOPIC)
    consumer_name = consumer_name_override or queue_cfg.get("consumer_name", DEFAULT_CONSUMER_NAME)

    # Queue-specific settings: bucket_name, polling, concurrency
    bucket_name: str = queue_cfg.get("bucket_name", "pawn-diarize-queue")
    polling_section: Dict[str, Any] = queue_cfg.get("polling", {})
    concurrency_section: Dict[str, Any] = queue_cfg.get("concurrency", {})

    # Map pawn-diarize s3 field names → pawn-queue builder parameters
    endpoint_url: str = s3_cfg.get("endpoint_url", "http://localhost:9000")
    use_ssl: bool = bool(s3_cfg.get("verify_ssl", s3_cfg.get("use_ssl", False)))

    builder = PawnQueueBuilder()
    builder = builder.s3(
        endpoint_url=endpoint_url,
        bucket_name=bucket_name,
        access_key=s3_cfg.get("access_key", s3_cfg.get("aws_access_key_id", "")),
        secret_key=s3_cfg.get("secret_key", s3_cfg.get("aws_secret_access_key", "")),
        region_name=s3_cfg.get("region", s3_cfg.get("region_name", "us-east-1")),
        use_ssl=use_ssl,
    )

    if polling_section:
        builder = builder.polling(**{
            k: v for k, v in polling_section.items()
            if k in (
                "interval_seconds",
                "max_messages_per_poll",
                "visibility_timeout_seconds",
                "lease_refresh_interval_seconds",
                "jitter_max_ms",
            )
        })

    if concurrency_section.get("strategy"):
        builder = builder.concurrency(strategy=concurrency_section["strategy"])

    # Model idle timeout: release loaded engines after N minutes of inactivity.
    # Configured under models.model_idle_timeout_minutes in pawnai.yaml.
    model_idle_timeout_minutes: float = float(
        cfg.get("model_idle_timeout_minutes", 10)
    )
    model_idle_timeout_seconds: float = model_idle_timeout_minutes * 60.0
    model_cache = ModelCache()

    logger.info(
        "Starting pawn-queue listener | topic=%r consumer=%r endpoint=%s bucket=%s "
        "model_idle_timeout=%.0f min",
        topic,
        consumer_name,
        endpoint_url,
        bucket_name,
        model_idle_timeout_minutes,
    )

    async with await builder.build() as pq:
        # Create topic if it doesn't exist yet
        try:
            await pq.create_topic(topic)
            logger.info("Topic %r created (or already exists)", topic)
        except Exception as exc:
            logger.warning("Could not create topic %r: %s", topic, exc)

        consumer = await pq.register_consumer(consumer_name, topics=[topic])
        handler = make_message_handler(
            cfg,
            model_cache=model_cache,
            model_idle_timeout_seconds=model_idle_timeout_seconds,
        )

        logger.info("Listening on topic %r as consumer %r …", topic, consumer_name)
        try:
            await consumer.listen(handler)
        except asyncio.CancelledError:
            logger.info("Listener cancelled — shutting down cleanly")
            model_cache.release()
