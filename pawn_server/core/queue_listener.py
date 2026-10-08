"""Queue listener for pawn-server.

Consumes messages from a pawn-queue S3-backed topic and dispatches them
to registered command handlers.  Results are persisted in the
``agent_runs`` table so they can be inspected later (the listener is
headless — no terminal).

Message format (published by pawn-diarize ``chain_agent``)::

    {
        "command": "run",
        "prompt": "Summarise session abc123",
        "session_id": "abc123",   // required — diarization session name
        "model": "openai:gpt-4o"  // optional per-message override
    }

The ``session_id`` is the diarization session name (not a conversation UUID).
It is passed directly to the sallm registry so that agent tools such as
``session_transcript`` can look up the correct transcript in the database.
Subsequent ``run`` messages for the same session continue the same
sallm conversation context.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, Optional

from pawn_agent.core.agent_runner import run_agent_turn
from pawn_agent.core.sallm_registry import SallmSessionRegistry

logger = logging.getLogger(__name__)

# Module-level registry — one sallm session per diarization session_id.
_registry = SallmSessionRegistry()

# ──────────────────────────────────────────────────────────────────────────────
# Per-command defaults
# ──────────────────────────────────────────────────────────────────────────────

COMMAND_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "run": {
        "prompt": None,
        "session_id": None,
        "model": None,
        "parent_run_id": None,
        "depth": 0,
        "event_id": None,
    },
    "vault_run": {
        "prompt": None,
        "session_id": None,
        "model": None,
        "request_id": None,
    },
    "session_completed": {"session_id": None, "force": False},
    # Gallery-linked People/ note refresh (appearances + optional facts).
    "speakers_refresh": {"session_id": None, "force": False},
}


def _merge_params(command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge *payload* over per-command defaults."""
    merged = dict(COMMAND_DEFAULTS[command])
    merged.update(payload)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# "run" handler
# ──────────────────────────────────────────────────────────────────────────────


async def _run_sallm(
    params: Dict[str, Any],
    cfg: Any,
    message_id: Optional[str] = None,
    *,
    command: str = "run",
    source: str = "queue",
) -> None:
    """Execute a ``run`` / ``vault_run`` command via the sallm session registry.

    Creates an ``agent_runs`` row immediately (so every attempt is tracked),
    then validates required fields.  On any failure the row is marked *failed*
    and the exception re-raised so the caller can nack the message.
    """
    prompt: Optional[str] = params.get("prompt") or None
    session_id: Optional[str] = params.get("session_id") or None
    model: Optional[str] = params.get("model") or None
    parent = params.get("parent_run_id") or None
    event_id = params.get("event_id") or None
    try:
        depth = int(params.get("depth") or 0)
    except (TypeError, ValueError):
        depth = 0

    await run_agent_turn(
        cfg=cfg,
        registry=_registry,
        message_id=message_id,
        source=source,
        command=command,
        prompt=prompt,
        session_id=session_id,
        model=model,
        parent_run_id=parent,
        depth=depth,
        event_id=event_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────────


async def dispatch(
    command: str,
    params: Dict[str, Any],
    cfg: Any,
    message_id: Optional[str] = None,
) -> None:
    """Route *command* to the correct handler.

    Raises :class:`ValueError` for any command not registered in
    :data:`COMMAND_DEFAULTS`.
    """
    if command not in COMMAND_DEFAULTS:
        raise ValueError(f"Unsupported command: {command!r}")

    if command == "run":
        await _run_sallm(params, cfg, message_id, command="run", source="queue")
        return
    if command == "vault_run":
        await _run_sallm(params, cfg, message_id, command="vault_run", source="vault")
        return
    if command == "session_completed":
        await _session_completed(params, cfg)
        return
    if command == "speakers_refresh":
        await _speakers_refresh(params, cfg)
        return

    raise NotImplementedError(f"Command {command!r} has no handler registered")


def _reject_self_run(cfg: Any, params: Dict[str, Any]) -> Optional[str]:
    """Return a policy reason when a child run must be dropped."""
    parent = params.get("parent_run_id")
    if not parent:
        return None
    from datetime import datetime, timedelta, timezone  # noqa: PLC0415

    from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415
    from pawn_agent.core.coworker.autonomy import reject_reason  # noqa: PLC0415

    autonomy = cfg.coworker.autonomy
    try:
        depth = int(params.get("depth") or 0)
    except (TypeError, ValueError):
        depth = 0
    event_id = params.get("event_id") or ""
    event_count = itemdb.count_runs_for_event(cfg.db_dsn, event_id) if event_id else 0
    day_count = itemdb.count_self_runs_since(
        cfg.db_dsn, datetime.now(timezone.utc) - timedelta(days=1)
    )
    duplicate = itemdb.recent_duplicate_run(
        cfg.db_dsn, str(params.get("prompt") or ""), str(params.get("session_id") or "")
    )
    reason = reject_reason(
        depth=depth,
        event_count=event_count,
        day_count=day_count,
        max_depth=int(autonomy.max_depth),
        max_per_event=int(autonomy.max_self_jobs_per_event),
        max_per_day=int(autonomy.max_self_jobs_per_day),
        duplicate=duplicate,
    )
    if reason:
        try:
            itemdb.record_decision(
                cfg.db_dsn,
                event_kind="self_queue",
                policy_decision="deny",
                event_id=event_id or None,
                proposed_action=str(params.get("prompt") or "")[:500],
                outcome=reason,
            )
        except Exception as exc:
            logger.warning("could not record self-queue denial: %s", exc)
    return reason


def _session_segment_count(cfg: Any, session_id: str) -> int:
    """Current transcription segment count for *session_id* (0 if unknown)."""
    try:
        from pawn_diarize.core.database import get_engine, init_db, load_session_state

        engine = get_engine(cfg.db_dsn)
        init_db(engine)
        *_rest, segment_count = load_session_state(session_id, engine)
        return int(segment_count)
    except Exception as exc:
        logger.debug("segment count for %s unavailable: %s", session_id, exc)
        return 0


def _already_session_completed(
    cfg: Any, session_id: str, segment_count: int
) -> bool:
    """True when a completed/running extract already covers this segment count.

    Guards duplicate finalize redelivery and overlapping queue workers. A later
    chunk that grows the transcript (higher segment count) is allowed through.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from pawn_agent.utils.db import AgentRun
    from pawn_core.database import get_engine

    marker = f"segments={segment_count}"
    stale_before = datetime.now(timezone.utc) - timedelta(hours=1)
    with Session(get_engine(cfg.db_dsn)) as db:
        rows = db.scalars(
            select(AgentRun)
            .where(
                AgentRun.session_id == session_id,
                AgentRun.command == "session_completed",
                AgentRun.status.in_(("completed", "running")),
            )
            .order_by(AgentRun.created_at.desc())
            .limit(20)
        ).all()
    for row in rows:
        blob = f"{row.prompt or ''}\n{row.response or ''}"
        if row.status == "completed" and marker in blob:
            return True
        if row.status == "running":
            started = row.started_at or row.created_at
            if started is not None:
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                if started >= stale_before:
                    # Another worker is mid-extract — avoid overlap.
                    return True
    return False


async def _session_completed(params: Dict[str, Any], cfg: Any) -> None:
    session_id = params.get("session_id") or None
    if not session_id:
        raise ValueError("session_completed requires session_id")
    if not getattr(cfg.coworker, "enabled", False):
        logger.info("session_completed for %s ignored; coworker disabled", session_id)
        return

    force = bool(params.get("force"))
    segment_count = _session_segment_count(cfg, session_id)
    if not force and _already_session_completed(cfg, session_id, segment_count):
        logger.info(
            "session_completed for %s skipped; already covered at %d segment(s)",
            session_id,
            segment_count,
        )
        try:
            await _speakers_refresh({"session_id": session_id}, cfg)
        except Exception as exc:
            logger.error(
                "speakers_refresh after skipped session_completed failed: %s",
                exc,
                exc_info=True,
            )
        return

    from pawn_agent.core.coworker.pipeline import process_session  # noqa: PLC0415
    from pawn_agent.utils.db import create_agent_run, update_agent_run  # noqa: PLC0415

    marker = f"segments={segment_count}"
    run_id = create_agent_run(
        cfg.db_dsn,
        source="queue",
        command="session_completed",
        prompt=marker,
        session_id=session_id,
        model=getattr(cfg, "chat_model_id", None) or "coworker",
    )
    update_agent_run(cfg.db_dsn, run_id, "running")
    try:
        result = await process_session(cfg, session_id)
        items = result.get("items") if isinstance(result, dict) else None
        update_agent_run(
            cfg.db_dsn,
            run_id,
            "completed",
            response=f"{marker} items={items}",
        )
    except Exception as exc:
        update_agent_run(cfg.db_dsn, run_id, "failed", error=str(exc)[:2000])
        raise

    # People bios refresh after the item extract pass (same event, isolated try).
    try:
        await _speakers_refresh({"session_id": session_id}, cfg)
    except Exception as exc:
        logger.error("speakers_refresh after session_completed failed: %s", exc, exc_info=True)


async def _speakers_refresh(params: Dict[str, Any], cfg: Any) -> None:
    """Update People/ notes for gallery speakers in a finished session."""
    session_id = params.get("session_id") or None
    if not session_id:
        raise ValueError("speakers_refresh requires session_id")
    force = bool(params.get("force"))
    from pawn_agent.core.people.refresh import refresh_people_for_session  # noqa: PLC0415

    result = await refresh_people_for_session(cfg, session_id, force=force)
    logger.info("speakers_refresh %s → %s", session_id, result)


# ──────────────────────────────────────────────────────────────────────────────
# Message handler
# ──────────────────────────────────────────────────────────────────────────────


def make_message_handler(
    cfg: Any,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """Return the async message handler for ``consumer.listen(handler)``.

    1. Reads ``payload["command"]`` to determine which function to call.
    2. Merges remaining payload over per-command defaults.
    3. Calls :func:`dispatch` (CPU-bound work runs in a thread executor).
    4. Acks on success, nacks (dead-letters) on failure.
    """

    async def handler(msg: Any) -> None:  # msg: pawn_queue.Message
        payload: Dict[str, Any] = dict(msg.payload)
        command: Optional[str] = payload.pop("command", None)

        if not command:
            logger.error("Message %s has no 'command' key — sending to dead-letter", msg.id)
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

        if command == "run":
            reason = _reject_self_run(cfg, params)
            if reason:
                logger.info("Message %s dropped by coworker policy: %s", msg.id, reason)
                await msg.ack()
                return

        try:
            await dispatch(command, params, cfg, message_id=msg.id)
            logger.info("Message %s completed successfully — ack", msg.id)
            await msg.ack()
        except Exception as exc:
            logger.error(
                "Message %s failed: %s — sending to dead-letter",
                msg.id,
                exc,
                exc_info=True,
            )
            await msg.nack()

    return handler


# ──────────────────────────────────────────────────────────────────────────────
# Listener bootstrap
# ──────────────────────────────────────────────────────────────────────────────

#: Default topic name when none is configured.
DEFAULT_TOPIC = "pawn-agent-jobs"
#: Default consumer registration name.
DEFAULT_CONSUMER_NAME = "pawn-agent-listener"


async def start_listener(
    cfg: Any,
    topic_override: Optional[str] = None,
    consumer_name_override: Optional[str] = None,
) -> None:
    """Set up pawn-queue and block until cancelled.

    Reads ``s3_config`` and ``queue_config`` from :class:`AgentConfig` to
    build a :class:`pawn_queue.PawnQueue` instance, registers a consumer,
    and calls ``consumer.listen(handler)`` which blocks until the asyncio
    task is cancelled (e.g. via ``KeyboardInterrupt``).
    """
    try:
        from pawn_queue import PawnQueueBuilder
    except ImportError as exc:
        raise ImportError("pawn-queue is not installed. Run: uv pip install pawn-queue") from exc

    queue_cfg: Optional[Dict[str, Any]] = cfg.queue_config
    if queue_cfg is None:
        raise RuntimeError(
            "No 'agent_queue:' section found in pawnai.yaml. "
            "Add an agent_queue: section with at minimum 'bucket_name'. "
            "S3 credentials are read from the top-level 's3:' section."
        )

    s3_cfg: Optional[Dict[str, Any]] = cfg.s3_config
    if not s3_cfg:
        raise RuntimeError(
            "No 's3:' section found in pawnai.yaml. "
            "The queue listener requires S3 credentials in the top-level 's3:' section."
        )

    topic = topic_override or queue_cfg.get("topic", DEFAULT_TOPIC)
    consumer_name = consumer_name_override or queue_cfg.get("consumer_name", DEFAULT_CONSUMER_NAME)

    bucket_name: str = queue_cfg.get("bucket_name", "pawn-agent-queue")
    polling_section: Dict[str, Any] = queue_cfg.get("polling", {})
    concurrency_section: Dict[str, Any] = queue_cfg.get("concurrency", {})

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
        builder = builder.polling(
            **{
                k: v
                for k, v in polling_section.items()
                if k
                in (
                    "interval_seconds",
                    "max_messages_per_poll",
                    "visibility_timeout_seconds",
                    "lease_refresh_interval_seconds",
                    "jitter_max_ms",
                )
            }
        )

    if concurrency_section.get("strategy"):
        builder = builder.concurrency(strategy=concurrency_section["strategy"])

    logger.info(
        "Starting pawn-server listener | topic=%r consumer=%r endpoint=%s bucket=%s",
        topic,
        consumer_name,
        endpoint_url,
        bucket_name,
    )

    async with await builder.build() as pq:
        try:
            await pq.create_topic(topic)
            logger.info("Topic %r created (or already exists)", topic)
        except Exception as exc:
            logger.warning("Could not create topic %r: %s", topic, exc)

        consumer = await pq.register_consumer(consumer_name, topics=[topic])
        handler = make_message_handler(cfg)

        logger.info("Listening on topic %r as consumer %r …", topic, consumer_name)
        try:
            from pawn_core.queue_control import listen_respecting_pause  # noqa: PLC0415

            await listen_respecting_pause(consumer, handler, pq._client, topic)
        except asyncio.CancelledError:
            logger.info("Listener cancelled — shutting down cleanly")
