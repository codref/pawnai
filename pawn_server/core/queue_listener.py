"""Queue listener for pawn-server.

Consumes messages from a pawn-queue S3-backed topic and dispatches them
to registered command handlers.  Results are persisted in the
``agent_runs`` table so they can be inspected later (the listener is
headless — no terminal).

Message format (published by pawn-diarize ``chain_agent``)::

    {
        "command": "run",
        "prompt": "Summarise session abc123 and push to SiYuan",
        "session_id": "abc123",   // required — diarization session name
        "model": "openai:gpt-4o"  // optional per-message override
    }

The ``session_id`` is the diarization session name (not a conversation UUID).
It is passed directly to the LangGraph registry so that agent tools such as
``query_conversation`` can look up the correct transcript in the database.
Subsequent ``run`` messages for the same session continue the same
LangGraph conversation context.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any, Callable, Coroutine, Dict, Optional

from pawn_agent.core.langgraph_registry import LangGraphSessionRegistry
from pawn_agent.utils.db import create_agent_run, update_agent_run
from pawn_agent.utils.model_utils import _apply_model_override

logger = logging.getLogger(__name__)

# Module-level registry — one LangGraph session per diarization session_id.
_registry = LangGraphSessionRegistry()

# ──────────────────────────────────────────────────────────────────────────────
# Per-command defaults
# ──────────────────────────────────────────────────────────────────────────────

COMMAND_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "run": {"prompt": None, "session_id": None, "model": None},
}


def _merge_params(command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge *payload* over per-command defaults."""
    merged = dict(COMMAND_DEFAULTS[command])
    merged.update(payload)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# "run" handler
# ──────────────────────────────────────────────────────────────────────────────


async def _run_langgraph(
    params: Dict[str, Any],
    cfg: Any,
    message_id: Optional[str] = None,
) -> None:
    """Execute a ``run`` command via the LangGraph session registry.

    Creates an ``agent_runs`` row immediately (so every attempt is tracked),
    then validates required fields.  On any failure the row is marked *failed*
    and the exception re-raised so the caller can nack the message.
    """
    prompt: Optional[str] = params.get("prompt") or None
    session_id: Optional[str] = params.get("session_id") or None
    model: Optional[str] = params.get("model") or None

    effective_cfg = cfg
    if model:
        effective_cfg = copy.copy(cfg)
        _apply_model_override(effective_cfg, model)

    run_id = create_agent_run(
        cfg.db_dsn,
        message_id=message_id,
        command="run",
        prompt=prompt,
        session_id=session_id,
        model=effective_cfg.pydantic_model,
    )
    update_agent_run(cfg.db_dsn, run_id, "running")

    try:
        if not prompt:
            raise ValueError("'prompt' is required for the 'run' command")
        if not session_id:
            raise ValueError(
                "'session_id' is required for the 'run' command — "
                "it must be the diarization session name used by agent tools"
            )

        reply = await _registry.handle_turn(session_id, prompt, effective_cfg, cfg.db_dsn)
        update_agent_run(cfg.db_dsn, run_id, "completed", response=reply)
    except Exception as exc:
        update_agent_run(cfg.db_dsn, run_id, "failed", error=str(exc))
        raise


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
        await _run_langgraph(params, cfg, message_id)
        return

    raise NotImplementedError(f"Command {command!r} has no handler registered")


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
            await dispatch(command, params, cfg, message_id=msg.id)
            logger.info("Message %s completed successfully — ack", msg.id)
            await msg.ack()
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
        raise ImportError(
            "pawn-queue is not installed. Run: uv pip install pawn-queue"
        ) from exc

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
            await consumer.listen(handler)
        except asyncio.CancelledError:
            logger.info("Listener cancelled — shutting down cleanly")
