"""Queue listener for pawn-server.

Consumes messages from a pawn-queue S3-backed topic and dispatches them
to the PydanticAgent.  Results are persisted in the ``agent_runs`` table
so they can be inspected later (the listener is headless — no terminal).

Message format::

    {
        "command": "run",
        "prompt": "Summarise session abc123 and push to SiYuan",
        "session_id": "abc123",   // optional
        "model": "openai:gpt-4o"  // optional
    }
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from copy import deepcopy
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Per-command defaults
# ──────────────────────────────────────────────────────────────────────────────

COMMAND_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "run": {
        "prompt": "",       # required — producer MUST supply
        "session_id": None,
        "model": None,
    },
}


def _history_mode(cfg: Any) -> str:
    return getattr(cfg, "history_mode", "raw")


def _history_kwargs(cfg: Any) -> Dict[str, Any]:
    return {
        "strip_thinking": getattr(cfg, "strip_thinking", True),
        "recent_turns": getattr(cfg, "history_recent_turns", 4),
        "replay_max_tokens": getattr(cfg, "history_replay_max_tokens", 8000),
        "max_text_chars": getattr(cfg, "history_max_text_chars", 500),
        "sanitize_leaked_thoughts": getattr(cfg, "history_sanitize_leaked_thoughts", True),
    }


def _merge_params(command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge *payload* over per-command defaults."""
    merged = dict(COMMAND_DEFAULTS[command])
    merged.update(payload)
    return merged


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

    Heavy work runs in a thread executor to avoid blocking the event loop.
    """
    if command != "run":
        raise ValueError(f"Unsupported command: {command!r}")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run_prompt, params, cfg, message_id)


# ──────────────────────────────────────────────────────────────────────────────
# Runners
# ──────────────────────────────────────────────────────────────────────────────

# Cache agents by (model, session_id) so session variables remain isolated.
_agent_cache: Dict[tuple[str, Optional[str]], Any] = {}


def _get_or_create_agent(
    cfg: Any,
    model_override: Optional[str] = None,
    *,
    session_id: Optional[str] = None,
) -> Any:
    """Return a cached PydanticAgent, creating one if needed.

    Session-bound agents are required for session variable persistence
    (e.g. ``listen_only``) because :class:`SessionVars` persists only when
    instantiated with a session id.
    """
    from pawn_agent.core.pydantic_agent import PydanticAgent  # noqa: PLC0415

    model_key = model_override or cfg.pydantic_model
    cache_key = (model_key, session_id)

    if cache_key not in _agent_cache:
        agent_cfg = cfg
        if model_override:
            from pawn_agent.utils.model_utils import _apply_model_override  # noqa: PLC0415

            agent_cfg = deepcopy(cfg)
            _apply_model_override(agent_cfg, model_override)
        _agent_cache[cache_key] = PydanticAgent(cfg=agent_cfg, session_id=session_id)

    return _agent_cache[cache_key]


def _run_prompt(
    params: Dict[str, Any],
    cfg: Any,
    message_id: Optional[str] = None,
) -> None:
    """Execute a prompt through the PydanticAgent and persist the result."""
    from pawn_agent.utils.db import create_agent_run, update_agent_run  # noqa: PLC0415
    from pawn_agent.core.session_store import append_turn, build_replay_history, load_history  # noqa: PLC0415

    prompt: str = params.get("prompt", "")
    session_id: Optional[str] = params.get("session_id")
    model_override: Optional[str] = params.get("model")

    effective_model = model_override or cfg.pydantic_model

    # Create a history record before any validation so every message is traceable
    run_id = create_agent_run(
        cfg.db_dsn,
        message_id=message_id,
        command="run",
        prompt=prompt,
        session_id=session_id,
        model=effective_model,
    )

    if not prompt:
        update_agent_run(
            cfg.db_dsn,
            run_id,
            "failed",
            error="Message is missing required 'prompt' field",
        )
        raise ValueError("Message is missing required 'prompt' field")

    # Mark as running
    update_agent_run(cfg.db_dsn, run_id, "running")

    # Prepend session hint so the agent can locate the right conversation
    effective_prompt = prompt
    if session_id:
        effective_prompt = f"[Session ID: {session_id}]\n{prompt}"

    # Load conversation history from the session store
    if _history_mode(cfg) == "raw":
        history = load_history(
            session_id or "",
            cfg.db_dsn,
            strip_thinking=getattr(cfg, "strip_thinking", True),
        )
    else:
        history = build_replay_history(
            session_id or "",
            cfg.db_dsn,
            **_history_kwargs(cfg),
        )

    try:
        agent = _get_or_create_agent(cfg, model_override, session_id=session_id)
        result = agent.run_sync(effective_prompt, message_history=history)
        response = result.output

        # Persist this turn — idempotent via source_id = message_id
        source_id = message_id or str(uuid.uuid4())
        append_turn(source_id, session_id or "", list(result.new_messages()), cfg.db_dsn)

        update_agent_run(cfg.db_dsn, run_id, "completed", response=response)
        logger.info("Run %s completed (%d chars)", run_id, len(response))
    except Exception as exc:
        update_agent_run(cfg.db_dsn, run_id, "failed", error=str(exc))
        raise


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
