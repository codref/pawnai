"""Queue listener for pawn-agent.

Subscribes to the ``agent-tasks`` topic (configurable) and processes each
message through the full agent pipeline:

1. Deserialise ``payload["request"]`` (free natural language) and
   ``payload.get("context", {})`` (structured hints like ``audio_path``).
2. Call :class:`~pawn_agent.agent.planner.AgentPlanner` to produce an
   explicit :class:`~pawn_agent.agent.planner.ExecutionPlan`.
3. Log the full plan before any execution begins.
4. Call :func:`~pawn_agent.agent.executor.execute_plan` to run each step.
5. ``msg.ack()`` on success, ``msg.nack()`` (dead-letter) on any error.

Usage::

    import asyncio
    from pawn_agent.config import AgentConfig
    from pawn_agent.listener import start_listener

    asyncio.run(start_listener(cfg, runner, planner))
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, Optional

from pawn_agent.agent.executor import execute_plan
from pawn_agent.agent.planner import AgentPlanner, ExecutionPlan
from pawn_agent.config import AgentConfig
from pawn_agent.skills.runner import SkillRunner

logger = logging.getLogger(__name__)


def make_message_handler(
    cfg: AgentConfig,
    planner: AgentPlanner,
    runner: SkillRunner,
) -> Callable[[Any], Coroutine[Any, Any, None]]:
    """Build the async message handler closure.

    Args:
        cfg:     Agent configuration.
        planner: Initialised :class:`AgentPlanner`.
        runner:  Initialised :class:`SkillRunner`.

    Returns:
        An async callable suitable for ``consumer.listen(handler)``.
    """

    async def handler(msg: Any) -> None:
        payload: Dict[str, Any] = dict(msg.payload)

        request: Optional[str] = payload.get("request")
        if not request or not request.strip():
            logger.error(
                "Message %s has no 'request' key or it is empty — sending to dead-letter",
                msg.id,
            )
            await msg.nack()
            return

        context: Dict[str, Any] = payload.get("context", {})
        if not isinstance(context, dict):
            logger.warning(
                "Message %s: 'context' is not a dict (%s) — treating as empty",
                msg.id,
                type(context).__name__,
            )
            context = {}

        logger.info(
            "Message %s received | request=%r | context_keys=%s",
            msg.id,
            request[:120],
            list(context),
        )

        try:
            # ── 1. Plan ───────────────────────────────────────────────────
            from pawn_agent.skills.registry import SkillRegistry  # avoid circular at module level

            skill_registry = runner._skills  # type: ignore[attr-defined]
            available_skills = skill_registry.describe_skills()

            plan: ExecutionPlan = planner.plan(
                user_request=request,
                available_skills=available_skills,
                context=context,
            )

            logger.info(
                "Message %s — plan produced (%d step(s)):\n%s",
                msg.id,
                len(plan.steps),
                json.dumps([s.model_dump() for s in plan.steps], indent=2),
            )

            # ── 2. Execute ────────────────────────────────────────────────
            result = await execute_plan(plan, runner, context=context, cfg=cfg)

            logger.info(
                "Message %s completed successfully — result_keys=%s",
                msg.id,
                list(result),
            )
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


async def start_listener(
    cfg: AgentConfig,
    planner: AgentPlanner,
    runner: SkillRunner,
) -> None:
    """Bootstrap pawn-queue and block until cancelled.

    Reads S3 and queue configuration from *cfg*, creates the topic if it
    doesn't exist, registers a consumer, and calls ``consumer.listen(handler)``
    which blocks until the asyncio task is cancelled (e.g. ``KeyboardInterrupt``).

    Args:
        cfg:     Active :class:`AgentConfig` instance.
        planner: Initialised :class:`AgentPlanner`.
        runner:  Initialised :class:`SkillRunner`.

    Raises:
        RuntimeError: If the ``queue:`` or ``s3:`` section is missing from the config.
        ImportError:  If ``pawn-queue`` is not installed.
    """
    try:
        from pawn_queue import PawnQueueBuilder  # lazy import
    except ImportError as exc:
        raise ImportError(
            "pawn-queue is not installed. Run: pip install pawn-queue"
        ) from exc

    queue_cfg: Optional[Dict[str, Any]] = cfg.get_queue_config()
    if queue_cfg is None:
        raise RuntimeError(
            "No 'queue:' section found in .pawnai.yml. "
            "Add a queue: section with at minimum 'bucket_name'."
        )

    s3_cfg: Dict[str, Any] = cfg.get_s3_config()
    if not s3_cfg:
        raise RuntimeError(
            "No 's3:' section found in .pawnai.yml. "
            "The agent listener requires S3 credentials in the top-level 's3:' section."
        )

    topic = cfg.topic
    consumer_name = cfg.consumer_name
    bucket_name: str = queue_cfg.get("bucket_name", "pawnai-queue")
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
        "Starting pawn-agent listener | topic=%r consumer=%r endpoint=%s bucket=%s",
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
        handler = make_message_handler(cfg, planner, runner)

        logger.info("Listening on topic %r as consumer %r …", topic, consumer_name)
        try:
            await consumer.listen(handler)
        except asyncio.CancelledError:
            logger.info("Listener cancelled — shutting down cleanly")
