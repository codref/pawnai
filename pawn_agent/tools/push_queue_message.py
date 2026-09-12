"""Queue publish (``queue_push`` CliTool)."""

from __future__ import annotations

from typing import Any

from pawn_agent.utils.config import AgentConfig


async def push_queue_message_impl(
    cfg: AgentConfig,
    target: str,
    command: str,
    payload: dict[str, Any],
) -> str:
    """Publish a message to the named *target*.

    Validates configuration, builds the queue, creates the topic, registers a
    producer, and publishes ``{ "command": command, **payload }``.

    Returns a short receipt string or an error message.
    """
    if not cfg.queue_producers:
        return "Error: no queue_producers configured in pawnai.yaml."
    producer_cfg = cfg.queue_producers.get(target)
    if not producer_cfg:
        known = ", ".join(cfg.queue_producers.keys())
        return f"Error: unknown queue target {target!r}. Known targets: {known}"
    if not command:
        return "Error: command is required."
    if not isinstance(payload, dict):
        return "Error: payload must be a JSON object (dict)."
    if "command" in payload:
        return "Error: payload must not contain a 'command' key."

    s3_cfg = cfg.s3_config
    if not s3_cfg:
        return "Error: no s3: section found in pawnai.yaml."

    try:
        from pawn_queue import PawnQueueBuilder
    except ImportError as exc:
        return f"Error: pawn-queue is not installed ({exc})."

    envelope = {"command": command, **payload}

    endpoint_url: str = s3_cfg.get("endpoint_url", "http://localhost:9000")
    use_ssl: bool = bool(s3_cfg.get("verify_ssl", s3_cfg.get("use_ssl", False)))

    builder = PawnQueueBuilder()
    builder = builder.s3(
        endpoint_url=endpoint_url,
        bucket_name=producer_cfg.bucket_name,
        access_key=s3_cfg.get("access_key", s3_cfg.get("aws_access_key_id", "")),
        secret_key=s3_cfg.get("secret_key", s3_cfg.get("aws_secret_access_key", "")),
        region_name=s3_cfg.get("region", s3_cfg.get("region_name", "us-east-1")),
        use_ssl=use_ssl,
    )

    polling_section = producer_cfg.polling or {}
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

    concurrency_section = producer_cfg.concurrency or {}
    if concurrency_section.get("strategy"):
        builder = builder.concurrency(strategy=concurrency_section["strategy"])

    try:
        async with await builder.build() as pq:
            await pq.create_topic(producer_cfg.topic)
            producer_name = producer_cfg.producer_name or f"pawn-agent-producer-{target}"
            producer = await pq.register_producer(producer_name)
            message_id = await producer.publish(producer_cfg.topic, envelope)
    except Exception as exc:
        return f"Error publishing to {target!r}: {exc}"

    return f"Published to target={target} topic={producer_cfg.topic} " f"message_id={message_id}"
