"""Publish jobs to the diarize listener queue.

Prefers ``queue_producers[api.upload_audio_target]`` (default ``diarize``).
When that named producer is missing — common when only ``diarize_queue:`` is
configured for ``pawn-diarize listen`` — falls back to publishing with the
``diarize_queue`` topic/bucket so Obsidian upload/reprocess still work.
"""

from __future__ import annotations

from typing import Any, Optional


async def publish_to_diarize_queue(
    cfg: Any,
    *,
    command: str,
    payload: dict[str, Any],
) -> str:
    """Publish ``{command, **payload}`` for the diarize worker.

    Returns a receipt string, or an ``Error: …`` message (same convention as
    :func:`push_queue_message_impl`).
    """
    from pawn_agent.tools.push_queue_message import push_queue_message_impl

    if "command" in payload:
        return "Error: payload must not contain a 'command' key."
    if not command:
        return "Error: command is required."

    target = str(getattr(getattr(cfg, "api", None), "upload_audio_target", "diarize") or "diarize")
    producers = getattr(cfg, "queue_producers", None) or {}
    if target in producers:
        return await push_queue_message_impl(
            cfg, target=target, command=command, payload=payload
        )

    dq = getattr(cfg, "diarize_queue", None)
    if dq is None:
        known = ", ".join(sorted(producers)) if producers else "(none)"
        return (
            f"Error: unknown queue target {target!r}. Known targets: {known}. "
            f"Add queue_producers.{target} (see docs/OBSIDIAN_AGENT.md) or configure "
            f"diarize_queue: for the listener."
        )

    return await _publish_via_diarize_queue(cfg, dq, command=command, payload=payload)


async def _publish_via_diarize_queue(
    cfg: Any,
    dq: Any,
    *,
    command: str,
    payload: dict[str, Any],
) -> str:
    try:
        from pawn_queue import PawnQueueBuilder
    except ImportError as exc:
        return f"Error: pawn-queue is not installed ({exc})."

    s3_cfg = getattr(cfg, "s3_config", None)
    if not s3_cfg and hasattr(cfg, "get_s3_config"):
        s3_cfg = cfg.get_s3_config()
    if not s3_cfg:
        return "Error: no s3: section found in pawnai.yaml."

    dq_data = dq.model_dump() if hasattr(dq, "model_dump") else dict(dq)
    topic = str(dq_data.get("topic") or "pawn-diarize-jobs")
    bucket = str(dq_data.get("bucket_name") or "")
    if not bucket:
        return "Error: diarize_queue.bucket_name is required."

    envelope = {"command": command, **payload}
    endpoint_url: str = s3_cfg.get("endpoint_url", "http://localhost:9000")
    use_ssl: bool = bool(s3_cfg.get("verify_ssl", s3_cfg.get("use_ssl", False)))

    builder = PawnQueueBuilder()
    builder = builder.s3(
        endpoint_url=endpoint_url,
        bucket_name=bucket,
        access_key=s3_cfg.get("access_key", s3_cfg.get("aws_access_key_id", "")),
        secret_key=s3_cfg.get("secret_key", s3_cfg.get("aws_secret_access_key", "")),
        region_name=s3_cfg.get("region", s3_cfg.get("region_name", "us-east-1")),
        use_ssl=use_ssl,
    )

    polling = dq_data.get("polling") or {}
    if isinstance(polling, dict) and polling:
        builder = builder.polling(
            **{
                k: v
                for k, v in polling.items()
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

    concurrency = dq_data.get("concurrency") or {}
    if isinstance(concurrency, dict) and concurrency.get("strategy"):
        builder = builder.concurrency(strategy=concurrency["strategy"])

    producer_name = "pawn-server-diarize-publisher"
    try:
        async with await builder.build() as pq:
            await pq.create_topic(topic)
            producer = await pq.register_producer(producer_name)
            message_id = await producer.publish(topic, envelope)
    except Exception as exc:
        return f"Error publishing to diarize_queue topic={topic!r}: {exc}"

    return (
        f"Published to diarize_queue topic={topic} bucket={bucket} "
        f"message_id={message_id}"
    )


def resolve_diarize_publish_source(cfg: Any) -> Optional[str]:
    """Return ``queue_producers:<name>`` or ``diarize_queue`` for diagnostics."""
    target = str(getattr(getattr(cfg, "api", None), "upload_audio_target", "diarize") or "diarize")
    producers = getattr(cfg, "queue_producers", None) or {}
    if target in producers:
        return f"queue_producers:{target}"
    if getattr(cfg, "diarize_queue", None) is not None:
        return "diarize_queue"
    return None
