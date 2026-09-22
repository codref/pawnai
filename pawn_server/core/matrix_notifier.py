"""Outbound Matrix notifier — consumes ``queue_producers.matrix`` notify messages.

Complements the inbound Matrix bot. SiYuan (and other producers) publish via
``queue_push``; this worker delivers short alerts to ``matrix_bot.notify_room_id``.

Prefer :func:`run_matrix_notifier_loop` with a shared client from
``start_matrix_bot`` so device identity is not duplicated.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def _deliver_notify(cfg: Any, client: Any, payload: dict[str, Any]) -> None:
    from pawn_server.core.matrix_bot import _send_text  # noqa: PLC0415

    room_id = getattr(cfg.matrix_bot, "notify_room_id", None) or ""
    if not room_id:
        raise RuntimeError("matrix_bot.notify_room_id is not configured — cannot deliver notify")
    text = str(payload.get("text") or "").strip()
    if not text:
        text = str(payload.get("message") or "").strip()
    if not text:
        raise ValueError("notify payload missing 'text'")
    event_id = await _send_text(client, room_id, text)
    logger.info(
        "Matrix notify delivered room=%s event_id=%s request_id=%s",
        room_id,
        event_id,
        payload.get("request_id"),
    )


async def run_matrix_notifier_loop(cfg: Any, client: Any) -> None:
    """Consume outbound notification queue using an already-logged-in client."""
    producers = cfg.queue_producers or {}
    target_name = getattr(cfg.siyuan_watcher, "matrix_target", None) or "matrix"
    producer_cfg = producers.get(target_name)
    if producer_cfg is None:
        if not producers:
            raise RuntimeError("No queue_producers configured for Matrix notifier")
        target_name, producer_cfg = next(iter(producers.items()))

    if not getattr(cfg.matrix_bot, "notify_room_id", None):
        raise RuntimeError("matrix_bot.notify_room_id is required for Matrix notifier")

    s3_cfg = cfg.s3_config
    if not s3_cfg:
        raise RuntimeError("s3: section required for Matrix notifier queue")

    from pawn_queue import PawnQueueBuilder  # noqa: PLC0415

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

    logger.info(
        "Matrix notifier listening topic=%s target=%s",
        producer_cfg.topic,
        target_name,
    )
    async with await builder.build() as pq:
        await pq.create_topic(producer_cfg.topic)
        consumer = await pq.register_consumer(
            "pawn-matrix-notifier",
            topics=[producer_cfg.topic],
        )

        async def handler(msg: Any) -> None:
            payload = dict(msg.payload)
            command = str(payload.pop("command", "") or "").strip().lower()
            if command not in ("notify", "alert", "message"):
                logger.warning("Matrix notifier ignoring unsupported command %r", command)
                await msg.ack()
                return
            try:
                await _deliver_notify(cfg, client, payload)
                await msg.ack()
            except Exception as exc:
                logger.error("Matrix notifier failed msg=%s: %s", msg.id, exc, exc_info=True)
                await msg.nack()

        await consumer.listen(handler)


async def start_matrix_notifier(cfg: Any) -> None:
    """Standalone notifier (own Matrix login). Prefer bot-shared loop in serve."""
    from pawn_server.core.matrix_bot import (  # noqa: PLC0415
        _build_client,
        _login,
        _patch_nio_sas_for_element,
        _require_nio,
        _validate_cfg,
    )

    _require_nio()
    _patch_nio_sas_for_element()
    mb = cfg.matrix_bot
    _validate_cfg(mb)
    client = _build_client(mb)
    try:
        await _login(client, mb)
        await run_matrix_notifier_loop(cfg, client)
    finally:
        await client.close()


def matrix_notifier_enabled(cfg: Any) -> bool:
    """True when outbound Matrix notify worker should run."""
    mb = cfg.matrix_bot
    if not mb.enabled:
        return False
    if not getattr(mb, "notify_room_id", None):
        return False
    producers = cfg.queue_producers or {}
    return bool(producers)
