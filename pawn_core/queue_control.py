"""Shared pawn-queue pause control and pause-aware listen loop."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

MessageHandler = Callable[[Any], Coroutine[Any, Any, None]]


def pause_marker_key(topic: str) -> str:
    """S3 key for the per-topic processing pause marker."""
    return f"{topic}/.paused"


async def read_pause_state(client: Any, topic: str) -> tuple[bool, Optional[str]]:
    """Return ``(paused, paused_at)`` for *topic* using the S3 pause marker."""
    key = pause_marker_key(topic)
    if not await client.object_exists(key):
        return False, None
    data = await client.get_object_json(key)
    if not isinstance(data, dict):
        return True, None
    return True, data.get("paused_at")


async def listen_respecting_pause(
    consumer: Any,
    handler: MessageHandler,
    client: Any,
    topic: str,
) -> None:
    """Like ``consumer.listen``, but skip claiming while ``{topic}/.paused`` exists.

    In-flight messages keep their leases refreshed so pause does not abandon
    work already claimed. New polls are suppressed until the marker is cleared
    (``pawn-server queue resume``).
    """
    try:
        interval = float(consumer._config.polling.interval_seconds)  # noqa: SLF001
    except Exception:
        interval = 5.0

    async def _poll_loop() -> None:
        was_paused = False
        while True:
            paused, _ = await read_pause_state(client, topic)
            if paused:
                if not was_paused:
                    logger.info(
                        "Queue topic %r is paused — not claiming new messages",
                        topic,
                    )
                    was_paused = True
                await asyncio.sleep(interval)
                continue
            if was_paused:
                logger.info("Queue topic %r resumed — claiming messages again", topic)
                was_paused = False

            try:
                messages = await consumer.poll()
                for msg in messages:
                    try:
                        await handler(msg)
                    except Exception as exc:
                        logger.error(
                            "Handler raised exception for message %s: %s",
                            msg.id,
                            exc,
                        )
                        if not getattr(msg, "_acked", False):
                            try:
                                await msg.nack()
                            except Exception:
                                pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Poll loop error: %s", exc)
            await asyncio.sleep(interval)

    await asyncio.gather(_poll_loop(), consumer._lease_refresher())  # noqa: SLF001
