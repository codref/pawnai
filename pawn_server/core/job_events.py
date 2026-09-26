"""In-process pub/sub for job status changes (feeds ``GET /v1/jobs/events``).

Only events raised inside this ``pawn-server`` process are delivered. Clients
must still reconcile with ``GET /v1/jobs`` after reconnecting.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_BACKLOG = 200


class JobEventBus:
    """Fan-out of job events to any number of asyncio subscribers."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        self._loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_BACKLOG)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def _deliver(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("job event subscriber backlog full; dropping event")

    def publish(self, event: dict[str, Any]) -> None:
        """Publish from the event loop or from a worker thread."""
        if not self._subscribers:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and running is self._loop:
            self._deliver(event)
        elif self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._deliver, event)


job_events = JobEventBus()


def publish_job_event(
    job_id: str,
    status: str,
    *,
    kind: str = "ask",
    conversation: Optional[str] = None,
    error_code: Optional[str] = None,
) -> None:
    job_events.publish(
        {
            "job_id": job_id,
            "status": status,
            "kind": kind,
            "conversation": conversation,
            "error_code": error_code,
        }
    )
