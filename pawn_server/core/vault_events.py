"""In-process ring of vault writes for ``GET /v1/vault/events``.

Only events raised inside this ``pawn-server`` process are delivered. A short
ring covers the gap between long-polls; if the client falls behind the ring,
the response sets ``resync`` so the plugin syncs once without a path list.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RING_SIZE = 100


class VaultEventBus:
    """Monotonic vault-write events plus waiters for long-poll."""

    def __init__(self, maxlen: int = _RING_SIZE) -> None:
        self._mu = threading.Lock()
        self._seq = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._waiters: set[asyncio.Event] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def seq(self) -> int:
        with self._mu:
            return self._seq

    def close(self) -> None:
        """Wake every long-poll so shutdown is not stuck on an open request.

        Safe to call from a signal handler: waiters are set on the loop.
        """
        self._closed = True
        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._wake)

    def snapshot(self, since: int) -> dict[str, Any]:
        """Events with ``seq`` greater than *since*, without waiting."""
        with self._mu:
            return self._snapshot_locked(since)

    def publish(
        self,
        paths: list[str],
        *,
        source: str,
        run_id: Optional[str] = None,
    ) -> int:
        """Append one event and wake long-polls. Returns the new seq (unchanged if empty)."""
        cleaned = _dedupe(paths)
        if not cleaned:
            return self.seq
        with self._mu:
            self._seq += 1
            event = {
                "seq": self._seq,
                "paths": cleaned,
                "source": source,
                "run_id": run_id,
            }
            self._events.append(event)
            seq = self._seq
        self._wake()
        logger.debug("vault event seq=%s source=%s paths=%s", seq, source, cleaned)
        return seq

    async def wait(self, since: int, timeout: float) -> dict[str, Any]:
        """Return a snapshot, waiting up to *timeout* seconds for a newer event."""
        self._loop = asyncio.get_running_loop()
        snap = self.snapshot(since)
        if snap["events"] or snap["resync"] or self._closed or timeout <= 0:
            return snap
        waiter = asyncio.Event()
        with self._mu:
            snap = self._snapshot_locked(since)
            if snap["events"] or snap["resync"] or self._closed:
                return snap
            self._waiters.add(waiter)
        try:
            await asyncio.wait_for(waiter.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            with self._mu:
                self._waiters.discard(waiter)
        return self.snapshot(since)

    def _snapshot_locked(self, since: int) -> dict[str, Any]:
        events = [dict(event) for event in self._events if event["seq"] > since]
        resync = False
        if self._events and since > 0:
            oldest = self._events[0]["seq"]
            if since + 1 < oldest:
                resync = True
                events = []
        for event in events:
            event["paths"] = list(event["paths"])
        return {"seq": self._seq, "resync": resync, "events": events}

    def _wake(self) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and running is self._loop:
            self._set_waiters()
        elif self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._set_waiters)

    def _set_waiters(self) -> None:
        for waiter in list(self._waiters):
            waiter.set()


def _dedupe(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in paths:
        key = (raw or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


vault_events = VaultEventBus()


def publish_vault_event(
    paths: list[str],
    *,
    source: str,
    run_id: Optional[str] = None,
) -> None:
    """Publish vault paths written by an agent turn or a task-note update."""
    vault_events.publish(paths, source=source, run_id=run_id)
