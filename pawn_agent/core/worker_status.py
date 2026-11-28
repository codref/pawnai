"""In-process heartbeat for background loops (scheduler, coworker crons).

Each loop records when it started, when it last ticked, and what it did, so
the portal's Diagnostics → Scheduler view can tell a stalled loop from an
idle one. State lives in this process only: a worker started with
``pawn-server serve --scheduler-only`` in another process does not show here
(its fires and runs still do, from the database).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

_LOCK = threading.Lock()
_STATE: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plain(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def started(name: str, **fields: Any) -> None:
    """A loop began: reset its record."""
    with _LOCK:
        _STATE[name] = {"running": True, "started_at": _now(), "ticks": 0, **_plain(fields)}


def tick(name: str, **fields: Any) -> None:
    """One loop iteration finished (``error`` is cleared unless passed)."""
    with _LOCK:
        row = _STATE.setdefault(name, {"running": True, "started_at": None, "ticks": 0})
        row["ticks"] = int(row.get("ticks") or 0) + 1
        row["last_tick_at"] = _now()
        row["error"] = None
        row.update(_plain(fields))


def update(name: str, **fields: Any) -> None:
    """Set fields without counting a tick (next fire times, job results)."""
    with _LOCK:
        _STATE.setdefault(name, {"running": True, "started_at": None, "ticks": 0}).update(
            _plain(fields)
        )


def stopped(name: str) -> None:
    with _LOCK:
        if name in _STATE:
            _STATE[name]["running"] = False
            _STATE[name]["stopped_at"] = _now()


def snapshot() -> dict[str, dict[str, Any]]:
    with _LOCK:
        return {k: dict(v) for k, v in _STATE.items()}


def reset() -> None:
    """Tests only."""
    with _LOCK:
        _STATE.clear()
