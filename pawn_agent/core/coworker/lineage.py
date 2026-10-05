"""Propagate run lineage into queue messages spawned by the agent."""

from __future__ import annotations

import os
from typing import Any, Optional


def child_lineage(environ: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """Fields a child queue message should carry. Empty when this is not a child."""
    env = environ if environ is not None else os.environ
    parent = (env.get("PAWN_PARENT_RUN_ID") or "").strip()
    if not parent:
        return {}
    try:
        depth = int(env.get("PAWN_RUN_DEPTH") or "0")
    except ValueError:
        depth = 0
    payload: dict[str, Any] = {"parent_run_id": parent, "depth": depth + 1}
    event_id = (env.get("PAWN_EVENT_ID") or "").strip()
    if event_id:
        payload["event_id"] = event_id
    return payload


class lineage_env:
    """Set lineage env vars for the duration of one agent turn."""

    def __init__(self, run_id: str, *, depth: int = 0, event_id: Optional[str] = None) -> None:
        self.run_id = run_id
        self.depth = depth
        self.event_id = event_id or run_id
        self._saved: dict[str, Optional[str]] = {}

    def __enter__(self) -> "lineage_env":
        for key in ("PAWN_PARENT_RUN_ID", "PAWN_RUN_DEPTH", "PAWN_EVENT_ID"):
            self._saved[key] = os.environ.get(key)
        os.environ["PAWN_PARENT_RUN_ID"] = self.run_id
        os.environ["PAWN_RUN_DEPTH"] = str(self.depth)
        os.environ["PAWN_EVENT_ID"] = self.event_id
        return self

    def __exit__(self, *_exc: object) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
