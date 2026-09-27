"""Decide what to do with a job left running across a server restart."""

from __future__ import annotations

from typing import Any, Optional


def recovery_plan(
    kind: Optional[str], instruction: Optional[str], payload: Optional[dict[str, Any]]
) -> str:
    """Return ``respawn`` or ``abandon``."""
    data = payload or {}
    if (kind or "ask") == "ask" and (instruction or "").strip() and not data.get("recovered"):
        return "respawn"
    return "abandon"
