"""Bridge sallm Tracer progress events into asyncio streams (SSE endpoints)."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, Optional


def describe_progress(kind: str, attrs: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Short human label for a Tracer event, or ``None`` to skip it."""
    attrs = attrs or {}
    if kind == "turn.start":
        return "Working…"
    if kind == "control":
        skill = str(attrs.get("sallm.control.skill") or "").strip()
        return f"Using skill {skill}…" if skill else None
    if kind == "tool":
        name = str(attrs.get("gen_ai.tool.name") or "").strip() or "tool"
        return f"Ran {name}"
    return None


ProgressCallback = Callable[[str, dict[str, Any]], None]


async def stream_turn(
    run: Callable[[ProgressCallback], Awaitable[Any]],
    *,
    keepalive_seconds: float = 10.0,
) -> AsyncIterator[tuple[str, Any]]:
    """Run ``run(on_progress)`` and yield ``(event, data)`` tuples.

    Events: ``progress`` (dict with ``kind`` / ``text``), ``keepalive`` (None),
    then exactly one of ``result`` (the awaited value) or ``error`` (str).
    ``on_progress`` is called from the agent worker thread.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def on_progress(kind: str, attrs: dict[str, Any]) -> None:
        text = describe_progress(kind, attrs)
        if text:
            loop.call_soon_threadsafe(queue.put_nowait, {"kind": kind, "text": text})

    task = asyncio.ensure_future(run(on_progress))
    getter: Optional[asyncio.Future[dict[str, Any]]] = None
    try:
        while True:
            if getter is None:
                getter = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait(
                {task, getter},
                timeout=keepalive_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if getter in done:
                yield "progress", getter.result()
                getter = None
                continue
            if task in done:
                break
            yield "keepalive", None
        getter.cancel()
        getter = None
        while not queue.empty():
            yield "progress", queue.get_nowait()
        exc = task.exception()
        if exc is not None:
            yield "error", str(exc)
        else:
            yield "result", task.result()
    finally:
        if getter is not None:
            getter.cancel()
