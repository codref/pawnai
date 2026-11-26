"""Create or upsert a Day-tab planner activity (``pawn: event`` note)."""

from __future__ import annotations

from typing import Any

from pawn_core.vault_config import vault_store_from_config
from pawn_server.core.planner import PlannerError, upsert_event


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def event_upsert_impl(
    cfg: Any,
    *,
    title: str,
    start: str = "",
    end: str = "",
    source: str = "manual",
    source_url: str = "",
    attendees: str = "",
    topics: str = "",
    organizer: str = "",
    location: str = "",
    meeting_url: str = "",
    session_id: str = "",
    body: str = "",
    event_id: str = "",
    store: Any = None,
) -> str:
    """Upsert one planner event. Returns a one-line receipt."""
    heading = (title or "").strip()
    if not heading:
        return "Error: --title is required"
    if store is None:
        store = vault_store_from_config(cfg)
    try:
        ev = upsert_event(
            cfg,
            title=heading,
            start=(start or "").strip(),
            end=(end or "").strip(),
            source=(source or "manual").strip() or "manual",
            source_url=(source_url or "").strip(),
            attendees=_split_csv(attendees) or None,
            topics=_split_csv(topics) or None,
            organizer=(organizer or "").strip(),
            location=(location or "").strip(),
            meeting_url=(meeting_url or "").strip(),
            session_id=(session_id or "").strip(),
            body=(body or "").strip(),
            event_id=(event_id or "").strip(),
            store=store,
        )
    except PlannerError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error upserting event: {exc}"

    eid = str(ev.get("id") or "").strip()
    path = str(ev.get("path") or "").strip()
    when = str(ev.get("start") or "").strip()
    end_s = str(ev.get("end") or "").strip()
    span = when
    if end_s:
        span = f"{when}–{end_s}" if when else end_s
    bits = [f"Upserted event {eid or path}"]
    if path:
        bits.append(path)
    if span:
        bits.append(span)
    return " ".join(bits)
