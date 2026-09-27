"""Open-loop detectors. The SQL wrappers feed the pure checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional


def _parse_due(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _created(item: dict[str, Any]) -> Optional[datetime]:
    raw = item.get("created_at")
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def is_me(owner: str, aliases: list[str]) -> bool:
    needle = (owner or "").strip().lower()
    if not needle:
        return False
    return any(needle == alias.strip().lower() for alias in aliases if alias.strip())


def my_commitments(
    items: list[dict[str, Any]],
    *,
    aliases: list[str],
    now: datetime,
    commitment_days: int = 7,
) -> list[dict[str, Any]]:
    found = []
    for item in items:
        if item.get("kind") != "commitment":
            continue
        if item.get("status") in {"filed", "dismissed"}:
            continue
        if not is_me(str(item.get("owner") or ""), aliases):
            continue
        due = _parse_due(item.get("due"))
        created = _created(item)
        overdue = due is not None and due <= now
        aged = created is not None and created <= now - timedelta(days=commitment_days)
        if overdue or aged:
            found.append(item)
    return found


def ownerless_decisions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if item.get("kind") == "decision"
        and not (item.get("owner") or "").strip()
        and item.get("status") not in {"filed", "dismissed"}
    ]


def recurring_questions(items: list[dict[str, Any]], *, minimum: int = 3) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if item.get("kind") == "open_question"
        and int(item.get("recurrence") or 0) >= minimum
        and item.get("status") not in {"dismissed", "filed"}
    ]


def stale_threads(
    threads: list[dict[str, Any]],
    *,
    now: datetime,
    stale_days: int = 14,
) -> list[dict[str, Any]]:
    cutoff = now - timedelta(days=stale_days)
    stale = []
    for thread in threads:
        if thread.get("status") != "active":
            continue
        last = thread.get("last_movement_at")
        if isinstance(last, str):
            try:
                last = datetime.fromisoformat(last)
            except ValueError:
                last = None
        if isinstance(last, datetime) and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if last is None or last <= cutoff:
            stale.append(thread)
    return stale


def thread_dict(row: Any) -> dict[str, Any]:
    return {
        "slug": row.slug,
        "name": row.name,
        "status": row.status,
        "last_movement_at": row.last_movement_at,
        "last_mention_at": row.last_mention_at,
        "open_items": row.open_items,
    }
