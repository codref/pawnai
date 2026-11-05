"""Day-plan helpers: compact planner day readout + ``## Plan`` on daily notes."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pawn_core.vault import VaultNotFound, VaultWriteDenied, normalize_vault_key
from pawn_core.vault_config import vault_store_from_config

_PLAN_SECTION_RE = re.compile(
    r"(?is)^##\s*Plan\s*\n+(.*?)(?=^##\s|\Z)",
    re.MULTILINE,
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _timezone_name(cfg: Any) -> str:
    return (getattr(getattr(cfg, "coworker", None), "timezone", "") or "UTC").strip() or "UTC"


def today_local(cfg: Any) -> str:
    """Local calendar day for coworker/planner timezone."""
    try:
        return datetime.now(ZoneInfo(_timezone_name(cfg))).date().isoformat()
    except ZoneInfoNotFoundError:
        return datetime.now(timezone.utc).date().isoformat()


def resolve_day(cfg: Any, date: Optional[str] = None) -> str:
    raw = (date or "").strip()
    if not raw:
        return today_local(cfg)
    if not _DATE_RE.match(raw):
        raise ValueError(f"date must be YYYY-MM-DD, got {raw!r}")
    return raw


def daily_note_key(cfg: Any, day: str) -> str:
    folder = str(getattr(cfg.coworker, "daily_dir", None) or "Pawn/Daily").strip().strip("/")
    return normalize_vault_key(f"{folder}/{day}.md")


def format_planner_day(payload: dict[str, Any]) -> str:
    """Render ``build_day`` output as compact LLM-readable text."""
    day = str(payload.get("date") or "").strip() or "?"
    lines: list[str] = [f"Day {day}", ""]

    blocks = list(payload.get("blocks") or [])
    if blocks:
        lines.append("## Timeline")
        for block in blocks:
            kind = str(block.get("kind") or "")
            if kind == "event":
                ev = block.get("event") or {}
                title = str(ev.get("title") or "untitled").strip()
                start = str(ev.get("start") or "").strip()
                end = str(ev.get("end") or "").strip()
                when = start
                if end:
                    when = f"{start}–{end}" if start else end
                focus = " [focus]" if ev.get("focus") else ""
                eid = str(ev.get("id") or "").strip()
                sid = str(ev.get("session_id") or "").strip()
                bits = [f"- event {when}: {title}{focus}".rstrip()]
                if eid:
                    bits.append(f"  id={eid}")
                if sid:
                    bits.append(f"  session={sid}")
                lines.extend(bits)
                for item in block.get("items") or []:
                    lines.append(_format_item_line(item, indent="  "))
            elif kind == "session":
                sess = block.get("session") or {}
                sid = str(sess.get("session_id") or sess.get("title") or "?").strip()
                created = str(sess.get("created_at") or "").strip()
                label = f"- session {sid}"
                if created:
                    label += f" ({created})"
                lines.append(label)
                for item in block.get("items") or []:
                    lines.append(_format_item_line(item, indent="  "))
        lines.append("")
    else:
        lines.append("## Timeline")
        lines.append("_No events or sessions._")
        lines.append("")

    other = list(payload.get("other_items") or [])
    lines.append("## Open items (not on timeline)")
    if other:
        for item in other:
            lines.append(_format_item_line(item))
    else:
        lines.append("_None._")
    lines.append("")

    suggestions = list(payload.get("suggestions") or [])
    if suggestions:
        lines.append("## Link suggestions")
        for sug in suggestions:
            ev_title = str(sug.get("event_title") or sug.get("event_id") or "?").strip()
            sess_title = str(sug.get("session_title") or sug.get("session_id") or "?").strip()
            score = sug.get("score")
            score_bit = f" score={score}" if score is not None else ""
            lines.append(f"- {ev_title} ↔ {sess_title}{score_bit}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _format_item_line(item: dict[str, Any], *, indent: str = "") -> str:
    kind = str(item.get("kind") or "item").strip()
    text = str(item.get("text") or "").strip() or "(empty)"
    short = str(item.get("short_id") or "").strip()
    thread = str(item.get("thread") or "").strip()
    interrupt = " !" if item.get("interrupt") else ""
    meta = []
    if short:
        meta.append(short)
    if thread:
        meta.append(thread)
    suffix = f" ({', '.join(meta)})" if meta else ""
    return f"{indent}- {kind}{interrupt}: {text}{suffix}"


def planner_day_impl(
    cfg: Any,
    *,
    date: Optional[str] = None,
    store: Any = None,
) -> str:
    """Load the merged day timeline and return compact text."""
    from pawn_server.core.planner import build_day  # noqa: PLC0415

    try:
        day = resolve_day(cfg, date)
    except ValueError as exc:
        return f"Error: {exc}"
    try:
        payload = build_day(cfg, day, store=store)
    except Exception as exc:
        return f"Error loading planner day: {exc}"
    return format_planner_day(payload)


def replace_plan_section(text: str, plan: str) -> str:
    """Replace or append ``## Plan`` in a daily note body."""
    body = text if text.endswith("\n") or not text else text + "\n"
    plan_body = (plan or "").rstrip() + "\n"
    match = _PLAN_SECTION_RE.search(body)
    if match:
        heading_end = match.start(1)
        end = match.end(1)
        return body[:heading_end] + plan_body + body[end:]
    base = body.rstrip()
    if not base.strip():
        return f"## Plan\n\n{plan_body}"
    return base + f"\n\n## Plan\n\n{plan_body}"


def day_plan_update_impl(
    cfg: Any,
    *,
    content: str,
    date: Optional[str] = None,
    store: Any = None,
) -> str:
    """Replace ``## Plan`` on ``Pawn/Daily/{date}.md``. Creates a stub if missing."""
    if store is None:
        store = vault_store_from_config(cfg)
    body = (content or "").strip()
    if not body:
        return "Error: plan content is empty"
    try:
        day = resolve_day(cfg, date)
    except ValueError as exc:
        return f"Error: {exc}"
    key = daily_note_key(cfg, day)
    try:
        existing = store.read(key)
    except VaultNotFound:
        existing = f"# {day}\n"
    except Exception as exc:
        return f"Error reading daily note: {exc}"

    try:
        updated = replace_plan_section(existing, body)
        store.write(key, updated)
    except VaultWriteDenied as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error writing plan: {exc}"

    try:
        from pawn_server.core.vault_events import publish_vault_event  # noqa: PLC0415

        publish_vault_event([key], source="day_plan")
    except Exception:
        pass
    return f"Updated ## Plan on {key}"
