"""Vault markdown for coworker items, threads, and the daily note."""

from __future__ import annotations

from typing import Any

from pawn_core.goals import slugify
from pawn_core.vault import dump_frontmatter, parse_frontmatter

TERMINAL_STATUSES = frozenset({"filed", "task", "dismissed"})


def render_item_note(
    *,
    item_id: str,
    short_id: str,
    status: str,
    kind: str,
    text: str,
    thread: str = "",
    quote: str = "",
    source_link: str = "",
    reason: str = "",
    interrupt: bool = False,
    related: list[str] | None = None,
    action: str = "",
) -> str:
    """Serialize one ``pawn: item`` note."""
    meta: dict[str, Any] = {
        "pawn": "item",
        "id": item_id,
        "short_id": short_id,
        "status": status,
        "kind": kind,
        "thread": thread or "",
        "action": action or "",
    }
    body_parts = [text.strip() or "(empty)"]
    if quote.strip():
        body_parts.append("")
        for line in quote.strip().splitlines():
            body_parts.append(f"> {line}")
    if source_link.strip():
        body_parts.append("")
        body_parts.append(f"Source: {source_link.strip()}")
    if interrupt and reason.strip():
        body_parts.append("")
        body_parts.append("## Why you were notified")
        body_parts.append(reason.strip())
    if related:
        body_parts.append("")
        body_parts.append("## Related")
        for line in related:
            body_parts.append(f"- {line}")
    return dump_frontmatter(meta, "\n".join(body_parts) + "\n")


def parse_item_note(text: str) -> dict[str, Any]:
    """Parse a ``pawn: item`` note. Raises ValueError when it is not one."""
    meta, body = parse_frontmatter(text or "")
    if meta.get("pawn") != "item":
        raise ValueError("frontmatter pawn must be 'item'")
    return {
        "id": str(meta.get("id") or "").strip(),
        "short_id": str(meta.get("short_id") or "").strip(),
        "status": str(meta.get("status") or "new").strip().lower(),
        "kind": str(meta.get("kind") or "").strip(),
        "thread": str(meta.get("thread") or "").strip(),
        "action": str(meta.get("action") or "").strip().lower(),
        "body": body,
        "meta": meta,
    }


def render_today(groups: dict[str, list[dict[str, Any]]]) -> str:
    """Render ``Pawn/Today.md``.

    *groups* keys: ``attention`` and ``filed``. Each item dict needs
    ``note_key``, ``kind``, ``text``, and ``thread``.
    """
    lines = ["# Today", ""]
    lines.append("## Needs you")
    attention = groups.get("attention") or []
    if not attention:
        lines.append("")
        lines.append("_Nothing waiting._")
    else:
        lines.append("")
        for item in attention:
            link = item.get("note_key") or item.get("short_id") or ""
            label = item.get("text") or ""
            thread = item.get("thread") or ""
            kind = item.get("kind") or "item"
            suffix = f" ({thread})" if thread else ""
            lines.append(f"- [[{link}]] {kind} — {label}{suffix}")
    lines.append("")
    lines.append("## Filed")
    filed = groups.get("filed") or []
    if not filed:
        lines.append("")
        lines.append("_Nothing new._")
    else:
        by_thread: dict[str, list[dict[str, Any]]] = {}
        for item in filed:
            by_thread.setdefault(item.get("thread") or "Unthreaded", []).append(item)
        for thread, items in by_thread.items():
            lines.append("")
            lines.append(f"### {thread}")
            for item in items:
                lines.append(f"- {item.get('text') or ''}")
    lines.append("")
    return dump_frontmatter({"pawn": "today"}, "\n".join(lines))


def append_thread_entry(existing: str, *, heading: str, line: str, title: str) -> str:
    """Append a bullet under *heading* in a thread note, creating the note if needed."""
    text = existing or ""
    if not text.strip():
        text = dump_frontmatter({"pawn": "thread", "title": title}, f"# {title}\n")
    block_heading = f"## {heading}"
    bullet = f"- {line.strip()}"
    if block_heading not in text:
        if not text.endswith("\n"):
            text += "\n"
        return text + f"\n{block_heading}\n{bullet}\n"
    parts = text.split(block_heading, 1)
    head, tail = parts[0], parts[1]
    next_heading = tail.find("\n## ")
    if next_heading == -1:
        section, rest = tail, ""
    else:
        section, rest = tail[: next_heading + 1], tail[next_heading + 1 :]
    if not section.endswith("\n"):
        section += "\n"
    return head + block_heading + section + bullet + "\n" + rest


def thread_note_key(threads_dir: str, thread_name: str) -> str:
    root = (threads_dir or "Pawn/Threads").strip("/")
    return f"{root}/{slugify(thread_name)}.md"
