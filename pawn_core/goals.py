"""Parse the user-owned Goals.md note.

A missing or invalid note means there are no active threads: items are filed
and nothing notifies. ``/goal``, ``/park``, and ``/goal apply`` are the chat
commands that write this file. ``note_write`` stays denied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pawn_core.vault import parse_frontmatter

_BULLET_RE = re.compile(r"^[-*]\s+([A-Za-z0-9_]+)\s*:\s*(.*)$")
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")


def slugify(name: str) -> str:
    """Filesystem-safe slug for a thread name."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "thread"


@dataclass
class GoalThread:
    """One active, parked, or recently closed thread."""

    name: str
    status: str = "active"
    why: str = ""
    movement: str = ""
    interrupt: str = ""
    note: str = ""
    do: str = ""
    closed: str = ""

    @property
    def slug(self) -> str:
        return slugify(self.name)


@dataclass
class Goals:
    """Parsed goals note."""

    threads: list[GoalThread] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)
    max_per_day: int = 5
    quiet_hours: Optional[str] = None
    timezone: Optional[str] = None
    autonomy: Optional[str] = None
    valid: bool = True

    @property
    def active(self) -> list[GoalThread]:
        return [thread for thread in self.threads if thread.status == "active"]

    @property
    def parked(self) -> list[GoalThread]:
        return [thread for thread in self.threads if thread.status == "parked"]

    def thread_named(self, name: str) -> Optional[GoalThread]:
        needle = (name or "").strip().lower()
        if not needle:
            return None
        for thread in self.threads:
            if thread.name.lower() == needle or thread.slug == slugify(needle):
                return thread
        return None


def empty_goals() -> Goals:
    """No active threads: file everything, notify nothing."""
    return Goals(valid=True)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _section_status(title: str) -> Optional[str]:
    lowered = title.strip().lower()
    if lowered == "active":
        return "active"
    if lowered == "parked":
        return "parked"
    if lowered.startswith("done"):
        return "done"
    return None


def parse_goals(text: str) -> Goals:
    """Parse a Goals.md body. Invalid ``pawn`` values yield an empty result."""
    meta, body = parse_frontmatter(text or "")
    pawn = meta.get("pawn")
    if pawn not in (None, "", "goals"):
        return Goals(valid=False)

    raw_notify = meta.get("notify")
    notify: dict[str, Any] = raw_notify if isinstance(raw_notify, dict) else {}
    max_per_day = notify.get("max_per_day", meta.get("max_per_day", 5))
    try:
        max_per_day_int = int(max_per_day)
    except (TypeError, ValueError):
        max_per_day_int = 5
    quiet = notify.get("quiet_hours") or meta.get("quiet_hours")
    timezone_name = meta.get("timezone")
    autonomy = meta.get("autonomy")
    goals = Goals(
        ignore=_as_str_list(meta.get("ignore")),
        max_per_day=max(0, max_per_day_int),
        quiet_hours=str(quiet).strip() if quiet else None,
        timezone=str(timezone_name).strip() if timezone_name else None,
        autonomy=str(autonomy).strip() if autonomy else None,
        valid=True,
    )

    status = "active"
    current: Optional[GoalThread] = None

    def _flush() -> None:
        nonlocal current
        if current is not None and current.name:
            goals.threads.append(current)
        current = None

    for raw_line in (body or "").splitlines():
        line = raw_line.strip()
        heading = _HEADING_RE.match(line)
        if heading:
            level, title = heading.group(1), heading.group(2).strip()
            if level == "##":
                _flush()
                mapped = _section_status(title)
                if mapped:
                    status = mapped
                continue
            _flush()
            current = GoalThread(name=title, status=status)
            continue
        if current is None:
            continue
        bullet = _BULLET_RE.match(line)
        if not bullet:
            continue
        key = bullet.group(1).lower()
        value = bullet.group(2).strip()
        if key == "why":
            current.why = value
        elif key == "movement":
            current.movement = value
        elif key == "interrupt":
            current.interrupt = value
        elif key == "note":
            current.note = value
        elif key == "do":
            current.do = value
        elif key == "closed":
            current.closed = value
    _flush()
    return goals


def load_goals_text(text: Optional[str]) -> Goals:
    """Parse *text*, or return empty goals when the note is missing."""
    if text is None:
        return empty_goals()
    parsed = parse_goals(text)
    if not parsed.valid:
        return Goals(valid=False)
    return parsed


class GoalInsertError(ValueError):
    """Raised when a thread cannot be spliced into Goals.md."""


_FM_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n?", re.DOTALL)
_H2_RE = re.compile(r"^##\s+(.+?)\s*$")


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return the raw frontmatter prefix and the body, preserving bytes."""
    match = _FM_RE.match(text or "")
    if match is None:
        return "", text or ""
    return text[: match.end()], text[match.end() :]


def _one_line(value: str) -> str:
    return " ".join((value or "").split())


def _thread_block(
    *,
    name: str,
    status: str,
    why: str,
    movement: str,
    interrupt: str,
    note: str,
    do: str,
) -> str:
    lines = [f"### {name}"]
    if status == "active":
        lines.append(f"- why: {_one_line(why)}")
        lines.append(f"- movement: {_one_line(movement)}")
        lines.append(f"- interrupt: {_one_line(interrupt)}")
    else:
        lines.append(f"- note: {_one_line(note)}")
        lines.append(f"- do: {_one_line(do)}")
    return "\n".join(lines)


def _splice_section(body: str, section: str, block: str) -> str:
    """Insert *block* at the end of a ``##`` section, creating the section if needed."""
    lines = body.splitlines(keepends=True)
    target = section.lower()
    start: Optional[int] = None
    for index, line in enumerate(lines):
        heading = _H2_RE.match(line.strip())
        if heading and heading.group(1).strip().lower() == target:
            start = index
            break
    chunk = block.strip() + "\n"
    if start is None:
        insert_at = len(lines)
        if target == "active":
            for index, line in enumerate(lines):
                heading = _H2_RE.match(line.strip())
                if heading is None:
                    continue
                title = heading.group(1).strip().lower()
                if title == "parked" or title.startswith("done"):
                    insert_at = index
                    break
        title = "Active" if target == "active" else "Parked"
        head = "".join(lines[:insert_at]).rstrip()
        tail = "".join(lines[insert_at:])
        if tail.startswith("\n"):
            tail = tail.lstrip("\n")
        section_text = f"## {title}\n\n{chunk}\n"
        if head:
            section_text = "\n\n" + section_text
        merged = head + section_text + tail
        return merged if merged.endswith("\n") else merged + "\n"

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _H2_RE.match(lines[index].strip()):
            end = index
            break
    head = "".join(lines[:end]).rstrip() + "\n\n"
    tail = "".join(lines[end:])
    return head + chunk + "\n" + tail


def insert_goal_thread(
    text: Optional[str],
    *,
    name: str,
    status: str,
    why: str = "",
    movement: str = "",
    interrupt: str = "",
    note: str = "",
    do: str = "",
) -> str:
    """Splice one thread into Goals.md without rewriting the rest of the note.

    A missing note becomes a valid ``pawn: goals`` file. An existing note whose
    ``pawn`` value is something other than goals is refused.
    """
    title = _one_line(name)
    if not title:
        raise GoalInsertError("A goal thread needs a name.")
    if len(title) > 120:
        title = title[:120].rstrip()
    kind = (status or "").strip().lower()
    if kind not in {"active", "parked"}:
        raise GoalInsertError("status must be active or parked.")

    raw = text if text and text.strip() else None
    if raw is None:
        prefix = "---\npawn: goals\n---\n\n"
        body = ""
    else:
        parsed = parse_goals(raw)
        if not parsed.valid:
            raise GoalInsertError("Goals.md is not a goals note, so it was left unchanged.")
        if parsed.thread_named(title) is not None:
            raise GoalInsertError(f"Goals.md already has a thread named '{title}'.")
        prefix, body = _split_frontmatter(raw)

    if kind == "parked" and not do.strip():
        do = "Link related notes and develop the idea. Do not notify."
    block = _thread_block(
        name=title,
        status=kind,
        why=why,
        movement=movement,
        interrupt=interrupt,
        note=note,
        do=do,
    )
    return prefix + _splice_section(body, kind, block)
