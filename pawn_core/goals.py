"""Parse the user-owned Goals.md note.

The agent reads this note and never writes it. A missing or invalid note
means there are no active threads: items are filed and nothing notifies.
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
