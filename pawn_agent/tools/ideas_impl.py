"""Write one idea note under ``Ideas/``.

The note is the user's. Frontmatter is ``tags: [idea]`` and ``status: inbox``
with no ``pawn: editable``, so later ``note_write`` calls cannot overwrite it.
The vault scanner does not extract idea notes, and nothing writes a
``Pawn/Ideas`` companion.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pawn_agent.utils.config import AgentConfig
from pawn_core.vault import VaultNotFound
from pawn_core.vault_config import vault_store_from_config

IDEAS_FOLDER = "Ideas"

_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_TITLE_MAX = 60


def idea_filename_title(title: str) -> str:
    """Filename stem matching the plugin Quick capture slug (first 60 chars)."""
    line = (title or "").splitlines()[0] if title else ""
    cleaned = _UNSAFE.sub("", line).strip().lstrip("#").strip()
    return cleaned[:_TITLE_MAX].strip() or "idea"


def _today(cfg: AgentConfig, today: Optional[date]) -> date:
    if today is not None:
        return today
    tzname = (getattr(cfg.coworker, "timezone", "") or "UTC").strip() or "UTC"
    try:
        return datetime.now(ZoneInfo(tzname)).date()
    except ZoneInfoNotFoundError:
        return datetime.now(timezone.utc).date()


def idea_note_key(cfg: AgentConfig, title: str, *, today: Optional[date] = None) -> str:
    """Vault key ``Ideas/{YYYY-MM-DD} {title}.md``."""
    day = _today(cfg, today).isoformat()
    stem = idea_filename_title(title)
    return f"{IDEAS_FOLDER}/{day} {stem}.md"


def render_idea_note(*, title: str, line: str) -> str:
    """Render the one-line idea note. The line is stored unchanged."""
    heading = idea_filename_title(title or line)
    body = (line or "").strip()
    lines = [
        "---",
        "tags: [idea]",
        "status: inbox",
        "---",
        "",
        f"# {heading}",
        "",
        body,
        "",
    ]
    return "\n".join(lines)


def capture_idea(
    cfg: AgentConfig,
    *,
    line: str,
    store: Any = None,
    today: Optional[date] = None,
) -> str:
    """Write the note once. An existing note at that path is left in place."""
    raw = (line or "").strip()
    if not raw:
        raise ValueError("/idea needs one line.")
    heading = idea_filename_title(raw)
    key = idea_note_key(cfg, heading, today=today)
    vault = store if store is not None else vault_store_from_config(cfg)
    try:
        vault.read(key)
    except VaultNotFound:
        pass
    else:
        return f"Already captured at {key}. Left the existing note in place."

    vault.write(key, render_idea_note(title=heading, line=raw), skip_guards=True)
    return f"Captured {key}"
