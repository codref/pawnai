"""Capture one idea note under the coworker watch folder.

The note is the user's. Frontmatter is ``tags: [idea]`` with no
``pawn: editable``, so later ``note_write`` calls cannot overwrite it.
The vault watcher still builds the ``Pawn/Ideas`` companion after the
quiet window. This module does not write that companion.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pawn_agent.utils.config import AgentConfig
from pawn_core.vault import VaultNotFound
from pawn_core.vault_config import vault_store_from_config

_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_BULLET = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
_WHY_MAX = 600
_SKETCH_MAX = 900
_QUESTION_MAX = 200
_QUESTIONS_MAX = 5
_TITLE_MAX = 60


def capture_folder(cfg: AgentConfig) -> str:
    """First coworker watch folder, or ``Ideas``."""
    folders = list(getattr(cfg.coworker, "watch_folders", None) or [])
    if folders and str(folders[0]).strip():
        return str(folders[0]).strip().strip("/")
    return "Ideas"


def idea_filename_title(title: str) -> str:
    """Filename stem matching the plugin Quick capture slug (first 60 chars)."""
    line = (title or "").splitlines()[0] if title else ""
    cleaned = _UNSAFE.sub("", line).strip().lstrip("#").strip()
    return cleaned[:_TITLE_MAX].strip() or "idea"


def _clip(text: str, limit: int) -> str:
    body = (text or "").strip()
    if len(body) <= limit:
        return body
    return body[: limit - 1].rstrip() + "…"


def _questions(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for line in (raw or "").splitlines():
            item = _BULLET.sub("", line).strip()
            if not item:
                continue
            out.append(_clip(item, _QUESTION_MAX))
            if len(out) >= _QUESTIONS_MAX:
                return out
    return out


def _today(cfg: AgentConfig, today: Optional[date]) -> date:
    if today is not None:
        return today
    tzname = (getattr(cfg.coworker, "timezone", "") or "UTC").strip() or "UTC"
    try:
        return datetime.now(ZoneInfo(tzname)).date()
    except ZoneInfoNotFoundError:
        return datetime.now(timezone.utc).date()


def idea_note_key(cfg: AgentConfig, title: str, *, today: Optional[date] = None) -> str:
    """Vault key ``{watch}/{YYYY-MM-DD} {title}.md``."""
    day = _today(cfg, today).isoformat()
    stem = idea_filename_title(title)
    return f"{capture_folder(cfg)}/{day} {stem}.md"


def render_idea_note(
    *,
    title: str,
    seed: str,
    why: str,
    sketch: str,
    open_questions: list[str],
) -> str:
    """Render the idea skeleton. The seed line is stored unchanged."""
    heading = idea_filename_title(title)
    questions = _questions(open_questions)
    lines = [
        "---",
        "tags: [idea]",
        "---",
        "",
        f"# {heading}",
        "",
        "## Seed",
        "",
        (seed or "").strip(),
        "",
        "## Why",
        "",
        _clip(why, _WHY_MAX),
        "",
        "## Sketch",
        "",
        _clip(sketch, _SKETCH_MAX),
        "",
        "## Open questions",
        "",
    ]
    for question in questions:
        lines.append(f"- {question}")
    lines.append("")
    return "\n".join(lines)


def idea_capture_impl(
    cfg: AgentConfig,
    *,
    title: str,
    seed: str,
    why: str,
    sketch: str,
    open_questions: list[str],
    store: Any = None,
    today: Optional[date] = None,
) -> str:
    """Write the skeleton once. An existing note at that path is left in place."""
    raw_seed = (seed or "").strip() or (title or "").strip()
    heading = idea_filename_title(title or raw_seed)
    if not raw_seed:
        raise ValueError("idea_capture needs the user's line as --seed.")
    if not (why or "").strip():
        raise ValueError("idea_capture needs --why filled from the user's line.")
    if not (sketch or "").strip():
        raise ValueError("idea_capture needs --sketch filled from the user's line.")
    questions = _questions(list(open_questions or []))
    if not questions:
        raise ValueError("idea_capture needs at least one --question from the user's line.")

    key = idea_note_key(cfg, heading, today=today)
    vault = store if store is not None else vault_store_from_config(cfg)
    try:
        vault.read(key)
    except VaultNotFound:
        pass
    else:
        return f"Already captured at {key}. Left the existing note in place."

    note = render_idea_note(
        title=heading,
        seed=raw_seed,
        why=why,
        sketch=sketch,
        open_questions=questions,
    )
    vault.write(key, note, skip_guards=True)
    return f"Captured {key}"
