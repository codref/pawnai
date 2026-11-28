"""Server-side prompt assembly for ``POST /v1/pawn/chat``.

The plugin sends structured context (active note, selection, @-mentioned
notes); this module turns it into one agent prompt. Notes without inline
content are read from the vault store when available. Notes with
``inline=False`` are path pointers only (body was attached earlier).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from pawn_core.vault import normalize_vault_key

logger = logging.getLogger(__name__)

NOTE_CHAR_LIMIT = 3000
SELECTION_CHAR_LIMIT = 6000
EXTRA_SECTION_CHAR_LIMIT = 8000
MAX_CONTEXT_NOTES = 8

_POINTER_HINT = (
    "Body attached earlier this conversation; " "use `note_read --path {key}` if needed."
)


@dataclass(frozen=True)
class NoteRef:
    path: str
    content: Optional[str] = None
    inline: bool = True


def _link(path: str) -> str:
    return f"[[{normalize_vault_key(path).removesuffix('.md')}]]"


def _vault_key(path: str) -> str:
    key = normalize_vault_key(path)
    if not key.lower().endswith(".md"):
        key = f"{key}.md"
    return key


async def resolve_notes(store: Any, notes: Sequence[NoteRef]) -> list[NoteRef]:
    """Fill missing ``content`` from the vault store (best effort).

    Notes with ``inline=False`` are left as path pointers; the store is not
    consulted.
    """
    out: list[NoteRef] = []
    for note in notes[:MAX_CONTEXT_NOTES]:
        if not note.inline:
            out.append(note)
            continue
        if note.content is not None or store is None:
            out.append(note)
            continue
        key = _vault_key(note.path)
        try:
            body = await asyncio.to_thread(store.read, key)
        except Exception as exc:
            logger.debug("context note %s unreadable: %s", key, exc)
            body = None
        out.append(NoteRef(path=note.path, content=body, inline=True))
    return out


def _body_section(note: NoteRef) -> str:
    """Render one note's ``###`` block (full body, pointer, or unavailable)."""
    key = _vault_key(note.path)
    if not note.inline:
        return f"### {_link(note.path)}\n{_POINTER_HINT.format(key=key)}"
    body = (note.content or "").strip()
    if not body:
        return f"### {_link(note.path)}\n(unavailable)"
    if len(body) > NOTE_CHAR_LIMIT:
        body = body[:NOTE_CHAR_LIMIT] + f"\n…(truncated; use note_read --path {key} for the rest)"
    return f"### {_link(note.path)}\n{body}"


def build_chat_prompt(
    message: str,
    *,
    active_note: Optional[NoteRef] = None,
    selection: Optional[str] = None,
    context: Sequence[NoteRef] = (),
    extra_sections: Sequence[tuple[str, str]] = (),
) -> str:
    """Compose the agent prompt; the bare message when there is no context.

    ``extra_sections`` are ``(heading, text)`` blocks that are not vault notes
    (a board snapshot); they come first, each capped separately.
    """
    message = (message or "").strip()
    sections: list[str] = []
    for heading, text in extra_sections:
        body = (text or "").strip()
        if body:
            sections.append(f"## {heading}\n{body[:EXTRA_SECTION_CHAR_LIMIT]}")
    if active_note is not None:
        path_line = f"Active note: {_link(active_note.path)} " f"(vault path `{active_note.path}`)"
        sections.append(path_line)
        if not active_note.inline:
            sections.append(_POINTER_HINT.format(key=_vault_key(active_note.path)))
        elif active_note.content:
            sections.append(_body_section(active_note))
    if selection and selection.strip():
        sel = selection.strip()[:SELECTION_CHAR_LIMIT]
        quoted = "\n".join(f"> {line}" for line in sel.splitlines())
        sections.append(f"Selected text in the active note:\n{quoted}")
    seen: set[str] = set()
    if active_note is not None:
        seen.add(normalize_vault_key(active_note.path))
    for note in context:
        key = normalize_vault_key(note.path)
        if key in seen:
            continue
        seen.add(key)
        sections.append(_body_section(note))
    if not sections:
        return message
    where = "the board and the user's vault" if extra_sections else "the user's Obsidian vault"
    return (
        f"{message}\n\n---\n"
        f"Context from {where} (reference material; "
        "the message above is the request):\n\n" + "\n\n".join(sections)
    )
