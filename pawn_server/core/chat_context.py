"""Server-side prompt assembly for ``POST /v1/pawn/chat``.

The plugin sends structured context (active note, selection, @-mentioned
notes); this module turns it into one agent prompt. Notes without inline
content are read from the vault store when available.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from pawn_core.vault import normalize_vault_key

logger = logging.getLogger(__name__)

NOTE_CHAR_LIMIT = 8000
SELECTION_CHAR_LIMIT = 6000
MAX_CONTEXT_NOTES = 8


@dataclass(frozen=True)
class NoteRef:
    path: str
    content: Optional[str] = None


def _link(path: str) -> str:
    return f"[[{normalize_vault_key(path).removesuffix('.md')}]]"


async def resolve_notes(store: Any, notes: Sequence[NoteRef]) -> list[NoteRef]:
    """Fill missing ``content`` from the vault store (best effort)."""
    out: list[NoteRef] = []
    for note in notes[:MAX_CONTEXT_NOTES]:
        if note.content is not None or store is None:
            out.append(note)
            continue
        key = normalize_vault_key(note.path)
        if not key.lower().endswith(".md"):
            key = f"{key}.md"
        try:
            body = await asyncio.to_thread(store.read, key)
        except Exception as exc:
            logger.debug("context note %s unreadable: %s", key, exc)
            body = None
        out.append(NoteRef(path=note.path, content=body))
    return out


def build_chat_prompt(
    message: str,
    *,
    active_note: Optional[NoteRef] = None,
    selection: Optional[str] = None,
    context: Sequence[NoteRef] = (),
) -> str:
    """Compose the agent prompt; the bare message when there is no context."""
    message = (message or "").strip()
    sections: list[str] = []
    if active_note is not None:
        sections.append(f"Active note: {_link(active_note.path)} (vault path `{active_note.path}`)")
    if selection and selection.strip():
        quoted = "\n".join(
            f"> {line}" for line in selection.strip()[:SELECTION_CHAR_LIMIT].splitlines()
        )
        sections.append(f"Selected text in the active note:\n{quoted}")
    notes: list[NoteRef] = []
    if active_note is not None and active_note.content:
        notes.append(active_note)
    seen = {normalize_vault_key(n.path) for n in notes}
    for note in context:
        key = normalize_vault_key(note.path)
        if key in seen:
            continue
        seen.add(key)
        notes.append(note)
    for note in notes:
        body = (note.content or "").strip()
        if not body:
            sections.append(f"### {_link(note.path)}\n(unavailable)")
            continue
        if len(body) > NOTE_CHAR_LIMIT:
            body = body[:NOTE_CHAR_LIMIT] + "\n…(truncated)"
        sections.append(f"### {_link(note.path)}\n{body}")
    if not sections:
        return message
    return (
        f"{message}\n\n---\n"
        "Context from the user's Obsidian vault (reference material; "
        "the message above is the request):\n\n" + "\n\n".join(sections)
    )
