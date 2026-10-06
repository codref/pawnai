"""Vault-backed person bios linked to the Speakers gallery.

Identity (voice) stays in Postgres ``speakers`` / enrollments.
Durable knowledge (summary, facts, appearances, tags) lives under
``People/{speaker_id}.md`` so every agent conversation key can read the same
record via tools — not per-chat sallm memory.

See ``docs/PEOPLE.md`` and ``docs/plans/speaker-people-vault.md``.
"""

from __future__ import annotations

from pawn_agent.core.people.notes import (
    append_facts,
    ensure_person_note,
    people_note_key,
    people_wiki_target,
    read_person_note,
    render_person_note,
    update_person_note,
)
from pawn_agent.core.people.refresh import refresh_people_for_session

__all__ = [
    "append_facts",
    "ensure_person_note",
    "people_note_key",
    "people_wiki_target",
    "read_person_note",
    "refresh_people_for_session",
    "render_person_note",
    "update_person_note",
]
