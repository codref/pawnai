"""Unit tests for Obsidian chat prompt assembly."""

from __future__ import annotations

import asyncio

from pawn_server.core.chat_context import (
    NOTE_CHAR_LIMIT,
    NoteRef,
    build_chat_prompt,
    resolve_notes,
)


def test_build_chat_prompt_without_context_is_bare() -> None:
    assert build_chat_prompt("hello") == "hello"
    prompt = build_chat_prompt(
        "hello",
        context=[
            NoteRef("A.md", "alpha"),
            NoteRef("A.md", "dup"),
            NoteRef("B.md", None),
        ],
    )
    assert "[[A]]" in prompt and "alpha" in prompt and "dup" not in prompt
    assert "[[B]]\n(unavailable)" in prompt


def test_build_chat_prompt_pointer_skips_body() -> None:
    prompt = build_chat_prompt(
        "hello",
        active_note=NoteRef("Projects/Roadmap.md", content="SECRET", inline=False),
        context=[NoteRef("Other.md", content="also secret", inline=False)],
    )
    assert "Active note: [[Projects/Roadmap]]" in prompt
    assert "SECRET" not in prompt
    assert "also secret" not in prompt
    assert "(unavailable)" not in prompt
    assert "note_read --path Projects/Roadmap.md" in prompt
    assert "[[Other]]" in prompt
    assert "note_read --path Other.md" in prompt


def test_build_chat_prompt_truncation_mentions_note_read() -> None:
    body = "x" * (NOTE_CHAR_LIMIT + 50)
    prompt = build_chat_prompt("q", context=[NoteRef("Long.md", body)])
    assert "note_read --path Long.md" in prompt
    assert "truncated" in prompt
    assert body not in prompt
    assert body[:NOTE_CHAR_LIMIT] in prompt


def test_resolve_notes_skips_store_when_not_inline() -> None:
    class Boom:
        def read(self, _key: str) -> str:
            raise AssertionError("vault must not be read for pointer notes")

    notes = asyncio.run(
        resolve_notes(
            Boom(),
            [
                NoteRef("A.md", inline=False),
                NoteRef("B.md", content=None, inline=True),
            ],
        )
    )
    assert notes[0] == NoteRef("A.md", content=None, inline=False)
    # inline + missing content + store that raises → content None
    assert notes[1].path == "B.md"
    assert notes[1].inline is True
    assert notes[1].content is None
