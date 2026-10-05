"""Tests for vault task protocol helpers."""

from __future__ import annotations

from pawn_server.core.vault_protocol import (
    build_obsidian_open_url,
    conversation_id_for_note,
    extract_wiki_links,
    get_section,
    instruction_hash,
    parse_task_note,
    render_task_note,
    set_section,
)


def test_instruction_hash_stable():
    a = instruction_hash("  Hello   World ")
    b = instruction_hash("hello world")
    assert a == b
    assert len(a) == 32


def test_conversation_id_for_note():
    assert conversation_id_for_note("Projects/Roadmap.md") == "note:Projects/Roadmap.md"
    assert conversation_id_for_note("/Pawn/Tasks/x.md") == "note:Pawn/Tasks/x.md"


def test_extract_wiki_links():
    text = "See [[Projects/Roadmap]] and [[Other|label]] for details."
    assert extract_wiki_links(text) == ["Projects/Roadmap", "Other"]


def test_section_get_set():
    body = "## Instruction\nDo thing\n\n## Context\nBackground\n"
    assert get_section(body, "Instruction") == "Do thing"
    updated = set_section(body, "Result", "Done.")
    assert "## Result\nDone." in updated
    assert get_section(updated, "Context") == "Background"


def test_parse_and_render_task_note():
    raw = """---
pawn: task
id: abc-123
status: todo
note: "[[Projects/Roadmap]]"
conversation: note:Projects/Roadmap.md
approved: false
---
## Instruction
Summarize the roadmap.

## Context
Quarterly planning.

## Result

"""
    parsed = parse_task_note(raw)
    assert parsed["id"] == "abc-123"
    assert parsed["status"] == "todo"
    assert parsed["approved"] is False
    assert parsed["conversation"] == "note:Projects/Roadmap.md"
    assert parsed["note_path"] == "Projects/Roadmap"
    assert "Summarize" in parsed["instruction"]
    assert "Quarterly" in parsed["context"]

    rendered = render_task_note(
        task_id="abc-123",
        status="review",
        instruction=parsed["instruction"],
        context=parsed["context"],
        result="Here is the summary.",
        conversation=parsed["conversation"],
        note_path="Projects/Roadmap",
        approved=False,
    )
    roundtrip = parse_task_note(rendered)
    assert roundtrip["status"] == "review"
    assert "summary" in roundtrip["result"]


def test_build_obsidian_open_url():
    url = build_obsidian_open_url("My Vault", "Pawn/Tasks/t1.md")
    assert url.startswith("obsidian://open?vault=")
    assert "Pawn%2FTasks%2Ft1.md" in url
