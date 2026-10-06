"""Unit tests for People/ vault note helpers (no live S3 / LLM)."""

from __future__ import annotations

from types import SimpleNamespace

from pawn_agent.core.people.extract import _normalize_people_payload
from pawn_agent.core.people.notes import (
    append_facts,
    ensure_person_note,
    parse_person_note,
    people_note_key,
    people_wiki_target,
    render_person_note,
    update_person_note,
)


class _MemStore:
    """Minimal vault stand-in for people note tests."""

    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    def read(self, key: str) -> str:
        from pawn_core.vault import VaultNotFound

        if key not in self.objects:
            raise VaultNotFound(key)
        return self.objects[key]

    def write(self, key: str, body: str, skip_guards: bool = False) -> None:
        self.objects[key] = body


def _cfg(people_dir: str = "People") -> SimpleNamespace:
    return SimpleNamespace(coworker=SimpleNamespace(people_dir=people_dir, me=["Davide"]))


def test_people_note_key_uses_speaker_id():
    cfg = _cfg()
    assert people_note_key(cfg, "Davide") == "People/Davide.md"
    assert people_wiki_target(cfg, "davide") == "People/davide"


def test_render_and_parse_roundtrip_preserves_notes():
    md = render_person_note(
        speaker_id="davide",
        display_name="Davide",
        aliases=["Dave"],
        tags=["cofounder"],
        summary="Cofounder.",
        facts=["2026-10-06: Likes S3."],
        appearances=["[[Pawn/Transcripts/x.md]] — 5m talk · 2026-10-06"],
        notes="Private reminder.",
    )
    note = parse_person_note(md, key="People/davide.md")
    assert note.speaker_id == "davide"
    assert note.display_name == "Davide"
    assert note.aliases == ["Dave"]
    assert "person" in note.tags
    assert "cofounder" in note.tags
    assert note.summary == "Cofounder."
    assert note.facts == ["2026-10-06: Likes S3."]
    assert note.notes == "Private reminder."


def test_update_preserves_user_notes_and_dedupes():
    store = _MemStore()
    cfg = _cfg()
    ensure_person_note(cfg, speaker_id="davide", display_name="Davide", store=store)
    # Seed user notes by rewriting once through parse/render path.
    raw = store.read("People/davide.md")
    note = parse_person_note(raw)
    store.write(
        "People/davide.md",
        render_person_note(
            speaker_id="davide",
            display_name="Davide",
            notes="Keep me",
            facts=["already here"],
        ),
    )
    update_person_note(
        cfg,
        speaker_id="davide",
        add_facts=["already here", "new fact"],
        add_appearances=["[[Pawn/Transcripts/a.md]]"],
        store=store,
    )
    updated = parse_person_note(store.read("People/davide.md"))
    assert updated.notes == "Keep me"
    assert updated.facts.count("already here") == 1
    assert "new fact" in updated.facts
    assert any("Transcripts/a.md" in a for a in updated.appearances)
    assert note.speaker_id == "davide"


def test_append_facts_adds_date_and_source():
    store = _MemStore()
    cfg = _cfg()
    ensure_person_note(cfg, speaker_id="alice", display_name="Alice", store=store)
    append_facts(
        cfg,
        "alice",
        ["Prefers async updates"],
        source_wiki="[[Pawn/Transcripts/meet.md]]",
        store=store,
    )
    note = parse_person_note(store.read("People/alice.md"))
    assert len(note.facts) == 1
    assert "Prefers async updates" in note.facts[0]
    assert "[[Pawn/Transcripts/meet.md]]" in note.facts[0]


def test_ensure_does_not_clobber():
    store = _MemStore()
    cfg = _cfg()
    key, created = ensure_person_note(cfg, speaker_id="bob", display_name="Bob", store=store)
    assert created
    store.write(key, render_person_note(speaker_id="bob", display_name="Bob", summary="Stay"))
    key2, created2 = ensure_person_note(cfg, speaker_id="bob", display_name="Robert", store=store)
    assert key2 == key
    assert not created2
    assert "Stay" in store.read(key)


def test_normalize_people_payload_filters_unknown_ids():
    raw = """
    {"people": [
      {"speaker_id": "davide", "facts": ["Owns infra"], "aliases": [], "tags": ["eng"], "summary": ""},
      {"speaker_id": "stranger", "facts": ["Nope"], "aliases": [], "tags": [], "summary": ""}
    ]}
    """
    out = _normalize_people_payload(raw, {"davide"})
    assert len(out) == 1
    assert out[0]["speaker_id"] == "davide"
    assert out[0]["facts"] == ["Owns infra"]
    assert out[0]["tags"] == ["eng"]


def test_format_speakers_section_wikilinks():
    from pawn_diarize.core.vault_transcript import format_speakers_section

    segs = [
        {
            "audio_file": "a.wav",
            "label": "SPEAKER_00",
            "start_time": 0.0,
            "end_time": 10.0,
            "text": "hi",
        }
    ]
    names = {("a.wav", "SPEAKER_00"): "Davide"}
    md = format_speakers_section(segs, names, people_links={"Davide": "People/davide"})
    assert "[[People/davide|Davide]]" in md
