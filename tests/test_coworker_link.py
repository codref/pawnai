"""Related-link query shaping and quality gates (no embeddings)."""

from pawn_agent.core.coworker.link import (
    compose_related_query,
    extract_anchors,
    format_related_label,
    select_related_hits,
)


def test_extract_anchors_keeps_proper_nouns():
    anchors = extract_anchors(
        "The team is waiting for a response from Miguel regarding an issue in Pitarosso.",
        quote="Miguel told me I'm waiting for his um his response",
    )
    assert "Miguel" in anchors
    assert "Pitarosso" in anchors
    assert "The" not in anchors
    assert "Team" not in anchors


def test_compose_related_query_includes_quote():
    q = compose_related_query("Waiting on Miguel", quote="his response on Pitarosso")
    assert "Waiting on Miguel" in q
    assert "Pitarosso" in q


def test_select_related_drops_far_and_unanchored_transcripts():
    hits = [
        {
            "source_kind": "transcript",
            "source_ref": "sportissimo-carlo-1007",
            "heading": "",
            "text": "we talked about the roadmap and hiring",
            "distance": 0.30,
        },
        {
            "source_kind": "transcript",
            "source_ref": "evo-kanban-earlier",
            "heading": "",
            "text": "Miguel still owes a response on Pitarosso",
            "distance": 0.32,
        },
        {
            "source_kind": "item",
            "source_ref": "Pawn/Items/2026-10-01-waiting-on-miguel-aabbccdd.md",
            "heading": "",
            "text": "Still waiting for Miguel on Pitarosso",
            "distance": 0.20,
        },
        {
            "source_kind": "transcript",
            "source_ref": "tech-huddle-1007",
            "heading": "",
            "text": "standup notes only",
            "distance": 0.60,
        },
    ]
    selected, recurrence = select_related_hits(
        hits,
        source_ref="evo-kanban-1008",
        anchors=["Miguel", "Pitarosso"],
        max_distance=0.45,
        limit=5,
    )
    refs = [h["source_ref"] for h in selected]
    assert refs[0].startswith("Pawn/Items/")
    assert "evo-kanban-earlier" in refs
    assert "sportissimo-carlo-1007" not in refs
    assert "tech-huddle-1007" not in refs
    assert recurrence == 1


def test_select_related_dedupes_source_ref():
    hits = [
        {
            "source_kind": "note",
            "source_ref": "Pawn/Notes/foo.md",
            "heading": "A",
            "text": "alpha",
            "distance": 0.40,
        },
        {
            "source_kind": "note",
            "source_ref": "Pawn/Notes/foo.md",
            "heading": "B",
            "text": "beta",
            "distance": 0.25,
        },
    ]
    selected, _rec = select_related_hits(
        hits,
        source_ref="other",
        anchors=[],
        max_distance=0.45,
        limit=5,
    )
    assert len(selected) == 1
    assert selected[0]["heading"] == "B"


def test_format_related_label_wikilinks_paths():
    assert (
        format_related_label({"source_ref": "Pawn/Items/x.md", "heading": ""})
        == "[[Pawn/Items/x.md]]"
    )
    assert format_related_label({"source_ref": "session-id", "heading": "Intro"}) == (
        "session-id — Intro"
    )
