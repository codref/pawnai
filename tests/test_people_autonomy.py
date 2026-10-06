"""Autonomy + approve path for people_refresh (no LLM / S3)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pawn_agent.core.coworker.autonomy import decide
from pawn_agent.core.people.notes import parse_person_note, render_person_note


def test_decide_people_refresh_limited_act_allows():
    cfg = SimpleNamespace(
        coworker=SimpleNamespace(
            autonomy=SimpleNamespace(
                mode="limited_act", auto_actions=["research", "people_refresh"]
            ),
        ),
        sallm=SimpleNamespace(state_dir=".sallm"),
    )
    with patch("pawn_agent.core.coworker.autonomy.is_paused", return_value=False):
        d = decide(cfg, "people_refresh", writes_outside_pawn=False)
    assert d.decision == "allow"


def test_decide_people_refresh_suggest_only_needs_approval():
    cfg = SimpleNamespace(
        coworker=SimpleNamespace(
            autonomy=SimpleNamespace(mode="suggest_only", auto_actions=["people_refresh"]),
        ),
        sallm=SimpleNamespace(state_dir=".sallm"),
    )
    with patch("pawn_agent.core.coworker.autonomy.is_paused", return_value=False):
        d = decide(cfg, "people_refresh", writes_outside_pawn=False)
    assert d.decision == "needs_approval"


def test_approve_people_update_applies_payload():
    from pawn_agent.core.coworker import actions

    store = MagicMock()
    store.write = MagicMock()
    applied: list = []

    def fake_apply(cfg, updates, store=None):
        applied.extend(updates)
        return ["People/davide.md"]

    item = {
        "id": "uuid-1",
        "short_id": "abcd1234",
        "kind": "people_update",
        "text": "updates",
        "thread": "People",
        "quote": "",
        "reason": "",
        "interrupt": True,
        "note_key": "Pawn/Items/abcd1234.md",
        "fingerprint": "fp",
        "payload": {
            "action_kind": "people_refresh",
            "updates": [{"speaker_id": "davide", "facts": ["2026-10-06: fact"]}],
        },
    }
    cfg = SimpleNamespace(db_dsn="sqlite://", coworker=SimpleNamespace())

    with (
        patch("pawn_agent.core.coworker.actions.itemdb.get_item", return_value=item),
        patch(
            "pawn_agent.core.coworker.actions.itemdb.update_item",
            return_value={**item, "status": "filed"},
        ),
        patch("pawn_agent.core.coworker.actions._store", return_value=store),
        patch(
            "pawn_agent.core.people.refresh.apply_people_updates",
            side_effect=fake_apply,
        ),
    ):
        receipt = asyncio.run(actions.apply_action(cfg, "abcd1234", "approve"))

    assert "Applied people update" in receipt
    assert applied and applied[0]["speaker_id"] == "davide"


def test_person_note_rejects_non_person_frontmatter():
    with pytest.raises(ValueError, match="pawn must be"):
        parse_person_note("---\npawn: editable\n---\n\n# X\n")


def test_render_includes_person_tag():
    md = render_person_note(speaker_id="x", display_name="X", tags=["vendor"])
    note = parse_person_note(md)
    assert note.tags[0] == "person"
    assert "vendor" in note.tags
