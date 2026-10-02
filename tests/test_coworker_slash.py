"""Slash commands for ideas and goals."""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import MagicMock

import pytest

from pawn_agent.core.coworker.review import extract_goals_block
from pawn_agent.core.coworker.slash import resolve_chat_message
from pawn_agent.tools.goals_impl import goal_propose_impl
from pawn_agent.tools.ideas_impl import capture_idea, render_idea_note
from pawn_agent.utils.config import AgentConfig
from pawn_core.goals import GoalInsertError, insert_goal_thread, parse_goals
from pawn_core.vault import VaultNotFound, VaultStore, VaultWriteDenied

SAMPLE = """---
pawn: goals
timezone: Europe/Rome
notify:
  max_per_day: 3
---

## Active

### Project X storage
- why: Blocking Friday.
- movement: A chosen option.
- interrupt: A new commitment.
- owner: Ada

## Parked

### Mobile inbox
- note: [[Ideas/Mobile inbox]]
- do: Link meetings.

## Done recently

### Vault transcripts
- closed: 2026-09-20
"""


class MemStore:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.skip: list[bool] = []

    def read(self, key: str) -> str:
        if key not in self.files:
            raise VaultNotFound(key)
        return self.files[key]

    def write(self, key: str, body: str, skip_guards: bool = False) -> None:
        self.files[key] = body
        self.skip.append(skip_guards)


def _cfg() -> AgentConfig:
    return AgentConfig(
        coworker={
            "goals_path": "Goals.md",
            "watch_folders": ["Ideas/"],
            "reviews_dir": "Pawn/Reviews",
            "timezone": "UTC",
            "enabled": True,
        }
    )


def test_insert_preserves_other_sections() -> None:
    updated = insert_goal_thread(
        SAMPLE,
        name="Multi-model",
        status="active",
        why="Ship more than one chat model.",
    )
    assert "timezone: Europe/Rome" in updated
    assert "- owner: Ada" in updated
    assert "### Vault transcripts" in updated
    assert "### Mobile inbox" in updated
    goals = parse_goals(updated)
    assert [thread.name for thread in goals.active] == ["Project X storage", "Multi-model"]
    added = goals.thread_named("Multi-model")
    assert added is not None
    assert added.why == "Ship more than one chat model."
    assert added.movement == ""
    assert added.interrupt == ""
    parked_at = updated.index("## Parked")
    added_at = updated.index("### Multi-model")
    assert added_at < parked_at


def test_insert_refuses_duplicate_and_invalid_note() -> None:
    with pytest.raises(GoalInsertError):
        insert_goal_thread(SAMPLE, name="project x storage", status="active", why="x")
    with pytest.raises(GoalInsertError):
        insert_goal_thread("---\npawn: task\n---\n## Active\n", name="Nope", status="active")


def test_insert_missing_note_is_a_goals_file() -> None:
    created = insert_goal_thread(None, name="Multi-model", status="parked")
    goals = parse_goals(created)
    assert goals.valid
    assert goals.parked[0].name == "Multi-model"
    assert "Do not notify" in goals.parked[0].do


def test_capture_idea_writes_once() -> None:
    store = MemStore()
    cfg = _cfg()
    line = "implement multi-model in pawnai"
    first = capture_idea(cfg, line=line, store=store, today=date(2026, 9, 29))
    key = "Ideas/2026-09-29 implement multi-model in pawnai.md"
    assert first == f"Captured {key}"
    body = store.files[key]
    assert "tags: [idea]" in body
    assert "status: inbox" in body
    assert "pawn: editable" not in body
    assert "## Seed" not in body
    assert f"# implement multi-model in pawnai\n\n{line}\n" in body
    assert store.skip == [True]
    store.files[key] = "kept"
    second = capture_idea(cfg, line=line, store=store, today=date(2026, 9, 29))
    assert "Already captured" in second
    assert store.files[key] == "kept"


def test_idea_note_is_denied_to_note_write() -> None:
    store = VaultStore(bucket="b", client=MagicMock())
    body = render_idea_note(title="Multi-model", line="implement multi-model")
    with pytest.raises(VaultWriteDenied):
        store.assert_writable("Ideas/2026-09-29 Multi-model.md", existing_body=body)


def test_goal_propose_stays_under_pawn() -> None:
    store = MemStore()
    store.files["Goals.md"] = SAMPLE
    cfg = _cfg()
    receipt = goal_propose_impl(
        cfg,
        name="Multi-model",
        why="One chat model is not enough.",
        movement="A chosen default and a way to override it.",
        interrupt="A request that names a model Pawn cannot run.",
        status="active",
        store=store,
    )
    key = "Pawn/Reviews/goal-proposal.md"
    assert key in receipt
    assert store.files["Goals.md"] == SAMPLE
    assert store.skip == [False]
    proposed = extract_goals_block(store.files[key])
    goals = parse_goals(proposed)
    assert goals.thread_named("Multi-model") is not None
    assert goals.thread_named("Project X storage") is not None


def test_slash_goal_writes_and_idea_rewrites() -> None:
    store = MemStore()
    cfg = _cfg()
    added = asyncio.run(
        resolve_chat_message(cfg, "/goal implement multi-model in pawnai", store=store)
    )
    assert added.mode == "reply"
    assert "Active" in added.text
    goals = parse_goals(store.files["Goals.md"])
    thread = goals.thread_named("implement multi-model in pawnai")
    assert thread is not None
    assert thread.why == "implement multi-model in pawnai"
    assert thread.movement == ""
    assert thread.interrupt == ""
    assert store.skip == [True]

    again = asyncio.run(
        resolve_chat_message(cfg, "/goal implement multi-model in pawnai", store=store)
    )
    assert "already has a thread" in again.text

    parked = asyncio.run(resolve_chat_message(cfg, "/park mobile inbox", store=store))
    assert parked.mode == "reply"
    assert parse_goals(store.files["Goals.md"]).thread_named("mobile inbox") is not None

    idea = asyncio.run(
        resolve_chat_message(cfg, "/idea implement multi-model in pawnai", store=store)
    )
    assert idea.mode == "reply"
    assert not idea.rewritten
    assert "idea_capture" not in idea.text
    idea_keys = [key for key in store.files if key.startswith("Ideas/")]
    assert len(idea_keys) == 1
    note = store.files[idea_keys[0]]
    assert "status: inbox" in note
    assert "implement multi-model in pawnai" in note
    assert "## Seed" not in note

    usage = asyncio.run(resolve_chat_message(cfg, "/idea", store=store))
    assert usage.mode == "reply"
    assert "Usage" in usage.text


def test_goal_apply_writes_the_proposal() -> None:
    store = MemStore()
    cfg = _cfg()
    goal_propose_impl(
        cfg,
        name="Multi-model",
        why="One model is not enough.",
        movement="A default plus an override.",
        interrupt="A named model Pawn cannot run.",
        store=store,
    )
    before = store.files.get("Goals.md")
    assert before is None
    applied = asyncio.run(resolve_chat_message(cfg, "/goal apply", store=store))
    assert applied.text.startswith("Wrote Goals.md")
    assert parse_goals(store.files["Goals.md"]).thread_named("Multi-model") is not None


def test_triage_requires_the_loop() -> None:
    cfg = _cfg()
    cfg.coworker.enabled = False
    plain = asyncio.run(resolve_chat_message(cfg, "file abcd1234"))
    assert plain.mode == "prompt"
    assert plain.text == "file abcd1234"

    cfg.coworker.enabled = True
    called: dict[str, str] = {}

    async def _apply(cfg, item_id, action, arg, **kwargs):  # noqa: ARG001
        called["action"] = action
        called["id"] = item_id
        return f"Filed {item_id}."

    from pawn_agent.core.coworker import actions

    original = actions.apply_action
    actions.apply_action = _apply  # type: ignore[assignment]
    try:
        receipt = asyncio.run(resolve_chat_message(cfg, "file abcd1234"))
    finally:
        actions.apply_action = original
    assert receipt.text == "Filed abcd1234."
    assert called == {"action": "file", "id": "abcd1234"}


def test_inbox_lists_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()

    def _list(dsn, **kwargs):  # noqa: ARG001
        return [
            {
                "short_id": "a1b2c3d4",
                "kind": "decision",
                "text": "Ship Friday",
                "thread": "X",
                "interrupt": True,
            },
            {
                "short_id": "eeeeeeee",
                "kind": "note",
                "text": "Quiet",
                "thread": "",
                "interrupt": False,
            },
        ]

    monkeypatch.setattr("pawn_agent.core.coworker.db.list_items", _list)
    reply = asyncio.run(resolve_chat_message(cfg, "/inbox"))
    assert "a1b2c3d4" in reply.text
    assert "Quiet" not in reply.text
