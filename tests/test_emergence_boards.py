"""The ``boards`` night job, board dreaming, and board slots in the directed pass."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from pawn_agent.core.emergence import board_night, dream, night, route
from tests.test_emergence_subjects import _vec

BASE = datetime(2026, 11, 1)


def _obs(i: int, vec: list[float], **extra: Any) -> dict[str, Any]:
    return {
        "id": f"o{i}",
        "text": f"note {i}",
        "speaker_name": "Mira",
        "source_ref": f"s{i}",
        "created_at": BASE + timedelta(days=i),
        "embedding": vec,
        **extra,
    }


def _cfg(**em: Any) -> SimpleNamespace:
    defaults = dict(
        route_ambiguous_distance=0.45,
        board_backfill_days=30,
        board_max_candidates=24,
        board_dream_pairs=6,
        board_dream_max=1,
    )
    return SimpleNamespace(db_dsn="x", emergence=SimpleNamespace(**{**defaults, **em}))


@pytest.fixture
def question() -> dict[str, Any]:
    return {
        "id": "Q1",
        "kind": "question",
        "title": "Why does team B slip?",
        "status": "active",
        "state_md": "",
        "fields": {"board": "b-1"},
        "centroid": _vec(0, 1),
    }


def test_backfill_links_judged_candidates_and_skips_cut_pairs(
    monkeypatch: pytest.MonkeyPatch, question: dict[str, Any]
) -> None:
    pool = [
        _obs(1, _vec(0, 1)),
        _obs(2, _vec(0, 1, 2)),
        _obs(3, _vec(5)),  # far away: not a candidate
        _obs(4, _vec(0, 1), text="cut by the user"),
        _obs(5, _vec(0, 1)),  # already on the board
    ]
    linked: list[tuple] = []
    events: list[tuple] = []
    prompts: list[str] = []
    monkeypatch.setattr(board_night.subj, "recent_observations", lambda dsn, **kw: pool)
    monkeypatch.setattr(
        board_night.subj, "subject_observations", lambda dsn, sid, **kw: [_obs(5, _vec(0, 1))]
    )
    monkeypatch.setattr(board_night.labels, "rejected_link_keys", lambda dsn: {"k"})
    monkeypatch.setattr(
        board_night.labels, "link_rejected", lambda keys, sid, text: text == "cut by the user"
    )

    async def ask(cfg: Any, prompt: str, **kw: Any) -> Any:
        prompts.append(prompt)
        return {
            "links": [{"n": 1, "role": "evidence_against"}, {"n": 2, "role": "bogus"}, {"n": 9}]
        }

    monkeypatch.setattr(board_night, "ask_json", ask)
    monkeypatch.setattr(
        board_night.subj, "link", lambda dsn, oid, sid, **kw: linked.append((oid, sid, kw)) or True
    )
    monkeypatch.setattr(board_night.subj, "add_event", lambda *a, **k: events.append((a, k)))
    out = asyncio.run(board_night.backfill(_cfg(), question, budget=None))
    assert "note 3" not in prompts[0] and "cut by the user" not in prompts[0]
    assert "note 5" in prompts[0]  # shown as already on the board, not as a candidate
    assert [o["id"] for o in out] == ["o1"]
    assert linked[0][:2] == ("o1", "Q1") and linked[0][2]["linked_by"] == "board"
    assert events[0][0][2] == "linked" and events[0][1]["causes"] == ["o1"]


def test_work_boards_dreams_writes_note_and_provokes(
    monkeypatch: pytest.MonkeyPatch, question: dict[str, Any]
) -> None:
    monkeypatch.setattr(board_night, "board_subjects", lambda dsn: [question])

    async def backfill(cfg: Any, s: dict, **kw: Any) -> list[dict]:
        return [
            {**_obs(1, _vec(0)), "role": "evidence_for"},
            {**_obs(2, _vec(0)), "role": "resolves"},
        ]

    async def dream_board(cfg: Any, store: Any, sid: str, **kw: Any) -> list[dict]:
        return [{"id": "H1", "title": "B waits on vendor API"}]

    async def ask(cfg: Any, prompt: str, **kw: Any) -> Any:
        return {"note": "Two notes came in; one answers it.", "answered": "partly"}

    posted: list[tuple] = []
    provoked: list[dict] = []
    monkeypatch.setattr(board_night, "backfill", backfill)
    monkeypatch.setattr(board_night.dream, "dream_board", dream_board)
    monkeypatch.setattr(board_night, "ask_json", ask)
    monkeypatch.setattr(
        board_night, "_post_note", lambda cfg, b, text, linked, hunches: posted.append((b, text))
    )
    monkeypatch.setattr(board_night.provocations, "add", lambda cfg, **kw: provoked.append(kw))
    out = asyncio.run(board_night.work_boards(_cfg(), object()))
    assert out == {"Q1": {"linked": 2, "hunches": 1, "board": "b-1"}}
    assert posted == [("b-1", "Two notes came in; one answers it.")]
    assert provoked[0]["kind"] == "board_update" and provoked[0]["subject_id"] == "Q1"
    assert "moved overnight" in provoked[0]["text"]


def test_quiet_board_gets_no_note(monkeypatch: pytest.MonkeyPatch, question: dict) -> None:
    async def nothing(*a: Any, **k: Any) -> list:
        return []

    monkeypatch.setattr(board_night, "board_subjects", lambda dsn: [question])
    monkeypatch.setattr(board_night, "backfill", nothing)
    monkeypatch.setattr(board_night.dream, "dream_board", nothing)
    monkeypatch.setattr(board_night, "_post_note", lambda *a: pytest.fail("no note expected"))
    assert asyncio.run(board_night.work_boards(_cfg(), object()))["Q1"]["linked"] == 0


def test_dream_board_uses_the_board_pool_and_parents_hunches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = [_obs(i, _vec(0, 1, 2 + i % 2), source_ref=f"s{i}") for i in range(6)]
    created: list[dict] = []

    async def ask(cfg: Any, prompt: str, **kw: Any) -> Any:
        if prompt.startswith("Be a strict"):
            return {"keep": True}
        return {"connected": True, "hypothesis": "h", "test": ""}

    monkeypatch.setattr(dream.subj, "subject_observations", lambda dsn, sid, limit=200: pool)
    monkeypatch.setattr(dream, "ask_json", ask)
    monkeypatch.setattr(dream.labels, "embed_many", lambda cfg, texts: [_vec(7)])
    monkeypatch.setattr(dream.subj, "list_subjects", lambda dsn, kinds=None: [])
    monkeypatch.setattr(
        dream.subj, "create_subject", lambda dsn, **f: created.append(f) or {"id": "H", **f}
    )
    monkeypatch.setattr(dream.subj, "link", lambda *a, **k: True)
    monkeypatch.setattr(dream.subj, "get_subject", lambda dsn, sid: None)
    monkeypatch.setattr(dream.vault_notes, "render_subject", lambda *a, **k: "k")
    monkeypatch.setattr(dream.provocations, "add", lambda *a, **k: None)
    made = asyncio.run(dream.dream_board(_cfg(), object(), "Q1", seed=2))
    assert len(made) == 1 and created[0]["parent_id"] == "Q1"
    assert not asyncio.run(dream.dream_board(_cfg(board_dream_max=0), object(), "Q1"))


def test_pairs_accept_a_closer_window() -> None:
    near = [_obs(1, _vec(0, 1, 2)), _obs(2, _vec(0, 1, 3))]
    assert dream._pairs(near, 4, seed=1) == []  # one day apart: the global rule says no
    assert dream._pairs(near, 4, seed=1, min_days=1, band=(0.0, 1.0))


def test_boards_job_is_registered_after_sync() -> None:
    from pawn_agent.utils.config import EmergenceConfig

    assert night.JOBS.index("boards") == night.JOBS.index("sync") + 1
    jobs = EmergenceConfig().night_jobs
    assert jobs.index("boards") < jobs.index("synthesis")


def test_directed_pass_always_offers_board_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    near = [
        {"id": f"s{i}", "kind": "thread", "title": f"t{i}", "fields": {}, "centroid": _vec(0)}
        for i in range(3)
    ]
    far = {
        "id": "Q1",
        "kind": "question",
        "title": "board q",
        "fields": {"board": "b-1"},
        "centroid": _vec(7),
    }
    monkeypatch.setattr(route.subj, "list_subjects", lambda dsn, statuses=None: near + [far])
    monkeypatch.setattr(route.labels, "rejected_link_keys", lambda dsn: set())
    offered: list[str] = []

    async def ask(cfg: Any, prompt: str, **kw: Any) -> Any:
        offered.append(prompt)
        return {"findings": []}

    monkeypatch.setattr(route, "ask_json", ask)
    window = SimpleNamespace(embedding=_vec(0), window=SimpleNamespace(text="talk", index=0))
    result = SimpleNamespace(source_ref="sX", windows=[window])
    cfg = SimpleNamespace(
        db_dsn="x", emergence=SimpleNamespace(directed_max_subjects=2, board_directed_slots=3)
    )
    asyncio.run(route.directed_pass(cfg, result))
    assert "board q" in offered[0]
    cfg.emergence.board_directed_slots = 0
    offered.clear()
    asyncio.run(route.directed_pass(cfg, result))
    assert "board q" not in offered[0]
