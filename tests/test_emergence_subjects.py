"""Emergence phases 3-7: pure logic (no database)."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from pawn_agent.core.emergence import dream, night, provocations, reflect, search, threads
from pawn_agent.core.emergence.llm import Budget, BudgetExhausted
from pawn_agent.core.emergence.route import parse_when
from pawn_agent.core.emergence.vault_notes import merge_note, split_sections


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-03-31", date(2026, 3, 31)),
        ("by end of Q1", date(2027, 3, 31)),
        ("end of Q4 2026", date(2026, 12, 31)),
        ("in March", date(2027, 3, 31)),
        ("December", date(2026, 12, 31)),
        ("next week", date(2026, 12, 5)),
        ("within a few weeks", date(2026, 12, 19)),
        ("end of year", date(2026, 12, 31)),
        ("someday", None),
        ("", None),
    ],
)
def test_parse_when(text, expected):
    assert parse_when(text, today=date(2026, 11, 28)) == expected


def _vec(*hot: int) -> list[float]:
    v = np.zeros(8)
    for i in hot:
        v[i] = 1.0
    return v.tolist()


def test_cluster_groups_similar_observations():
    obs = [
        {"id": "a", "embedding": _vec(0, 1)},
        {"id": "b", "embedding": _vec(0, 1, 2)},
        {"id": "c", "embedding": _vec(5)},
        {"id": "d", "embedding": _vec(0, 1)},
        {"id": "e", "embedding": None},
    ]
    groups = threads.cluster(obs, max_distance=0.3)
    assert [[o["id"] for o in g] for g in groups] == [["a", "b", "d"], ["c"]]


def test_merge_note_replaces_managed_and_keeps_user_sections():
    existing = (
        "---\npawn: thread\nstatus: proposed\nmine: keep\n---\n\n# Old title\n\n"
        "## State\n\nold state\n\n## Filed\n\n- filed by coworker\n\n## Notes\n\nmy notes\n"
    )
    out = merge_note(
        existing,
        meta={"pawn": "thread", "status": "active"},
        title="New title",
        managed=[("State", "new state"), ("Timeline", "")],
    )
    assert "mine: keep" in out and "status: active" in out
    assert "old state" not in out and "new state" in out
    assert "- filed by coworker" in out and "my notes" in out
    assert out.index("## State") < out.index("## Timeline") < out.index("## Filed")
    assert "_(nothing yet)_" in out
    _, sections = split_sections(out.split("---\n", 2)[2])
    assert [name for name, _ in sections] == ["State", "Timeline", "Filed", "Notes"]


def test_parse_verdict():
    assert provocations.parse_verdict("useful p3a1") == ("useful", "p3a1")
    assert provocations.parse_verdict("Not now P3A1") == ("not_now", "p3a1")
    assert provocations.parse_verdict("not-now p3a1") == ("not_now", "p3a1")
    assert provocations.parse_verdict("later p3a1") == ("not_now", "p3a1")
    assert provocations.parse_verdict("useful") is None
    assert provocations.parse_verdict("file a1b2c3d4") is None


def test_heuristic_score_orders_kinds_and_features():
    broken = provocations.score("assumption_broken", {"tension": 1.0})
    hunch = provocations.score("hypothesis", {"novelty": 1.0})
    goal_momentum = provocations.score("momentum", {"goal_thread": 1.0, "momentum": 3.0})
    plain_momentum = provocations.score("momentum", {"momentum": 3.0})
    assert broken > goal_momentum > plain_momentum > hunch


def test_learned_score_uses_weights():
    n = len(provocations._vector("momentum", {}))
    weights = np.zeros(n + 1)
    weights[list(provocations.KIND_WEIGHT).index("hypothesis")] = 5.0
    assert provocations.score("hypothesis", {}, weights) > provocations.score(
        "momentum", {}, weights
    )


def test_period_for():
    today = date(2026, 11, 28)  # a Saturday
    assert reflect.period_for("week", today) == ("2026-W47", date(2026, 11, 16), date(2026, 11, 23))
    assert reflect.period_for("month", today) == ("2026-10", date(2026, 10, 1), date(2026, 11, 1))
    assert reflect.period_for("quarter", today) == ("2026-Q3", date(2026, 7, 1), date(2026, 10, 1))
    assert reflect.period_for("quarter", date(2026, 2, 3))[0] == "2025-Q4"


def test_search_score():
    gold = [
        {"text": "Freeze the API on Monday", "quote": "", "positive": True},
        {"text": "Davide sends the deck", "quote": "", "positive": True},
        {"text": "We chatted about lunch", "quote": "", "positive": False},
    ]
    found = [{"text": "Freeze the API on Monday."}, {"text": "We chatted about lunch."}]
    s = search.score(found, gold)
    assert (s["hits"], s["false_alarms"], s["misses"]) == (1, 1, 1)
    assert s["f1"] == 0.5
    assert search.score([], [])["f1"] == 0.0


def test_dream_pairs_respect_time_and_distance():
    base = datetime(2026, 9, 1)
    obs = [
        {"id": "a", "source_ref": "s1", "created_at": base, "embedding": _vec(0, 1, 2)},
        {
            "id": "b",
            "source_ref": "s2",
            "created_at": base + timedelta(days=30),
            "embedding": _vec(0, 1, 3),
        },
        {
            "id": "c",
            "source_ref": "s3",
            "created_at": base + timedelta(days=1),
            "embedding": _vec(0, 1, 2),
        },
        {
            "id": "d",
            "source_ref": "s1",
            "created_at": base + timedelta(days=60),
            "embedding": _vec(0, 1, 3),
        },
    ]
    pairs = dream._pairs(obs, 10, seed=1)
    assert pairs
    for a, b, distance in pairs:
        assert a["source_ref"] != b["source_ref"]
        assert abs((a["created_at"] - b["created_at"]).days) >= 7
        assert 0.3 <= distance <= 0.65


def test_budget():
    budget = Budget(max_calls=2, max_seconds=60)
    budget.spend()
    budget.spend()
    assert budget.exhausted
    with pytest.raises(BudgetExhausted):
        budget.spend()


def test_run_night_isolates_failures_and_budget(monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []

    async def job(name, cfg, store, budget, today):  # noqa: ARG001
        seen.append(name)
        if name == "clustering":
            raise RuntimeError("boom")
        if name == "synthesis":
            raise BudgetExhausted("spent")
        return name.upper()

    monkeypatch.setattr(night, "_job", job)
    cfg = SimpleNamespace(emergence=SimpleNamespace(night_jobs=None), db_dsn="unused")
    results = asyncio.run(
        night.run_night(
            cfg,
            store=object(),
            jobs=["sync", "clustering", "synthesis", "people"],
            budget=Budget(max_calls=5),
        )
    )
    assert results["sync"] == "SYNC" and results["people"] == "PEOPLE"
    assert results["clustering"].startswith("error: boom")
    assert results["synthesis"] == "stopped: budget"
    exhausted = Budget(max_calls=0)
    results = asyncio.run(
        night.run_night(cfg, store=object(), jobs=["clustering", "commitments"], budget=exhausted)
    )
    assert results["clustering"] == "skipped: budget" and results["commitments"] == "COMMITMENTS"


def test_chat_routes_provocation_verdicts(monkeypatch: pytest.MonkeyPatch):
    from pawn_agent.core.coworker import slash

    directive = slash.classify_chat_text("useful p3a1", triage=True)
    assert (directive.kind, directive.command, directive.arg, directive.extra) == (
        "direct",
        "provocation",
        "p3a1",
        "useful",
    )
    assert slash.classify_chat_text("useful p3a1", triage=False).kind == "prompt"
    monkeypatch.setattr(provocations, "apply_verdict", lambda cfg, sid, v: f"{v}:{sid}")
    assert slash._run_direct(SimpleNamespace(), directive, store=None) == "useful:p3a1"


def test_meeting_prep_includes_ledger(monkeypatch: pytest.MonkeyPatch):
    from pawn_agent.core.emergence import ledger
    from pawn_server.core import meeting_prep

    monkeypatch.setattr(ledger, "meeting_brief", lambda cfg, names: "**They owe**\n- Luca: RFC")
    cfg = SimpleNamespace(
        emergence=SimpleNamespace(enabled=True), coworker=SimpleNamespace(goals_path="Goals.md")
    )
    prompt = meeting_prep.build_prep_prompt(cfg, path="p.md", event={"attendees": ["Luca"]})
    assert "Pawn ledger for these attendees" in prompt and "Luca: RFC" in prompt
    cfg.emergence.enabled = False
    assert "Pawn ledger" not in meeting_prep.build_prep_prompt(
        cfg, path="p.md", event={"attendees": ["Luca"]}
    )


def test_briefing_provocations_only_when_enabled(monkeypatch: pytest.MonkeyPatch):
    from pawn_server.core import coworker_worker

    off = SimpleNamespace(emergence=SimpleNamespace(enabled=False))
    assert coworker_worker._provocations_for(off, "2026-11-28") == []
    monkeypatch.setattr(
        provocations, "select_for_day", lambda cfg, day: [{"text": "t", "short_id": "p1"}]
    )
    on = SimpleNamespace(emergence=SimpleNamespace(enabled=True))
    assert coworker_worker._provocations_for(on, "2026-11-28")[0]["short_id"] == "p1"
    lines = provocations.render_section([{"text": "Revisit X?", "short_id": "p1"}])
    assert lines[0] == "## Provocations" and "- Revisit X? `p1`" in lines


def test_speaker_comes_from_the_transcript_not_the_model():
    from pawn_agent.core.coworker.extract import Window, speaker_at
    from pawn_agent.core.emergence.pipeline import verify

    body = "[00:10.00] Ansgar:\n  We freeze the API on Monday.\n[00:40.00] Davide:\n  Fine by me.\n"
    assert speaker_at(body, body.index("Fine")) == "Davide"
    assert speaker_at(body, 0) == "Ansgar"
    obs = {"quote": "We freeze the API on Monday", "speaker_name": "Ansgram"}
    verify(obs, Window(index=0, text=body, char_start=0), body, 0.8)
    assert obs["status"] == "live" and obs["speaker_name"] == "Ansgar"
    fake = {"quote": "nothing like this was said here at all", "speaker_name": "Ansgram"}
    verify(fake, Window(index=0, text=body, char_start=0), body, 0.8)
    assert fake["status"] == "rejected" and fake["speaker_name"] == "Ansgram"


def test_timeline_dedupe_and_raw_labels():
    from pawn_agent.core.emergence.reflect import _RAW_LABEL
    from pawn_agent.core.emergence.vault_notes import dedupe

    rows = [
        {"text": "No automated tests.", "source_ref": "m1", "start_sec": 2930.0, "type": "concern"},
        {"text": "no automated  tests.", "source_ref": "m1", "start_sec": 2930.4, "type": "fact"},
        {"text": "No automated tests.", "source_ref": "m2", "start_sec": 10.0, "type": "fact"},
    ]
    assert [r["source_ref"] for r in dedupe(rows)] == ["m1", "m2"]
    assert _RAW_LABEL.match("SPEAKER_01") and _RAW_LABEL.match("speaker 3")
    assert not _RAW_LABEL.match("Ansgar")


def test_fingerprint_key_ignores_changing_text():
    a = provocations.fingerprint("recurring_concern", None, "Luca raised 3 concerns", "person:luca")
    b = provocations.fingerprint("recurring_concern", None, "Luca raised 5 concerns", "person:luca")
    c = provocations.fingerprint("recurring_concern", None, "Luca raised 3 concerns")
    d = provocations.fingerprint("recurring_concern", None, "Luca raised 5 concerns")
    assert a == b and c != d


def test_dream_stops_at_the_nightly_cap(monkeypatch: pytest.MonkeyPatch):
    base = datetime(2026, 9, 1)
    pool = [
        {
            "id": str(i),
            "text": f"t{i}",
            "source_ref": f"s{i}",
            "created_at": base + timedelta(days=10 * i),
            "embedding": _vec(0, 1, 2 + i % 2),
        }
        for i in range(8)
    ]
    calls: list[str] = []

    async def ask(cfg, prompt, **kw):  # noqa: ARG001
        calls.append(prompt[:10])
        if prompt.startswith("Be a strict"):
            return {"keep": True}
        return {"connected": True, "hypothesis": f"h{len(calls)}", "test": ""}

    created: list[dict] = []
    monkeypatch.setattr(dream.subj, "recent_observations", lambda dsn, days: pool)
    monkeypatch.setattr(dream, "ask_json", ask)
    monkeypatch.setattr(dream.labels, "embed_many", lambda cfg, texts: [_vec(len(created) % 8)])
    monkeypatch.setattr(dream.subj, "list_subjects", lambda dsn, kinds=None: [])
    monkeypatch.setattr(
        dream.subj,
        "create_subject",
        lambda dsn, **f: created.append(f) or {"id": str(len(created)), **f},
    )
    monkeypatch.setattr(dream.subj, "link", lambda *a, **k: True)
    monkeypatch.setattr(dream.subj, "get_subject", lambda dsn, sid: None)
    monkeypatch.setattr(dream.vault_notes, "render_subject", lambda *a, **k: "k")
    monkeypatch.setattr(dream.provocations, "add", lambda *a, **k: None)
    cfg = SimpleNamespace(
        emergence=SimpleNamespace(dream_pairs=10, dream_max_hypotheses=2), db_dsn="x"
    )
    assert asyncio.run(dream.dream(cfg, object(), seed=3)) == 2
    assert len(created) == 2
