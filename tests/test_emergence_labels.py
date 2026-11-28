"""Emergence Phase 2: labels, review-note import, ranker, few-shot."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pawn_agent.core.emergence import labels, passes, prompts, ranker
from pawn_agent.core.emergence.pipeline import observe_source

NOTE = """---
pawn: extract-compare
source_ref: meet-1
model: m
reviewed: {reviewed}
---

# Extraction compare: meet-1

## Observation items (3)

- [x] [02:45] **commitment** Davide will share the output. — owner: Davide · *in both* · rank 0.81
    > So I will share the output.
- [ ] [1:00:27] **open_question** Is Q2 realistic? — owner: Ansgar
    > is Q2 even realistic
- [X] [10:00] **decision** Quality first in Q1. · rank 0.40

## Single-prompt items (1)

- [ ] [04:36] **decision** Quality must be the top priority for Q1. — owner: Ansgar, due 2026-01-01 · *in both*
    > quality needs to be top of mind for Q1

## Other observations (stored, not items)

### concern (2)

- [x] [13:41] Politics beat evidence. — Davide · rank 0.90
- [ ] [15:16] Complexity is high. — Ansgar

## Rejected (2)

Tick any that were real (the quote check was wrong).

- [x] **decision** Hire two contractors. (quote_not_found, match 0.2)
- [ ] **commitment** Fake promise. (no_quote, match 0.0)
"""


def _cfg(**em: Any) -> SimpleNamespace:
    base = dict(
        enabled=True,
        fast_model="",
        window_chars=2000,
        window_overlap=300,
        triage_threshold=0.35,
        quote_min_score=0.8,
        types=["decision", "commitment"],
        always_types=["decision", "commitment"],
        think_types=[],
        few_shot=True,
        few_shot_accept=3,
        few_shot_reject=2,
        ranker=True,
        ranker_min_labels=30,
        ranker_min_score=0.0,
        ranker_min_auc=0.7,
        train_on_bulk=False,
    )
    base.update(em)
    return SimpleNamespace(
        emergence=SimpleNamespace(**base),
        coworker=SimpleNamespace(reviews_dir="Pawn/Reviews"),
        chat_model_id="test@m",
        db_dsn="unused",
    )


def test_parse_review_note_sections_and_suffixes():
    meta, entries = labels.parse_review_note(NOTE.format(reviewed="false"))
    assert meta["source_ref"] == "meet-1"
    got = [(e["section"], e["type"], e["text"], e["ticked"], e["start_sec"]) for e in entries]
    assert got == [
        ("observations", "commitment", "Davide will share the output.", True, 165),
        ("observations", "question", "Is Q2 realistic?", False, 3627),
        ("observations", "decision", "Quality first in Q1.", True, 600),
        ("single", "decision", "Quality must be the top priority for Q1.", False, 276),
        ("other", "concern", "Politics beat evidence.", True, 821),
        ("other", "concern", "Complexity is high.", False, 916),
        ("rejected", "decision", "Hire two contractors.", True, None),
        ("rejected", "commitment", "Fake promise.", False, None),
    ]
    assert entries[0]["quote"] == "So I will share the output."


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    monkeypatch.setattr(labels, "embed_many", lambda cfg, texts: [[0.1] * 4 for _ in texts])
    monkeypatch.setattr(labels, "upsert_label", lambda dsn, **f: rows.append(f) or f)
    return rows


def test_import_unreviewed_note_keeps_only_ticks(captured):
    counts = labels.import_review_note(_cfg(), "k", NOTE.format(reviewed="false"))
    assert counts == {"accept": 4, "reject": 0, "skipped": 4}
    assert {r["verdict"] for r in captured} == {"accept"}
    assert {r["surface"] for r in captured} == {"review"}


def test_import_reviewed_note_rejects_unticked_except_rejected_section(captured):
    counts = labels.import_review_note(_cfg(), "k", NOTE.format(reviewed="true"))
    assert counts == {"accept": 4, "reject": 3, "skipped": 1}
    single = next(r for r in captured if r["extractor"] == "single")
    assert single["verdict"] == "reject"
    # Same text in the same source → same target id on re-import.
    again: list[dict[str, Any]] = []
    labels.upsert_label = lambda dsn, **f: again.append(f) or f  # type: ignore[assignment]
    labels.import_review_note(_cfg(), "k2", NOTE.format(reviewed="true"))
    assert [r["target_id"] for r in again] == [r["target_id"] for r in captured]


def test_record_item_label_maps_actions(captured):
    item = {
        "id": "i1",
        "kind": "open_question",
        "text": "Is Q2 real?",
        "quote": "q",
        "owner": "",
        "source_ref": "s",
        "payload": {"start_sec": 12.0, "quote_score": 0.9},
    }
    labels.record_item_label(_cfg(), item, "todo", surface="inbox")
    labels.record_item_label(_cfg(), item, "ignore", surface="matrix")
    labels.record_item_label(_cfg(), item, "later", surface="inbox")
    labels.record_item_label(_cfg(), {**item, "kind": "proposal"}, "file", surface="inbox")
    labels.record_item_label(_cfg(enabled=False), item, "todo", surface="inbox")
    assert [(r["verdict"], r["surface"], r["type"]) for r in captured] == [
        ("accept", "inbox", "question"),
        ("reject", "matrix", "question"),
    ]
    assert captured[0]["target_kind"] == "item" and captured[0]["start_sec"] == 12.0


def test_record_item_label_never_raises(monkeypatch: pytest.MonkeyPatch):
    def boom(dsn, **f):
        raise RuntimeError("no table")

    monkeypatch.setattr(labels, "embed_many", lambda cfg, texts: [None])
    monkeypatch.setattr(labels, "upsert_label", boom)
    labels.record_item_label(
        _cfg(), {"id": "i", "kind": "decision", "text": "x"}, "delete", surface="inbox"
    )


def test_bulk_deletes_are_labelled_bulk(monkeypatch: pytest.MonkeyPatch):
    from pawn_agent.core.coworker import actions

    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        labels,
        "record_item_label",
        lambda cfg, item, action, *, surface, embed=True: seen.append((surface, embed)),
    )
    monkeypatch.setattr(actions, "_store", lambda cfg, store: None)
    monkeypatch.setattr(actions.itemdb, "get_item", lambda dsn, i: {"id": i, "fingerprint": i, "short_id": i})
    monkeypatch.setattr(actions.itemdb, "add_suppression", lambda *a: None)
    monkeypatch.setattr(actions.itemdb, "update_item", lambda *a, **k: None)
    monkeypatch.setattr(actions, "_delete_note", lambda *a, **k: None)
    cfg = _cfg()
    for ids, all_open in ((["a"], False), (["a", "b"], False), (["a"], True)):
        monkeypatch.setattr(actions, "_resolve_delete_ids", lambda cfg, **k: list(ids))
        asyncio.run(actions.delete_items(cfg, ids=ids, all_open=all_open))
    assert seen == [("inbox", True), ("bulk", False), ("bulk", False), ("bulk", False)]


def test_record_api_label_validates(captured):
    with pytest.raises(ValueError):
        labels.record_api_label(_cfg(), {"type": "nonsense", "text": "x"})
    with pytest.raises(ValueError):
        labels.record_api_label(_cfg(), {"type": "decision", "text": " "})
    with pytest.raises(ValueError):
        labels.record_api_label(_cfg(), {"type": "decision", "text": "x", "verdict": "maybe"})
    labels.record_api_label(_cfg(), {"type": "open_question", "text": "Why?"})
    assert captured[-1]["verdict"] == "missed" and captured[-1]["type"] == "question"


# ── ranker ───────────────────────────────────────────────────────────────────


def _rows(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        accept = i % 2 == 0
        vec = rng.normal(size=16) + (2.0 if accept else -2.0) * np.eye(16)[0]
        rows.append(
            {
                "type": "decision",
                "verdict": "accept" if accept else "reject",
                "features": {"quote_score": 1.0, "has_owner": accept},
                "embedding": vec.tolist(),
            }
        )
    return rows


def test_auc_basic():
    y = np.array([1, 1, 0, 0])
    assert ranker.auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 1.0
    assert ranker.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 0.0
    assert ranker.auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5


def test_ranker_learns_separable_labels(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(ranker, "model_path", lambda cfg: tmp_path / "ranker.json")
    meta = ranker.train(_cfg(), _rows(60), label_total=60)
    assert meta["status"] == "trained"
    assert meta["auc"] > 0.9
    model = ranker.load(_cfg())
    assert model["label_total"] == 60 and len(model["w"]) == 16 + 8 + 2
    good = {"type": "decision", "embedding": (np.eye(16)[0] * 3).tolist(), "fields": {}}
    bad = {"type": "decision", "embedding": (-np.eye(16)[0] * 3).tolist(), "fields": {}}
    assert ranker.score(model, good) > ranker.score(model, bad)
    assert ranker.score(None, good) is None
    assert ranker.score(model, {"type": "decision", "embedding": None}) is None


def test_ranker_needs_enough_labels(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(ranker, "model_path", lambda cfg: tmp_path / "ranker.json")
    meta = ranker.train(_cfg(), _rows(10), label_total=10)
    assert meta["status"] == "too_few_labels"
    assert "w" not in ranker.load(_cfg())


def test_ensure_fresh_retrains_on_new_labels(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(ranker, "model_path", lambda cfg: tmp_path / "ranker.json")
    total = {"n": 60}
    trained: list[int] = []
    monkeypatch.setattr(labels, "label_counts", lambda dsn: {"total": total["n"], "by": []})
    monkeypatch.setattr(labels, "training_rows", lambda dsn, include_bulk=False: _rows(total["n"]))
    real_train = ranker.train

    def counting(cfg, rows=None, *, label_total=None):
        trained.append(label_total)
        return real_train(cfg, rows, label_total=label_total)

    monkeypatch.setattr(ranker, "train", counting)
    ranker.ensure_fresh(_cfg())
    ranker.ensure_fresh(_cfg())
    total["n"] = 62
    model = ranker.ensure_fresh(_cfg())
    assert trained == [60, 62]
    assert "w" in model and model["label_total"] == 62


# ── few-shot + pipeline ──────────────────────────────────────────────────────


def test_examples_block():
    assert prompts.examples_block("decision", None) == ""
    assert prompts.examples_block("decision", {"accept": [], "reject": []}) == ""
    block = prompts.examples_block(
        "decision", {"accept": [{"text": "Ship Friday"}], "reject": [{"text": "We chatted"}]}
    )
    assert 'Wanted:\n- "Ship Friday"' in block and 'Not wanted:\n- "We chatted"' in block
    prompt = prompts.typed_prompt("decision", "SRC", "", {"accept": [{"text": "Ship Friday"}]})
    assert prompt.index("Ship Friday") < prompt.index("SOURCE:")


def _transcript() -> str:
    out = []
    for i in range(80):
        sec = i * 30
        out.append(
            f"[{sec // 60:02d}:{sec % 60:05.2f}] Davide:\n"
            f"  routine status update on roadmap and staffing, number {i}.\n"
        )
    out.append("[40:00.00] Mira:\n  Decision: we freeze the API on Monday.\n")
    return "".join(out)


def test_pipeline_uses_examples_and_ranks(monkeypatch: pytest.MonkeyPatch):
    from pawn_agent.core.emergence import pipeline

    prompts_seen: list[str] = []

    async def run(cfg, prompt, system_prompt=None, **kw):  # noqa: ARG001
        prompts_seen.append(prompt)
        if prompt.startswith("Score"):
            return '{"decision": 0.9, "commitment": 0.1}'
        if "freeze the API" in prompt.split("SOURCE:\n", 1)[1] and "Find every decision" in prompt:
            return json.dumps(
                {
                    "observations": [
                        {"text": "Freeze API Monday.", "quote": "we freeze the API on Monday"}
                    ]
                }
            )
        return '{"observations": []}'

    monkeypatch.setattr(passes, "llm_run", run)
    monkeypatch.setattr(labels, "label_counts", lambda dsn: {"total": 3, "by": []})
    monkeypatch.setattr(labels, "embed_many", lambda cfg, texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(
        labels,
        "few_shot_examples",
        lambda dsn, t, v, n_accept, n_reject: {
            "accept": [{"text": f"past {t} kept"}],
            "reject": [],
        },
    )
    monkeypatch.setattr(ranker, "ensure_fresh", lambda cfg: {"status": "trained", "auc": 0.8})
    monkeypatch.setattr(ranker, "score", lambda model, obs: 0.77)
    result = asyncio.run(
        observe_source(
            _cfg(), source_kind="session", source_ref="s", source_text=_transcript(), persist=False
        )
    )
    typed = [p for p in prompts_seen if p.startswith("Find every")]
    assert typed and all("past decision kept" in p or "past commitment kept" in p for p in typed)
    assert result.stats["few_shot_prompts"] == len(typed)
    assert result.stats["ranker"] == {"status": "trained", "auc": 0.8}
    live = result.live
    assert live and live[0]["embedding"] == [1.0, 0.0] and live[0]["rank_score"] == 0.77
    assert pipeline.project_items(result.observations)[0]["rank_score"] == 0.77


def test_few_shot_off_without_labels(monkeypatch: pytest.MonkeyPatch):
    prompts_seen: list[str] = []

    async def run(cfg, prompt, system_prompt=None, **kw):  # noqa: ARG001
        prompts_seen.append(prompt)
        return '{"decision": 0.9, "commitment": 0.9}' if prompt.startswith("Score") else "{}"

    calls: list[str] = []
    monkeypatch.setattr(passes, "llm_run", run)
    monkeypatch.setattr(labels, "label_counts", lambda dsn: {"total": 0, "by": []})
    monkeypatch.setattr(labels, "embed_many", lambda cfg, texts: calls.append("embed") or [])
    result = asyncio.run(
        observe_source(
            _cfg(ranker=False),
            source_kind="session",
            source_ref="s",
            source_text=_transcript(),
            persist=False,
        )
    )
    assert result.stats["few_shot_prompts"] == 0
    assert not calls
    assert not any("How this user judged" in p for p in prompts_seen)


def test_rank_filter_only_with_good_ranker():
    from pawn_agent.core.coworker.pipeline import _rank_filter

    items = [{"rank_score": 0.9}, {"rank_score": 0.1}, {"rank_score": None}]
    trained = {"ranker": {"status": "trained", "auc": 0.8}}
    assert len(_rank_filter(_cfg(), list(items), dict(trained))) == 3  # min_score 0
    stats = dict(trained)
    kept = _rank_filter(_cfg(ranker_min_score=0.5), list(items), stats)
    assert [i["rank_score"] for i in kept] == [0.9, None] and stats["ranker_filtered"] == 1
    weak = {"ranker": {"status": "trained", "auc": 0.6}}
    assert len(_rank_filter(_cfg(ranker_min_score=0.5), list(items), weak)) == 3
    untrained = {"ranker": {"status": "too_few_labels"}}
    assert len(_rank_filter(_cfg(ranker_min_score=0.5), list(items), untrained)) == 3


def test_compare_report_round_trips_through_parser(monkeypatch: pytest.MonkeyPatch):
    from pawn_agent.core.emergence.compare import Comparison, render_report
    from pawn_agent.core.emergence.pipeline import ObserveResult

    observed = ObserveResult("session", "s1", "r", "h", "m")
    observed.observations = [
        {
            "type": "concern",
            "status": "live",
            "text": "Politics beat evidence.",
            "speaker_name": "Davide",
            "start_sec": 821.0,
            "rank_score": 0.9,
        },
    ]
    observed.stats = {"model": "m"}
    items = [
        {
            "kind": "decision",
            "text": "Freeze API.",
            "owner": "Luca",
            "quote": "freeze it",
            "start_sec": 60.0,
            "rank_score": 0.66,
        }
    ]
    report = render_report(Comparison("s1", [], 0, 0.0, observed, items))
    assert "reviewed: false" in report
    ticked = re.sub(r"- \[ \]", "- [x]", report)
    _, entries = labels.parse_review_note(ticked)
    assert [(e["type"], e["text"], e["ticked"]) for e in entries] == [
        ("decision", "Freeze API.", True),
        ("concern", "Politics beat evidence.", True),
    ]
