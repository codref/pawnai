"""End-to-end emergence loop against a real PostgreSQL (skipped by default).

Set ``EMERGENCE_TEST_DSN`` to an empty, migrated database (``alembic
upgrade head``) to run. The model is a scripted fake and embeddings are a
deterministic bag of words, so routing and clustering are predictable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

DSN = os.environ.get("EMERGENCE_TEST_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="EMERGENCE_TEST_DSN not set")


class Store:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def read(self, key: str) -> str:
        return self.files[key]

    def write(self, key: str, text: str) -> None:
        self.files[key] = text

    def exists(self, key: str) -> bool:
        return key in self.files

    def list(self, prefix: str = "", *, suffix: str = ".md") -> list[str]:
        return sorted(k for k in self.files if k.startswith(prefix) and k.endswith(suffix))


_STOP = {"the", "a", "we", "on", "and", "to", "of", "is", "it", "i", "will", "by", "in", "for"}


def bag(text: str) -> list[float]:
    vec = np.zeros(1024)
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _STOP:
            continue
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 1024] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm if norm else vec).tolist()


MEETING_1 = """[00:10.00] Mira:
  We decide to freeze the API on Monday, assuming team B is ready by Monday.
[00:40.00] Davide:
  I will send the migration plan to Luca by 2026-12-01.
[01:10.00] Luca:
  I am worried the vendor contract renewal is too expensive.
[01:40.00] Mira:
  I predict the vendor contract renewal slips to Q1.
"""

MEETING_2 = """[00:05.00] Ansgar:
  The vendor contract renewal costs keep rising, I am worried.
[00:30.00] Luca:
  Vendor contract renewal is blocked by legal review, which worries me.
[01:00.00] Mira:
  Team B is not ready, they need two more weeks.
"""


def fake_model(meeting_texts: dict[str, list[dict]]):
    async def run(cfg, prompt, system_prompt=None, **kw):  # noqa: ARG001
        if prompt.startswith("Score how likely"):
            keys = re.findall(r'"(\w+)": 0\.0', prompt.split("Kinds:")[0])
            return json.dumps({k: 0.9 for k in keys})
        m = re.match(r"Find every (\w+)", prompt)
        if m:
            source = prompt.split("SOURCE:\n", 1)[1]
            out = [o for o in meeting_texts.get(m.group(1), []) if o["quote"] in source]
            return json.dumps({"observations": out})
        if "Below are open subjects" in prompt:
            source = prompt.split("SOURCE:\n", 1)[1]
            subjects = prompt.split("Subjects:\n", 1)[1].split("\n\nReturn JSON")[0]
            ev = []
            for line in subjects.splitlines():
                n = re.match(r"(?i)(\d+)\. \[decision\] .*freeze", line)
                if n and "Team B is not ready" in source:
                    ev.append(
                        {
                            "subject": int(n.group(1)),
                            "role": "evidence_against",
                            "assumption": 0,
                            "text": "Team B is not ready.",
                            "quote": "Team B is not ready, they need two more weeks.",
                        }
                    )
            return json.dumps({"evidence": ev})
        if "Attach each observation" in prompt:
            return json.dumps({"links": []})
        if "seem to form one" in prompt:
            return json.dumps(
                {
                    "title": "Vendor contract renewal risk",
                    "state": "Costs rising.",
                    "question": "Will the vendor renewal close in time?",
                    "coherent": True,
                }
            )
        if "You keep a running summary" in prompt:
            return json.dumps({"state": "Summary rewritten.", "answered": False, "answer": ""})
        if "It rests on these assumptions" in prompt:
            return json.dumps(
                {
                    "assumptions": [
                        {"index": 0, "state": "broken", "why": "Mira: team B is not ready."}
                    ]
                }
            )
        if "Someone predicted" in prompt:
            return json.dumps({"outcome": "true", "why": "Renewal moved to January."})
        if "List statements that contradict" in prompt:
            return json.dumps({"contradictions": []})
        if "Two things a manager heard" in prompt:
            return json.dumps({"connected": False})
        if prompt.startswith("Write a "):
            return json.dumps(
                {"summary": "A week.", "worked": [], "didnt": [], "watch": [], "worry_then": ""}
            )
        return "{}"

    return run


def _cfg() -> SimpleNamespace:
    em = SimpleNamespace(
        enabled=True,
        fast_model="",
        window_chars=12000,
        window_overlap=1500,
        triage_threshold=0.35,
        quote_min_score=0.8,
        types=["decision", "commitment", "concern", "prediction", "fact"],
        always_types=["decision", "commitment"],
        think_types=[],
        few_shot=False,
        few_shot_accept=3,
        few_shot_reject=2,
        ranker=False,
        ranker_min_labels=30,
        ranker_min_score=0.0,
        ranker_min_auc=0.7,
        train_on_bulk=False,
        vault_root="Pawn",
        directed=True,
        directed_max_subjects=12,
        route_max_distance=0.32,
        route_margin=0.04,
        route_ambiguous_distance=0.45,
        cluster_distance=0.6,
        cluster_min_observations=3,
        cluster_min_sources=2,
        cluster_window_days=30,
        proposed_expiry_days=21,
        decision_review_days=30,
        commitment_days=7,
        night_max_calls=400,
        night_max_minutes=60,
        night_jobs=None,
        brief_max_per_day=3,
        dream_pairs=0,
        people_tone=False,
    )
    return SimpleNamespace(
        emergence=em,
        coworker=SimpleNamespace(
            me=["Davide"],
            goals_path="Goals.md",
            threads_dir="Pawn/Threads",
            reviews_dir="Pawn/Reviews",
            people_dir="People",
        ),
        chat_model_id="fake@model",
        db_dsn=DSN,
    )


OBS = {
    "decision": [
        {
            "text": "Freeze the API on Monday.",
            "speaker": "Mira",
            "quote": "We decide to freeze the API on Monday, assuming team B is ready by Monday.",
            "choice": "Freeze the API on Monday",
            "assumptions": ["Team B is ready by Monday"],
            "owner": "Mira",
        }
    ],
    "commitment": [
        {
            "text": "Davide sends the migration plan to Luca.",
            "speaker": "Davide",
            "quote": "I will send the migration plan to Luca by 2026-12-01.",
            "debtor": "Davide",
            "creditor": "Luca",
            "what": "Send the migration plan",
            "due": "2026-12-01",
        }
    ],
    "concern": [
        {
            "text": "Vendor contract renewal is too expensive.",
            "speaker": "Luca",
            "quote": "I am worried the vendor contract renewal is too expensive.",
        },
        {
            "text": "Vendor contract renewal costs keep rising.",
            "speaker": "Ansgar",
            "quote": "The vendor contract renewal costs keep rising, I am worried.",
        },
        {
            "text": "Vendor contract renewal is blocked by legal review.",
            "speaker": "Luca",
            "quote": "Vendor contract renewal is blocked by legal review, which worries me.",
        },
    ],
    "prediction": [
        {
            "text": "Vendor contract renewal slips to Q1.",
            "speaker": "Mira",
            "quote": "I predict the vendor contract renewal slips to Q1.",
            "claim": "renewal slips",
            "who": "Mira",
            "by_when": "2026-11-30",
        }
    ],
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine, text

    with create_engine(DSN).begin() as c:
        for table in (
            "em_provocations",
            "em_subject_events",
            "em_links",
            "em_subjects",
            "em_labels",
            "em_observations",
            "em_windows",
            "em_sources",
        ):
            c.execute(text(f"DELETE FROM {table}"))
    from pawn_agent.core.emergence import labels, llm, passes

    monkeypatch.setattr(labels, "embed_many", lambda cfg, texts: [bag(t) for t in texts])
    model = fake_model(OBS)
    monkeypatch.setattr(passes, "llm_run", model)
    monkeypatch.setattr(llm, "llm_run", model)
    return _cfg(), Store()


def test_full_loop(env):
    from pawn_agent.core.emergence import ledger, night, provocations, route
    from pawn_agent.core.emergence import subjects as subj
    from pawn_agent.core.emergence.pipeline import observe_source

    cfg, store = env
    store.write(
        "Goals.md",
        "---\npawn: goals\n---\n\n## Active\n\n### API freeze\n- why: Ship the platform on time.\n",
    )

    async def process(ref: str, text: str) -> dict:
        result = await observe_source(cfg, source_kind="session", source_ref=ref, source_text=text)
        return await route.after_observe(cfg, result)

    first = asyncio.run(process("m1", MEETING_1))
    assert first["ledger"] == 3  # decision, commitment, prediction
    second = asyncio.run(process("m2", MEETING_2))
    assert second["directed"] == 1  # "Team B is not ready" against the freeze decision

    decision = subj.list_subjects(DSN, kinds=("decision",))[0]
    evidence = subj.subject_observations(DSN, decision["id"])
    assert any(o["role"] == "evidence_against" for o in evidence)

    today = date.today() + timedelta(days=40)
    results = asyncio.run(
        night.run_night(
            cfg,
            store=store,
            today=today,
            jobs=[
                "sync",
                "clustering",
                "synthesis",
                "assumptions",
                "commitments",
                "predictions",
                "contradictions",
                "people",
                "retros",
                "provocations",
            ],
        )
    )
    assert not [
        v for v in results.values() if isinstance(v, str) and v.startswith("error")
    ], results

    threads = subj.list_subjects(DSN, kinds=("thread",))
    titles = {t["title"]: t for t in threads}
    assert "API freeze" in titles and titles["API freeze"]["origin"] == "goal"
    emergent = titles["Vendor contract renewal risk"]
    assert emergent["status"] == "proposed" and emergent["note_key"] in store.files
    assert subj.list_subjects(DSN, kinds=("question",))[0]["status"] == "proposed"

    decision = subj.get_subject(DSN, decision["id"])
    assert decision["fields"]["assumption_health"]["0"]["state"] == "broken"
    assert "**broken**" in store.files[decision["note_key"]]

    commitment = subj.list_subjects(DSN, kinds=("commitment",))[0]
    assert commitment["fields"]["overdue"] is True
    assert "## I owe" in store.files["Pawn/Commitments.md"]
    assert "Send the migration plan" in store.files["Pawn/Commitments.md"]

    prediction = subj.list_subjects(DSN, kinds=("prediction",))[0]
    assert prediction["status"] == "pending"
    assert prediction["fields"]["proposed_resolution"] == "true"

    assert any(k.startswith("Pawn/People/") for k in store.files)

    picked = provocations.select_for_day(cfg, (today + timedelta(days=1)).isoformat())
    assert 1 <= len(picked) <= 3
    kinds = {p["kind"] for p in provocations.list_recent(cfg, days=90)}
    assert {"assumption_broken", "overdue_mine", "emergent_thread", "prediction_resolved"} <= kinds
    assert picked[0]["kind"] == "assumption_broken"  # highest base weight

    prov = next(
        p for p in provocations.list_recent(cfg, days=90) if p["kind"] == "prediction_resolved"
    )
    receipt = provocations.apply_verdict(cfg, prov["short_id"], "useful")
    assert "came true" in receipt
    assert subj.get_subject(DSN, prediction["id"])["fields"]["confirmed"] == "true"

    thread_prov = next(
        p for p in provocations.list_recent(cfg, days=90) if p["kind"] == "emergent_thread"
    )
    provocations.apply_verdict(cfg, thread_prov["short_id"], "useful")
    assert subj.get_subject(DSN, emergent["id"])["status"] == "active"

    # A verdict changed the status; the next sync writes it to the note.
    key = emergent["note_key"]
    assert "status: proposed" in store.files[key]
    asyncio.run(night.run_night(cfg, store=store, today=today, jobs=["sync"]))
    assert "status: active" in store.files[key]
    # The user edits a note's status; the next sync applies it.
    store.write(key, store.files[key].replace("status: active", "status: dismissed"))
    asyncio.run(night.run_night(cfg, store=store, today=today, jobs=["sync"]))
    assert subj.get_subject(DSN, emergent["id"])["status"] == "dismissed"

    brief = ledger.meeting_brief(cfg, ["Luca"])
    assert "Vendor contract renewal" in brief and "You owe them" in brief

    asyncio.run(night.run_night(cfg, store=store, today=today, jobs=["predictions"]))
    assert "Calibration" in store.files["Pawn/Predictions.md"]
    assert "Mira: 1/1 came true" in store.files["Pawn/Predictions.md"]
