"""Emergence Phase 1: triage, typed extraction, verification, projection."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from pawn_agent.core.emergence import passes, pipeline
from pawn_agent.core.emergence.compare import Comparison, render_report
from pawn_agent.core.emergence.pipeline import observe_source, project_items
from pawn_agent.core.emergence.types import enabled_types


def _transcript(minutes: int) -> str:
    out = []
    for i in range(minutes * 2):
        sec = i * 30
        who = "Davide" if i % 2 == 0 else "Mira"
        out.append(
            f"[{sec // 60:02d}:{sec % 60:05.2f}] {who}:\n  routine status update on the roadmap, staffing and vendors, number {i}.\n"
        )
    return "".join(out)


def _with_late_decision(minutes: int = 90) -> str:
    text = _transcript(minutes)
    late = "[85:00.00] Mira:\n  Decision: we freeze the API on Monday and Luca owns it.\n"
    return text.replace("[85:00.00] Davide:\n", late + "[85:00.01] Davide:\n", 1)


def _cfg(**overrides: Any) -> SimpleNamespace:
    em = dict(
        enabled=True,
        fast_model="",
        window_chars=2000,
        window_overlap=300,
        triage_threshold=0.35,
        quote_min_score=0.8,
        types=["decision", "commitment", "question", "prediction"],
    )
    em.update(overrides)
    return SimpleNamespace(
        emergence=SimpleNamespace(**em), chat_model_id="test@model", db_dsn="unused"
    )


class FakeModel:
    """Triage says 'decision' only where the marker is; typed returns fixtures."""

    def __init__(self, typed: dict[str, list[dict]], marker: str = "freeze the API"):
        self.typed = typed
        self.marker = marker
        self.calls: list[str] = []

    async def __call__(self, cfg, prompt, system_prompt=None, **kw):  # noqa: ARG002
        source = prompt.split("SOURCE:\n", 1)[1]
        if prompt.startswith("Score how likely"):
            self.calls.append("triage")
            hit = self.marker in source
            return json.dumps(
                {
                    "decision": 0.9 if hit else 0.05,
                    "commitment": 0.1,
                    "question": 0.0,
                    "prediction": 0.6 if hit else 0.0,
                }
            )
        name = re.match(r"Find every (\w+)", prompt).group(1)
        self.calls.append(name)
        if self.marker not in source:
            return '{"observations": []}'
        return json.dumps({"observations": self.typed.get(name, [])})


DECISION = {
    "text": "Freeze the API on Monday.",
    "context": "Mira decided; Luca owns the freeze.",
    "speaker": "Mira",
    "quote": "we freeze the API on Monday and Luca owns it",
    "confidence": 0.9,
    "choice": "Freeze the API on Monday",
    "rationale": "",
    "assumptions": ["Team B is ready by Monday"],
    "owner": "Luca",
    "alternatives": [],
}
FABRICATED = {
    "text": "Hire two contractors.",
    "quote": "we agreed to hire two contractors next quarter",
    "owner": "Davide",
}
PREDICTION = {
    "text": "API freeze holds through Q1.",
    "quote": "freeze the API on Monday",
    "who": "Mira",
    "claim": "freeze holds",
}


def test_enabled_types_keeps_canonical_order():
    assert enabled_types(["fact", "decision", "bogus"]) == ["decision", "fact"]
    assert "decision" in enabled_types(None)


def test_parse_triage_variants():
    types = ["decision", "fact"]
    assert passes.parse_triage('{"decision": 0.7, "fact": "0.2"}', types) == {
        "decision": 0.7,
        "fact": 0.2,
    }
    assert passes.parse_triage('```json\n{"scores": {"Decision": 80}}\n```', types) == {
        "decision": 0.8,
        "fact": 0.0,
    }
    assert passes.parse_triage("no json here", types) is None
    assert passes.parse_triage('{"other": 1}', types) is None


def test_triage_fails_open_on_bad_reply(monkeypatch: pytest.MonkeyPatch):
    async def run(cfg, prompt, system_prompt=None, **kw):  # noqa: ARG001
        return "I think there is a decision"

    monkeypatch.setattr(passes, "llm_run", run)
    scores, ok = asyncio.run(passes.triage(None, ["decision", "fact"], "x", ""))
    assert not ok
    assert scores == {"decision": 1.0, "fact": 1.0}


def test_normalize_observation_shapes_fields():
    obs = passes.normalize_observation(DECISION, "decision")
    assert obs is not None
    assert obs["type"] == "decision"
    assert obs["extractor"] == "decision.v2"
    assert obs["fields"]["assumptions"] == ["Team B is ready by Monday"]
    assert obs["fields"]["owner"] == "Luca"
    assert obs["speaker_name"] == "Mira"
    assert passes.normalize_observation({"text": " "}, "decision") is None
    q = passes.normalize_observation({"text": "Who?", "blocking": "false"}, "question")
    assert q["fields"]["blocking"] is False


def test_observe_source_end_to_end(monkeypatch: pytest.MonkeyPatch):
    model = FakeModel({"decision": [DECISION, FABRICATED], "prediction": [PREDICTION]})
    monkeypatch.setattr(passes, "llm_run", model)
    result = asyncio.run(
        observe_source(
            _cfg(),
            source_kind="session",
            source_ref="s1",
            source_text=_with_late_decision(),
            persist=False,
        )
    )
    stats = result.stats
    assert stats["windows"] > 5
    # Only windows with the marker run typed prompts.
    assert stats["typed_calls"] < stats["triage_calls"] * 2
    assert stats["windows_skipped"] >= stats["windows"] - 2
    live = result.live
    assert {o["type"] for o in live} == {"decision", "prediction"}
    decision = next(o for o in live if o["type"] == "decision")
    assert decision["start_sec"] == pytest.approx(85 * 60)
    assert decision["quote_score"] > 0.9
    assert decision["model"] == "test@model"
    # Overlap duplicates merged to one decision; fabricated quote rejected.
    assert sum(1 for o in live if o["type"] == "decision") == 1
    assert {o["reject_reason"] for o in result.rejected} == {"quote_not_found"}
    assert all(o["text"] == "Hire two contractors." for o in result.rejected)


def test_observe_source_raises_when_every_call_fails(monkeypatch: pytest.MonkeyPatch):
    async def down(cfg, prompt, system_prompt=None, **kw):  # noqa: ARG001
        raise RuntimeError("model down")

    monkeypatch.setattr(passes, "llm_run", down)
    with pytest.raises(passes.PassError):
        asyncio.run(
            observe_source(
                _cfg(),
                source_kind="session",
                source_ref="s",
                source_text=_transcript(30),
                persist=False,
            )
        )


def test_observe_source_persists(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(passes, "llm_run", FakeModel({"decision": [DECISION]}))
    saved: dict[str, Any] = {}

    def save_run(dsn, result, *, pipeline_version):
        saved.update(dsn=dsn, result=result, version=pipeline_version)
        return "src"

    from pawn_agent.core.emergence import store

    monkeypatch.setattr(store, "save_run", save_run)
    asyncio.run(
        observe_source(
            _cfg(), source_kind="session", source_ref="s", source_text=_with_late_decision()
        )
    )
    assert saved["dsn"] == "unused"
    assert saved["version"] == pipeline.PIPELINE_VERSION
    assert saved["result"].live


def test_annotations_reach_every_prompt(monkeypatch: pytest.MonkeyPatch):
    from pawn_agent.core.coworker.pipeline import _source_with_annotations

    prompts: list[str] = []

    async def run(cfg, prompt, system_prompt=None, **kw):  # noqa: ARG001
        prompts.append(prompt)
        return '{"decision": 0, "commitment": 0, "question": 0, "prediction": 0}'

    monkeypatch.setattr(passes, "llm_run", run)
    source = _source_with_annotations(_transcript(40), "Luca is the tech lead.")
    asyncio.run(
        observe_source(
            _cfg(), source_kind="session", source_ref="s", source_text=source, persist=False
        )
    )
    assert len(prompts) > 3
    assert all("Luca is the tech lead." in p for p in prompts)
    assert all("## User annotations" not in p.split("SOURCE:\n", 1)[1] for p in prompts)


def test_project_items_maps_types():
    observations = [
        {
            "id": "o1",
            "type": "decision",
            "status": "live",
            "text": "Freeze API",
            "fields": {"owner": "Luca"},
            "start_sec": 30.0,
            "quote": "q",
            "quote_score": 1.0,
            "window": 0,
        },
        {
            "id": "o2",
            "type": "commitment",
            "status": "live",
            "text": "Send RFC",
            "fields": {"debtor": "Davide", "due": "2026-12-01"},
            "start_sec": 10.0,
        },
        {
            "id": "o3",
            "type": "commitment",
            "status": "live",
            "text": "Soon",
            "fields": {"debtor": "Mira", "due": "next week"},
            "start_sec": None,
        },
        {
            "id": "o4",
            "type": "question",
            "status": "live",
            "text": "Is Q2 real?",
            "fields": {"asked_by": "Mira"},
        },
        {"id": "o5", "type": "prediction", "status": "live", "text": "Slips", "fields": {}},
        {"id": "o6", "type": "decision", "status": "rejected", "text": "Fake", "fields": {}},
    ]
    items = project_items(observations)
    assert [i["observation_id"] for i in items] == ["o2", "o1", "o3", "o4"]
    by_id = {i["observation_id"]: i for i in items}
    assert by_id["o1"]["owner"] == "Luca"
    assert by_id["o2"]["owner"] == "Davide"
    assert by_id["o2"]["due"] == "2026-12-01"
    assert by_id["o3"]["due"] == ""
    assert by_id["o4"]["kind"] == "open_question"
    assert by_id["o4"]["owner"] == "Mira"


def test_coworker_extract_uses_observations_when_enabled(monkeypatch: pytest.MonkeyPatch):
    from pawn_agent.core.coworker import pipeline as cw

    monkeypatch.setattr(passes, "llm_run", FakeModel({"decision": [DECISION]}))
    from pawn_agent.core.emergence import store

    monkeypatch.setattr(store, "save_run", lambda *a, **k: "src")
    items, summary = asyncio.run(
        cw._extract(
            _cfg(), source_kind="session", source_ref="s", source_text=_with_late_decision()
        )
    )
    assert len(items) == 1
    assert items[0]["observation_id"]
    assert "observations" in summary
    payload = cw._item_payload(items[0])
    assert payload["observation_type"] == "decision"
    assert payload["start_sec"] == pytest.approx(85 * 60)

    called: list[str] = []

    async def legacy(cfg, text):  # noqa: ARG001
        called.append("legacy")
        return []

    monkeypatch.setattr(cw, "extract_items", legacy)
    off = _cfg(enabled=False)
    asyncio.run(cw._extract(off, source_kind="session", source_ref="s", source_text="x"))
    assert called == ["legacy"]


def test_model_cfg_applies_fast_model(monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []

    def apply(cfg, raw):
        seen.append(raw)
        cfg.chat_model_id = raw

    monkeypatch.setattr("pawn_agent.utils.model_catalog.apply_model_selection", apply)
    cfg = _cfg(fast_model="ollama@small")
    effective, model_id = pipeline.model_cfg(cfg)
    assert model_id == "ollama@small"
    assert effective is not cfg
    assert cfg.chat_model_id == "test@model"
    same, default_id = pipeline.model_cfg(_cfg())
    assert default_id == "test@model"


def test_render_report_lists_both_sides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        passes,
        "llm_run",
        FakeModel({"decision": [DECISION, FABRICATED], "prediction": [PREDICTION]}),
    )
    observed = asyncio.run(
        observe_source(
            _cfg(),
            source_kind="session",
            source_ref="s1",
            source_text=_with_late_decision(),
            persist=False,
        )
    )
    baseline = [
        {"kind": "decision", "text": "Freeze the API on Monday.", "quote": "", "start_sec": 5100},
        {"kind": "commitment", "text": "Davide sends the deck.", "quote": "", "owner": "Davide"},
    ]
    report = render_report(
        Comparison(
            source_ref="s1",
            baseline=baseline,
            baseline_calls=7,
            baseline_seconds=12.0,
            observed=observed,
            items=project_items(observed.observations),
        )
    )
    assert "pawn: extract-compare" in report
    assert "| Items | 2 | 1 |" in report
    assert "| Found only here | 1 | 0 |" in report
    assert "[1:25:00] **decision** Freeze the API on Monday. — owner: Luca · *in both*" in report
    assert "Davide sends the deck." in report
    assert "### prediction (1)" in report
    assert "## Rejected (1)" in report


def test_passes_turn_thinking_off(monkeypatch: pytest.MonkeyPatch):
    seen: list[tuple[str, Any]] = []
    model = FakeModel({"decision": [DECISION], "prediction": [PREDICTION]})

    async def run(cfg, prompt, system_prompt=None, **kw):
        match = re.match(r"Find every (\w+)", prompt)
        seen.append((match.group(1) if match else "triage", kw.get("think")))
        return await model(cfg, prompt, system_prompt)

    monkeypatch.setattr(passes, "llm_run", run)
    for think_types in ([], ["decision"]):
        seen.clear()
        asyncio.run(
            observe_source(
                _cfg(think_types=think_types),
                source_kind="session",
                source_ref="s",
                source_text=_with_late_decision(),
                persist=False,
            )
        )
        assert {t for kind, t in seen if kind == "triage"} == {False}
        assert {t for kind, t in seen if kind == "prediction"} == {False}
        assert {t for kind, t in seen if kind == "decision"} == {bool(think_types)}


def test_lenient_json_parsing():
    from pawn_agent.core.coworker.jsonutil import parse_json_value

    assert parse_json_value('{"observations": [{"text": "a",},],}') == {
        "observations": [{"text": "a"}]
    }
    assert parse_json_value('{"text": "line one\nline two"}') == {"text": "line one\nline two"}
    assert parse_json_value('{"a": 1}\n{"b": 2}') == {"a": 1}
    assert parse_json_value('Here you go:\n```json\n{"a": [1, 2,]}\n```') == {"a": [1, 2]}


def test_extract_type_retries_once_on_bad_json(monkeypatch: pytest.MonkeyPatch):
    replies = iter(['{"observations": [{"text": "x" "quote": "y"}]}', '{"observations": []}'])
    prompts_seen: list[str] = []

    async def run(cfg, prompt, system_prompt=None, **kw):  # noqa: ARG001
        prompts_seen.append(prompt)
        return next(replies)

    monkeypatch.setattr(passes, "llm_run", run)
    assert asyncio.run(passes.extract_type(None, "decision", "src", "")) == []
    assert len(prompts_seen) == 2
    assert "not valid JSON" in prompts_seen[1]

    async def always_bad(cfg, prompt, system_prompt=None, **kw):  # noqa: ARG001
        prompts_seen.append(prompt)
        return "not json at all"

    prompts_seen.clear()
    monkeypatch.setattr(passes, "llm_run", always_bad)
    assert asyncio.run(passes.extract_type(None, "decision", "src", "")) == []
    assert len(prompts_seen) == 2


def test_always_types_bypass_triage(monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []

    async def run(cfg, prompt, system_prompt=None, **kw):  # noqa: ARG001
        match = re.match(r"Find every (\w+)", prompt)
        if not match:
            return '{"decision": 0.0, "commitment": 0.0, "question": 0.0, "prediction": 0.0}'
        seen.append(match.group(1))
        return '{"observations": []}'

    monkeypatch.setattr(passes, "llm_run", run)
    asyncio.run(
        observe_source(
            _cfg(always_types=["commitment"]),
            source_kind="session",
            source_ref="s",
            source_text=_transcript(40),
            persist=False,
        )
    )
    assert seen and set(seen) == {"commitment"}
