"""Windowed coworker extraction: windows, quote location, merge, timestamps."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from pawn_agent.core.coworker import extract
from pawn_agent.core.coworker.extract import (
    locate_quote,
    merge_items,
    split_annotations,
    start_seconds_at,
    window_source,
)
from pawn_agent.core.coworker.notes import format_offset, render_item_note


def _transcript(minutes: int, *, line: str = "we talk about the roadmap and staffing") -> str:
    """One speaker turn every 30 s, alternating speakers."""
    out = []
    for i in range(minutes * 2):
        sec = i * 30
        who = "Davide" if i % 2 == 0 else "Mira"
        out.append(f"[{sec // 60:02d}:{sec % 60:05.2f}] {who}:\n  {line} number {i}.\n")
    return "".join(out)


def _cfg(size: int = 2000, overlap: int = 300) -> SimpleNamespace:
    return SimpleNamespace(
        coworker=SimpleNamespace(extract_window_chars=size, extract_window_overlap=overlap)
    )


def test_short_source_is_one_window():
    text = _transcript(2)
    windows = window_source(text, max_chars=12000)
    assert len(windows) == 1
    assert windows[0].text == text


def test_windows_cover_source_with_overlap_on_turns():
    text = _transcript(90)
    windows = window_source(text, max_chars=2000, overlap_chars=300)
    assert len(windows) > 5
    for win in windows:
        assert len(win.text) <= 2000
        assert win.text.startswith("[")  # cut on a speaker turn
        assert text[win.char_start : win.char_start + len(win.text)] == win.text
    # Contiguous coverage with overlap: each window starts before the last ended.
    for prev, nxt in zip(windows, windows[1:]):
        prev_end = prev.char_start + len(prev.text)
        assert nxt.char_start < prev_end
        assert nxt.char_start > prev.char_start
    last = windows[-1]
    assert last.char_start + len(last.text) == len(text)


def test_oversized_turn_is_hard_split():
    text = "[00:00.00] Davide:\n" + ("  word " * 50 + "\n") * 40
    windows = window_source(text, max_chars=1000, overlap_chars=100)
    assert len(windows) > 1
    assert all(len(w.text) <= 1000 for w in windows)
    assert windows[-1].char_start + len(windows[-1].text) == len(text)


def test_paragraph_source_windows():
    text = "\n\n".join(f"Paragraph {i}. " + "text " * 60 for i in range(30))
    windows = window_source(text, max_chars=1500, overlap_chars=200)
    assert len(windows) > 3
    assert windows[-1].char_start + len(windows[-1].text) == len(text)


def test_split_annotations_roundtrip():
    from pawn_agent.core.coworker.pipeline import _source_with_annotations

    joined = _source_with_annotations("[00:00.00] A:\n  hi\n", "Mira is CEO.")
    ann, body = split_annotations(joined)
    assert ann == "Mira is CEO."
    assert body == "[00:00.00] A:\n  hi\n"
    assert split_annotations("plain") == ("", "plain")


def test_locate_quote_ignores_speaker_formatting():
    text = (
        "[10:00.00] Davide:\n  We should not move the bucket before the deploy.\n"
        "[10:30.00] Mira:\n  yes, keep it on our S3 through Friday at least.\n"
    )
    quote = "We should not move the bucket before the deploy.\nMira: yes, keep it on our S3"
    offset, score = locate_quote(text, quote)
    assert offset is not None
    assert score > 0.8
    assert start_seconds_at(text, offset) == 600.0


def test_locate_quote_missing():
    offset, score = locate_quote(
        "[00:00.00] A:\n  hello there\n", "completely unrelated claim words"
    )
    assert score < 0.5
    assert locate_quote("", "x") == (None, 0.0)


def test_merge_keeps_richer_duplicate_and_sorts_by_time():
    items = [
        {
            "kind": "decision",
            "text": "Keep recordings on S3.",
            "quote": "keep it",
            "context": "",
            "quote_score": 0.5,
            "start_sec": 900.0,
            "window": 2,
        },
        {
            "kind": "decision",
            "text": "Keep the recordings on S3.",
            "quote": "keep it on S3",
            "context": "longer context",
            "quote_score": 0.9,
            "start_sec": 905.0,
            "window": 3,
        },
        {
            "kind": "commitment",
            "text": "Keep recordings on S3.",
            "quote": "",
            "context": "",
            "start_sec": 10.0,
            "window": 0,
        },
    ]
    merged = merge_items(items)
    assert [m["kind"] for m in merged] == ["commitment", "decision"]
    assert merged[1]["context"] == "longer context"


def _fake_llm(responses: dict[str, list[dict]]):
    """Return items whose marker appears in the prompt's SOURCE."""

    async def run(cfg, prompt, system_prompt=None):  # noqa: ARG001
        source = prompt.split("SOURCE:\n", 1)[1]
        found = [item for marker, batch in responses.items() if marker in source for item in batch]
        return json.dumps({"items": found})

    return run


def test_extract_reads_the_end_of_a_long_meeting(monkeypatch: pytest.MonkeyPatch):
    text = _transcript(90)
    late = "[85:00.00] Mira:\n  Decision: we freeze the API on Monday and Luca owns it.\n"
    text = text.replace("[85:00.00] Davide:\n", late + "[85:00.01] Davide:\n", 1)
    assert "freeze the API" in text
    calls: list[str] = []
    fake = _fake_llm(
        {
            "freeze the API": [
                {
                    "kind": "decision",
                    "text": "Freeze the API on Monday.",
                    "context": "Mira decided; Luca owns it.",
                    "owner": "Luca",
                    "quote": "we freeze the API on Monday and Luca owns it",
                }
            ]
        }
    )

    async def counting(cfg, prompt, system_prompt=None):
        calls.append(prompt)
        return await fake(cfg, prompt, system_prompt)

    monkeypatch.setattr(extract, "llm_run", counting)
    items = asyncio.run(extract.extract_items(_cfg(), text))
    assert len(calls) > 5
    assert "part 1 of" in calls[0]
    assert len(items) == 1  # overlap duplicates merged
    item = items[0]
    assert item["owner"] == "Luca"
    assert item["start_sec"] == pytest.approx(85 * 60)
    assert item["quote_score"] > 0.9


def test_extract_annotations_reach_every_window(monkeypatch: pytest.MonkeyPatch):
    from pawn_agent.core.coworker.pipeline import _source_with_annotations

    prompts: list[str] = []

    async def run(cfg, prompt, system_prompt=None):  # noqa: ARG001
        prompts.append(prompt)
        return '{"items": []}'

    monkeypatch.setattr(extract, "llm_run", run)
    source = _source_with_annotations(_transcript(60), "Luca is the tech lead.")
    asyncio.run(extract.extract_items(_cfg(), source))
    assert len(prompts) > 3
    assert all("Luca is the tech lead." in p for p in prompts)
    assert all("## User annotations" not in p.split("SOURCE:\n", 1)[1] for p in prompts)


def test_extract_single_window_keeps_annotated_source(monkeypatch: pytest.MonkeyPatch):
    from pawn_agent.core.coworker.pipeline import _source_with_annotations

    prompts: list[str] = []

    async def run(cfg, prompt, system_prompt=None):  # noqa: ARG001
        prompts.append(prompt)
        return '{"items": []}'

    monkeypatch.setattr(extract, "llm_run", run)
    source = _source_with_annotations(_transcript(2), "Luca is the tech lead.")
    asyncio.run(extract.extract_items(_cfg(size=12000), source))
    assert len(prompts) == 1
    assert "part 1 of" not in prompts[0]
    assert prompts[0].split("SOURCE:\n", 1)[1] == source + "\n"


def test_extract_skips_failed_window_but_raises_when_all_fail(monkeypatch: pytest.MonkeyPatch):
    text = _transcript(60)
    state = {"n": 0}

    async def flaky(cfg, prompt, system_prompt=None):  # noqa: ARG001
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("model down")
        return '{"items": []}'

    monkeypatch.setattr(extract, "llm_run", flaky)
    assert asyncio.run(extract.extract_items(_cfg(), text)) == []

    async def down(cfg, prompt, system_prompt=None):  # noqa: ARG001
        raise RuntimeError("model down")

    monkeypatch.setattr(extract, "llm_run", down)
    with pytest.raises(RuntimeError):
        asyncio.run(extract.extract_items(_cfg(), text))


def test_item_note_shows_offset():
    assert format_offset(None) == ""
    assert format_offset(65.4) == "01:05"
    assert format_offset(3725) == "1:02:05"
    note = render_item_note(
        item_id="i",
        short_id="s",
        status="new",
        kind="decision",
        text="Freeze the API.",
        source_link="[[Pawn/Transcripts/x.md]]",
        start_sec=5100.0,
    )
    assert "Source: [[Pawn/Transcripts/x.md]] at 1:25:00" in note
    assert "start_sec: 5100.0" in note
    plain = render_item_note(item_id="i", short_id="s", status="new", kind="decision", text="x")
    assert "start_sec" not in plain
