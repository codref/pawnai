"""process_session prepends vault ## Annotations into extract source."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pawn_agent.core.coworker import pipeline
from pawn_diarize.core.vault_transcript import user_annotations


def test_user_annotations_skips_stub():
    stub = "# Session\n\n## Annotations\n\n_(Add notes and tags here.)_\n\n## Transcript\nHi\n"
    assert user_annotations(stub) == ""
    assert user_annotations("") == ""
    assert user_annotations("## Annotations\n\nMira owns Project X.\n") == "Mira owns Project X."


def test_source_with_annotations_prepends():
    out = pipeline._source_with_annotations("SPEAKER: hi", "Mira is CEO.")
    assert out.startswith("## User annotations\nMira is CEO.")
    assert "## Transcript\nSPEAKER: hi" in out
    assert pipeline._source_with_annotations("only", "") == "only"
    assert pipeline._source_with_annotations("only", "  ") == "only"


def test_process_session_includes_annotations(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(db_dsn="postgresql://unused")
    store = MagicMock()
    store.read.return_value = (
        "# Meet\n\n## Annotations\n\nDavide = SPEAKER_00; Project X.\n\n## Transcript\nx\n"
    )
    monkeypatch.setattr(pipeline, "_vault_note_key", lambda *_a, **_k: "Pawn/Transcripts/s.md")
    monkeypatch.setattr(
        "pawn_agent.utils.transcript.fetch_transcript",
        lambda *_a, **_k: "SPEAKER_00: Ship Friday.",
    )
    captured: dict[str, Any] = {}

    async def _process_source(cfg, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return {"source_ref": kwargs["source_ref"], "items": 0, "interrupts": 0}

    monkeypatch.setattr(pipeline, "process_source", AsyncMock(side_effect=_process_source))

    result = asyncio.run(pipeline.process_session(cfg, "meet-1", store=store))
    assert result["items"] == 0
    text = captured["source_text"]
    assert "## User annotations" in text
    assert "Davide = SPEAKER_00" in text
    assert "## Transcript" in text
    assert "Ship Friday" in text
    assert captured["source_kind"] == "session"
    assert captured["source_ref"] == "meet-1"


def test_process_session_skips_stub_annotations(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(db_dsn="postgresql://unused")
    store = MagicMock()
    store.read.return_value = "# Meet\n\n## Annotations\n\n_(Add notes and tags here.)_\n"
    monkeypatch.setattr(pipeline, "_vault_note_key", lambda *_a, **_k: "Pawn/Transcripts/s.md")
    monkeypatch.setattr(
        "pawn_agent.utils.transcript.fetch_transcript",
        lambda *_a, **_k: "SPEAKER: hi",
    )
    captured: dict[str, Any] = {}

    async def _process_source(cfg, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return {"source_ref": "s", "items": 0, "interrupts": 0}

    monkeypatch.setattr(pipeline, "process_source", AsyncMock(side_effect=_process_source))
    asyncio.run(pipeline.process_session(cfg, "s", store=store))
    assert captured["source_text"] == "SPEAKER: hi"
