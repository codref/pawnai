"""POST /v1/sessions/{id}/extract helper."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pawn_server.core import sessions_api


def _cfg(*, coworker_enabled: bool = True) -> Any:
    return SimpleNamespace(
        db_dsn="postgresql://unused",
        coworker=SimpleNamespace(enabled=coworker_enabled),
    )


def test_extract_items_and_people(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    monkeypatch.setattr(
        "pawn_agent.core.coworker.pipeline.process_session",
        AsyncMock(return_value={"source_ref": "s1", "items": 2, "interrupts": 0}),
    )
    monkeypatch.setattr(
        "pawn_agent.core.people.refresh.refresh_people_for_session",
        AsyncMock(return_value={"session_id": "s1", "updated": 1}),
    )
    out = asyncio.run(sessions_api.extract_session(cfg, "s1", items=True, people=True))
    assert out["session_id"] == "s1"
    assert out["items"]["items"] == 2
    assert out["people"]["updated"] == 1


def test_extract_people_only_when_coworker_off(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(coworker_enabled=False)
    monkeypatch.setattr(
        "pawn_agent.core.people.refresh.refresh_people_for_session",
        AsyncMock(return_value={"session_id": "s1", "updated": 0}),
    )
    out = asyncio.run(sessions_api.extract_session(cfg, "s1", items=False, people=True))
    assert out["items"] is None
    assert out["people"]["updated"] == 0


def test_extract_items_requires_coworker(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(coworker_enabled=False)
    with pytest.raises(sessions_api.SessionsApiError) as exc:
        asyncio.run(sessions_api.extract_session(cfg, "s1", items=True, people=False))
    assert exc.value.status_code == 400


def test_extract_requires_a_flag() -> None:
    with pytest.raises(sessions_api.SessionsApiError) as exc:
        asyncio.run(sessions_api.extract_session(_cfg(), "s1", items=False, people=False))
    assert exc.value.status_code == 422
