"""Tests for event_upsert CliTool helper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from pawn_core.vault import VaultNotFound
from pawn_agent.tools.event_upsert_impl import event_upsert_impl
from pawn_agent.utils.config import PlannerConfig
from pawn_core.config import VaultConfig


class _MemStore:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def read(self, key: str) -> str:
        if key not in self.files:
            raise VaultNotFound(key)
        return self.files[key]

    def write(self, key: str, body: str, skip_guards: bool = False) -> None:
        self.files[key] = body

    def delete(self, key: str) -> None:
        self.files.pop(key, None)

    def list(self, prefix: str = "", *, suffix: str = ".md") -> list[str]:
        p = (prefix or "").strip().strip("/")
        out = []
        for key in self.files:
            if p and not (key == p or key.startswith(p + "/")):
                continue
            if suffix and not key.endswith(suffix):
                continue
            out.append(key)
        return sorted(out)


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        vault=VaultConfig(agent_root="Pawn"),
        coworker=SimpleNamespace(timezone="UTC"),
        planner=PlannerConfig(events_dir="{agent_root}/Events"),
    )


def test_event_upsert_creates_timed_activity() -> None:
    cfg = _cfg()
    store = _MemStore()
    with patch("pawn_server.core.vault_events.publish_vault_event"):
        receipt = event_upsert_impl(
            cfg,
            title="Standup",
            start="2026-11-06T09:00",
            end="2026-11-06T09:30",
            attendees="Ada, Bob",
            store=store,
        )
    assert "Upserted event" in receipt
    assert "2026-11-06T09:00" in receipt
    keys = [k for k in store.files if k.startswith("Pawn/Events/")]
    assert len(keys) == 1
    body = store.read(keys[0])
    assert "pawn: event" in body
    assert "Standup" in body
    assert "2026-11-06T09:00" in body
    assert "Ada" in body


def test_event_upsert_requires_title() -> None:
    assert event_upsert_impl(_cfg(), title="  ", store=_MemStore()).startswith("Error:")


def test_event_upsert_idempotent_on_title_start() -> None:
    cfg = _cfg()
    store = _MemStore()
    with patch("pawn_server.core.vault_events.publish_vault_event"):
        first = event_upsert_impl(
            cfg,
            title="1:1",
            start="2026-11-06T14:00",
            end="2026-11-06T14:30",
            store=store,
        )
        second = event_upsert_impl(
            cfg,
            title="1:1",
            start="2026-11-06T14:00",
            end="2026-11-06T15:00",
            location="Room A",
            store=store,
        )
    assert "Upserted event" in first
    assert "Upserted event" in second
    keys = [k for k in store.files if k.startswith("Pawn/Events/")]
    assert len(keys) == 1
    body = store.read(keys[0])
    assert "2026-11-06T15:00" in body
    assert "Room A" in body
