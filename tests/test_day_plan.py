"""Tests for day_plan CliTool helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from pawn_core.vault import VaultNotFound
from pawn_agent.tools.day_plan_impl import (
    daily_note_key,
    day_plan_update_impl,
    format_planner_day,
    planner_day_impl,
    replace_plan_section,
)


class _MemStore:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def read(self, key: str) -> str:
        if key not in self.files:
            raise VaultNotFound(key)
        return self.files[key]

    def write(self, key: str, body: str, skip_guards: bool = False) -> None:
        self.files[key] = body


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        coworker=SimpleNamespace(
            timezone="UTC",
            daily_dir="Pawn/Daily",
            today_path="Pawn/Today.md",
        )
    )


def test_format_planner_day_compact() -> None:
    text = format_planner_day(
        {
            "date": "2026-11-05",
            "blocks": [
                {
                    "kind": "event",
                    "event": {
                        "id": "evt_1",
                        "title": "Standup",
                        "start": "2026-11-05T09:00",
                        "end": "2026-11-05T09:30",
                        "focus": True,
                        "session_id": "meet-1",
                    },
                    "items": [
                        {
                            "kind": "commitment",
                            "text": "Ship day_plan",
                            "short_id": "abcd1234",
                            "thread": "Pawn",
                            "interrupt": True,
                        }
                    ],
                },
                {
                    "kind": "session",
                    "session": {"session_id": "call-2", "created_at": "2026-11-05T11:00:00Z"},
                    "items": [],
                },
            ],
            "other_items": [
                {"kind": "open_question", "text": "Who owns S3?", "short_id": "q1q1q1q1"}
            ],
            "suggestions": [
                {
                    "event_title": "Standup",
                    "session_title": "meet-1",
                    "score": 0.8,
                }
            ],
        }
    )
    assert "Day 2026-11-05" in text
    assert "Standup" in text
    assert "[focus]" in text
    assert "Ship day_plan" in text
    assert "call-2" in text
    assert "Who owns S3?" in text
    assert "Standup ↔ meet-1" in text


def test_format_planner_day_empty() -> None:
    text = format_planner_day({"date": "2026-11-05", "blocks": [], "other_items": []})
    assert "_No events or sessions._" in text
    assert "_None._" in text


def test_replace_plan_section_append_and_replace() -> None:
    base = "# 2026-11-05\n\n## Open loops\n\n- a\n"
    with_plan = replace_plan_section(base, "- first priority")
    assert "## Plan" in with_plan
    assert "- first priority" in with_plan
    assert "## Open loops" in with_plan
    updated = replace_plan_section(with_plan, "- revised")
    assert "- revised" in updated
    assert "- first priority" not in updated
    assert updated.count("## Plan") == 1


def test_day_plan_update_creates_stub() -> None:
    cfg = _cfg()
    store = _MemStore()
    with patch("pawn_server.core.vault_events.publish_vault_event"):
        receipt = day_plan_update_impl(
            cfg,
            content="- Focus on tests",
            date="2026-11-05",
            store=store,
        )
    key = daily_note_key(cfg, "2026-11-05")
    assert key in receipt
    body = store.read(key)
    assert body.startswith("# 2026-11-05")
    assert "## Plan" in body
    assert "- Focus on tests" in body


def test_day_plan_update_replaces_existing_plan() -> None:
    cfg = _cfg()
    store = _MemStore()
    key = daily_note_key(cfg, "2026-11-05")
    store.write(key, "# 2026-11-05\n\n## Plan\n\n- old\n\n## Notes\n\nkeep\n")
    with patch("pawn_server.core.vault_events.publish_vault_event"):
        day_plan_update_impl(cfg, content="- new", date="2026-11-05", store=store)
    body = store.read(key)
    assert "- new" in body
    assert "- old" not in body
    assert "## Notes" in body
    assert "keep" in body


def test_day_plan_update_rejects_empty() -> None:
    assert day_plan_update_impl(_cfg(), content="  ", store=_MemStore()).startswith("Error:")


def test_planner_day_impl_uses_build_day() -> None:
    cfg = _cfg()
    fake = {
        "date": "2026-11-05",
        "blocks": [],
        "other_items": [],
        "suggestions": [],
    }
    with patch("pawn_server.core.planner.build_day", return_value=fake) as build:
        text = planner_day_impl(cfg, date="2026-11-05")
    build.assert_called_once()
    assert "Day 2026-11-05" in text


def test_planner_day_impl_bad_date() -> None:
    assert planner_day_impl(_cfg(), date="11-05").startswith("Error:")
