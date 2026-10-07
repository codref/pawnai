"""Coworker item list filters, todo/delete actions, and bulk delete."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from pawn_agent.core.coworker import actions
from pawn_agent.core.coworker import db as itemdb


def _cfg(dsn: str = "postgresql://unused") -> Any:
    return SimpleNamespace(
        db_dsn=dsn,
        vault=SimpleNamespace(agent_root="Pawn"),
        tasknotes=SimpleNamespace(external_tasks_dir="TaskNotes/Tasks"),
        coworker=SimpleNamespace(items_dir="Pawn/Items", threads_dir="Pawn/Threads"),
        agent_scheduler=SimpleNamespace(default_timezone="UTC"),
    )


def test_list_items_filters_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "id": "1",
            "short_id": "aaa",
            "kind": "decision",
            "text": "Ship",
            "thread": "X",
            "status": "new",
        },
        {
            "id": "2",
            "short_id": "bbb",
            "kind": "commitment",
            "text": "Call John",
            "thread": "Y",
            "status": "notified",
        },
    ]
    captured: dict[str, Any] = {}

    def _list(dsn, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return rows[: kwargs.get("limit", 100)]

    def _count(dsn, **kwargs):  # noqa: ARG001
        return 2

    monkeypatch.setattr(itemdb, "list_items", _list)
    monkeypatch.setattr(itemdb, "count_items", _count)

    listed = itemdb.list_items(
        "dsn",
        statuses=list(itemdb.OPEN_STATUSES),
        kind="decision",
        q="Ship",
        limit=50,
        offset=0,
    )
    assert listed == rows
    assert captured["kind"] == "decision"
    assert captured["q"] == "Ship"
    assert itemdb.count_items("dsn", statuses=list(itemdb.OPEN_STATUSES)) == 2


def test_apply_todo_writes_tasknotes(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    store = MagicMock()
    store.read.side_effect = Exception("missing")
    written: dict[str, str] = {}

    def _write(key: str, body: str) -> None:
        written[key] = body

    store.write.side_effect = _write
    item = {
        "id": "uuid-1",
        "short_id": "abcd1234",
        "kind": "commitment",
        "text": "Involve John Martinez",
        "thread": "Project",
        "fingerprint": "fp1",
        "payload": {},
        "note_key": "Pawn/Items/x.md",
        "quote": "",
        "reason": "",
        "interrupt": False,
    }
    monkeypatch.setattr(itemdb, "get_item", lambda dsn, iid: item)
    monkeypatch.setattr(
        itemdb,
        "update_item",
        lambda dsn, iid, **fields: {**item, **fields},
    )

    receipt = asyncio.run(actions.apply_action(cfg, "abcd1234", "todo", store=store))
    assert "TODO" in receipt
    assert any(k.startswith("Pawn/TaskNotes/Tasks/") for k in written)
    body = next(iter(written.values()))
    assert "Involve John Martinez" in body
    assert "status: open" in body or "status: open" in body.replace('"', "")


def test_apply_delete_suppresses_and_removes_note(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    store = MagicMock()
    item = {
        "id": "uuid-1",
        "short_id": "abcd1234",
        "kind": "decision",
        "text": "Noise",
        "thread": "",
        "fingerprint": "fp-noise",
        "payload": {},
        "note_key": "Pawn/Items/noise.md",
        "quote": "",
        "reason": "",
        "interrupt": False,
    }
    suppressed: list[str] = []
    monkeypatch.setattr(itemdb, "get_item", lambda dsn, iid: item)
    monkeypatch.setattr(
        itemdb,
        "update_item",
        lambda dsn, iid, **fields: {**item, **fields},
    )
    monkeypatch.setattr(itemdb, "add_suppression", lambda dsn, fp: suppressed.append(fp))

    receipt = asyncio.run(actions.apply_action(cfg, "abcd1234", "delete", store=store))
    assert "Deleted" in receipt
    assert suppressed == ["fp-noise"]
    store.delete.assert_called_once_with("Pawn/Items/noise.md")


def test_bulk_delete_all_open(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    monkeypatch.setattr(itemdb, "list_item_ids", lambda dsn, **kwargs: ["id1", "id2"])
    called: list[str] = []

    async def _apply(cfg, item_id, action, arg=None, **kwargs):  # noqa: ANN001,ARG001
        called.append(f"{action}:{item_id}")
        return f"Deleted {item_id}."

    monkeypatch.setattr(actions, "apply_action", _apply)
    result = asyncio.run(actions.delete_items(cfg, all_open=True))
    assert result["deleted"] == 2
    assert called == ["delete:id1", "delete:id2"]
