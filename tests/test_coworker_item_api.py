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
        tasknotes=SimpleNamespace(
            external_tasks_dir="TaskNotes/Tasks",
            tasks_dir="",
            views_dir="",
            projects_dir="",
            proposals_dir="",
            ident_tag="task",
            week_starts_on=1,
            me=[],
            display_name="",
            timezone="",
        ),
        coworker=SimpleNamespace(items_dir="Pawn/Items", threads_dir="Pawn/Threads", me=[]),
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

    def _write(key: str, body: str, **_kwargs: Any) -> None:
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
    body = next(v for k, v in written.items() if k.startswith("Pawn/TaskNotes/Tasks/"))
    assert "Involve John Martinez" in body
    assert "status: open" in body or "status: open" in body.replace('"', "")
    assert "project:" not in body.split("---", 2)[1]


def test_apply_todo_with_project_writes_stub_and_board(monkeypatch: pytest.MonkeyPatch) -> None:
    from pawn_core.vault import VaultError

    cfg = _cfg()
    store = MagicMock()
    store.read.side_effect = VaultError("missing")
    store.list.return_value = []
    written: dict[str, str] = {}

    def _write(key: str, body: str, **_kwargs: Any) -> None:
        written[key] = body

    store.write.side_effect = _write
    item = {
        "id": "uuid-1",
        "short_id": "abcd1234",
        "kind": "commitment",
        "text": "Scaffold OCI resources",
        "thread": "Hiring",
        "fingerprint": "fp1",
        "payload": {},
        "note_key": "Pawn/Items/x.md",
        "quote": "",
        "reason": "",
        "interrupt": False,
    }
    monkeypatch.setattr(itemdb, "get_item", lambda dsn, iid: item)
    updated_payload: dict[str, Any] = {}

    def _update(dsn, iid, **fields):  # noqa: ANN001,ARG001
        updated_payload.update(fields.get("payload") or {})
        return {**item, **fields}

    monkeypatch.setattr(itemdb, "update_item", _update)

    receipt = asyncio.run(
        actions.apply_action(cfg, "abcd1234", "todo", "Hiring pipeline", store=store)
    )
    assert "Hiring pipeline" in receipt
    assert any(k.startswith("Pawn/TaskNotes/Tasks/") for k in written)
    assert any(k.startswith("Pawn/TaskNotes/Projects/") for k in written)
    assert any(k.endswith(".base") for k in written)
    task_body = next(v for k, v in written.items() if k.startswith("Pawn/TaskNotes/Tasks/"))
    assert "project: Hiring pipeline" in task_body or 'project: "Hiring pipeline"' in task_body
    assert "[[Hiring pipeline]]" in task_body or "projects:" in task_body
    assert updated_payload.get("todo_project") == "Hiring pipeline"


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
