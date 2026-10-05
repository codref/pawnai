"""Notify transport records an audit row without a Matrix producer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from pawn_server.core import notify as notify_mod


def test_notify_skips_matrix_when_unconfigured(monkeypatch):
    recorded = {}

    def _record(dsn, **kwargs):
        recorded.update(kwargs)
        return "id"

    monkeypatch.setattr("pawn_agent.core.coworker.db.record_decision", _record)
    cfg = SimpleNamespace(
        db_dsn="postgresql://unused",
        vault=SimpleNamespace(obsidian_vault_name=""),
        coworker=SimpleNamespace(
            matrix_target="matrix", notify=SimpleNamespace(ntfy_url="", topic="", token="")
        ),
        queue_producers=None,
    )
    asyncio.run(notify_mod.notify(cfg, kind="coworker_item", text="hello", item_id="abc"))
    assert recorded["event_kind"] == "notify"
    assert recorded["outcome"] == "sent"


def test_ntfy_posts(monkeypatch):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    cfg = SimpleNamespace(
        coworker=SimpleNamespace(
            notify=SimpleNamespace(ntfy_url="https://ntfy.example", topic="pawn", token="")
        )
    )
    assert notify_mod._ntfy(cfg, text="hi", link="") is None
