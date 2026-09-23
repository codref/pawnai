"""Watcher tick unit tests (mocked SiYuan + DB)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pawn_server.core.siyuan_watcher import discover_and_enqueue, run_siyuan_watcher_tick


def _cfg(**kwargs):
    base = dict(
        db_dsn="postgresql+psycopg://x",
        siyuan_url="http://127.0.0.1:6806",
        siyuan_token="t",
        siyuan_notebook="nb1",
        siyuan_watcher=SimpleNamespace(
            enabled=True,
            poll_interval_seconds=10,
            settle_seconds=0.0,  # tests claim immediately unless overridden
            mention_token="@pawn",
            discover_mentions=True,  # exercise SQL discovery path in unit tests
            notebook_allowlist=[],
            max_ref_depth=1,
            max_context_blocks=40,
            matrix_target="matrix",
            max_claims_per_tick=2,
        ),
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_discover_bootstrap_skips_historical() -> None:
    cfg = _cfg()
    client = MagicMock()
    client.query_sql.return_value = [
        {
            "id": "b1",
            "parent_id": "p1",
            "root_id": "r1",
            "box": "nb1",
            "content": "@pawn old",
            "markdown": "@pawn old",
            "updated": "20260101000000",
        }
    ]
    with (
        patch(
            "pawn_server.core.siyuan_watcher.get_siyuan_poll_watermark",
            return_value="",
        ),
        patch("pawn_server.core.siyuan_watcher.set_siyuan_poll_watermark") as set_wm,
        patch("pawn_server.core.siyuan_watcher.upsert_siyuan_agent_request") as upsert,
    ):
        n = discover_and_enqueue(cfg, client)
    assert n == 0
    set_wm.assert_called_once_with(cfg.db_dsn, "20260101000000")
    upsert.assert_not_called()


def test_discover_enqueues_new_block() -> None:
    cfg = _cfg()
    client = MagicMock()
    client.query_sql.return_value = [
        {
            "id": "b1",
            "parent_id": "p1",
            "root_id": "r1",
            "box": "nb1",
            "content": "@pawn do it",
            "markdown": "@pawn do it",
            "updated": "20260921120000",
        }
    ]
    client.get_block_attrs.return_value = {}
    with (
        patch(
            "pawn_server.core.siyuan_watcher.get_siyuan_poll_watermark",
            return_value="20260920000000",
        ),
        patch("pawn_server.core.siyuan_watcher.set_siyuan_poll_watermark"),
        patch(
            "pawn_server.core.siyuan_watcher.upsert_siyuan_agent_request",
            return_value="req-1",
        ) as upsert,
        patch("pawn_server.core.siyuan_watcher.new_request_id", return_value="req-1"),
    ):
        n = discover_and_enqueue(cfg, client)
    assert n == 1
    upsert.assert_called_once()
    client.set_block_attrs.assert_called()


def test_discover_waits_for_settle() -> None:
    from pawn_agent.core.siyuan_protocol import InstructionSettleTracker
    from pawn_server.core import siyuan_watcher as sw

    cfg = _cfg()
    cfg.siyuan_watcher.settle_seconds = 45.0
    client = MagicMock()
    client.query_sql.return_value = [
        {
            "id": "b1",
            "parent_id": "p1",
            "root_id": "r1",
            "box": "nb1",
            "content": "@pawn still typing",
            "markdown": "@pawn still typing",
            "updated": "20260921120000",
        }
    ]
    client.get_block_attrs.return_value = {}
    tracker = InstructionSettleTracker()
    with (
        patch.object(sw, "_SETTLE", tracker),
        patch(
            "pawn_server.core.siyuan_watcher.get_siyuan_poll_watermark",
            return_value="20260920000000",
        ),
        patch("pawn_server.core.siyuan_watcher.set_siyuan_poll_watermark") as set_wm,
        patch("pawn_server.core.siyuan_watcher.upsert_siyuan_agent_request") as upsert,
    ):
        n1 = discover_and_enqueue(cfg, client)
        n2 = discover_and_enqueue(cfg, client)
    assert n1 == 0
    assert n2 == 0
    upsert.assert_not_called()
    # Unsettled blocks must not advance the watermark.
    set_wm.assert_not_called()


def test_instruction_settle_tracker() -> None:
    from pawn_agent.core.siyuan_protocol import InstructionSettleTracker

    t = InstructionSettleTracker()
    assert t.is_settled("b", "h1", settle_seconds=10, now=100.0) is False
    assert t.is_settled("b", "h1", settle_seconds=10, now=105.0) is False
    assert t.is_settled("b", "h1", settle_seconds=10, now=110.0) is True
    # Edit resets the timer.
    assert t.is_settled("b", "h2", settle_seconds=10, now=111.0) is False
    assert t.is_settled("b", "h2", settle_seconds=10, now=120.0) is False
    assert t.is_settled("b", "h2", settle_seconds=10, now=121.0) is True


def test_upsert_rewrites_queued_on_hash_change() -> None:
    """Mid-edit race: same trigger still queued, new hash must not PK-collide."""
    from pawn_agent.utils import db as dbmod

    existing = SimpleNamespace(
        id="old-id",
        trigger_block_id="t1",
        status="queued",
        instruction_hash="hash-old",
        instruction_text="@pawn a",
        source_updated="1",
        parent_block_id="p",
        root_id="r",
        notebook_id="n",
        conversation_id="siyuan:r",
        created_at=None,
        updated_at=None,
    )

    class _Q:
        def __init__(self, rows):
            self._rows = rows

        def filter_by(self, **kwargs):
            self._kwargs = kwargs
            return self

        def order_by(self, *_a, **_k):
            return self

        def one_or_none(self):
            # First call is by trigger+hash — miss.
            if self._kwargs.get("instruction_hash") == "hash-new":
                return None
            return None

        def first(self):
            return existing

    class _Session:
        def query(self, _model):
            return _Q([])

        def get(self, _model, key):
            return existing if key == "old-id" else None

        def add(self, _row):
            raise AssertionError("should rewrite queued row, not insert")

    class _CM:
        def __enter__(self):
            return _Session()

        def __exit__(self, *args):
            return False

    with patch.object(dbmod, "_get_session", return_value=_CM()):
        rid = dbmod.upsert_siyuan_agent_request(
            "dsn",
            request_id="old-id",  # same id SiYuan attr still holds
            trigger_block_id="t1",
            parent_block_id="p",
            root_id="r",
            notebook_id="n",
            instruction_hash="hash-new",
            conversation_id="siyuan:r",
            instruction_text="@pawn suggest me a followup",
            source_updated="2",
        )
    assert rid == "old-id"
    assert existing.instruction_hash == "hash-new"
    assert existing.instruction_text == "@pawn suggest me a followup"


def test_watcher_tick_claims_queued() -> None:
    import asyncio

    cfg = _cfg()
    req = SimpleNamespace(
        id="req-1",
        trigger_block_id="t1",
        parent_block_id="p1",
        root_id="r1",
        conversation_id="siyuan:r1",
        instruction_text="@pawn hi",
        output_block_id=None,
        indexed_at=None,
    )
    client = MagicMock()
    registry = MagicMock()
    with (
        patch(
            "pawn_server.core.siyuan_watcher.client_from_agent_config",
            return_value=client,
        ),
        patch(
            "pawn_server.core.siyuan_watcher.discover_and_enqueue",
            return_value=0,
        ),
        patch(
            "pawn_server.core.siyuan_watcher.list_siyuan_agent_requests",
            return_value=[req],
        ),
        patch(
            "pawn_server.core.siyuan_watcher.claim_siyuan_agent_request",
            return_value=True,
        ),
        patch(
            "pawn_server.core.siyuan_watcher.execute_claimed_request",
            new_callable=AsyncMock,
        ) as exec_req,
    ):
        stats = asyncio.run(run_siyuan_watcher_tick(cfg, registry=registry))
    assert stats["claimed"] == 1
    exec_req.assert_awaited_once()


def test_watcher_tick_skips_discovery_when_disabled() -> None:
    import asyncio

    cfg = _cfg()
    cfg.siyuan_watcher.discover_mentions = False
    client = MagicMock()
    registry = MagicMock()
    with (
        patch(
            "pawn_server.core.siyuan_watcher.client_from_agent_config",
            return_value=client,
        ),
        patch(
            "pawn_server.core.siyuan_watcher.discover_and_enqueue",
            return_value=99,
        ) as discover,
        patch(
            "pawn_server.core.siyuan_watcher.list_siyuan_agent_requests",
            return_value=[],
        ),
    ):
        stats = asyncio.run(run_siyuan_watcher_tick(cfg, registry=registry))
    assert stats["discovered"] == 0
    discover.assert_not_called()


def test_execute_claimed_request_does_not_rewrite_callout() -> None:
    import asyncio

    from pawn_server.core.siyuan_watcher import execute_claimed_request

    cfg = _cfg()
    client = MagicMock()
    client.get_block_kramdown.return_value = "parent"
    client.append_block.return_value = "out-1"
    request = SimpleNamespace(
        id="req-1",
        trigger_block_id="callout1",
        parent_block_id="parent1",
        root_id="root1",
        conversation_id="siyuan:root1",
        instruction_text="> [!TIP] 🤖 T\n> @pawn do it\n",
        instruction_hash="abc",
        output_block_id=None,
    )
    registry = MagicMock()
    result = SimpleNamespace(response="done", run_id="run-1")

    with (
        patch(
            "pawn_server.core.siyuan_watcher.get_siyuan_agent_request",
            return_value=request,
        ),
        patch("pawn_server.core.siyuan_watcher.update_siyuan_agent_request"),
        patch(
            "pawn_server.core.siyuan_watcher.run_agent_turn",
            new_callable=AsyncMock,
            return_value=result,
        ),
        patch(
            "pawn_server.core.siyuan_watcher._notify_matrix",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        asyncio.run(execute_claimed_request(cfg, "req-1", registry=registry, client=client))

    client.update_block.assert_not_called()
