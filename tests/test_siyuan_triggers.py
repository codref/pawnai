"""Tests for POST /v1/siyuan/triggers and accept_siyuan_trigger."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pawn_agent.core.sallm_registry import SallmSessionRegistry
from pawn_agent.core.siyuan_protocol import instruction_hash
from pawn_server.core.siyuan_triggers import (
    SiyuanTriggerError,
    accept_siyuan_trigger,
)


def _cfg(**kwargs):
    base = dict(
        db_dsn="postgresql+psycopg://x",
        siyuan_url="http://127.0.0.1:6806",
        siyuan_token="t",
        siyuan_notebook="nb1",
        api_token="secret",
        api_host="127.0.0.1",
        api_port=8000,
        api_model_idle_timeout_minutes=10.0,
        siyuan_watcher=SimpleNamespace(
            enabled=True,
            poll_interval_seconds=10,
            settle_seconds=0.0,
            mention_token="@pawn",
            discover_mentions=False,
            notebook_allowlist=[],
            max_ref_depth=1,
            max_context_blocks=40,
            matrix_target="matrix",
            max_claims_per_tick=2,
        ),
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _resolved(**kwargs):
    defaults = dict(
        trigger_block_id="callout1",
        parent_block_id="parent1",
        root_id="root1",
        notebook_id="nb1",
        instruction_text="> [!TIP] 🤖 T\n> @pawn do it\n> more\n",
        source_updated="20260922120000",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_accept_schedules_execute_for_new_request() -> None:
    cfg = _cfg()
    client = MagicMock()
    registry = SallmSessionRegistry()
    resolved = _resolved()
    ih = instruction_hash(resolved.instruction_text)

    with (
        patch(
            "pawn_server.core.siyuan_triggers.resolve_pawn_trigger",
            return_value=resolved,
        ),
        patch(
            "pawn_server.core.siyuan_triggers.resolve_notebook_allowlist",
            return_value=["nb1"],
        ),
        patch(
            "pawn_server.core.siyuan_triggers.upsert_siyuan_agent_request",
            return_value="req-1",
        ),
        patch(
            "pawn_server.core.siyuan_triggers.get_siyuan_agent_request",
            return_value=SimpleNamespace(
                id="req-1",
                status="queued",
                instruction_hash=ih,
                conversation_id="siyuan:root1",
            ),
        ),
        patch(
            "pawn_server.core.siyuan_triggers.claim_siyuan_agent_request",
            return_value=True,
        ),
        patch(
            "pawn_server.core.siyuan_triggers.execute_claimed_request",
            new_callable=AsyncMock,
        ) as exec_req,
        patch("asyncio.create_task") as create_task,
    ):

        def _swallow(coro, *args, **kwargs):
            if hasattr(coro, "close"):
                coro.close()
            return MagicMock()

        create_task.side_effect = _swallow
        result = asyncio.run(accept_siyuan_trigger(cfg, "child1", registry=registry, client=client))
    assert result.started is True
    assert result.request_id == "req-1"
    assert result.status == "claimed"
    assert result.conversation_id == "siyuan:root1"
    create_task.assert_called_once()
    client.set_block_attrs.assert_called()
    # Background runner not invoked synchronously (create_task mocked).
    exec_req.assert_not_called()


def test_accept_idempotent_when_review() -> None:
    cfg = _cfg()
    client = MagicMock()
    registry = SallmSessionRegistry()
    resolved = _resolved()
    ih = instruction_hash(resolved.instruction_text)

    with (
        patch(
            "pawn_server.core.siyuan_triggers.resolve_pawn_trigger",
            return_value=resolved,
        ),
        patch(
            "pawn_server.core.siyuan_triggers.resolve_notebook_allowlist",
            return_value=["nb1"],
        ),
        patch(
            "pawn_server.core.siyuan_triggers.upsert_siyuan_agent_request",
            return_value="req-1",
        ),
        patch(
            "pawn_server.core.siyuan_triggers.get_siyuan_agent_request",
            return_value=SimpleNamespace(
                id="req-1",
                status="review",
                instruction_hash=ih,
                conversation_id="siyuan:root1",
            ),
        ),
        patch("pawn_server.core.siyuan_triggers.claim_siyuan_agent_request") as claim,
        patch("asyncio.create_task") as create_task,
    ):
        result = asyncio.run(
            accept_siyuan_trigger(cfg, "callout1", registry=registry, client=client)
        )
    assert result.started is False
    assert result.status == "review"
    claim.assert_not_called()
    create_task.assert_not_called()


def test_accept_rejects_disallowed_notebook() -> None:
    cfg = _cfg()
    resolved = _resolved(notebook_id="other")
    with (
        patch(
            "pawn_server.core.siyuan_triggers.resolve_pawn_trigger",
            return_value=resolved,
        ),
        patch(
            "pawn_server.core.siyuan_triggers.resolve_notebook_allowlist",
            return_value=["nb1"],
        ),
        pytest.raises(SiyuanTriggerError) as excinfo,
    ):
        asyncio.run(
            accept_siyuan_trigger(
                cfg, "callout1", registry=SallmSessionRegistry(), client=MagicMock()
            )
        )
    assert excinfo.value.status_code == 403


def test_accept_404_when_unresolved() -> None:
    cfg = _cfg()
    with (
        patch(
            "pawn_server.core.siyuan_triggers.resolve_pawn_trigger",
            return_value=None,
        ),
        pytest.raises(SiyuanTriggerError) as excinfo,
    ):
        asyncio.run(
            accept_siyuan_trigger(
                cfg, "missing", registry=SallmSessionRegistry(), client=MagicMock()
            )
        )
    assert excinfo.value.status_code == 404


def test_api_siyuan_triggers_requires_bearer() -> None:
    from pawn_server.core import api_server

    cfg = _cfg()
    api_server._cfg = cfg  # noqa: SLF001
    client = TestClient(api_server.app)
    resp = client.post("/v1/siyuan/triggers", json={"block_id": "b1"})
    assert resp.status_code == 401


def test_api_siyuan_triggers_202() -> None:
    from pawn_server.core import api_server
    from pawn_server.core.siyuan_triggers import SiyuanTriggerResult

    cfg = _cfg()
    api_server._cfg = cfg  # noqa: SLF001

    async def _fake_accept(*_a, **_k):
        return SiyuanTriggerResult(
            request_id="req-9",
            status="claimed",
            conversation_id="siyuan:root1",
            trigger_block_id="callout1",
            started=True,
        )

    with patch(
        "pawn_server.core.siyuan_triggers.accept_siyuan_trigger",
        new=_fake_accept,
    ):
        client = TestClient(api_server.app)
        resp = client.post(
            "/v1/siyuan/triggers",
            json={"block_id": "callout1"},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["request_id"] == "req-9"
    assert body["started"] is True
    assert body["conversation_id"] == "siyuan:root1"
