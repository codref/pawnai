"""Vault watcher tick tests with in-memory vault store."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pawn_core.vault import VaultStore
from pawn_server.core.vault_protocol import render_task_note
from pawn_server.core.vault_watcher import run_vault_watcher_tick
from tests.test_vault_store import FakeS3Client


@dataclass
class _Cfg:
    vault: MagicMock
    vault_watcher: MagicMock
    db_dsn: str = "sqlite:///:memory:"


@pytest.fixture
def vault_setup(monkeypatch):
    client = FakeS3Client()
    store = VaultStore(bucket="b", agent_root="Pawn", client=client)
    task_id = "task-uuid-1"
    body = render_task_note(
        task_id=task_id,
        status="todo",
        instruction="List open items.",
        context="",
        result="",
        conversation="note:Pawn/Tasks/task-uuid-1.md",
    )
    store.write(f"Pawn/Tasks/{task_id}.md", body)

    vault = MagicMock()
    vault.s3 = MagicMock()
    vault.s3.bucket = "b"
    vault.s3.prefix = ""
    vault.s3.endpoint_url = None
    vault.s3.access_key = None
    vault.s3.secret_key = None
    vault.s3.region = None
    vault.s3.path_style = True
    vault.s3.verify_ssl = True
    vault.bucket = "b"  # property-compatible shortcut used by helpers
    vault.agent_root = "Pawn"
    vault.obsidian_vault_name = "TestVault"

    watcher = MagicMock()
    watcher.max_claims_per_tick = 3
    watcher.matrix_target = "matrix"

    cfg = _Cfg(vault=vault, vault_watcher=watcher)

    monkeypatch.setattr(
        "pawn_server.core.vault_watcher.vault_store_from_config",
        lambda _cfg: store,
    )

    upsert_calls: list[str] = []

    def _upsert(dsn, **kwargs):
        upsert_calls.append(kwargs.get("task_id", ""))
        return kwargs["task_id"]

    def _claim(dsn, tid):
        return True

    monkeypatch.setattr("pawn_server.core.vault_watcher.upsert_vault_task", _upsert)
    monkeypatch.setattr("pawn_server.core.vault_watcher.claim_vault_task", _claim)
    monkeypatch.setattr("pawn_server.core.vault_watcher.get_vault_task", lambda *a, **k: None)
    monkeypatch.setattr(
        "pawn_server.core.vault_watcher.get_vault_task_by_key", lambda *a, **k: None
    )
    monkeypatch.setattr("pawn_server.core.vault_watcher.update_vault_task", lambda *a, **k: None)

    execute = AsyncMock(return_value=MagicMock(task_id=task_id, status="review", result="ok"))
    monkeypatch.setattr("pawn_server.core.vault_watcher.execute_vault_task", execute)

    registry = MagicMock()
    return cfg, registry, execute, upsert_calls, task_id


def test_watcher_tick_executes_todo(vault_setup):
    cfg, registry, execute, upsert_calls, task_id = vault_setup
    stats = asyncio.run(run_vault_watcher_tick(cfg, registry=registry))
    assert stats["executed"] == 1
    assert upsert_calls == [task_id]
    execute.assert_awaited_once()
    assert execute.await_args.kwargs["write_result_to_vault"] is True
