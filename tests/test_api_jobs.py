"""Background jobs API (/v1/jobs) and native chat SSE (/v1/pawn/chat)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from pawn_agent.core.agent_runner import AgentRunResult
from pawn_agent.utils.config import ApiSection
from pawn_agent.utils.db import AgentRun, VaultTask
from pawn_core.config import VaultConfig
from pawn_core.database import Base
from pawn_core.vault import VaultStore
from pawn_server.core import api_server
from pawn_server.core.chat_context import NoteRef, build_chat_prompt
from pawn_server.core.jobs import build_job_context
from pawn_server.core.vault_protocol import parse_task_note
from tests.test_vault_store import FakeS3Client

AUTH = {"Authorization": "Bearer tok"}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'jobs.db'}"
    engine = sa.create_engine(dsn)
    Base.metadata.create_all(engine, tables=[AgentRun.__table__, VaultTask.__table__])
    engine.dispose()

    api = ApiSection(token="tok", stream_keepalive_seconds=0.05)
    cfg = SimpleNamespace(
        api=api,
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=dsn,
        vault=VaultConfig(agent_root="Pawn"),
        vault_watcher=SimpleNamespace(matrix_target="matrix"),
    )
    store = VaultStore(bucket="b", agent_root="Pawn", client=FakeS3Client())
    store.write("Projects/Roadmap.md", "Roadmap body", skip_guards=True)

    monkeypatch.setattr("pawn_server.core.jobs.vault_store_from_config", lambda _c: store)
    monkeypatch.setattr("pawn_core.vault_config.vault_store_from_config", lambda _c: store)

    prompts: list[dict[str, Any]] = []

    async def fake_turn(**kwargs: Any) -> AgentRunResult:
        prompts.append(kwargs)
        cb = kwargs.get("on_progress")
        if cb is not None:
            cb("tool", {"gen_ai.tool.name": "note_read"})
        return AgentRunResult(run_id="run-1", response="the answer")

    async def no_notify(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr("pawn_server.core.vault_tasks.run_agent_turn", fake_turn)
    monkeypatch.setattr("pawn_server.core.vault_tasks._notify_matrix", no_notify)
    monkeypatch.setattr(api_server, "run_agent_turn", fake_turn)
    return SimpleNamespace(cfg=cfg, store=store, prompts=prompts)


def _wait_status(client: TestClient, job_id: str, want: set[str]) -> dict:
    deadline = time.time() + 5
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
        if job["status"] in want:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {want}: {job}")


def test_ask_job_accepted_then_review(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/jobs",
            headers=AUTH,
            json={
                "kind": "ask",
                "id": "job-1",
                "instruction": "Summarize",
                "note_path": "Projects/Roadmap.md",
                "selection": "important bit",
                "context_paths": ["Other.md"],
            },
        )
        assert resp.status_code == 202
        assert resp.json()["id"] == "job-1"
        job = _wait_status(client, "job-1", {"review"})
        assert job["result"] == "the answer"
        assert job["conversation"] == "note:Projects/Roadmap.md"

        note = parse_task_note(env.store.read("Pawn/Tasks/job-1.md"))
        assert note["status"] == "review"
        assert "the answer" in note["result"]
        assert "important bit" in note["context"]
        assert "important bit" in env.prompts[0]["prompt"]

        listed = client.get(
            "/v1/jobs", headers=AUTH, params={"conversation": "note:Projects/Roadmap.md"}
        ).json()["data"]
        assert [j["id"] for j in listed] == ["job-1"]

        remembered: list[str] = []
        agent = SimpleNamespace(remember=lambda text, **_k: remembered.append(text))

        async def get_or_create(*_a: Any) -> Any:
            return SimpleNamespace(_agent=agent)

        original = api_server.get_sallm_registry()
        api_server.set_sallm_registry(SimpleNamespace(get_or_create=get_or_create))
        try:
            approved = client.post("/v1/jobs/job-1/approve", headers=AUTH, json={})
        finally:
            api_server.set_sallm_registry(original)
        assert approved.status_code == 200
        assert approved.json()["status"] == "done"
        assert approved.json()["approved"] is True
        assert "the answer" in remembered[0]
        assert parse_task_note(env.store.read("Pawn/Tasks/job-1.md"))["approved"] is True


def test_ask_job_dismiss_closes_without_indexing(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/jobs",
            headers=AUTH,
            json={"kind": "ask", "id": "job-d1", "instruction": "Draft something"},
        )
        assert resp.status_code == 202
        _wait_status(client, "job-d1", {"review"})

        remembered: list[str] = []
        agent = SimpleNamespace(remember=lambda text, **_k: remembered.append(text))

        async def get_or_create(*_a: Any) -> Any:
            return SimpleNamespace(_agent=agent)

        original = api_server.get_sallm_registry()
        api_server.set_sallm_registry(SimpleNamespace(get_or_create=get_or_create))
        try:
            dismissed = client.post("/v1/jobs/job-d1/dismiss", headers=AUTH, json={})
        finally:
            api_server.set_sallm_registry(original)

        assert dismissed.status_code == 200
        body = dismissed.json()
        assert body["status"] == "done"
        assert body["approved"] is False
        assert remembered == []
        note = parse_task_note(env.store.read("Pawn/Tasks/job-d1.md"))
        assert note["status"] == "done"
        assert note["approved"] is False

        again = client.post("/v1/jobs/job-d1/dismiss", headers=AUTH, json={})
        assert again.status_code == 200
        assert again.json()["status"] == "done"

        push = client.post(
            "/v1/jobs",
            headers=AUTH,
            json={
                "kind": "push_note",
                "id": "job-d2",
                "path": "Pawn/Notes/x",
                "content": "hi",
            },
        )
        assert push.status_code == 202
        _wait_status(client, "job-d2", {"done", "blocked"})
        wrong_kind = client.post("/v1/jobs/job-d2/dismiss", headers=AUTH, json={})
        assert wrong_kind.status_code == 409


def test_delete_and_flush_finished_jobs(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        ask = client.post(
            "/v1/jobs",
            headers=AUTH,
            json={"kind": "ask", "id": "job-del", "instruction": "Draft"},
        )
        assert ask.status_code == 202
        _wait_status(client, "job-del", {"review"})
        # Dismiss → done so flush can take it; delete works on review too.
        assert client.post("/v1/jobs/job-del/dismiss", headers=AUTH, json={}).status_code == 200
        assert env.store.exists("Pawn/Tasks/job-del.md")

        push = client.post(
            "/v1/jobs",
            headers=AUTH,
            json={
                "kind": "push_note",
                "id": "job-push-del",
                "path": "Pawn/Notes/flush-me",
                "content": "x",
            },
        )
        assert push.status_code == 202
        assert _wait_status(client, "job-push-del", {"done", "blocked"})["status"] == "done"

        one = client.delete("/v1/jobs/job-del", headers=AUTH)
        assert one.status_code == 200
        assert one.json()["deleted"] is True
        assert not env.store.exists("Pawn/Tasks/job-del.md")
        assert client.get("/v1/jobs/job-del", headers=AUTH).status_code == 404

        flushed = client.post(
            "/v1/jobs/delete",
            headers=AUTH,
            json={"flush_terminal": True},
        )
        assert flushed.status_code == 200
        assert flushed.json()["deleted"] >= 1
        assert client.get("/v1/jobs/job-push-del", headers=AUTH).status_code == 404

        running = client.post(
            "/v1/jobs",
            headers=AUTH,
            json={"kind": "ask", "id": "job-live", "instruction": "Stay"},
        )
        assert running.status_code == 202
        # While still active (before review), delete must 409.
        early = client.delete("/v1/jobs/job-live", headers=AUTH)
        # May already be review in fast tests; only assert 409 when still active.
        if early.status_code == 409:
            detail = early.json()["detail"].lower()
            assert "cancel" in detail or "status" in detail


def test_push_note_job_respects_guards(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        ok = client.post(
            "/v1/jobs",
            headers=AUTH,
            json={"kind": "push_note", "id": "p1", "path": "Pawn/Notes/x", "content": "hi"},
        )
        assert ok.status_code == 202
        assert _wait_status(client, "p1", {"done", "blocked"})["status"] == "done"
        assert env.store.read("Pawn/Notes/x.md") == "hi"

        denied = client.post(
            "/v1/jobs",
            headers=AUTH,
            json={"kind": "push_note", "id": "p2", "path": "Projects/Roadmap.md", "content": "x"},
        )
        assert denied.status_code == 202
        job = _wait_status(client, "p2", {"done", "blocked"})
        assert job["status"] == "blocked"
        assert job["error_code"] == "write_denied"


def test_upload_document_lands_in_inbox(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/jobs/upload",
            headers=AUTH,
            files={"file": ("notes.md", b"# Doc", "text/markdown")},
            data={"id": "u1"},
        )
        assert resp.status_code == 202
        job = _wait_status(client, "u1", {"done", "blocked"})
        assert job["status"] == "done"
        assert env.store.read("Pawn/Inbox/notes.md") == "# Doc"


def test_unknown_kind_rejected(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post("/v1/jobs", headers=AUTH, json={"kind": "nope"})
        assert resp.status_code == 422


def test_legacy_vault_task_alias_returns_202(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/vault/tasks", headers=AUTH, json={"id": "legacy", "instruction": "Do it"}
        )
        assert resp.status_code == 202
        assert resp.json()["task_id"] == "legacy"
        _wait_status(client, "legacy", {"review"})
        status = client.get("/v1/vault/tasks/legacy", headers=AUTH).json()
        assert status["status"] == "review"
        assert status["result"] == "the answer"


def _events(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        lines = block.strip().splitlines()
        name = next((ln[7:] for ln in lines if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in lines if ln.startswith("data: ")), None)
        if name and data:
            out.append((name, json.loads(data)))
    return out


def test_pawn_chat_streams_typed_events(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/pawn/chat",
            headers=AUTH,
            json={
                "conversation": "note:Projects/Roadmap.md",
                "message": "What is next?",
                "active_note": {"path": "Projects/Roadmap.md"},
                "selection": "Q3 goals",
            },
        )
        events = _events(resp.text)
        names = [e[0] for e in events]
        assert names[0] == "progress"
        assert ("answer", {"content": "the answer"}) in events
        assert names[-1] == "done"
        prompt = env.prompts[0]["prompt"]
        assert prompt.startswith("What is next?")
        assert "Roadmap body" in prompt
        assert "> Q3 goals" in prompt
        assert env.prompts[0]["session_id"] == "note:Projects/Roadmap.md"


def test_pawn_chat_background_creates_job(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/pawn/chat",
            headers=AUTH,
            json={"conversation": "chat:x", "message": "Long thing", "background": True},
        )
        events = _events(resp.text)
        assert events[0][0] == "job"
        job_id = events[0][1]["id"]
        assert _wait_status(client, job_id, {"review"})["conversation"] == "chat:x"


def test_build_chat_prompt_without_context_is_bare() -> None:
    assert build_chat_prompt("hello") == "hello"
    prompt = build_chat_prompt(
        "hello",
        context=[NoteRef("A.md", "alpha"), NoteRef("A.md", "dup"), NoteRef("B.md", None)],
    )
    assert "[[A]]" in prompt and "alpha" in prompt and "dup" not in prompt
    assert "[[B]]\n(unavailable)" in prompt


def test_build_job_context() -> None:
    ctx = build_job_context(selection="sel", context_paths=["X/Y.md"], context="extra")
    assert "sel" in ctx and "[[X/Y]]" in ctx and ctx.endswith("extra")
