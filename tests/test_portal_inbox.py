"""Portal Inbox routes (/v1/portal/inbox/*): ideas, research captures, counts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from pawn_agent.core.agent_runner import AgentRunResult
from pawn_agent.utils.config import ApiSection, CaptureConfig, CoworkerConfig, PortalConfig
from pawn_agent.utils.db import AgentRun, VaultTask
from pawn_core.config import VaultConfig
from pawn_core.database import Base
from pawn_core.vault import VaultStore, parse_frontmatter
from pawn_server.core import api_server
from pawn_server.core.inbox_api import idea_line, with_status
from pawn_server.core.portal_models import PORTAL_TABLES
from tests.test_vault_store import FakeS3Client

AUTH = {"Authorization": "Bearer tok"}

IDEA = "---\ntags: [idea]\nstatus: inbox\n---\n\n# Solar dryer\n\nBuild a solar fruit dryer.\n"
CAPTURE = (
    "---\npawn: capture\nstatus: inbox\nkind: text\ncreated: '2026-11-20T10:00:00Z'\n"
    "collection: ''\nhint: books\nentity: ''\ntype: ''\nsource_url: https://example.com/a\n---\n\n"
    "# A book\n\n## Snippets\n"
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'portal.db'}"
    engine = sa.create_engine(dsn)
    Base.metadata.create_all(engine, tables=[AgentRun.__table__, VaultTask.__table__, *PORTAL_TABLES])
    engine.dispose()
    cfg = SimpleNamespace(
        api=ApiSection(token="tok", stream_keepalive_seconds=0.05, portal_dir=""),
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=dsn,
        vault=VaultConfig(agent_root="Pawn"),
        portal=PortalConfig(),
        coworker=CoworkerConfig(),
        capture=CaptureConfig(),
        vault_watcher=SimpleNamespace(matrix_target="matrix"),
    )
    store = VaultStore(bucket="b", agent_root="Pawn", client=FakeS3Client())
    store.write("Ideas/2026-11-20 Solar dryer.md", IDEA, skip_guards=True)
    store.write(
        "Ideas/2026-11-19 Old.md", IDEA.replace("inbox", "later").replace("Solar dryer", "Old"), skip_guards=True
    )
    store.write("Pawn/Captures/Inbox/2026-11-20 A book-s1.md", CAPTURE, skip_guards=True)
    store.write(
        "Pawn/Captures/Inbox/2026-11-18 Filed-s2.md", CAPTURE.replace("status: inbox", "status: filed"), skip_guards=True
    )
    monkeypatch.setattr("pawn_core.vault_config.vault_store_from_config", lambda _c: store)
    monkeypatch.setattr(
        "pawn_agent.core.coworker.db.count_items", lambda *_a, **_k: 3
    )
    prompts: list[dict[str, Any]] = []

    async def fake_turn(**kwargs: Any) -> AgentRunResult:
        prompts.append(kwargs)
        return AgentRunResult(run_id="run-1", response="ok")

    monkeypatch.setattr(api_server, "run_agent_turn", fake_turn)
    return SimpleNamespace(cfg=cfg, store=store, prompts=prompts)


def test_helpers() -> None:
    assert idea_line("# T\n\nline one\nline two") == "line one\nline two"
    out = with_status(IDEA, "dropped")
    assert parse_frontmatter(out)[0]["status"] == "dropped"
    assert out.endswith("Build a solar fruit dryer.\n")
    assert "status: goal" in with_status("---\ntags: [idea]\n---\nbody", "goal")


def test_counts_and_lists(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        assert client.get("/v1/portal/inbox/counts").status_code == 401
        assert client.get("/v1/portal/inbox/counts", headers=AUTH).json() == {
            "items": 3,
            "ideas": 1,
            "captures": 1,
        }
        ideas = client.get("/v1/portal/inbox/ideas", headers=AUTH).json()["ideas"]
        assert [i["title"] for i in ideas] == ["Solar dryer"]
        assert ideas[0]["line"] == "Build a solar fruit dryer."
        every = client.get("/v1/portal/inbox/ideas?status=all", headers=AUTH).json()["ideas"]
        assert len(every) == 2
        caps = client.get("/v1/portal/inbox/captures", headers=AUTH).json()["captures"]
        assert [(c["title"], c["hint"], c["snippet_kind"]) for c in caps] == [("A book", "books", "text")]
        assert len(client.get("/v1/portal/inbox/captures?scope=all", headers=AUTH).json()["captures"]) == 2


def test_idea_actions(env: SimpleNamespace) -> None:
    path = "Ideas/2026-11-20 Solar dryer.md"
    with TestClient(api_server.create_app(env.cfg)) as client:
        bad = client.post("/v1/portal/inbox/ideas/action", headers=AUTH, json={"path": path, "action": "nope"})
        assert bad.status_code == 422
        outside = client.post(
            "/v1/portal/inbox/ideas/action", headers=AUTH, json={"path": "Pawn/x.md", "action": "keep"}
        )
        assert outside.status_code == 422

        goal = client.post(
            "/v1/portal/inbox/ideas/action",
            headers=AUTH,
            json={"path": path, "action": "goal", "title": "Dry fruit"},
        )
        assert goal.status_code == 200, goal.text
        assert goal.json()["written"] == "Goals.md"
        goals = env.store.read("Goals.md")
        assert "Dry fruit" in goals and "[[Ideas/2026-11-20 Solar dryer]]" in goals
        assert parse_frontmatter(env.store.read(path))[0]["status"] == "goal"

        task = client.post("/v1/portal/inbox/ideas/action", headers=AUTH, json={"path": path, "action": "task"})
        key = task.json()["written"]
        assert key == "Pawn/TaskNotes/Tasks/Solar dryer.md"
        assert "Source: [[Ideas/2026-11-20 Solar dryer]]" in env.store.read(key)
        again = client.post("/v1/portal/inbox/ideas/action", headers=AUTH, json={"path": path, "action": "task"})
        assert again.json()["written"] == "Pawn/TaskNotes/Tasks/Solar dryer 2.md"

        drop = client.post("/v1/portal/inbox/ideas/action", headers=AUTH, json={"path": path, "action": "drop"})
        assert drop.json()["status"] == "dropped"
        assert client.get("/v1/portal/inbox/ideas", headers=AUTH).json()["ideas"] == []

        missing = client.post(
            "/v1/portal/inbox/ideas/action", headers=AUTH, json={"path": "Ideas/none.md", "action": "keep"}
        )
        assert missing.status_code == 404


def test_quick_capture(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post("/v1/portal/inbox/ideas", headers=AUTH, json={"line": "Learn to juggle"})
        assert resp.status_code == 201, resp.text
        assert "Learn to juggle" in resp.json()["receipt"]
        titles = [i["title"] for i in client.get("/v1/portal/inbox/ideas", headers=AUTH).json()["ideas"]]
        assert "Learn to juggle" in titles
        assert client.post("/v1/portal/inbox/ideas", headers=AUTH, json={"line": " "}).status_code == 422


def test_item_side_chat_is_grounded(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    item = {
        "id": "it1",
        "kind": "commitment",
        "status": "new",
        "text": "Send the budget to Ada",
        "quote": "I'll send it Friday",
        "reason": "matches goal Budget",
        "source_kind": "session",
        "source_ref": "standup",
        "note_key": "Pawn/Items/it1.md",
    }
    monkeypatch.setattr("pawn_agent.core.coworker.db.get_item", lambda _dsn, _id: item)
    with TestClient(api_server.create_app(env.cfg)) as client:
        chat = client.post(
            "/v1/portal/chats",
            headers=AUTH,
            json={"origin": {"kind": "item", "ref": "it1"}},
        ).json()
        resp = client.post(f"/v1/portal/chats/{chat['id']}/messages", headers=AUTH, json={"message": "why?"})
        assert resp.status_code == 200, resp.text
    prompt = env.prompts[0]["prompt"]
    assert "Send the budget to Ada" in prompt
    assert "I'll send it Friday" in prompt
