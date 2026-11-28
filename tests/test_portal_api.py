"""Portal API (/v1/portal/*): chats, mirror streaming, folders, tags, files."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from pawn_agent.core.agent_runner import AgentRunResult
from pawn_agent.utils.config import ApiSection, PortalConfig
from pawn_agent.utils.db import AgentRun, VaultTask
from pawn_core.config import VaultConfig
from pawn_core.database import Base
from pawn_core.vault import VaultStore
from pawn_server.core import api_server
from pawn_server.core.portal_api import build_tag_tree, list_folder, rename_tag
from pawn_server.core.portal_models import PORTAL_TABLES
from pawn_server.core.vault_events import publish_vault_event
from tests.test_vault_store import FakeS3Client

AUTH = {"Authorization": "Bearer tok"}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'portal.db'}"
    engine = sa.create_engine(dsn)
    Base.metadata.create_all(
        engine, tables=[AgentRun.__table__, VaultTask.__table__, *PORTAL_TABLES]
    )
    engine.dispose()

    cfg = SimpleNamespace(
        api=ApiSection(token="tok", stream_keepalive_seconds=0.05, portal_dir=""),
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=dsn,
        vault=VaultConfig(agent_root="Pawn"),
        portal=PortalConfig(),
        vault_watcher=SimpleNamespace(matrix_target="matrix"),
    )
    store = VaultStore(bucket="b", agent_root="Pawn", client=FakeS3Client())
    store.write("Pawn/Analyses/s1.md", "# Analysis\n", skip_guards=True)
    store.write("People/Ada.md", "Ada bio", skip_guards=True)
    store.write(".obsidian/app.json", "{}", skip_guards=True)
    monkeypatch.setattr("pawn_core.vault_config.vault_store_from_config", lambda _c: store)
    # planner binds the name at import; patch it there too (order-independent).
    monkeypatch.setattr("pawn_server.core.planner.vault_store_from_config", lambda _c: store)

    prompts: list[dict[str, Any]] = []

    async def fake_turn(**kwargs: Any) -> AgentRunResult:
        prompts.append(kwargs)
        publish_vault_event(["Pawn/Analyses/new.md"], source="agent", run_id="run-1")
        return AgentRunResult(run_id="run-1", response="the answer")

    monkeypatch.setattr(api_server, "run_agent_turn", fake_turn)
    return SimpleNamespace(cfg=cfg, store=store, prompts=prompts)


def _sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event:
            out.append((event, data))
    return out


def _chat(client: TestClient, **body: Any) -> dict:
    resp = client.post("/v1/portal/chats", headers=AUTH, json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _say(client: TestClient, chat_id: str, message: str) -> list[tuple[str, dict]]:
    resp = client.post(
        f"/v1/portal/chats/{chat_id}/messages", headers=AUTH, json={"message": message}
    )
    assert resp.status_code == 200, resp.text
    return _sse(resp.text)


def test_requires_token(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        assert client.get("/v1/portal/chats").status_code == 401


def test_chat_stream_mirrors_messages_and_written_files(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        chat = _chat(client)
        assert chat["conversation_key"] == f"chat:{chat['id']}"
        events = _say(client, chat["id"], "Analyze the standup please")
        names = [e for e, _ in events]
        assert names == ["message", "answer", "files", "message", "done"]
        assert dict(events)["files"]["paths"] == ["Pawn/Analyses/new.md"]
        assert env.prompts[0]["session_id"] == chat["conversation_key"]
        assert env.prompts[0]["source"] == "portal"

        full = client.get(f"/v1/portal/chats/{chat['id']}", headers=AUTH).json()
        assert full["title"] == "Analyze the standup please"
        roles = [(m["role"], m["content"]) for m in full["messages"]]
        assert roles == [("user", "Analyze the standup please"), ("assistant", "the answer")]
        assert full["messages"][1]["context"]["written_files"] == ["Pawn/Analyses/new.md"]


def test_plain_pawn_chat_unchanged(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/pawn/chat", headers=AUTH, json={"conversation": "chat:x", "message": "hi"}
        )
        assert [e for e, _ in _sse(resp.text)] == ["answer", "done"]
        assert env.prompts[0]["source"] == "obsidian"


def test_side_chat_first_turn_includes_origin(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        parent = _chat(client, title="Main")
        side = _chat(
            client,
            parent_chat_id=parent["id"],
            origin={"kind": "file", "ref": "People/Ada.md", "quote": "Ada bio"},
        )
        _say(client, side["id"], "Who is this?")
        prompt = env.prompts[0]["prompt"]
        assert "Ada bio" in prompt and "[[People/Ada]]" in prompt
        full = client.get(f"/v1/portal/chats/{parent['id']}", headers=AUTH).json()
        assert [c["id"] for c in full["side_chats"]] == [side["id"]]


def test_side_chat_origin_ref_filter_and_promote(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        plain = _chat(client, title="Plain")
        side = _chat(client, origin={"kind": "file", "ref": "People/Ada.md", "quote": "Ada"})
        listed = client.get(
            "/v1/portal/chats", headers=AUTH, params={"origin_ref": "People/Ada.md"}
        ).json()["chats"]
        assert [c["id"] for c in listed] == [side["id"]]
        resp = client.patch(f"/v1/portal/chats/{side['id']}", headers=AUTH, json={"promoted": True})
        assert resp.status_code == 200, resp.text
        assert resp.json()["origin"] == {
            "kind": "file",
            "ref": "People/Ada.md",
            "quote": "Ada",
            "promoted": True,
        }
        clip = client.post(
            "/v1/portal/clips", headers=AUTH, json={"content_md": "quoted", "title": "Q"}
        ).json()
        got = client.get(f"/v1/portal/clips/{clip['id']}", headers=AUTH).json()
        assert got["content_md"] == "quoted"
        assert client.get("/v1/portal/clips/nope", headers=AUTH).status_code == 404
        refused = client.patch(
            f"/v1/portal/chats/{plain['id']}", headers=AUTH, json={"promoted": True}
        )
        assert refused.status_code == 422


def test_fork_copies_mirror_up_to_message(env: SimpleNamespace, monkeypatch) -> None:
    calls: list[dict] = []

    def fake_fork(cfg: Any, src: str, dst: str, *, turns: int, fallback: Any) -> int:
        calls.append({"src": src, "dst": dst, "turns": turns, "fallback": list(fallback)})
        return turns

    monkeypatch.setattr("pawn_agent.core.sallm_fork.fork_session", fake_fork)
    with TestClient(api_server.create_app(env.cfg)) as client:
        chat = _chat(client, title="Plan", tags=["work"])
        _say(client, chat["id"], "one")
        _say(client, chat["id"], "two")
        msgs = client.get(f"/v1/portal/chats/{chat['id']}", headers=AUTH).json()["messages"]
        resp = client.post(
            f"/v1/portal/chats/{chat['id']}/fork",
            headers=AUTH,
            json={"message_id": msgs[1]["id"]},
        )
        assert resp.status_code == 201, resp.text
        fork = resp.json()
        assert fork["message_count"] == 2 and fork["sallm_seeded"] == 2
        assert fork["tags"] == ["work"] and fork["parent_chat_id"] == chat["id"]
        assert calls[0]["src"] == chat["conversation_key"]
        assert calls[0]["fallback"] == [("user", "one"), ("assistant", "the answer")]


def test_folders_tree_move_and_delete(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        work = client.post("/v1/portal/folders", headers=AUTH, json={"name": "Work"}).json()
        acme = client.post(
            "/v1/portal/folders", headers=AUTH, json={"name": "Acme", "parent_id": work["id"]}
        ).json()
        # Cycles are refused.
        bad = client.patch(
            f"/v1/portal/folders/{work['id']}", headers=AUTH, json={"parent_id": acme["id"]}
        )
        assert bad.status_code == 422
        chat = _chat(client, title="a", folder_id=acme["id"])
        tree = client.get("/v1/portal/folders", headers=AUTH).json()
        assert tree["folders"][0]["children"][0]["chat_count"] == 1

        client.delete(f"/v1/portal/folders/{acme['id']}", headers=AUTH)
        got = client.get(f"/v1/portal/chats/{chat['id']}", headers=AUTH).json()
        assert got["folder_id"] == work["id"]

        moved = client.post(
            "/v1/portal/chats/move", headers=AUTH, json={"ids": [chat["id"]], "folder_id": None}
        ).json()
        assert moved["moved"] == 1
        unfiled = client.get("/v1/portal/chats?unfiled=true", headers=AUTH).json()["chats"]
        assert [c["id"] for c in unfiled] == [chat["id"]]


def test_sorting_and_tag_filter(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        _chat(client, title="beta", tags=["work/acme"])
        _chat(client, title="Alpha", tags=["home"])
        _chat(client, title="gamma")

        def titles(qs: str) -> list[str]:
            return [
                c["title"]
                for c in client.get(f"/v1/portal/chats?{qs}", headers=AUTH).json()["chats"]
            ]

        assert titles("sort=name&order=asc") == ["Alpha", "beta", "gamma"]
        assert titles("sort=created&order=desc") == ["gamma", "Alpha", "beta"]
        assert titles("sort=tag&order=asc") == ["Alpha", "beta", "gamma"]
        assert titles("tag=work") == ["beta"]
        assert client.get("/v1/portal/chats?sort=bogus", headers=AUTH).status_code == 422


def test_tag_tree_and_rename(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        chat = _chat(client, title="x", tags=["#work/acme/q4", "work"])
        tree = client.get("/v1/portal/tags", headers=AUTH).json()["chats"]
        assert tree[0]["path"] == "work" and tree[0]["count"] == 1
        assert tree[0]["children"][0]["children"][0]["path"] == "work/acme/q4"
        out = client.post(
            "/v1/portal/tags/rename", headers=AUTH, json={"from": "work/acme", "to": "clients/acme"}
        ).json()
        assert out["changed"] == 1
        got = client.get(f"/v1/portal/chats/{chat['id']}", headers=AUTH).json()
        assert got["tags"] == ["clients/acme/q4", "work"]


def test_search_clips_and_annotations(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        chat = _chat(client, title="Budget talk")
        _say(client, chat["id"], "what about the zebra budget")
        msgs = client.get(f"/v1/portal/chats/{chat['id']}", headers=AUTH).json()["messages"]
        clip = client.post(
            "/v1/portal/clips",
            headers=AUTH,
            json={"chat_id": chat["id"], "message_ids": [m["id"] for m in msgs], "tags": ["fin"]},
        ).json()
        assert "zebra budget" in clip["content_md"]
        ann = client.post(
            "/v1/portal/annotations",
            headers=AUTH,
            json={
                "target_kind": "message",
                "target_ref": msgs[1]["id"],
                "anchor": {"exact": "answer", "prefix": "the ", "suffix": ""},
                "body_md": "zebra check later",
            },
        ).json()
        listed = client.get(
            f"/v1/portal/annotations?target_kind=message&target_ref={msgs[1]['id']}", headers=AUTH
        ).json()["annotations"]
        assert [a["id"] for a in listed] == [ann["id"]]
        bad = client.post(
            "/v1/portal/annotations",
            headers=AUTH,
            json={"target_kind": "planet", "target_ref": "x"},
        )
        assert bad.status_code == 422

        hits = client.get("/v1/portal/search?q=zebra", headers=AUTH).json()
        assert hits["messages"][0]["chat_title"] == "Budget talk"
        assert len(hits["clips"]) == 1 and len(hits["annotations"]) == 1

        saved = client.post(f"/v1/portal/clips/{clip['id']}/save", headers=AUTH, json={}).json()
        assert saved["key"].startswith("Pawn/Scratch/Clips/")
        body = env.store.read(saved["key"])
        assert "zebra budget" in body and "pawn: editable" in body


def test_annotations_filters_unlink_and_search(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        chat = _chat(client, title="Quotes")
        _say(client, chat["id"], "tell me")
        msgs = client.get(f"/v1/portal/chats/{chat['id']}", headers=AUTH).json()["messages"]
        side = _chat(client, title="side")

        def make(ref: str, exact: str, **extra: Any) -> dict:
            resp = client.post(
                "/v1/portal/annotations",
                headers=AUTH,
                json={
                    "target_kind": "message",
                    "target_ref": ref,
                    "anchor": {"exact": exact, "prefix": "", "suffix": ""},
                    **extra,
                },
            )
            assert resp.status_code == 201, resp.text
            return resp.json()

        a = make(msgs[0]["id"], "tell", color="yellow", side_chat_id=side["id"])
        b = make(msgs[1]["id"], "quokka habitat")
        make("other-message", "nothing")

        both = client.get(
            "/v1/portal/annotations",
            params=[
                ("target_kind", "message"),
                ("target_ref", msgs[0]["id"]),
                ("target_ref", msgs[1]["id"]),
            ],
            headers=AUTH,
        ).json()["annotations"]
        assert {x["id"] for x in both} == {a["id"], b["id"]}

        # Omitted fields stay; an explicit null clears.
        kept = client.patch(
            f"/v1/portal/annotations/{a['id']}", headers=AUTH, json={"body_md": "note"}
        ).json()
        assert kept["side_chat_id"] == side["id"] and kept["color"] == "yellow"
        cleared = client.patch(
            f"/v1/portal/annotations/{a['id']}", headers=AUTH, json={"color": None}
        ).json()
        assert cleared["color"] is None and cleared["side_chat_id"] == side["id"]

        # Search matches the highlighted quote and says which chat it is in.
        hits = client.get("/v1/portal/search?q=quokka", headers=AUTH).json()["annotations"]
        assert [h["id"] for h in hits] == [b["id"]]
        assert hits[0]["context"]["chat_id"] == chat["id"]
        assert hits[0]["context"]["chat_title"] == "Quotes"

        # Deleting the side chat unlinks the annotation but keeps it.
        assert client.delete(f"/v1/portal/chats/{side['id']}", headers=AUTH).status_code == 200
        again = client.get(
            f"/v1/portal/annotations?target_ref={msgs[0]['id']}", headers=AUTH
        ).json()["annotations"]
        assert again[0]["side_chat_id"] is None and again[0]["body_md"] == "note"


def test_files_list_read_write_move_delete(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        root = client.get("/v1/portal/files", headers=AUTH).json()
        assert {f["name"] for f in root["folders"]} == {"Pawn", "People"}
        assert root["scratch_dir"] == "Pawn/Scratch"

        doc = client.get("/v1/portal/files/content?key=People/Ada.md", headers=AUTH).json()
        assert doc["content"] == "Ada bio" and doc["writable"] is False
        denied = client.put(
            "/v1/portal/files/content",
            headers=AUTH,
            json={"key": "People/Ada.md", "content": "x", "etag": doc["etag"]},
        )
        assert denied.status_code == 403
        meta = client.get("/v1/portal/files/content?key=.obsidian/app.json", headers=AUTH)
        assert meta.status_code == 404

        created = client.put(
            "/v1/portal/files/content",
            headers=AUTH,
            json={"key": "Pawn/Scratch/n.md", "content": "v1"},
        ).json()
        again = client.put(
            "/v1/portal/files/content",
            headers=AUTH,
            json={"key": "Pawn/Scratch/n.md", "content": "v1"},
        )
        assert again.status_code == 409
        client.put(
            "/v1/portal/files/content",
            headers=AUTH,
            json={"key": "Pawn/Scratch/n.md", "content": "v2", "etag": created["etag"]},
        )
        stale = client.put(
            "/v1/portal/files/content",
            headers=AUTH,
            json={"key": "Pawn/Scratch/n.md", "content": "v3", "etag": created["etag"]},
        )
        assert stale.status_code == 409

        moved = client.post(
            "/v1/portal/files/move",
            headers=AUTH,
            json={"src": "Pawn/Scratch/n.md", "dst": "Pawn/Scratch/sub/m.md"},
        )
        assert moved.status_code == 200, moved.text
        assert env.store.read("Pawn/Scratch/sub/m.md") == "v2"
        assert not env.store.exists("Pawn/Scratch/n.md")

        up = client.post(
            "/v1/portal/files/upload",
            headers=AUTH,
            files={"file": ("pic.png", b"\x89PNG", "image/png")},
        ).json()
        assert up["key"] == "Pawn/Scratch/pic.png"
        raw = client.get(f"/v1/portal/files/content?key={up['key']}", headers=AUTH)
        assert raw.content == b"\x89PNG" and raw.headers["content-type"] == "image/png"

        assert client.delete("/v1/portal/files?key=People/Ada.md", headers=AUTH).status_code == 403
        assert client.delete(f"/v1/portal/files?key={up['key']}", headers=AUTH).status_code == 200


def test_workspace_roundtrip(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        assert client.get("/v1/portal/workspace", headers=AUTH).json()["layout"] is None
        client.put("/v1/portal/workspace", headers=AUTH, json={"layout": {"grid": 1}})
        client.put("/v1/portal/workspace", headers=AUTH, json={"layout": {"grid": 2}})
        assert client.get("/v1/portal/workspace", headers=AUTH).json()["layout"] == {"grid": 2}


def test_delete_chat_resets_sallm(env: SimpleNamespace, monkeypatch) -> None:
    reset: list[str] = []

    async def fake_reset(key: str, dsn: str = "") -> None:
        reset.append(key)

    monkeypatch.setattr(api_server.get_sallm_registry(), "reset", fake_reset)
    with TestClient(api_server.create_app(env.cfg)) as client:
        chat = _chat(client)
        _say(client, chat["id"], "hi")
        assert client.delete(f"/v1/portal/chats/{chat['id']}", headers=AUTH).status_code == 200
        assert reset == [chat["conversation_key"]]
        assert client.get(f"/v1/portal/chats/{chat['id']}", headers=AUTH).status_code == 404


def test_helpers() -> None:
    tree = build_tag_tree([["a/b", "a/c"], ["a/b/d"]])
    assert tree[0]["count"] == 2
    assert [c["path"] for c in tree[0]["children"]] == ["a/b", "a/c"]
    assert rename_tag(["a/b", "x"], "a", "z") == ["z/b", "x"]
    assert rename_tag(["x"], "a", "z") is None
    listing = list_folder(["P/a.md", "P/s/b.md", "Q/c.md", ".obsidian/x"], "P")
    assert [f["name"] for f in listing["folders"]] == ["s"]
    assert [f["path"] for f in listing["files"]] == ["P/a.md"]


def test_files_list_has_ages_and_tags(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        root = client.get("/v1/portal/files", headers=AUTH).json()
        pawn = next(f for f in root["folders"] if f["name"] == "Pawn")
        assert pawn["modified"] and pawn["modified"].endswith("Z")
        people = client.get("/v1/portal/files?prefix=People", headers=AUTH).json()
        assert people["files"][0]["modified"] and people["files"][0]["tags"] == []


def test_file_tags_frontmatter_and_db(env: SimpleNamespace) -> None:
    env.store.write("Pawn/Scratch/n.md", "---\ntags: [old]\n---\n# Note\n", skip_guards=True)
    with TestClient(api_server.create_app(env.cfg)) as client:
        # Writable note: tags land in frontmatter.
        out = client.put(
            "/v1/portal/files/tags",
            headers=AUTH,
            json={"key": "Pawn/Scratch/n.md", "tags": ["#work/sec", "old"]},
        ).json()
        assert out["source"] == "frontmatter" and out["tags"] == ["work/sec", "old"]
        assert "work/sec" in env.store.read("Pawn/Scratch/n.md")

        # Read-only note: tags go to the DB; the file is untouched.
        out = client.put(
            "/v1/portal/files/tags", headers=AUTH, json={"key": "People/Ada.md", "tags": ["people"]}
        ).json()
        assert out["source"] == "db" and out["tags"] == ["people"]
        assert env.store.read("People/Ada.md") == "Ada bio"
        doc = client.get("/v1/portal/files/content?key=People/Ada.md", headers=AUTH).json()
        assert doc["tags"] == ["people"]

        # Binary: DB tags, moved and dropped with the file.
        up = client.post(
            "/v1/portal/files/upload",
            headers=AUTH,
            files={"file": ("doc.bin", b"\x00\x01", "application/octet-stream")},
        ).json()
        client.put("/v1/portal/files/tags", headers=AUTH, json={"key": up["key"], "tags": ["bin"]})
        client.post(
            "/v1/portal/files/move",
            headers=AUTH,
            json={"src": up["key"], "dst": "Pawn/Scratch/moved.bin"},
        )
        moved = client.get("/v1/portal/files/tags?key=Pawn/Scratch/moved.bin", headers=AUTH)
        assert moved.json()["tags"] == ["bin"]

        client.post("/v1/portal/files/reindex", headers=AUTH)
        tree = client.get("/v1/portal/tags", headers=AUTH).json()
        assert {n["path"] for n in tree["files"]} == {"bin", "old", "people", "work"}

        hits = client.get("/v1/portal/search?q=%23people", headers=AUTH).json()
        assert [f["key"] for f in hits["files"]] == ["People/Ada.md"]

        renamed = client.post(
            "/v1/portal/tags/rename", headers=AUTH, json={"from": "work", "to": "job"}
        ).json()
        assert renamed["changed"] >= 1
        assert "job/sec" in env.store.read("Pawn/Scratch/n.md")

        client.delete("/v1/portal/files?key=Pawn/Scratch/moved.bin", headers=AUTH)
        gone = client.get("/v1/portal/files/tags?key=Pawn/Scratch/moved.bin", headers=AUTH)
        assert gone.json()["tags"] == []


def test_search_files_full_text(env: SimpleNamespace) -> None:
    env.store.write("Pawn/Scratch/plan.md", "# Plan\nThe quokka migration\n", skip_guards=True)
    with TestClient(api_server.create_app(env.cfg)) as client:
        counts = client.post("/v1/portal/files/reindex", headers=AUTH).json()
        assert counts["indexed"] >= 3
        hits = client.get("/v1/portal/search?q=quokka", headers=AUTH).json()
        assert hits["files"][0]["key"] == "Pawn/Scratch/plan.md"
        assert hits["files"][0]["title"] == "Plan" and "quokka" in hits["files"][0]["snippet"]
        assert client.get("/v1/portal/search?q=Ada.md", headers=AUTH).json()["files"]


def test_search_includes_sessions(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_page(cfg: Any, **kwargs: Any) -> dict:  # noqa: ANN001
        assert kwargs["q"] == "budget"
        return {
            "data": [
                {
                    "session_id": "weekly",
                    "title": "Budget review",
                    "summary": "the budget holds",
                    "created_at": "2026-11-27T10:00:00",
                }
            ],
            "facets": {},
        }

    monkeypatch.setattr("pawn_server.core.sessions_api.list_sessions_page", fake_page)
    with TestClient(api_server.create_app(env.cfg)) as client:
        hits = client.get("/v1/portal/search?q=budget", headers=AUTH).json()
        assert hits["sessions"][0]["session_id"] == "weekly"
        assert hits["sessions"][0]["title"] == "Budget review"
        assert "budget" in hits["sessions"][0]["snippet"]
        tagged = client.get("/v1/portal/search?q=%23people", headers=AUTH).json()
        assert tagged["sessions"] == []


def test_auto_name_after_first_answer(env: SimpleNamespace, monkeypatch) -> None:
    calls: list[str] = []

    def fake_complete(cfg: Any, messages: list[dict], **_kw: Any) -> str:
        calls.append(messages[0]["content"])
        return '```json\n{"title": "Zebra budget review.", "tags": ["Finance/Budget"]}\n```'

    monkeypatch.setattr("pawn_server.core.background_llm.complete_text", fake_complete)
    with TestClient(api_server.create_app(env.cfg)) as client:
        chat = _chat(client)
        events = _say(client, chat["id"], "what about the zebra budget")
        assert [e for e, _ in events][-2:] == ["chat", "done"]
        assert dict(events)["chat"] == {
            "id": chat["id"],
            "title": "Zebra budget review",
            "tags": ["finance/budget"],
        }
        assert "zebra budget" in calls[0]

        # Not due again until auto_name_every messages later.
        _say(client, chat["id"], "more")
        assert len(calls) == 1

        # A manual rename sticks; tags may still be refined.
        client.patch(f"/v1/portal/chats/{chat['id']}", headers=AUTH, json={"title": "Mine"})
        for _ in range(3):
            _say(client, chat["id"], "again")
        got = client.get(f"/v1/portal/chats/{chat['id']}", headers=AUTH).json()
        assert len(calls) == 2
        assert got["title"] == "Mine" and got["title_auto"] is False and got["tags_auto"] is True


def test_split_query() -> None:
    from pawn_server.core.portal_api import split_query

    assert split_query("budget #work/acme  #x q4") == ("budget q4", ["work/acme", "x"])


def test_event_annotation_kind(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        resp = client.post(
            "/v1/portal/annotations",
            headers=AUTH,
            json={"target_kind": "event", "target_ref": "ev1", "body_md": "prep note"},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["target_kind"] == "event"


def test_side_chat_session_origin_without_transcript(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        side = _chat(
            client,
            origin={"kind": "session", "ref": "sess-1", "t": 75, "quote": "the budget"},
        )
        _say(client, side["id"], "Who said this?")
        prompt = env.prompts[0]["prompt"]
        assert "the budget" in prompt
        assert "Session id: sess-1" in prompt
        assert "session_transcript --session-id sess-1" in prompt
        assert "1:15" in prompt


def test_side_chat_session_origin_inlines_transcript(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_get(_cfg: Any, sid: str) -> dict[str, Any]:
        return {
            "session_id": sid,
            "title": "Product weekly",
            "created_at": "2026-11-27T10:02:00",
            "speakers": ["Ada", "SPEAKER_00"],
            "transcript_path": "Pawn/Transcripts/sess-1.md",
        }

    monkeypatch.setattr("pawn_server.core.sessions_api.get_session", fake_get)
    with TestClient(api_server.create_app(env.cfg)) as client:
        side = _chat(client, origin={"kind": "session", "ref": "sess-1", "t": 75})
        _say(client, side["id"], "Summarize")
        prompt = env.prompts[0]["prompt"]
        assert "Product weekly" in prompt
        assert "Ada, SPEAKER_00" in prompt
        assert "1:15" in prompt
        assert "session_transcript" not in prompt
        assert "[[Pawn/Transcripts/sess-1]]" in prompt


def test_side_chat_event_origin_inlines_note(env: SimpleNamespace) -> None:
    env.store.write(
        "Pawn/Events/2026-11-27-standup-ab12.md",
        "---\npawn: event\nid: ab12\ntitle: Standup\nstart: '2026-11-27T09:00:00'\n---\n\n"
        "# Standup\n\n## Prep\n\nRead the brief.\n",
        skip_guards=True,
    )
    with TestClient(api_server.create_app(env.cfg)) as client:
        side = _chat(client, origin={"kind": "event", "ref": "ab12"})
        _say(client, side["id"], "What should I prepare?")
        prompt = env.prompts[0]["prompt"]
        assert "Event: Standup" in prompt
        assert "[[Pawn/Events/2026-11-27-standup-ab12]]" in prompt
        assert "Read the brief." in prompt


def test_side_chat_day_origin_names_the_skill(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        side = _chat(client, origin={"kind": "day", "ref": "2026-11-27"})
        _say(client, side["id"], "Plan it")
        prompt = env.prompts[0]["prompt"]
        assert "Day 2026-11-27" in prompt
        assert "Use the day_plan skill." in prompt


def test_side_chat_speaker_origin_inlines_people_note(env: SimpleNamespace) -> None:
    with TestClient(api_server.create_app(env.cfg)) as client:
        side = _chat(client, origin={"kind": "speaker", "ref": "Ada"})
        _say(client, side["id"], "Who is this?")
        prompt = env.prompts[0]["prompt"]
        assert "Ada" in prompt
        assert "[[People/Ada]]" in prompt
