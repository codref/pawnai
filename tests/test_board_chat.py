"""Board chat: snapshot on every turn, BoardProfile, ghosts collected by run id."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from pawn_agent.core.agent_runner import AgentRunResult
from pawn_agent.core.board_profile import BoardProfile
from pawn_server.core import api_server, boards
from pawn_server.core.chat_context import build_chat_prompt
from tests.test_portal_api import AUTH, _sse, env  # noqa: F401  (fixture)


def _board_world(
    env: SimpleNamespace, monkeypatch: Any  # noqa: F811
) -> tuple[TestClient, dict, dict]:  # noqa: F811
    client = TestClient(api_server.create_app(env.cfg))
    resp = client.post("/v1/portal/boards", headers=AUTH, json={"title": "Vendor risk"})
    assert resp.status_code == 201, resp.text
    board = resp.json()
    chat = client.post(f"/v1/portal/boards/{board['name']}/chat", headers=AUTH).json()
    return client, board, chat


def test_board_chat_turn_carries_snapshot_and_returns_ghosts(
    env: SimpleNamespace, monkeypatch: Any  # noqa: F811
) -> None:
    client, board, chat = _board_world(env, monkeypatch)
    turns: list[dict[str, Any]] = []

    async def fake_turn(**kwargs: Any) -> AgentRunResult:
        turns.append(kwargs)
        if len(turns) == 1:
            boards.propose(
                env.cfg.db_dsn, board["name"], "card", {"title": "Renewal at risk"}, run_id="run-b1"
            )
        return AgentRunResult(run_id=f"run-b{len(turns)}", response="Proposed a note.")

    monkeypatch.setattr(api_server, "run_agent_turn", fake_turn)
    with client:
        resp = client.post(
            f"/v1/portal/chats/{chat['id']}/messages",
            headers=AUTH,
            json={"message": "what connects these?", "board_selection": ["item:x"]},
        )
        events = _sse(resp.text)
        kinds = [e for e, _ in events]
        assert "board" in kinds and kinds.index("board") > kinds.index("answer")
        payload = dict(events)["board"]
        assert payload["board"] == board["name"] and payload["proposals"][0]["kind"] == "card"
        first = turns[0]
        assert isinstance(first["voice"], BoardProfile) and first["voice"].board == board["name"]
        assert "## Board" in first["prompt"] and "Vendor risk" in first["prompt"]
        assert first["prompt"].startswith("what connects these?")
        # Every turn, not only the first, sees the board.
        resp = client.post(
            f"/v1/portal/chats/{chat['id']}/messages", headers=AUTH, json={"message": "and now?"}
        )
        assert "board" not in [e for e, _ in _sse(resp.text)]
        assert "## Board" in turns[1]["prompt"]
        full = client.get(f"/v1/portal/chats/{chat['id']}", headers=AUTH).json()
        user0, answer0 = full["messages"][0], full["messages"][1]
        assert user0["context"]["board_selection"] == ["item:x"]
        assert answer0["context"]["board_proposals"] == [payload["proposals"][0]["id"]]
        # The ghost is on the board until decided.
        graph = client.get(f"/v1/portal/brainmaps/{board['name']}", headers=AUTH).json()
        ghost = next(n for n in graph["nodes"] if n["proposed"])
        assert ghost["kind"] == "note" and graph["meta"]["chat_id"] == chat["id"]
        decided = client.post(
            f"/v1/portal/boards/{board['name']}/proposals/{ghost['meta']['proposal_id']}",
            headers=AUTH,
            json={"verdict": "accept"},
        ).json()
        assert decided["proposal"]["status"] == "accepted"
        graph = client.get(f"/v1/portal/brainmaps/{board['name']}", headers=AUTH).json()
        assert [n["title"] for n in graph["nodes"] if n["kind"] == "note"] == ["Renewal at risk"]


def test_ordinary_chat_has_no_board(env: SimpleNamespace) -> None:  # noqa: F811
    with TestClient(api_server.create_app(env.cfg)) as client:
        chat = client.post("/v1/portal/chats", headers=AUTH, json={}).json()
        client.post(f"/v1/portal/chats/{chat['id']}/messages", headers=AUTH, json={"message": "hi"})
        assert env.prompts[0]["voice"] is None
        assert "## Board" not in env.prompts[0]["prompt"]


def test_boards_crud_over_http(env: SimpleNamespace) -> None:  # noqa: F811
    with TestClient(api_server.create_app(env.cfg)) as client:
        made = client.post(
            "/v1/portal/boards",
            headers=AUTH,
            json={"title": "A", "kind": "lens", "spec": {"q": "x"}},
        ).json()
        name = made["name"]
        listed = client.get("/v1/portal/boards", headers=AUTH).json()["boards"]
        assert [b["name"] for b in listed] == ["items", name]
        got = client.patch(
            f"/v1/portal/boards/{name}", headers=AUTH, json={"title": "B", "pinned": True}
        ).json()
        assert got["title"] == "B" and got["pinned"] is True
        assert (
            client.patch(f"/v1/portal/boards/{name}", headers=AUTH, json={"bad": 1}).status_code
            == 422
        )
        assert client.get("/v1/portal/brainmaps/nope/layout", headers=AUTH).status_code == 404
        assert client.get(f"/v1/portal/brainmaps/{name}/layout", headers=AUTH).status_code == 200
        assert client.post(f"/v1/portal/boards/{name}/seen", headers=AUTH).status_code == 200
        assert client.delete(f"/v1/portal/boards/{name}", headers=AUTH).json() == {"deleted": name}
        assert client.get(f"/v1/portal/boards/{name}", headers=AUTH).status_code == 404
        assert client.post("/v1/portal/boards", headers=AUTH, json={"title": ""}).status_code == 400


def test_extra_sections_come_first_and_are_capped() -> None:
    prompt = build_chat_prompt("q?", selection="sel", extra_sections=[("Board", "x" * 9000)])
    assert prompt.startswith("q?\n\n---\nContext from the board")
    assert prompt.index("## Board") < prompt.index("Selected text")
    assert "x" * 8001 not in prompt
    assert build_chat_prompt("q?", extra_sections=[("Board", "  ")]) == "q?"


def test_board_profile_fingerprint_and_prompt() -> None:
    p = BoardProfile("b-1")
    assert p.fingerprint() == ("board", "b-1")
    assert "board_propose" in p.system_extra() and "b-1" in p.system_extra()
    assert {"board_read", "board_propose"} <= set(p.tools())
    assert "board" in p.skills().names()
