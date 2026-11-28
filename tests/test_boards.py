"""User boards: CRUD, lens/manual/question builders, notes, proposals, snapshot.

Portal tables run on SQLite; emergence helpers are monkeypatched like
``test_brainmap`` (em_* tables need pgvector).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa

from pawn_core.database import Base
from pawn_server.core import boards
from pawn_server.core import brainmap as bm
from pawn_server.core.portal_models import PORTAL_TABLES
from tests.test_brainmap import world  # noqa: F401  (fixture)


@pytest.fixture
def cfg(tmp_path: Path) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'boards.db'}"
    Base.metadata.create_all(sa.create_engine(dsn), tables=PORTAL_TABLES)
    return SimpleNamespace(
        db_dsn=dsn,
        emergence=SimpleNamespace(enabled=True, route_ambiguous_distance=0.45),
    )


def _ids(graph: dict[str, Any]) -> set[str]:
    return {n["id"] for n in graph["nodes"]}


def test_list_always_has_items_board(cfg: SimpleNamespace) -> None:
    listed = boards.list_boards(cfg.db_dsn)
    assert [b["name"] for b in listed] == ["items"]
    assert listed[0]["kind"] == "items"
    with pytest.raises(bm.BrainmapError):
        boards.delete_board(cfg.db_dsn, "items")
    with pytest.raises(bm.BrainmapError):
        boards.create_board(cfg, title="  ", kind="manual")
    with pytest.raises(bm.BrainmapError):
        boards.create_board(cfg, title="x", kind="nope")


def test_manual_board_pins_context_hidden_and_notes(
    cfg: SimpleNamespace, world: SimpleNamespace  # noqa: F811
) -> None:
    b = boards.create_board(cfg, title="Acme 1:1", nodes=["item:it1"], tags=["Work"])
    assert b["name"].startswith("b-") and b["tags"] == ["Work"]
    g = boards.build(cfg, b["name"])
    ids = _ids(g)
    # The pinned item, its owner, its tape, and (one hop) the thread it feeds.
    assert {"item:it1", "person:mira", "session:weekly", "subject:t1"} <= ids
    assert "item:it2" not in ids
    assert g["meta"]["kind"] == "manual" and g["proposals"] == []
    boards.remove_nodes(cfg.db_dsn, b["name"], ["session:weekly"])
    assert "session:weekly" not in _ids(boards.build(cfg, b["name"]))
    boards.add_pins(cfg.db_dsn, b["name"], ["session:weekly"])  # pinning un-hides
    assert "session:weekly" in _ids(boards.build(cfg, b["name"]))

    note = boards.add_note(cfg.db_dsn, b["name"], title="Mira is overloaded", sources=["item:it1"])
    tied = boards.tie_note(cfg.db_dsn, b["name"], f"note:{note['id']}", "subject:t1")
    assert tied["edge"]["id"] == f"notesrc:{note['id']}:subject:t1"
    with pytest.raises(bm.BrainmapError):
        boards.tie_note(cfg.db_dsn, b["name"], f"note:{note['id']}", "subject:t1")
    g = boards.build(cfg, b["name"])
    edges = {e["id"] for e in g["edges"]}
    assert f"note:{note['id']}" in _ids(g)
    assert {f"notesrc:{note['id']}:item:it1", f"notesrc:{note['id']}:subject:t1"} <= edges
    boards.cut_note_source(cfg.db_dsn, b["name"], f"notesrc:{note['id']}:subject:t1")
    boards.remove_nodes(cfg.db_dsn, b["name"], [f"note:{note['id']}"])
    assert f"note:{note['id']}" not in _ids(boards.build(cfg, b["name"]))


def test_items_board_refuses_membership(cfg: SimpleNamespace) -> None:
    with pytest.raises(bm.BrainmapError):
        boards.add_pins(cfg.db_dsn, "items", ["item:it1"])
    with pytest.raises(bm.BrainmapError):
        boards.add_note(cfg.db_dsn, "items", title="x")


def test_lens_board_from_query_and_subjects(
    cfg: SimpleNamespace, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    monkeypatch.setattr(boards.labels, "embed_many", lambda c, texts: [[1.0, 0.0]] * len(texts))
    monkeypatch.setattr(boards.subj, "recent_observations", lambda dsn, **kw: [])
    monkeypatch.setattr(boards, "_evidence_obs", lambda dsn, ids, per_subject=25: ["o1"])
    b = boards.create_board(cfg, title="Acme", kind="lens", spec={"q": "acme launch", "junk": 1})
    assert b["spec"] == {"q": "acme launch"}
    ids = _ids(boards.build(cfg, b["name"]))
    # t1 is near the query; its child question and its evidence (item 1) come along.
    assert {"subject:t1", "subject:q1", "item:it1"} <= ids
    assert "item:it2" not in ids
    by_person = boards.create_board(cfg, title="Mira", kind="lens", spec={"people": ["mira"]})
    assert "item:it1" in _ids(boards.build(cfg, by_person["name"]))
    patched = boards.patch_board(cfg.db_dsn, by_person["name"], spec={"people": ["nobody"]})
    assert patched["spec"] == {"people": ["nobody"]}
    assert "item:it1" not in _ids(boards.build(cfg, by_person["name"]))


def test_proposals_show_as_ghosts_and_accept_through_gestures(
    cfg: SimpleNamespace, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    b = boards.create_board(cfg, title="Acme", nodes=["item:it1", "item:it2", "subject:q1"])
    name = b["name"]
    s = boards.propose(
        cfg.db_dsn,
        name,
        "string",
        {"from": "item:it1", "to": "subject:q1", "role": "updates"},
        run_id="run-1",
    )
    again = boards.propose(cfg.db_dsn, name, "string", {"from": "subject:q1", "to": "item:it1"})
    assert again["id"] == s["id"]  # the same ghost once
    card = boards.propose(cfg.db_dsn, name, "card", {"title": "Risk", "sources": ["item:it2"]})
    pin = boards.propose(cfg.db_dsn, name, "pin", {"node": "subject:g1", "reason": "goal"})
    group = boards.propose(
        cfg.db_dsn, name, "group", {"title": "Launch", "nodes": ["item:it1", "item:it2"]}
    )
    g = boards.build(cfg, name)
    ghosts = {x["id"]: x for x in g["suggestions"]}
    assert ghosts[f"prop:{s['id']}"]["role"] == "updates"
    nodes = {n["id"]: n for n in g["nodes"]}
    assert nodes[f"prop:{card['id']}"]["proposed"] and nodes[f"prop:{card['id']}"]["kind"] == "note"
    assert nodes["subject:g1"]["meta"]["proposal_id"] == pin["id"]
    assert nodes[f"prop:{group['id']}"]["kind"] == "thread"
    assert [p["id"] for p in boards.proposals_for_run(cfg.db_dsn, "run-1")] == [s["id"]]

    tied: list[tuple] = []
    monkeypatch.setattr(
        boards.bm,
        "tie",
        lambda c, f, t, r, surface: tied.append((f, t, r, surface)) or {"receipt": "ok"},
    )
    boards.decide_proposal(cfg, name, s["id"], "confirm")
    assert tied == [("item:it1", "subject:q1", "updates", "board_chat")]
    with pytest.raises(bm.BrainmapError):
        boards.decide_proposal(cfg, name, s["id"], "accept")  # already decided
    boards.decide_proposal(cfg, name, card["id"], "accept")
    boards.decide_proposal(cfg, name, pin["id"], "accept")
    lassoed: list[dict] = []
    monkeypatch.setattr(
        boards.bm,
        "lasso",
        lambda c, ids, **kw: lassoed.append({"ids": ids, **kw})
        or {"subject": {"id": "subject:new", "title": "Launch"}},
    )
    boards.decide_proposal(cfg, name, group["id"], "accept")
    assert lassoed[0]["surface"] == "board_chat" and lassoed[0]["ids"] == ["item:it1", "item:it2"]
    after = boards.get_board(cfg.db_dsn, name)
    assert "subject:g1" in after["pins"] and "subject:new" in after["pins"]
    assert after["notes"][0]["id"] == card["id"] and after["notes"][0]["sources"] == ["item:it2"]
    assert boards.list_proposals(cfg.db_dsn, name) == []


def test_burning_a_string_writes_a_reject_label(
    cfg: SimpleNamespace, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    b = boards.create_board(cfg, title="Acme", nodes=["item:it1", "subject:q1"])
    p = boards.propose(cfg.db_dsn, b["name"], "string", {"from": "item:it1", "to": "subject:q1"})
    recorded: list[dict] = []
    monkeypatch.setattr(boards.labels, "record_link_label", lambda c, **kw: recorded.append(kw))
    res = boards.decide_proposal(cfg, b["name"], p["id"], "snip")
    assert res["proposal"]["status"] == "rejected"
    assert recorded[0]["verdict"] == "reject" and recorded[0]["surface"] == "board_chat"
    assert recorded[0]["observation"]["id"] == "o1"


def test_new_since_last_look(
    cfg: SimpleNamespace, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    b = boards.create_board(cfg, title="Acme", nodes=["item:it1"])
    seen = datetime.fromisoformat(b["seen_at"])
    for lk in world.links:
        lk["created_at"] = (
            seen + timedelta(hours=3) if lk["id"] == "L1" else seen - timedelta(days=1)
        )
    g = boards.build(cfg, b["name"])
    edge = next(e for e in g["edges"] if e["id"] == "link:L1")
    assert edge.get("new") is True
    assert next(n for n in g["nodes"] if n["id"] == "item:it1")["meta"].get("new") is True
    boards.mark_seen(cfg.db_dsn, b["name"])
    for lk in world.links:
        lk["created_at"] = seen
    g = boards.build(cfg, b["name"])
    assert not any(e.get("new") for e in g["edges"])


def test_snapshot_puts_selection_first_and_caps(
    cfg: SimpleNamespace, world: SimpleNamespace  # noqa: F811
) -> None:
    b = boards.create_board(cfg, title="Acme", nodes=["item:it1", "item:it2"])
    text = boards.snapshot(cfg, b["name"], ["item:it2", "nope"])
    assert text.startswith(f"Board “Acme” (id {b['name']}, kind manual)")
    sel = text.index("SELECTED")
    assert text.index("[item:it2]") > sel and text.index("[item:it2]") < text.index("\nCards:")
    assert "[item:it1]" in text and "nope" not in text
    assert len(boards.snapshot(cfg, b["name"], [], limit_chars=300)) <= 300
    detail = boards.card_detail(cfg, b["name"], "item:it1")
    assert detail.startswith("[item:it1] ITEM") and "Strings:" in detail
    with pytest.raises(bm.BrainmapError):
        boards.card_detail(cfg, b["name"], "item:zzz")


def test_board_chat_is_created_once_and_night_note_lands(cfg: SimpleNamespace) -> None:
    b = boards.create_board(cfg, title="Acme")
    chat = boards.ensure_chat(cfg.db_dsn, b["name"])
    assert chat["origin"] == {"kind": "board", "ref": b["name"], "quote": "Acme"}
    assert boards.ensure_chat(cfg.db_dsn, b["name"])["id"] == chat["id"]
    assert boards.get_board(cfg.db_dsn, b["name"])["chat_id"] == chat["id"]
    assert boards.board_for_origin(chat["origin"]) == b["name"]
    assert boards.board_for_origin({"kind": "file", "ref": "x", "board": "b-1"}) == "b-1"
    assert boards.board_for_origin({"kind": "board", "ref": "b-1", "promoted": True}) is None
    assert boards.board_for_origin(None) is None
    mid = boards.post_night_note(
        cfg.db_dsn, b["name"], "Two notes came in.", observation_ids=["o1"]
    )
    from pawn_core.database import _get_session  # noqa: PLC0415
    from pawn_server.core.portal_models import PortalMessage  # noqa: PLC0415

    with _get_session(cfg.db_dsn) as db:
        msg = db.get(PortalMessage, mid)
        assert msg.role == "assistant" and msg.content.startswith("**While you slept.**")
        assert msg.context["night"] is True and msg.context["observation_ids"] == ["o1"]
    boards.delete_board(cfg.db_dsn, b["name"])
    assert not boards.exists(cfg.db_dsn, b["name"])


def test_board_from_existing_chat(cfg: SimpleNamespace) -> None:
    from pawn_core.database import _get_session  # noqa: PLC0415
    from pawn_server.core.portal_api import new_chat  # noqa: PLC0415

    with _get_session(cfg.db_dsn) as db:
        chat_id = new_chat(db, title="Vendor talk").id
    b = boards.create_board(cfg, title="Vendor", chat_id=chat_id)
    assert b["chat_id"] == chat_id
    assert boards.ensure_chat(cfg.db_dsn, b["name"])["origin"] == {"board": b["name"]}
    with pytest.raises(bm.BrainmapError):
        boards.create_board(cfg, title="x", chat_id="missing")


def test_question_board_creates_subject(
    cfg: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[dict] = []
    updated: list[tuple] = []

    def create_subject(dsn: str, **f: Any) -> dict:
        created.append(f)
        return {"id": "Q1", **f, "fields": dict(f.get("fields") or {})}

    monkeypatch.setattr(boards.labels, "embed_many", lambda c, texts: [[0.0, 1.0]])
    monkeypatch.setattr(boards.subj, "create_subject", create_subject)
    monkeypatch.setattr(
        boards.subj, "get_subject", lambda dsn, sid: {"id": sid, "kind": "question", "fields": {}}
    )
    monkeypatch.setattr(
        boards.subj, "update_subject", lambda dsn, sid, **f: updated.append((sid, f))
    )
    b = boards.create_board(cfg, title="", kind="question", question="Why does team B slip?")
    assert (
        b["kind"] == "question"
        and b["subject_id"] == "Q1"
        and b["title"] == "Why does team B slip?"
    )
    assert created[0]["kind"] == "question" and created[0]["status"] == "active"
    assert created[0]["centroid"] == [0.0, 1.0]
    assert updated == [("Q1", {"fields": {"board": b["name"]}})]
    off = SimpleNamespace(db_dsn=cfg.db_dsn, emergence=SimpleNamespace(enabled=False))
    with pytest.raises(bm.BrainmapError) as err:
        boards.create_board(off, title="q", kind="question")
    assert err.value.status == 409


def test_board_propose_tool_validates_ids_and_tags_run(
    cfg: SimpleNamespace, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    from pawn_agent.tools.board_impl import board_propose_impl, board_read_impl  # noqa: PLC0415

    b = boards.create_board(cfg, title="Acme", nodes=["item:it1", "subject:q1"])
    monkeypatch.setenv("PAWN_PARENT_RUN_ID", "run-9")
    out = board_propose_impl(
        cfg, board=b["name"], kind="string", frm="item:it1", to="subject:q1", reason="same week"
    )
    assert "Proposed string" in out and "not done yet" in out
    assert boards.proposals_for_run(cfg.db_dsn, "run-9")[0]["payload"]["reason"] == "same week"
    with pytest.raises(ValueError, match="not on board"):
        board_propose_impl(cfg, board=b["name"], kind="string", frm="item:it1", to="item:zzz")
    with pytest.raises(ValueError, match="role"):
        board_propose_impl(
            cfg, board=b["name"], kind="string", frm="item:it1", to="subject:q1", role="loves"
        )
    with pytest.raises(ValueError):
        board_propose_impl(cfg, board=b["name"], kind="group", title="x", nodes=["item:it1"])
    boards.add_pins(cfg.db_dsn, b["name"], ["item:it2"])
    with pytest.raises(ValueError, match="group"):
        board_propose_impl(cfg, board=b["name"], kind="string", frm="item:it1", to="item:it2")
    boards.remove_nodes(cfg.db_dsn, b["name"], ["item:it2"])
    pinned = board_propose_impl(cfg, board=b["name"], kind="pin", nodes=["item:it2", "item:it1"])
    assert "proposed pin item:it2" in pinned and "item:it1 is already on the board" in pinned
    assert "SELECTED" in board_read_impl(cfg, board=b["name"], selection=["item:it1"])
