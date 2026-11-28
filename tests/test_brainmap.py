"""Brainmap board: graph building, string verdicts, cut-aware routing, layout.

DB helpers are monkeypatched (em_* tables need pgvector); the layout table
runs on SQLite like the other portal tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa

from pawn_agent.core.emergence import labels
from pawn_agent.core.emergence import route as em_route
from pawn_core.database import Base
from pawn_server.core import brainmap as bm
from pawn_server.core.portal_models import PORTAL_TABLES


def _cfg(enabled: bool = True, dsn: str = "postgresql://unused") -> SimpleNamespace:
    return SimpleNamespace(
        db_dsn=dsn, emergence=SimpleNamespace(enabled=enabled, route_ambiguous_distance=0.45)
    )


def _item(i: int, **extra: Any) -> dict[str, Any]:
    base = {
        "id": f"it{i}",
        "kind": "commitment",
        "text": f"Item {i}",
        "status": "new",
        "owner": None,
        "due": None,
        "quote": None,
        "context": None,
        "thread": None,
        "source_kind": "session",
        "source_ref": "weekly",
        "note_key": None,
        "payload": {"observation_id": f"o{i}", "start_sec": 10.0 * i},
    }
    base.update(extra)
    return base


def _subject(sid: str, kind: str = "thread", **extra: Any) -> dict[str, Any]:
    base = {
        "id": sid,
        "kind": kind,
        "title": f"Subject {sid}",
        "slug": f"subject-{sid}",
        "origin": "emergent",
        "status": "active",
        "state_md": "",
        "fields": {},
        "momentum": 0.1,
        "source_ref": None,
        "parent_id": None,
        "note_key": None,
        "centroid": None,
    }
    base.update(extra)
    return base


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Two items, a thread with a child question, one dismissed hunch."""
    items = [
        _item(1, owner="Mira", thread="Acme launch"),
        _item(2, payload={"observation_id": "o2", "merged_observation_ids": ["o2b"]}),
        _item(3, payload={}),  # legacy: no observation
    ]
    subjects = {
        "t1": _subject("t1", centroid=[1.0, 0.0]),
        "q1": _subject("q1", "question", parent_id="t1", centroid=[0.0, 1.0]),
        "h1": _subject("h1", "hypothesis", status="dismissed"),
        "g1": _subject("g1", origin="goal", slug="acme-launch", title="Acme launch"),
    }
    links = [
        {
            "id": "L1",
            "observation_id": "o1",
            "subject_id": "t1",
            "role": "mentions",
            "linked_by": "embedding",
            "score": 0.2,
        },
        {
            "id": "L2",
            "observation_id": "o2b",
            "subject_id": "q1",
            "role": "evidence_for",
            "linked_by": "user",
            "score": None,
        },
        {
            "id": "L3",
            "observation_id": "o1",
            "subject_id": "h1",
            "role": "evidence_for",
            "linked_by": "dream",
            "score": None,
        },
    ]
    observations = {
        "o1": {
            "id": "o1",
            "type": "commitment",
            "text": "Item 1",
            "status": "live",
            "embedding": [1.0, 0.0],
            "source_ref": "weekly",
        },
        "o2": {
            "id": "o2",
            "type": "decision",
            "text": "Item 2",
            "status": "live",
            "embedding": [0.1, 1.0],
            "source_ref": "weekly",
        },
    }
    rejected: set[str] = set()
    monkeypatch.setattr(bm.itemdb, "list_items", lambda dsn, **kw: list(items))
    monkeypatch.setattr(bm.itemdb, "list_open_items", lambda dsn: list(items))
    monkeypatch.setattr(
        bm.itemdb, "get_item", lambda dsn, iid: next((i for i in items if i["id"] == iid), None)
    )
    monkeypatch.setattr(
        bm, "_links_for", lambda dsn, ids: [lk for lk in links if lk["observation_id"] in set(ids)]
    )
    monkeypatch.setattr(
        bm,
        "_subjects_by_id",
        lambda dsn, ids: {k: v for k, v in subjects.items() if k in set(ids) or k == "t1"},
    )
    monkeypatch.setattr(bm, "_evidence_counts", lambda dsn, ids: {k: 3 for k in ids})
    monkeypatch.setattr(
        bm,
        "_observations",
        lambda dsn, ids: {k: v for k, v in observations.items() if k in set(ids)},
    )

    def list_subjects(dsn: str, *, kinds: Any = None, statuses: Any = None) -> list[dict[str, Any]]:
        return [
            s
            for s in subjects.values()
            if (not kinds or s["kind"] in kinds) and (not statuses or s["status"] in statuses)
        ]

    monkeypatch.setattr(bm.subj, "list_subjects", list_subjects)
    monkeypatch.setattr(bm.subj, "get_subject", lambda dsn, sid: subjects.get(sid))
    monkeypatch.setattr(bm.labels, "rejected_link_keys", lambda dsn: rejected)
    return SimpleNamespace(
        items=items, subjects=subjects, links=links, observations=observations, rejected=rejected
    )


def _ids(graph: dict[str, Any]) -> set[str]:
    return {n["id"] for n in graph["nodes"]}


def test_items_board_shape(world: SimpleNamespace) -> None:
    g = bm.build_items_board(_cfg())
    ids = _ids(g)
    assert {"item:it1", "item:it2", "item:it3", "subject:t1", "subject:q1", "subject:g1"} <= ids
    assert "subject:h1" not in ids  # dismissed hunches drop off the board
    assert {"person:mira", "session:weekly"} <= ids
    edges = {e["id"]: e for e in g["edges"]}
    assert edges["link:L1"] == {
        **edges["link:L1"],
        "from": "item:it1",
        "to": "subject:t1",
        "by": "pawn",
        "cuttable": True,
    }
    # Merged observation ids still hang the string from the item.
    assert edges["link:L2"]["from"] == "item:it2" and edges["link:L2"]["by"] == "user"
    assert edges["parent:q1"]["to"] == "subject:t1"
    assert edges["goal:it1"] == {
        **edges["goal:it1"],
        "to": "subject:g1",
        "by": "goal",
        "cuttable": False,
    }
    assert edges["derived:item:it1->person:mira"]["cuttable"] is False
    item1 = next(n for n in g["nodes"] if n["id"] == "item:it1")
    assert item1["meta"]["observation_id"] == "o1" and "todo" in item1["actions"]
    assert g["emergence_enabled"] is True


def test_suggestions_skip_linked_and_rejected(world: SimpleNamespace) -> None:
    g = bm.build_items_board(_cfg())
    sugg = {(s["from"], s["to"]) for s in g["suggestions"]}
    # o1 is already tied to t1; o2 is close to q1 but already linked via o2b? no: o2 → q1 is new.
    assert ("item:it1", "subject:t1") not in sugg
    assert ("item:it2", "subject:q1") in sugg
    world.rejected.add(labels.link_key("q1", "Item 2"))
    g = bm.build_items_board(_cfg())
    assert ("item:it2", "subject:q1") not in {(s["from"], s["to"]) for s in g["suggestions"]}


def test_board_without_emergence_is_read_only(world: SimpleNamespace) -> None:
    g = bm.build_items_board(_cfg(enabled=False))
    ids = _ids(g)
    assert "thread:acme-launch" in ids and not any(i.startswith("subject:") for i in ids)
    assert g["suggestions"] == [] and g["emergence_enabled"] is False
    with pytest.raises(bm.BrainmapError) as err:
        bm.tie(_cfg(enabled=False), "item:it1", "subject:t1", "mentions")
    assert err.value.status == 409


def test_tie_links_observation_and_labels(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    linked: list[tuple] = []
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(
        bm.subj, "link", lambda dsn, oid, sid, **kw: linked.append((oid, sid, kw)) or True
    )
    monkeypatch.setattr(
        bm,
        "_link_row",
        lambda dsn, oid, sid: {"id": "NEW", "role": "evidence_for", "linked_by": "user"},
    )
    monkeypatch.setattr(
        bm.labels, "upsert_label", lambda dsn, **f: recorded.append(f) or {"id": "x"}
    )
    res = bm.tie(_cfg(), "subject:q1", "item:it1", "evidence_for")
    assert linked[0][:2] == ("o1", "q1") and linked[0][2]["linked_by"] == "user"
    assert res["edge"] == {
        **res["edge"],
        "id": "link:NEW",
        "from": "item:it1",
        "to": "subject:q1",
        "by": "user",
    }
    assert recorded[0]["target_kind"] == "link" and recorded[0]["verdict"] == "accept"
    assert recorded[0]["surface"] == "brainmap" and recorded[0]["embedding"] is None
    with pytest.raises(bm.BrainmapError):
        bm.tie(_cfg(), "item:it3", "subject:t1", "mentions")  # legacy item
    with pytest.raises(bm.BrainmapError):
        bm.tie(_cfg(), "person:mira", "subject:t1", "mentions")
    with pytest.raises(bm.BrainmapError):
        bm.tie(_cfg(), "item:it1", "item:it2", "mentions")


def test_tie_hub_files_under_parent_without_cycles(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    updates: list[tuple] = []
    monkeypatch.setattr(bm.subj, "update_subject", lambda dsn, sid, **f: updates.append((sid, f)))
    monkeypatch.setattr(bm.subj, "add_event", lambda *a, **k: None)
    res = bm.tie(_cfg(), "subject:g1", "subject:t1", "parent")
    assert updates == [("g1", {"parent_id": "t1"})]
    assert res["edge"]["id"] == "parent:g1"
    with pytest.raises(bm.BrainmapError):
        bm.tie(_cfg(), "subject:t1", "subject:q1", "parent")  # q1 is already under t1


def test_cut_writes_reject_and_routing_respects_it(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[dict[str, Any]] = []
    deleted: list[str] = []

    class FakeDb:
        def get(self, _model: Any, ref: str) -> Any:
            return SimpleNamespace(
                observation_id="o1", subject_id="t1", role="mentions", linked_by="embedding"
            )

        def delete(self, row: Any) -> None:
            deleted.append(row.subject_id)

    class FakeSession:
        def __enter__(self) -> FakeDb:
            return FakeDb()

        def __exit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(bm, "_get_session", lambda dsn: FakeSession())
    monkeypatch.setattr(bm.subj, "add_event", lambda *a, **k: None)
    monkeypatch.setattr(
        bm.labels, "upsert_label", lambda dsn, **f: recorded.append(f) or {"id": "x"}
    )
    res = bm.cut(_cfg(), "link:L1")
    assert deleted == ["t1"] and "Cut" in res["receipt"]
    label = recorded[0]
    assert label["verdict"] == "reject" and label["target_id"] == labels.link_key("t1", "Item 1")
    assert label["before"] == {"role": "mentions", "linked_by": "embedding"}
    with pytest.raises(bm.BrainmapError):
        bm.cut(_cfg(), "derived:item:it1->person:mira")

    # Routing: the cut pair is skipped even though t1 is the clear nearest subject.
    cand = _subject("t1", centroid=[1.0, 0.0])
    monkeypatch.setattr(em_route.subj, "list_subjects", lambda dsn, **kw: [cand])
    monkeypatch.setattr(em_route.labels, "rejected_link_keys", lambda dsn: {label["target_id"]})
    routed_links: list[tuple] = []
    monkeypatch.setattr(
        em_route.subj, "link", lambda dsn, oid, sid, **kw: routed_links.append((oid, sid))
    )
    obs = [
        {
            "id": "new-o1",
            "text": "Item 1",
            "type": "commitment",
            "status": "live",
            "embedding": [1.0, 0.0],
        },
        {
            "id": "other",
            "text": "Something else",
            "type": "fact",
            "status": "live",
            "embedding": [1.0, 0.01],
        },
    ]
    routed = asyncio.run(em_route.route(_cfg(), obs))
    assert routed == {"other": "t1"} and routed_links == [("other", "t1")]


def test_suggestion_verdicts(world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(
        bm.labels, "upsert_label", lambda dsn, **f: recorded.append(f) or {"id": "x"}
    )
    monkeypatch.setattr(bm.subj, "link", lambda *a, **k: True)
    monkeypatch.setattr(
        bm,
        "_link_row",
        lambda dsn, oid, sid: {"id": "S1", "role": "evidence_for", "linked_by": "user"},
    )
    snip = bm.suggestion_verdict(_cfg(), "sug:o2:q1", "snip")
    assert snip["edge"] is None and recorded[-1]["verdict"] == "reject"
    ok = bm.suggestion_verdict(_cfg(), "sug:o2:q1", "confirm")
    assert ok["edge"]["from"] == "item:it2" and ok["edge"]["role"] == "evidence_for"
    assert recorded[-1]["verdict"] == "accept"
    with pytest.raises(bm.BrainmapError):
        bm.suggestion_verdict(_cfg(), "bogus", "confirm")


def test_lasso_creates_user_thread(world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []

    def create_subject(dsn: str, **fields: Any) -> dict[str, Any]:
        s = _subject("new", **fields)
        world.subjects["new"] = s
        created.append(fields)
        return s

    linked: list[tuple] = []
    parents: list[tuple] = []
    monkeypatch.setattr(bm.subj, "create_subject", create_subject)
    monkeypatch.setattr(
        bm.subj, "link", lambda dsn, oid, sid, **kw: linked.append((oid, sid)) or True
    )
    monkeypatch.setattr(bm.subj, "update_subject", lambda dsn, sid, **f: parents.append((sid, f)))
    monkeypatch.setattr(bm.subj, "add_event", lambda *a, **k: None)
    monkeypatch.setattr(
        bm,
        "_link_row",
        lambda dsn, oid, sid: None if (oid, sid) not in linked else {"id": f"{oid}-{sid}"},
    )
    monkeypatch.setattr(bm.labels, "upsert_label", lambda dsn, **f: {"id": "x"})
    res = bm.lasso(
        _cfg(),
        ["item:it1", "item:it2", "item:it3", "person:mira", "subject:q1"],
        title="  Staging   capacity ",
    )
    assert created[0]["origin"] == "user" and created[0]["title"] == "Staging capacity"
    assert linked == [("o1", "new"), ("o2", "new")]
    assert parents == [("q1", {"parent_id": "new"})]
    assert {e["from"] for e in res["edges"]} == {"item:it1", "item:it2", "subject:q1"}
    assert res["subject"]["id"] == "subject:new"
    with pytest.raises(bm.BrainmapError):
        bm.lasso(_cfg(), ["item:it1"], title="  ")


def test_subject_label_only_for_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(labels, "upsert_label", lambda dsn, **f: recorded.append(f) or {"id": "x"})
    s = _subject("t9", status="proposed")
    assert labels.record_subject_label(_cfg(), s, "done") is None
    labels.record_subject_label(_cfg(), s, "dismissed", surface="brainmap")
    labels.record_subject_label(_cfg(), s, "active")
    assert [(r["verdict"], r["surface"]) for r in recorded] == [
        ("reject", "brainmap"),
        ("accept", "api"),
    ]
    assert all(r["embedding"] is None and r["target_kind"] == "subject" for r in recorded)


def test_link_key_survives_reprocessing() -> None:
    assert labels.link_key("t1", "We freeze the API, Monday!") == labels.link_key(
        "t1", "we freeze the api monday"
    )
    assert labels.link_key("t1", "a") != labels.link_key("t2", "a")


def test_layout_round_trip(tmp_path: Path) -> None:
    dsn = f"sqlite:///{tmp_path / 'boards.db'}"
    Base.metadata.create_all(sa.create_engine(dsn), tables=PORTAL_TABLES)
    assert bm.get_layout(dsn, "items") == {"positions": {}, "viewport": None, "updated_at": None}
    saved = bm.put_layout(
        dsn,
        "items",
        {
            "item:it1": {"x": 10, "y": -5.5, "pinned": True},
            "bad": {"x": "nope"},
            "subject:t1": {"x": 1, "y": 2},
        },
        {"x": 1, "y": 2, "zoom": 0.7},
    )
    assert saved["positions"] == {
        "item:it1": {"x": 10.0, "y": -5.5, "pinned": True},
        "subject:t1": {"x": 1.0, "y": 2.0},
    }
    again = bm.put_layout(dsn, "items", {"item:it1": {"x": 0, "y": 0}}, None)
    assert again["positions"] == {"item:it1": {"x": 0.0, "y": 0.0}}
    assert bm.get_layout(dsn, "items")["positions"] == {"item:it1": {"x": 0.0, "y": 0.0}}
    assert bm.get_layout(dsn, "other")["positions"] == {}


def test_unknown_board(tmp_path: Path) -> None:
    dsn = f"sqlite:///{tmp_path / 'boards.db'}"
    Base.metadata.create_all(sa.create_engine(dsn), tables=PORTAL_TABLES)
    with pytest.raises(bm.BrainmapError) as err:
        bm.build_board(_cfg(dsn=dsn), "nope")
    assert err.value.status == 404
