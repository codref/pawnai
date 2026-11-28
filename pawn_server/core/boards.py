"""User boards: brainmap corkboards you create, fill, and talk to.

A board is *seed + pins - hidden* (see :class:`PortalBoard`):

- ``lens``     — a saved query (``spec``) that keeps itself up to date
- ``manual``   — only what is pinned
- ``question`` — an em_subjects question the night shift works on (v2)
- ``items``    — the built-in Items board (built by :mod:`brainmap`)

Meaning stays in the emergence tables (links, parents, subjects); a board only
stores membership, note cards, layout and its chat. The board chat and the
night shift never change a board directly: they write
``portal_board_proposals`` rows that show as ghost cards and strings until the
user accepts (through the same tie / lasso / pin code as a gesture, so
``em_labels`` keep training, surface ``board_chat``) or burns them.

Extra node ids on top of :mod:`brainmap`'s: ``note:{id}`` (an accepted note
card) and ``prop:{id}`` (a proposal). Edge ids: ``notesrc:{note}:{node}`` and
``propsrc:{proposal}:{node}``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Iterable, Optional

from sqlalchemy import select

from pawn_agent.core.coworker import db as itemdb
from pawn_agent.core.emergence import labels
from pawn_agent.core.emergence import subjects as subj
from pawn_agent.utils.db import EmLink, EmObservation, _get_session
from pawn_server.core import brainmap as bm
from pawn_server.core.brainmap import BrainmapError

logger = logging.getLogger(__name__)

KINDS = ("lens", "manual", "question")
PROPOSAL_KINDS = ("string", "card", "group", "pin")
SURFACE = "board_chat"
MAX_NODES = 400
SNAPSHOT_CHARS = 8000
LENS_DISTANCE = 0.45


def _now() -> datetime:
    return bm._now()


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    try:
        return datetime.fromisoformat(str(value)).replace(tzinfo=None)
    except ValueError:
        return None


def _models() -> Any:
    from pawn_server.core import portal_models  # noqa: PLC0415

    return portal_models


def _session(dsn: str) -> Any:
    from pawn_core.database import _get_session as core_session  # noqa: PLC0415

    return core_session(dsn)


def _clean_ids(ids: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for i in ids or []:
        v = str(i or "").strip()[:300]
        if v and v not in out:
            out.append(v)
    return out


# ── rows ─────────────────────────────────────────────────────────────────────


def board_out(row: Any) -> dict[str, Any]:
    return {
        "name": row.name,
        "title": row.title or row.name,
        "kind": row.kind or "manual",
        "spec": row.spec or {},
        "pins": list(row.pins or []),
        "hidden": list(row.hidden or []),
        "notes": list(row.notes or []),
        "folder_id": row.folder_id,
        "tags": list(row.tags or []),
        "pinned": bool(row.pinned),
        "archived": bool(row.archived),
        "chat_id": row.chat_id,
        "subject_id": row.subject_id,
        "seen_at": _iso(row.seen_at),
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def _items_row(db: Any) -> Any:
    m = _models()
    row = db.scalar(select(m.PortalBoard).where(m.PortalBoard.name == "items"))
    if row is None:
        row = m.PortalBoard(
            id=str(uuid.uuid4()), name="items", title="Items", kind="items", created_at=_now()
        )
        db.add(row)
        db.flush()
    return row


def _row(db: Any, name: str) -> Any:
    m = _models()
    if name == "items":
        return _items_row(db)
    row = db.scalar(select(m.PortalBoard).where(m.PortalBoard.name == name))
    if row is None:
        raise BrainmapError(f"no such board {name!r}", 404)
    return row


def exists(dsn: str, name: str) -> bool:
    if name == "items":
        return True
    m = _models()
    with _session(dsn) as db:
        return db.scalar(select(m.PortalBoard.id).where(m.PortalBoard.name == name)) is not None


def get_board(dsn: str, name: str) -> dict[str, Any]:
    with _session(dsn) as db:
        return board_out(_row(db, name))


def list_boards(dsn: str, *, archived: bool = False) -> list[dict[str, Any]]:
    m = _models()
    with _session(dsn) as db:
        _items_row(db)
        rows = db.scalars(select(m.PortalBoard).where(m.PortalBoard.archived == archived)).all()
        out = [board_out(r) for r in rows]
    out.sort(key=lambda b: (b["kind"] != "items", -(_ts(b["updated_at"] or b["created_at"]))))
    return out


def _ts(value: Optional[str]) -> float:
    t = _parse_time(value)
    return t.timestamp() if t else 0.0


def create_board(
    cfg: Any,
    *,
    title: str,
    kind: str = "manual",
    spec: Optional[dict[str, Any]] = None,
    nodes: Optional[list[str]] = None,
    folder_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    chat_id: Optional[str] = None,
    question: str = "",
    subject_id: Optional[str] = None,
    store: Any = None,
) -> dict[str, Any]:
    """Create a board; a ``question`` board also gets (or reuses) its subject."""
    from pawn_server.core.portal_api import clean_tags  # noqa: PLC0415

    if kind not in KINDS:
        raise BrainmapError(f"kind must be one of {', '.join(KINDS)}")
    name_title = " ".join((title or question or "").split())[:200]
    if not name_title:
        raise BrainmapError("Name the board.")
    dsn = cfg.db_dsn
    if kind == "question":
        subject_id = _question_subject(cfg, question or name_title, subject_id, store)
    m = _models()
    name = f"b-{uuid.uuid4().hex[:10]}"
    now = _now()
    with _session(dsn) as db:
        if chat_id:
            chat = db.get(m.PortalChat, chat_id)
            if chat is None:
                raise BrainmapError("no such chat", 404)
            # The chat becomes this board's chat; it stays in the sidebar.
            chat.origin = {**(chat.origin or {}), "board": name}
        row = m.PortalBoard(
            id=str(uuid.uuid4()),
            name=name,
            title=name_title,
            kind=kind,
            spec=_clean_spec(spec or {}),
            pins=_clean_ids(nodes or []),
            hidden=[],
            notes=[],
            folder_id=folder_id,
            tags=clean_tags(tags or []),
            chat_id=chat_id,
            subject_id=subject_id,
            seen_at=now,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
        db.flush()
        if subject_id:
            s = subj.get_subject(dsn, subject_id)
            if s is not None and not (s.get("fields") or {}).get("board"):
                subj.update_subject(
                    dsn, subject_id, fields={**(s.get("fields") or {}), "board": name}
                )
        return board_out(row)


def _question_subject(cfg: Any, question: str, subject_id: Optional[str], store: Any) -> str:
    dsn = cfg.db_dsn
    if not bm.emergence_enabled(cfg):
        raise BrainmapError("Question boards need the emergence layer (emergence.enabled).", 409)
    if subject_id:
        s = subj.get_subject(dsn, subject_id)
        if s is None:
            raise BrainmapError("no such subject", 404)
        if s["kind"] not in ("question", "thread", "hypothesis", "decision"):
            raise BrainmapError("A question board grows around a question, thread or decision.")
        return str(s["id"])
    text = " ".join(question.split())[:300]
    vector = (labels.embed_many(cfg, [text]) or [None])[0]
    s = subj.create_subject(
        dsn,
        kind="question",
        title=text,
        origin="user",
        status="active",
        state_md="",
        fields={"from": "board"},
        centroid=vector,
    )
    if store is not None:
        try:
            from pawn_agent.core.emergence import vault_notes  # noqa: PLC0415

            vault_notes.render_subject(cfg, store, s)
        except Exception as exc:  # the board must not fail on a vault hiccup
            logger.warning("question note not rendered for %s: %s", s["id"], exc)
    return str(s["id"])


_SPEC_KEYS = {
    "q": str,
    "subject_ids": list,
    "subject_kinds": list,
    "subject_statuses": list,
    "item_kinds": list,
    "item_statuses": list,
    "people": list,
    "sources": list,
    "threads": list,
    "days": int,
    "expand": bool,
}


def _clean_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, typ in _SPEC_KEYS.items():
        if key not in spec or spec[key] in (None, "", []):
            continue
        value = spec[key]
        if typ is list:
            value = _clean_ids(value if isinstance(value, list) else [value])
        elif typ is int:
            try:
                value = max(1, min(int(value), 3650))
            except (TypeError, ValueError):
                continue
        elif typ is bool:
            value = bool(value)
        else:
            value = str(value).strip()[:500]
        out[key] = value
    return out


def patch_board(dsn: str, name: str, **fields: Any) -> dict[str, Any]:
    from pawn_server.core.portal_api import clean_tags  # noqa: PLC0415

    with _session(dsn) as db:
        row = _row(db, name)
        if fields.get("title") is not None:
            t = " ".join(str(fields["title"]).split())[:200]
            if t:
                row.title = t
        if fields.get("unfile"):
            row.folder_id = None
        elif fields.get("folder_id") is not None:
            row.folder_id = fields["folder_id"]
        if fields.get("tags") is not None:
            row.tags = clean_tags(fields["tags"])
        for flag in ("pinned", "archived"):
            if fields.get(flag) is not None:
                setattr(row, flag, bool(fields[flag]))
        if fields.get("spec") is not None and row.kind == "lens":
            row.spec = _clean_spec(fields["spec"])
        row.updated_at = _now()
        db.flush()
        return board_out(row)


def delete_board(dsn: str, name: str) -> dict[str, Any]:
    if name == "items":
        raise BrainmapError("The Items board is built in.")
    m = _models()
    with _session(dsn) as db:
        row = _row(db, name)
        for p in db.scalars(
            select(m.PortalBoardProposal).where(m.PortalBoardProposal.board == name)
        ):
            db.delete(p)
        if row.chat_id:
            chat = db.get(m.PortalChat, row.chat_id)
            if chat is not None and chat.origin:
                # The conversation stays as an ordinary chat.
                chat.origin = {**chat.origin, "promoted": True}
        subject_id = row.subject_id
        db.delete(row)
    if subject_id:
        s = subj.get_subject(dsn, subject_id)
        if s is not None and (s.get("fields") or {}).get("board") == name:
            fields = {k: v for k, v in (s.get("fields") or {}).items() if k != "board"}
            subj.update_subject(dsn, subject_id, fields=fields)
    return {"deleted": name}


def add_pins(dsn: str, name: str, node_ids: list[str]) -> dict[str, Any]:
    ids = [i for i in _clean_ids(node_ids) if not i.startswith(("prop:", "note:"))]
    with _session(dsn) as db:
        row = _row(db, name)
        if row.kind == "items":
            raise BrainmapError("The Items board shows open Items; make a board to pin cards.")
        pins = list(row.pins or [])
        row.pins = pins + [i for i in ids if i not in pins]
        row.hidden = [h for h in (row.hidden or []) if h not in ids]
        row.updated_at = _now()
        db.flush()
        return board_out(row)


def remove_nodes(dsn: str, name: str, node_ids: list[str]) -> dict[str, Any]:
    ids = _clean_ids(node_ids)
    props = [i[5:] for i in ids if i.startswith("prop:")]
    for pid in props:
        try:
            _set_proposal_status(dsn, name, pid, "rejected")
        except BrainmapError:
            pass
    with _session(dsn) as db:
        row = _row(db, name)
        if row.kind == "items":
            raise BrainmapError("The Items board shows open Items; triage them instead.")
        notes = {f"note:{n['id']}" for n in row.notes or []}
        row.notes = [n for n in row.notes or [] if f"note:{n['id']}" not in ids]
        rest = [i for i in ids if not i.startswith("prop:") and i not in notes]
        row.pins = [p for p in row.pins or [] if p not in rest]
        hidden = list(row.hidden or [])
        row.hidden = hidden + [i for i in rest if i not in hidden]
        row.updated_at = _now()
        db.flush()
        return board_out(row)


def mark_seen(dsn: str, name: str) -> dict[str, Any]:
    with _session(dsn) as db:
        row = _row(db, name)
        row.seen_at = _now()
        db.flush()
        return board_out(row)


def ensure_chat(dsn: str, name: str) -> dict[str, Any]:
    """The board's chat, created on first use (origin ``{kind: board, ref}``)."""
    from pawn_server.core.portal_api import _chat_out, new_chat  # noqa: PLC0415

    m = _models()
    with _session(dsn) as db:
        row = _row(db, name)
        if row.chat_id:
            chat = db.get(m.PortalChat, row.chat_id)
            if chat is not None:
                return _chat_out(chat)
        chat = new_chat(
            db,
            title=f"Board: {row.title or name}"[:200],
            origin={"kind": "board", "ref": name, "quote": row.title or name},
        )
        row.chat_id = chat.id
        db.flush()
        return _chat_out(chat)


def board_for_origin(origin: Optional[dict[str, Any]]) -> Optional[str]:
    """Board name a chat belongs to (board chat, or a chat a board was made from)."""
    if not origin:
        return None
    if origin.get("board"):
        return str(origin["board"])
    if origin.get("kind") == "board" and origin.get("ref") and not origin.get("promoted"):
        return str(origin["ref"])
    return None


# ── notes (accepted note cards) ──────────────────────────────────────────────


def add_note(
    dsn: str,
    name: str,
    *,
    title: str,
    body: str = "",
    sources: Optional[list[str]] = None,
    note_id: Optional[str] = None,
) -> dict[str, Any]:
    note = {
        "id": note_id or uuid.uuid4().hex[:12],
        "title": " ".join((title or "").split())[:200] or "Note",
        "body": (body or "").strip()[:4000],
        "sources": _clean_ids(sources or []),
        "created_at": _iso(_now()),
    }
    with _session(dsn) as db:
        row = _row(db, name)
        if row.kind == "items":
            raise BrainmapError("Make a board to keep note cards.")
        row.notes = list(row.notes or []) + [note]
        row.updated_at = _now()
        db.flush()
    return note


def tie_note(dsn: str, name: str, frm: str, to: str) -> dict[str, Any]:
    note_end, other = (frm, to) if frm.startswith("note:") else (to, frm)
    if other.startswith("note:"):
        raise BrainmapError("Two notes make a list: lasso them instead.")
    nid = note_end[5:]
    with _session(dsn) as db:
        row = _row(db, name)
        notes = [dict(n) for n in row.notes or []]
        for n in notes:
            if n["id"] == nid:
                if other in n.get("sources", []):
                    raise BrainmapError("Already tied.", 409)
                n["sources"] = list(n.get("sources", [])) + [other]
                break
        else:
            raise BrainmapError("no such note", 404)
        row.notes = [dict(n) for n in notes]
        row.updated_at = _now()
        db.flush()
    eid = f"notesrc:{nid}:{other}"
    return {
        "edge": bm._edge(eid, note_end, other, "mentions", "user", cuttable=True),
        "receipt": "Pinned the note to it.",
    }


def cut_note_source(dsn: str, name: str, edge_id: str) -> dict[str, Any]:
    _, nid, other = edge_id.split(":", 2)
    with _session(dsn) as db:
        row = _row(db, name)
        notes = [dict(n) for n in row.notes or []]
        for n in notes:
            if n["id"] == nid:
                n["sources"] = [s for s in n.get("sources", []) if s != other]
        row.notes = notes
        row.updated_at = _now()
        db.flush()
    return {"receipt": "Unpinned."}


# ── proposals ────────────────────────────────────────────────────────────────


def proposal_out(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "board": row.board,
        "chat_id": row.chat_id,
        "run_id": row.run_id,
        "origin": row.origin,
        "kind": row.kind,
        "payload": row.payload or {},
        "status": row.status,
        "created_at": _iso(row.created_at),
        "decided_at": _iso(row.decided_at),
    }


def propose(
    dsn: str,
    name: str,
    kind: str,
    payload: dict[str, Any],
    *,
    run_id: Optional[str] = None,
    origin: str = "chat",
) -> dict[str, Any]:
    if kind not in PROPOSAL_KINDS:
        raise BrainmapError(f"kind must be one of {', '.join(PROPOSAL_KINDS)}")
    m = _models()
    with _session(dsn) as db:
        row = _row(db, name)
        # The same open proposal twice is one ghost.
        for p in db.scalars(
            select(m.PortalBoardProposal).where(
                m.PortalBoardProposal.board == name,
                m.PortalBoardProposal.status == "proposed",
                m.PortalBoardProposal.kind == kind,
            )
        ):
            if _same_payload(kind, p.payload or {}, payload):
                return proposal_out(p)
        prop = m.PortalBoardProposal(
            id=uuid.uuid4().hex[:12],
            board=name,
            chat_id=row.chat_id,
            run_id=run_id,
            origin=origin,
            kind=kind,
            payload=payload,
            status="proposed",
            created_at=_now(),
        )
        db.add(prop)
        db.flush()
        return proposal_out(prop)


def _same_payload(kind: str, a: dict[str, Any], b: dict[str, Any]) -> bool:
    if kind == "string":
        return {a.get("from"), a.get("to")} == {b.get("from"), b.get("to")}
    if kind == "pin":
        return a.get("node") == b.get("node")
    return (a.get("title") or "").strip().lower() == (b.get("title") or "").strip().lower()


def list_proposals(
    dsn: str,
    name: str,
    *,
    status: Optional[str] = "proposed",
    run_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    m = _models()
    with _session(dsn) as db:
        stmt = select(m.PortalBoardProposal).where(m.PortalBoardProposal.board == name)
        if status:
            stmt = stmt.where(m.PortalBoardProposal.status == status)
        if run_id:
            stmt = stmt.where(m.PortalBoardProposal.run_id == run_id)
        rows = db.scalars(stmt.order_by(m.PortalBoardProposal.created_at)).all()
        return [proposal_out(r) for r in rows]


def proposals_for_run(dsn: str, run_id: str) -> list[dict[str, Any]]:
    m = _models()
    with _session(dsn) as db:
        rows = db.scalars(
            select(m.PortalBoardProposal).where(m.PortalBoardProposal.run_id == run_id)
        ).all()
        return [proposal_out(r) for r in rows]


def _set_proposal_status(dsn: str, name: str, pid: str, status: str) -> dict[str, Any]:
    m = _models()
    with _session(dsn) as db:
        row = db.get(m.PortalBoardProposal, pid)
        if row is None or row.board != name:
            raise BrainmapError("no such proposal", 404)
        if row.status != "proposed":
            raise BrainmapError(f"Already {row.status}.", 409)
        row.status = status
        row.decided_at = _now()
        db.flush()
        return proposal_out(row)


def _get_proposal(dsn: str, name: str, pid: str) -> dict[str, Any]:
    m = _models()
    with _session(dsn) as db:
        row = db.get(m.PortalBoardProposal, pid)
        if row is None or row.board != name:
            raise BrainmapError("no such proposal", 404)
        if row.status != "proposed":
            raise BrainmapError(f"Already {row.status}.", 409)
        return proposal_out(row)


def decide_proposal(
    cfg: Any, name: str, pid: str, verdict: str, *, store: Any = None
) -> dict[str, Any]:
    """Accept (through the gesture code) or burn one proposal."""
    if verdict in ("confirm", "accept", "approve"):
        verdict = "accept"
    elif verdict in ("snip", "reject", "dismiss", "burn"):
        verdict = "reject"
    else:
        raise BrainmapError("verdict must be accept or reject")
    dsn = cfg.db_dsn
    prop = _get_proposal(dsn, name, pid)
    kind, p = prop["kind"], prop["payload"]
    receipt = "Burned. Pawn will learn from it."
    if verdict == "reject":
        if kind == "string":
            _reject_string_label(cfg, p)
        _set_proposal_status(dsn, name, pid, "rejected")
        return {"receipt": receipt, "proposal": {**prop, "status": "rejected"}}
    board = get_board(dsn, name)
    if kind == "string":
        frm, to = str(p.get("from") or ""), str(p.get("to") or "")
        if frm.startswith("note:") or to.startswith("note:"):
            receipt = tie_note(dsn, name, frm, to)["receipt"]
        else:
            role = str(p.get("role") or "")
            if role not in bm.LINK_ROLES:
                role = "mentions"
            try:
                receipt = bm.tie(cfg, frm, to, role, surface=SURFACE)["receipt"]
            except BrainmapError as exc:
                if exc.status != 409:  # already tied: accepting is a no-op
                    raise
                receipt = "Already tied."
    elif kind == "card":
        add_note(
            dsn,
            name,
            title=str(p.get("title") or ""),
            body=str(p.get("body") or ""),
            sources=list(p.get("sources") or []),
            note_id=pid,
        )
        receipt = "Pinned the note."
    elif kind == "group":
        res = bm.lasso(
            cfg,
            list(p.get("nodes") or []),
            title=str(p.get("title") or ""),
            store=store,
            surface=SURFACE,
        )
        if board["kind"] != "items":
            add_pins(dsn, name, [res["subject"]["id"]])
        receipt = f"Grouped under “{res['subject']['title']}”."
    elif kind == "pin":
        add_pins(dsn, name, [str(p.get("node") or "")])
        receipt = "Pinned."
    _set_proposal_status(dsn, name, pid, "accepted")
    return {"receipt": receipt, "proposal": {**prop, "status": "accepted"}}


def _reject_string_label(cfg: Any, p: dict[str, Any]) -> None:
    """A burned obs ↔ subject string is a ``link`` reject, like a snipped ghost."""
    try:
        a = bm._resolve(cfg.db_dsn, str(p.get("from") or ""))
        b = bm._resolve(cfg.db_dsn, str(p.get("to") or ""))
    except BrainmapError:
        return
    if {a[0], b[0]} != {"obs", "subject"}:
        return
    oid, sid = (a[1], b[1]) if a[0] == "obs" else (b[1], a[1])
    subject = subj.get_subject(cfg.db_dsn, sid)
    obs = bm._observations(cfg.db_dsn, [oid]).get(oid)
    if subject is None or obs is None:
        return
    labels.record_link_label(
        cfg,
        subject=subject,
        observation=obs,
        verdict="reject",
        surface=SURFACE,
        role=str(p.get("role") or "suggested"),
    )


# ── building ─────────────────────────────────────────────────────────────────


class _Assembly:
    """Nodes and edges of one board as they are collected."""

    def __init__(self, cfg: Any, hidden: set[str]) -> None:
        self.cfg = cfg
        self.dsn = cfg.db_dsn
        self.hidden = hidden
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.obs_owner: dict[str, str] = {}  # observation id → node id
        self.subject_ids: list[str] = []
        self._open_items: Optional[list[dict[str, Any]]] = None
        self._owner_of: Optional[dict[str, dict[str, Any]]] = None

    def full(self) -> bool:
        return len(self.nodes) >= MAX_NODES

    def owner_of(self) -> dict[str, dict[str, Any]]:
        if self._owner_of is None:
            self._open_items = itemdb.list_open_items(self.dsn)
            self._owner_of = {}
            for item in self._open_items:
                for oid in bm._item_obs_ids(item):
                    self._owner_of.setdefault(oid, item)
        return self._owner_of

    def add_item(self, item: dict[str, Any]) -> None:
        nid = f"item:{item['id']}"
        if nid in self.hidden or nid in self.nodes or self.full():
            return
        node = bm.item_node(item)
        self.nodes[nid] = node
        for oid in bm._item_obs_ids(item):
            self.obs_owner.setdefault(oid, nid)
        bm.add_item_context(self.nodes, self.edges, item)
        for k in [k for k in self.nodes if k in self.hidden]:
            self.nodes.pop(k, None)

    def add_observations(self, oids: Iterable[str]) -> None:
        want = [o for o in dict.fromkeys(oids) if o and f"obs:{o}" not in self.hidden]
        owners = self.owner_of()
        loose = []
        for oid in want:
            item = owners.get(oid)
            if item is not None:
                self.add_item(item)
            else:
                loose.append(oid)
        for oid, o in bm._observations(self.dsn, loose).items():
            if o["status"] != "live" or self.full():
                continue
            nid = f"obs:{oid}"
            self.nodes[nid] = bm.obs_node(o)
            self.obs_owner.setdefault(oid, nid)

    def add_subject_ids(self, ids: Iterable[str]) -> None:
        for sid in ids:
            if sid and sid not in self.subject_ids and f"subject:{sid}" not in self.hidden:
                self.subject_ids.append(sid)

    def add_ref(self, node_id: str, notes: dict[str, dict[str, Any]]) -> None:
        kind, _, ref = node_id.partition(":")
        if not ref or node_id in self.hidden:
            return
        if kind == "item":
            item = itemdb.get_item(self.dsn, ref)
            if item is not None:
                self.add_item(item)
        elif kind == "subject":
            self.add_subject_ids([ref])
        elif kind == "obs":
            self.add_observations([ref])
        elif kind == "person":
            self.nodes.setdefault(
                node_id, {**bm.person_node(ref.replace("-", " ").title()), "id": node_id}
            )
        elif kind == "session":
            self.nodes.setdefault(node_id, bm.session_node(ref))
        elif kind == "note" and ref in notes:
            self.nodes[node_id] = note_node(notes[ref])


def note_node(n: dict[str, Any], *, proposal: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    nid = f"prop:{proposal['id']}" if proposal else f"note:{n['id']}"
    return {
        "id": nid,
        "kind": "note",
        "title": n.get("title") or "Note",
        "body": n.get("body") or None,
        "quote": None,
        "status": "proposed" if proposal else "active",
        "origin": proposal["origin"] if proposal else "user",
        "proposed": bool(proposal),
        "meta": {
            "note_id": None if proposal else n["id"],
            "created_at": None if proposal else n.get("created_at"),
            "proposal_id": proposal["id"] if proposal else None,
            "sources": list(n.get("sources") or []),
            "reason": (proposal or {}).get("payload", {}).get("reason"),
        },
        "actions": ["accept", "dismiss"] if proposal else ["remove"],
    }


def _evidence_obs(dsn: str, subject_ids: list[str], per_subject: int = 25) -> list[str]:
    out: list[str] = []
    if not subject_ids:
        return out
    with _get_session(dsn) as db:
        for sid in subject_ids:
            rows = db.execute(
                select(EmLink.observation_id)
                .join(EmObservation, EmObservation.id == EmLink.observation_id)
                .where(EmLink.subject_id == sid, EmObservation.status == "live")
                .order_by(EmLink.created_at.desc())
                .limit(per_subject)
            ).all()
            out += [r[0] for r in rows]
    return out


def _children(dsn: str, subject_ids: list[str]) -> list[dict[str, Any]]:
    ids = set(subject_ids)
    if not ids:
        return []
    return [
        s
        for s in subj.list_subjects(dsn)
        if s.get("parent_id") in ids and s.get("status") not in ("dismissed", "expired")
    ]


def _seed_lens(a: _Assembly, spec: dict[str, Any]) -> None:
    dsn, cfg = a.dsn, a.cfg
    days = int(spec.get("days") or 120)
    # Subjects named, by kind/status, or near the query.
    sids = list(spec.get("subject_ids") or [])
    if spec.get("subject_kinds") or spec.get("subject_statuses"):
        found = subj.list_subjects(
            dsn,
            kinds=spec.get("subject_kinds") or None,
            statuses=spec.get("subject_statuses") or subj.OPEN_STATUSES,
        )
        sids += [s["id"] for s in found[:60]]
    q = (spec.get("q") or "").strip()
    q_vec = (labels.embed_many(cfg, [q]) or [None])[0] if q else None
    if q_vec is not None:
        hubs = [
            s
            for s in subj.list_subjects(dsn, kinds=bm.HUB_KINDS, statuses=subj.OPEN_STATUSES)
            if s.get("centroid") is not None
        ]
        sids += [s["id"] for d, s in subj.nearest(q_vec, hubs)[:12] if d <= LENS_DISTANCE]
    if sids and bm.emergence_enabled(cfg):
        children = _children(dsn, sids)
        a.add_subject_ids(sids + [c["id"] for c in children])
        a.add_observations(_evidence_obs(dsn, sids))
    # Items by kind / person / source / goal thread.
    item_filters = any(spec.get(k) for k in ("item_kinds", "people", "sources", "threads"))
    if item_filters or spec.get("item_statuses"):
        items = itemdb.list_items(
            dsn, statuses=spec.get("item_statuses") or list(itemdb.OPEN_STATUSES), limit=400
        )
        kinds = set(spec.get("item_kinds") or [])
        people = {p.lower() for p in spec.get("people") or []}
        sources = set(spec.get("sources") or [])
        threads = {subj.slugify(t) for t in spec.get("threads") or []}
        for item in items:
            if kinds and item.get("kind") not in kinds:
                continue
            if people and (item.get("owner") or "").strip().lower() not in people:
                continue
            if sources and item.get("source_ref") not in sources:
                continue
            if threads and subj.slugify(item.get("thread") or "") not in threads:
                continue
            a.add_item(item)
    # Observations near the query.
    if q_vec is not None and bm.emergence_enabled(cfg):
        pool = subj.recent_observations(dsn, days=days, limit=3000)
        a.add_observations(
            o["id"]
            for d, o in subj.nearest(q_vec, pool, key="embedding")[:30]
            if d <= LENS_DISTANCE
        )
    if q and q_vec is None:
        # No embeddings: fall back to a text match on Items.
        for item in itemdb.list_items(dsn, statuses=list(itemdb.OPEN_STATUSES), q=q, limit=60):
            a.add_item(item)


def _seed_question(a: _Assembly, subject_id: str) -> None:
    s = subj.get_subject(a.dsn, subject_id)
    if s is None:
        return
    children = _children(a.dsn, [subject_id])
    a.add_subject_ids([subject_id] + [c["id"] for c in children])
    a.add_observations(_evidence_obs(a.dsn, [subject_id], per_subject=60))
    hyp_ids = [c["id"] for c in children if c["kind"] == "hypothesis"]
    a.add_observations(_evidence_obs(a.dsn, hyp_ids, per_subject=6))


def build(cfg: Any, name: str, *, focus: str = "", limit: int = 150) -> dict[str, Any]:
    """The graph of one user board, in the shape of :func:`brainmap.build_items_board`."""
    dsn = cfg.db_dsn
    board = get_board(dsn, name)
    if board["kind"] == "items":
        return bm.build_board(cfg, "items", focus=focus, limit=limit)
    enabled = bm.emergence_enabled(cfg)
    hidden = set(board["hidden"])
    notes = {n["id"]: n for n in board["notes"]}
    a = _Assembly(cfg, hidden)
    spec = board["spec"] or {}
    if board["kind"] == "lens":
        _seed_lens(a, spec)
    elif board["kind"] == "question" and board["subject_id"]:
        _seed_question(a, board["subject_id"])
    for nid in board["pins"]:
        a.add_ref(nid, notes)
    for n in notes.values():
        a.nodes[f"note:{n['id']}"] = note_node(n)
    if focus and focus not in a.nodes:
        a.add_ref(focus, notes)

    links: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    if enabled:
        links = bm._links_for(dsn, a.obs_owner)
        if spec.get("expand", True):
            # One hop: what the cards' observations feed.
            a.add_subject_ids(lk["subject_id"] for lk in links)
        subjects = bm._subjects_by_id(dsn, a.subject_ids) if a.subject_ids else {}
        subjects = {
            k: v
            for k, v in subjects.items()
            if v.get("status") != "dismissed"
            and f"subject:{k}" not in hidden
            and (k in a.subject_ids or spec.get("expand", True))
        }
        counts = bm._evidence_counts(dsn, subjects)
        for s in subjects.values():
            n = bm.subject_node(s, evidence_count=counts.get(s["id"], 0), enabled=enabled)
            a.nodes[n["id"]] = n
            parent = s.get("parent_id")
            if parent and parent in subjects:
                a.edges[f"parent:{s['id']}"] = bm._edge(
                    f"parent:{s['id']}",
                    n["id"],
                    f"subject:{parent}",
                    "parent",
                    "pawn",
                    cuttable=True,
                )
        for lk in links:
            frm = a.obs_owner.get(lk["observation_id"])
            to = f"subject:{lk['subject_id']}"
            if not frm or to not in a.nodes or frm not in a.nodes:
                continue
            if any(e["from"] == frm and e["to"] == to for e in a.edges.values()):
                continue
            a.edges[f"link:{lk['id']}"] = bm._edge(
                f"link:{lk['id']}",
                frm,
                to,
                lk["role"],
                bm._by(lk["linked_by"]),
                cuttable=True,
                score=lk["score"],
            )
        present = set(a.nodes)
        raw = bm._suggestions(
            cfg, {o: n for o, n in a.obs_owner.items() if n in a.nodes}, links, a.nodes
        )
        for k in set(a.nodes) - present:
            a.nodes.pop(k, None)
        suggestions = [s for s in raw if s["to"] in present]
    # Person and session cards only stay if something on the board ties to them.
    for nid, n in list(a.nodes.items()):
        if n["kind"] in ("person", "session") and nid not in board["pins"]:
            if not any(e["to"] == nid or e["from"] == nid for e in a.edges.values()):
                a.nodes.pop(nid, None)
    for n in notes.values():
        for src in n.get("sources") or []:
            if src in a.nodes:
                eid = f"notesrc:{n['id']}:{src}"
                a.edges[eid] = bm._edge(
                    eid, f"note:{n['id']}", src, "mentions", "user", cuttable=True
                )
    proposals = list_proposals(dsn, name)
    _add_proposals(a, proposals, suggestions, notes)
    _mark_new(a, board, links)
    nodes = list(a.nodes.values())
    truncated = len(nodes) > limit * 2 or len(a.nodes) >= MAX_NODES
    return {
        "board": name,
        "meta": {
            "name": name,
            "title": board["title"],
            "kind": board["kind"],
            "chat_id": board["chat_id"],
            "subject_id": board["subject_id"],
            "seen_at": board["seen_at"],
            "spec": spec,
        },
        "nodes": nodes,
        "edges": list(a.edges.values()),
        "suggestions": suggestions,
        "proposals": proposals,
        "emergence_enabled": enabled,
        "truncated": truncated,
    }


def _resolve_ghost(
    a: _Assembly, node_id: str, notes: dict[str, dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """A card for a proposed pin, without pulling in its context."""
    kind, _, ref = node_id.partition(":")
    if kind == "item":
        item = itemdb.get_item(a.dsn, ref)
        return bm.item_node(item) if item else None
    if kind == "subject":
        s = subj.get_subject(a.dsn, ref)
        return bm.subject_node(s, enabled=bm.emergence_enabled(a.cfg)) if s else None
    if kind == "obs":
        o = bm._observations(a.dsn, [ref]).get(ref)
        return bm.obs_node(o) if o else None
    if kind == "session":
        return bm.session_node(ref)
    if kind == "person":
        return {**bm.person_node(ref.replace("-", " ").title()), "id": node_id}
    return None


def _add_proposals(
    a: _Assembly,
    proposals: list[dict[str, Any]],
    suggestions: list[dict[str, Any]],
    notes: dict[str, dict[str, Any]],
) -> None:
    for p in proposals:
        pl, pid = p["payload"], p["id"]
        if p["kind"] in ("card", "group"):
            members = list(pl.get("sources") or pl.get("nodes") or [])
            if p["kind"] == "card":
                node = note_node(pl, proposal=p)
            else:
                node = {
                    **note_node(pl, proposal=p),
                    "kind": "thread",
                    "meta": {**note_node(pl, proposal=p)["meta"], "members": members},
                }
            a.nodes[node["id"]] = node
            for src in members:
                if src in a.nodes:
                    eid = f"propsrc:{pid}:{src}"
                    a.edges[eid] = bm._edge(
                        eid,
                        node["id"],
                        src,
                        "mentions" if p["kind"] == "card" else "parent",
                        "pawn",
                        cuttable=False,
                    )
        elif p["kind"] == "pin":
            nid = str(pl.get("node") or "")
            if not nid or nid in a.nodes:
                continue
            ghost = _resolve_ghost(a, nid, notes)
            if ghost is None:
                continue
            a.nodes[nid] = {
                **ghost,
                "proposed": True,
                "meta": {**ghost.get("meta", {}), "proposal_id": pid, "reason": pl.get("reason")},
                "actions": ["accept", "dismiss"],
            }
        elif p["kind"] == "string":
            frm, to = str(pl.get("from") or ""), str(pl.get("to") or "")
            if frm in a.nodes and to in a.nodes:
                suggestions.append(
                    {
                        "id": f"prop:{pid}",
                        "from": frm,
                        "to": to,
                        "role": pl.get("role") or "mentions",
                        "reason": pl.get("reason") or "Proposed in the board chat",
                        "score": None,
                        "by": p["origin"],
                    }
                )


def _mark_new(a: _Assembly, board: dict[str, Any], links: list[dict[str, Any]]) -> None:
    seen = _parse_time(board["seen_at"])
    if seen is None:
        return
    new_nodes: set[str] = set()
    for lk in links:
        eid = f"link:{lk['id']}"
        created = _parse_time(lk.get("created_at"))
        if eid in a.edges and created and created > seen:
            a.edges[eid]["new"] = True
            new_nodes.add(a.edges[eid]["from"])
    for nid, n in a.nodes.items():
        if nid.startswith("note:"):
            created = _parse_time(n.get("meta", {}).get("created_at"))
            if created and created > seen:
                new_nodes.add(nid)
    # Subjects created after the last look (a hypothesis dreamt overnight).
    if a.subject_ids:
        for sid, s in bm._subjects_by_id(a.dsn, [i for i in a.subject_ids]).items():
            created = _parse_time(s.get("created_at"))
            if created and created > seen and f"subject:{sid}" in a.nodes:
                new_nodes.add(f"subject:{sid}")
    for nid in new_nodes:
        if nid in a.nodes:
            a.nodes[nid]["meta"] = {**a.nodes[nid].get("meta", {}), "new": True}


# ── the board as text (board chat context) ───────────────────────────────────


def _card_line(n: dict[str, Any]) -> str:
    meta = n.get("meta") or {}
    bits = [f"[{n['id']}]", n["kind"].upper(), f"“{(n.get('title') or '')[:160]}”"]
    status = n.get("status")
    if status and status not in ("active", "live", "new"):
        bits.append(f"({status})")
    facts = []
    for key in ("owner", "due", "by_when", "speaker", "source_ref"):
        if meta.get(key):
            facts.append(f"{key}={meta[key]}")
    if meta.get("momentum") is not None:
        facts.append(
            f"momentum={meta['momentum']:.2f}" if isinstance(meta["momentum"], (int, float)) else ""
        )
    if meta.get("evidence_count"):
        facts.append(f"evidence={meta['evidence_count']}")
    shaky = [
        x["text"][:80]
        for x in meta.get("assumptions") or []
        if x.get("health") in ("weak", "broken")
    ]
    if shaky:
        facts.append("shaky assumptions: " + "; ".join(shaky))
    if meta.get("new"):
        facts.append("NEW since last look")
    if n.get("proposed"):
        facts.append("PROPOSED (not accepted)")
    line = " ".join(bits)
    if facts:
        line += " — " + ", ".join(f for f in facts if f)
    body = (n.get("quote") or n.get("body") or "").strip().replace("\n", " ")
    if body:
        line += f"\n    > {body[:260]}"
    return line


def snapshot(
    cfg: Any, name: str, selection: Optional[list[str]] = None, *, limit_chars: int = SNAPSHOT_CHARS
) -> str:
    """The board as compact text: selection first, then the rest by degree."""
    graph = build(cfg, name) if name != "items" else bm.build_board(cfg, "items")
    meta = graph.get("meta") or {}
    nodes = {n["id"]: n for n in graph["nodes"]}
    degree: dict[str, int] = {k: 0 for k in nodes}
    for e in graph["edges"]:
        degree[e["from"]] = degree.get(e["from"], 0) + 1
        degree[e["to"]] = degree.get(e["to"], 0) + 1
    selected = [i for i in (selection or []) if i in nodes]
    head = [
        f"Board “{meta.get('title') or name}” (id {name}, kind {meta.get('kind') or 'items'}), "
        f"{len(nodes)} cards, {len(graph['edges'])} strings."
    ]
    if meta.get("kind") == "question" and meta.get("subject_id"):
        q = nodes.get(f"subject:{meta['subject_id']}")
        if q:
            head.append(f"The board's question: “{q['title']}” ({q.get('status')}).")
    out = list(head)
    if selected:
        out.append("\nSELECTED by the user (the question is about these):")
        out += [_card_line(nodes[i]) for i in selected]
        sel = set(selected)
        strings = [e for e in graph["edges"] if e["from"] in sel or e["to"] in sel]
        if strings:
            out.append("Strings touching the selection:")
            out += [f"- {e['from']} --{e['role']}--> {e['to']}" for e in strings[:40]]
    rest = sorted((i for i in nodes if i not in selected), key=lambda i: -degree.get(i, 0))
    out.append("\nCards:")
    used = sum(len(x) + 1 for x in out)
    shown = 0
    for i in rest:
        line = _card_line(nodes[i])
        if used + len(line) > limit_chars * 0.7:
            break
        out.append(line)
        used += len(line) + 1
        shown += 1
    if shown < len(rest):
        out.append(
            f"… {len(rest) - shown} more cards (board_read --board {name} --node ID for one)."
        )
    strings = [
        f"- {e['from']} --{e['role']}--> {e['to']}" + (" (NEW)" if e.get("new") else "")
        for e in graph["edges"]
        if e["from"] not in selected and e["to"] not in selected
    ]
    if strings:
        out.append("\nStrings:")
        for s in strings:
            if used + len(s) > limit_chars * 0.9:
                out.append("… more strings")
                break
            out.append(s)
            used += len(s) + 1
    ghosts = graph.get("suggestions") or []
    if ghosts:
        out.append("\nOpen ghost strings (suggested, not accepted):")
        out += [
            f"- {g['from']} ~~{g['role']}~~> {g['to']}: {g.get('reason') or ''}"[:200]
            for g in ghosts[:15]
        ]
    text = "\n".join(out)
    return text[:limit_chars]


def card_detail(cfg: Any, name: str, node_id: str) -> str:
    """One card in depth: its text, its strings, and a subject's evidence."""
    graph = build(cfg, name) if name != "items" else bm.build_board(cfg, "items")
    nodes = {n["id"]: n for n in graph["nodes"]}
    n = nodes.get(node_id)
    if n is None:
        raise BrainmapError(f"{node_id} is not on board {name}", 404)
    lines = [_card_line(n)]
    if n.get("body") and n.get("quote"):
        lines.append(f"Context: {str(n['body'])[:1200]}")
    strings = [e for e in graph["edges"] if node_id in (e["from"], e["to"])]
    if strings:
        lines.append("Strings:")
        for e in strings:
            other = e["to"] if e["from"] == node_id else e["from"]
            title = (nodes.get(other) or {}).get("title", "")
            lines.append(f"- --{e['role']}--> [{other}] “{title[:100]}”")
    kind, _, ref = node_id.partition(":")
    if kind == "subject":
        obs = subj.subject_observations(cfg.db_dsn, ref, limit=30)
        if obs:
            lines.append("Evidence (oldest first):")
            for o in obs:
                when = _iso(o.get("created_at")) or ""
                lines.append(
                    f"- [obs:{o['id']}] ({o.get('role')}) {o.get('text', '')[:200]}"
                    f" — {o.get('speaker_name') or '?'}, {o.get('source_ref') or ''} {when[:10]}"
                )
    return "\n".join(lines)


def post_night_note(
    dsn: str,
    name: str,
    text: str,
    *,
    observation_ids: Optional[list[str]] = None,
    subject_ids: Optional[list[str]] = None,
) -> Optional[str]:
    """Append the night shift's "While you slept" note to the board chat."""
    from sqlalchemy import func  # noqa: PLC0415

    m = _models()
    chat = ensure_chat(dsn, name)
    with _session(dsn) as db:
        last = db.scalar(
            select(func.max(m.PortalMessage.seq)).where(m.PortalMessage.chat_id == chat["id"])
        )
        now = _now()
        msg = m.PortalMessage(
            id=uuid.uuid4().hex,
            chat_id=chat["id"],
            seq=(last + 1) if last is not None else 0,
            role="assistant",
            content=f"**While you slept.** {text.strip()}",
            context={
                "night": True,
                "observation_ids": list(observation_ids or []),
                "subject_ids": list(subject_ids or []),
            },
            created_at=now,
        )
        db.add(msg)
        row = db.get(m.PortalChat, chat["id"])
        if row is not None:
            row.last_message_at = now
            row.updated_at = now
        db.flush()
        return msg.id
