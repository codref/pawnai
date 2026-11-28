"""Brainmap boards: a corkboard graph over Items and the emergence web.

A board is a set of nodes (cards) and edges (red string) built from the
database for one board id. The first board, ``items``, pins open coworker
Items and pulls in what their observations feed: em_subjects (threads,
questions, decisions, commitments, predictions, hypotheses) through
``em_links``, the subjects' parent threads, goal threads named on the item,
the people involved and the session tape.

Every mutation is a human verdict and writes an ``em_labels`` row with
surface ``brainmap`` (see :mod:`pawn_agent.core.emergence.labels`):

- tie a string        → ``em_links`` row (``linked_by=user``) + ``link`` accept
- cut a string        → link deleted + ``link`` reject (routing skips the pair)
- ghost string ✓ / ✂  → same as tie / cut
- lasso               → a user thread, linked to every lassoed observation

Node ids: ``item:{id}``, ``subject:{id}``, ``obs:{id}`` (spilled evidence),
``person:{slug}``, ``session:{ref}``, ``thread:{slug}`` (a goal thread with no
subject row, when emergence is off). Edge ids: ``link:{em_links.id}``,
``parent:{child_subject_id}``, ``goal:{item_id}``, ``derived:{from}->{to}``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from sqlalchemy import func, select

from pawn_agent.core.coworker import db as itemdb
from pawn_agent.core.emergence import labels
from pawn_agent.core.emergence import subjects as subj
from pawn_agent.utils.db import EmLink, EmObservation, EmSource, EmSubject, _get_session

logger = logging.getLogger(__name__)

LINK_ROLES = ("mentions", "evidence_for", "evidence_against", "updates", "resolves", "contradicts")
HUB_KINDS = ("thread", "question", "decision", "commitment", "prediction", "hypothesis")
SUGGEST_KINDS = ("thread", "question", "decision", "commitment", "prediction")
PROPOSAL_KINDS = ("proposal", "people_update", "schedule_proposal")
ITEM_VERBS = ["todo", "file", "task", "later", "ignore", "delete"]
PROPOSAL_VERBS = ["approve", "reject", "delete"]
HEALTH = {"holds": "ok", "weakened": "weak", "broken": "broken"}
MAX_SUGGESTIONS = 40


class BrainmapError(Exception):
    """A request the board cannot honour; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def emergence_enabled(cfg: Any) -> bool:
    return bool(getattr(getattr(cfg, "emergence", None), "enabled", False))


def _em(cfg: Any, name: str, default: Any) -> Any:
    return getattr(getattr(cfg, "emergence", None), name, default)


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _require_enabled(cfg: Any) -> None:
    if not emergence_enabled(cfg):
        raise BrainmapError(
            "Emergence is off: strings are read-only. Set emergence.enabled to tie and cut.", 409
        )


# ── nodes ────────────────────────────────────────────────────────────────────


def person_id(name: str) -> str:
    return f"person:{subj.slugify(name, 40)}"


def _item_obs_ids(item: dict[str, Any]) -> list[str]:
    payload = item.get("payload") or {}
    ids = [payload.get("observation_id")] + list(payload.get("merged_observation_ids") or [])
    return [str(i) for i in ids if i]


def item_node(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload") or {}
    is_open = item.get("status") in itemdb.OPEN_STATUSES
    if not is_open:
        actions: list[str] = []
    elif item.get("kind") in PROPOSAL_KINDS:
        actions = list(PROPOSAL_VERBS)
    else:
        actions = list(ITEM_VERBS)
    obs_ids = _item_obs_ids(item)
    return {
        "id": f"item:{item['id']}",
        "kind": "item",
        "title": item.get("text") or "",
        "body": item.get("context") or None,
        "quote": item.get("quote") or None,
        "status": item.get("status") or "new",
        "origin": None,
        "proposed": False,
        "meta": {
            "item_id": item["id"],
            "item_kind": item.get("kind"),
            "observation_id": obs_ids[0] if obs_ids else None,
            "source_ref": item.get("source_ref"),
            "start_sec": payload.get("start_sec"),
            "owner": item.get("owner"),
            "due": item.get("due"),
            "note_key": item.get("note_key"),
            "thread": item.get("thread"),
        },
        "actions": actions,
    }


def _assumptions(fields: dict[str, Any]) -> list[dict[str, str]]:
    raw = fields.get("assumptions") or []
    health = fields.get("assumption_health") or {}
    out = []
    for i, text in enumerate(raw if isinstance(raw, list) else []):
        entry = health.get(str(i)) or {}
        state = entry.get("state") if isinstance(entry, dict) else str(entry)
        out.append({"text": str(text), "health": HEALTH.get(str(state or ""), "unchecked")})
    return out


def subject_actions(s: dict[str, Any], enabled: bool = True) -> list[str]:
    if not enabled:
        return []
    status, kind = s.get("status"), s.get("kind")
    if status in ("dismissed", "expired"):
        return []
    if status == "proposed":
        return ["accept", "dismiss"]
    out: list[str] = []
    if kind == "thread":
        out += ["spill", "done", "dismiss"]
    elif kind == "decision":
        out += ["spill", "council", "done"]
    elif kind == "question":
        out += ["spill", "resolved", "dismiss"]
    elif kind in ("commitment", "prediction"):
        out += ["done", "dismiss"]
    elif kind == "hypothesis":
        out += ["spill", "dismiss"]
    return out


def subject_node(
    s: dict[str, Any], *, evidence_count: int = 0, enabled: bool = True
) -> dict[str, Any]:
    fields = {k: v for k, v in (s.get("fields") or {}).items() if not str(k).startswith("_")}
    return {
        "id": f"subject:{s['id']}",
        "kind": s["kind"],
        "title": s.get("title") or "",
        "body": s.get("state_md") or None,
        "quote": fields.get("quote") or None,
        "status": s.get("status") or "active",
        "origin": s.get("origin"),
        "proposed": s.get("status") == "proposed",
        "meta": {
            "subject_id": s["id"],
            "momentum": s.get("momentum"),
            "evidence_count": evidence_count,
            "assumptions": _assumptions(fields),
            "direction": fields.get("direction"),
            "due": fields.get("due"),
            "by_when": fields.get("by_when"),
            "owner": fields.get("owner") or fields.get("debtor"),
            "speaker": fields.get("speaker") or None,
            "note_key": s.get("note_key"),
            "source_ref": s.get("source_ref"),
        },
        "actions": subject_actions(s, enabled),
    }


def obs_node(o: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"obs:{o['id']}",
        "kind": "scrap",
        "title": o.get("text") or "",
        "body": o.get("context") or None,
        "quote": o.get("quote") or None,
        "status": o.get("status") or "live",
        "origin": None,
        "proposed": False,
        "meta": {
            "observation_id": o["id"],
            "source_ref": o.get("source_ref"),
            "start_sec": o.get("start_sec"),
            "speaker": o.get("speaker_name") or None,
        },
        "actions": [],
    }


def _edge(
    eid: str,
    frm: str,
    to: str,
    role: str,
    by: str,
    *,
    cuttable: bool,
    score: Optional[float] = None,
) -> dict[str, Any]:
    return {
        "id": eid,
        "from": frm,
        "to": to,
        "role": role,
        "by": by,
        "cuttable": cuttable,
        "score": score,
    }


def _by(linked_by: str) -> str:
    return "user" if linked_by == "user" else "pawn"


# ── queries ──────────────────────────────────────────────────────────────────


def _links_for(dsn: str, obs_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Links from live observations in *obs_ids*."""
    ids = list(set(obs_ids))
    if not ids:
        return []
    with _get_session(dsn) as db:
        rows = db.execute(
            select(EmLink, EmObservation.status)
            .join(EmObservation, EmObservation.id == EmLink.observation_id)
            .where(EmLink.observation_id.in_(ids), EmObservation.status == "live")
        ).all()
        return [
            {
                "id": link.id,
                "observation_id": link.observation_id,
                "subject_id": link.subject_id,
                "role": link.role,
                "linked_by": link.linked_by,
                "score": link.score,
                "created_at": link.created_at,
            }
            for link, _status in rows
        ]


def _subjects_by_id(dsn: str, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    want = set(i for i in ids if i)
    out: dict[str, dict[str, Any]] = {}
    # Follow parent_id up a few levels so hubs hang from their threads.
    for _ in range(4):
        missing = want - set(out)
        if not missing:
            break
        with _get_session(dsn) as db:
            rows = db.scalars(select(EmSubject).where(EmSubject.id.in_(list(missing)))).all()
            found = [subj.subject_dict(r) for r in rows]
        for s in found:
            out[s["id"]] = s
        want = set(out) | {s["parent_id"] for s in found if s.get("parent_id")}
    return out


def _evidence_counts(dsn: str, subject_ids: Iterable[str]) -> dict[str, int]:
    ids = list(set(subject_ids))
    if not ids:
        return {}
    with _get_session(dsn) as db:
        rows = db.execute(
            select(EmLink.subject_id, func.count())
            .join(EmObservation, EmObservation.id == EmLink.observation_id)
            .where(EmLink.subject_id.in_(ids), EmObservation.status == "live")
            .group_by(EmLink.subject_id)
        ).all()
        return {sid: int(n) for sid, n in rows}


def _observations(dsn: str, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    want = list(set(ids))
    if not want:
        return {}
    with _get_session(dsn) as db:
        rows = db.execute(
            select(EmObservation, EmSource)
            .join(EmSource, EmSource.id == EmObservation.source_id, isouter=True)
            .where(EmObservation.id.in_(want))
        ).all()
        out = {}
        for obs, source in rows:
            out[obs.id] = {
                "id": obs.id,
                "type": obs.type,
                "text": obs.text,
                "context": obs.context or "",
                "quote": obs.quote or "",
                "speaker_name": obs.speaker_name or "",
                "start_sec": obs.start_sec,
                "status": obs.status,
                "source_ref": source.ref if source is not None else "",
                "embedding": (
                    [float(x) for x in obs.embedding] if obs.embedding is not None else None
                ),
            }
        return out


# ── the items board ──────────────────────────────────────────────────────────


def person_node(name: str) -> dict[str, Any]:
    return {
        "id": person_id(name),
        "kind": "person",
        "title": name,
        "status": "active",
        "proposed": False,
        "meta": {},
        "actions": [],
    }


def session_node(ref: str) -> dict[str, Any]:
    return {
        "id": f"session:{ref}",
        "kind": "session",
        "title": ref,
        "status": "active",
        "proposed": False,
        "meta": {"source_ref": ref},
        "actions": [],
    }


def add_item_context(
    nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]], item: dict[str, Any]
) -> None:
    """The item's owner and session tape, wired to its card."""
    nid = f"item:{item['id']}"
    owner = (item.get("owner") or "").strip()
    if owner:
        pid = person_id(owner)
        nodes.setdefault(pid, person_node(owner))
        edges[f"derived:{nid}->{pid}"] = _edge(
            f"derived:{nid}->{pid}", nid, pid, "owner", "derived", cuttable=False
        )
    ref = item.get("source_ref") or ""
    if ref and item.get("source_kind") == "session":
        sid = f"session:{ref}"
        nodes.setdefault(sid, session_node(ref))
        edges[f"derived:{nid}->{sid}"] = _edge(
            f"derived:{nid}->{sid}", nid, sid, "source", "derived", cuttable=False
        )


def build_items_board(
    cfg: Any,
    *,
    statuses: Optional[list[str]] = None,
    kinds: Optional[list[str]] = None,
    focus: str = "",
    limit: int = 150,
) -> dict[str, Any]:
    dsn = cfg.db_dsn
    enabled = emergence_enabled(cfg)
    limit = min(max(1, limit), 400)
    kind_filter = set(kinds or [])
    items = itemdb.list_items(
        dsn, statuses=statuses or list(itemdb.OPEN_STATUSES), limit=limit + 1 + len(kind_filter)
    )
    if kind_filter:
        items = [i for i in items if i.get("kind") in kind_filter]
    truncated = len(items) > limit
    items = items[:limit]
    if focus.startswith("item:") and not any(f"item:{i['id']}" == focus for i in items):
        extra = itemdb.get_item(dsn, focus[5:])
        if extra is not None:
            items.append(extra)

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    obs_owner: dict[str, str] = {}  # observation id → item node id
    for item in items:
        node = item_node(item)
        nodes[node["id"]] = node
        for oid in _item_obs_ids(item):
            obs_owner.setdefault(oid, node["id"])
        add_item_context(nodes, edges, item)

    suggestions: list[dict[str, Any]] = []
    goal_slugs = {subj.slugify(i["thread"]): i for i in items if i.get("thread")}
    if enabled:
        links = _links_for(dsn, obs_owner)
        goal_subjects = (
            [
                s
                for s in subj.list_subjects(dsn, kinds=("thread",))
                if s.get("origin") == "goal" and s["slug"] in goal_slugs
            ]
            if goal_slugs
            else []
        )
        subjects = _subjects_by_id(
            dsn, [lk["subject_id"] for lk in links] + [s["id"] for s in goal_subjects]
        )
        subjects = {k: v for k, v in subjects.items() if v.get("status") != "dismissed"}
        counts = _evidence_counts(dsn, subjects)
        for s in subjects.values():
            n = subject_node(s, evidence_count=counts.get(s["id"], 0), enabled=enabled)
            nodes[n["id"]] = n
            parent = s.get("parent_id")
            if parent and parent in subjects:
                edges[f"parent:{s['id']}"] = _edge(
                    f"parent:{s['id']}",
                    n["id"],
                    f"subject:{parent}",
                    "parent",
                    "pawn",
                    cuttable=True,
                )
        for lk in links:
            frm = obs_owner.get(lk["observation_id"])
            to = f"subject:{lk['subject_id']}"
            if (
                not frm
                or to not in nodes
                or any(e["from"] == frm and e["to"] == to for e in edges.values())
            ):
                continue
            edges[f"link:{lk['id']}"] = _edge(
                f"link:{lk['id']}",
                frm,
                to,
                lk["role"],
                _by(lk["linked_by"]),
                cuttable=True,
                score=lk["score"],
            )
        by_slug = {s["slug"]: s for s in goal_subjects}
        for item in items:
            s = by_slug.get(subj.slugify(item.get("thread") or ""))
            if s is None:
                continue
            frm, to = f"item:{item['id']}", f"subject:{s['id']}"
            if to in nodes and not any(e["from"] == frm and e["to"] == to for e in edges.values()):
                edges[f"goal:{item['id']}"] = _edge(
                    f"goal:{item['id']}", frm, to, "mentions", "goal", cuttable=False
                )
        suggestions = _suggestions(cfg, obs_owner, links, nodes)
    else:
        # Without the emergence layer, goal threads still come from the items.
        for slug, item in goal_slugs.items():
            tid = f"thread:{slug}"
            nodes.setdefault(
                tid,
                {
                    "id": tid,
                    "kind": "thread",
                    "title": item["thread"],
                    "status": "active",
                    "origin": "goal",
                    "proposed": False,
                    "meta": {},
                    "actions": [],
                },
            )
        for item in items:
            if item.get("thread"):
                tid = f"thread:{subj.slugify(item['thread'])}"
                edges[f"goal:{item['id']}"] = _edge(
                    f"goal:{item['id']}",
                    f"item:{item['id']}",
                    tid,
                    "mentions",
                    "goal",
                    cuttable=False,
                )

    return {
        "board": "items",
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "suggestions": suggestions,
        "emergence_enabled": enabled,
        "truncated": truncated,
    }


def _suggestions(
    cfg: Any,
    obs_owner: dict[str, str],
    links: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ghost strings: nearest open subject per board observation, not yet tied."""
    dsn = cfg.db_dsn
    candidates = [
        s
        for s in subj.list_subjects(dsn, kinds=SUGGEST_KINDS, statuses=subj.OPEN_STATUSES)
        if s.get("centroid") is not None
    ]
    if not candidates or not obs_owner:
        return []
    max_d = float(_em(cfg, "route_ambiguous_distance", 0.45))
    linked = {(lk["observation_id"], lk["subject_id"]) for lk in links}
    rejected = labels.rejected_link_keys(dsn)
    observations = _observations(dsn, obs_owner)
    out: list[dict[str, Any]] = []
    for oid, obs in observations.items():
        if obs["status"] != "live" or obs.get("embedding") is None:
            continue
        frm = obs_owner[oid]
        picked = 0
        for d, s in subj.nearest(obs["embedding"], candidates):
            if d > max_d or picked >= 2:
                break
            if (oid, s["id"]) in linked or labels.link_rejected(rejected, s["id"], obs["text"]):
                continue
            to = f"subject:{s['id']}"
            if to not in nodes:
                nodes[to] = subject_node(s, evidence_count=0)
            out.append(
                {
                    "id": f"sug:{oid}:{s['id']}",
                    "from": frm,
                    "to": to,
                    "role": "mentions" if s["kind"] == "thread" else "evidence_for",
                    "reason": f"Close to {s['kind']} “{s['title'][:60]}” (d={d:.2f})",
                    "score": round(float(d), 3),
                }
            )
            picked += 1
    out.sort(key=lambda x: x["score"])
    return out[:MAX_SUGGESTIONS]


BOARDS: dict[str, Callable[..., dict[str, Any]]] = {"items": build_items_board}


def build_board(cfg: Any, board: str, **params: Any) -> dict[str, Any]:
    """Built-in boards by name; user boards (lens, manual, question) via :mod:`boards`."""
    builder = BOARDS.get(board)
    if builder is not None:
        graph = builder(cfg, **params)
        if board == "items":
            graph.setdefault("meta", {"name": "items", "title": "Items", "kind": "items"})
            graph.setdefault("proposals", [])
        return graph
    from pawn_server.core import boards  # noqa: PLC0415

    return boards.build(
        cfg, board, focus=params.get("focus") or "", limit=params.get("limit") or 150
    )


# ── spill ("pull the thread") ────────────────────────────────────────────────


def spill(cfg: Any, subject_id: str, *, limit: int = 40) -> dict[str, Any]:
    """The subject's evidence that is not an open Item, as scrap cards."""
    dsn = cfg.db_dsn
    if subj.get_subject(dsn, subject_id) is None:
        raise BrainmapError("no such subject", 404)
    on_board = {oid for item in itemdb.list_open_items(dsn) for oid in _item_obs_ids(item)}
    with _get_session(dsn) as db:
        rows = db.execute(
            select(EmLink, EmObservation, EmSource)
            .join(EmObservation, EmObservation.id == EmLink.observation_id)
            .join(EmSource, EmSource.id == EmObservation.source_id, isouter=True)
            .where(EmLink.subject_id == subject_id, EmObservation.status == "live")
            .order_by(EmLink.created_at.desc())
            .limit(limit + len(on_board))
        ).all()
        found = [
            (
                link.id,
                link.role,
                link.linked_by,
                {
                    "id": obs.id,
                    "text": obs.text,
                    "context": obs.context,
                    "quote": obs.quote,
                    "speaker_name": obs.speaker_name,
                    "start_sec": obs.start_sec,
                    "status": obs.status,
                    "source_ref": source.ref if source is not None else "",
                },
            )
            for link, obs, source in rows
        ]
    nodes, edges = [], []
    for link_id, role, linked_by, o in found:
        if o["id"] in on_board or len(nodes) >= limit:
            continue
        n = obs_node(o)
        nodes.append(n)
        edges.append(
            _edge(
                f"link:{link_id}",
                n["id"],
                f"subject:{subject_id}",
                role,
                _by(linked_by),
                cuttable=True,
            )
        )
    return {"nodes": nodes, "edges": edges}


# ── mutations ────────────────────────────────────────────────────────────────


def _resolve(dsn: str, node_id: str) -> tuple[str, str]:
    """``("obs", observation_id)`` or ``("subject", subject_id)``."""
    kind, _, ref = node_id.partition(":")
    if kind == "subject" and ref:
        return "subject", ref
    if kind == "obs" and ref:
        return "obs", ref
    if kind == "item" and ref:
        item = itemdb.get_item(dsn, ref)
        if item is None:
            raise BrainmapError("no such item", 404)
        ids = _item_obs_ids(item)
        if not ids:
            raise BrainmapError(
                "This item predates the observation layer, so there is nothing to link."
            )
        return "obs", ids[0]
    raise BrainmapError("People and tapes are wired by Pawn from the transcript.")


def _subject(dsn: str, subject_id: str) -> dict[str, Any]:
    s = subj.get_subject(dsn, subject_id)
    if s is None:
        raise BrainmapError("no such subject", 404)
    return s


def _observation(dsn: str, oid: str) -> dict[str, Any]:
    obs = _observations(dsn, [oid]).get(oid)
    if obs is None:
        raise BrainmapError("no such observation", 404)
    return obs


def _link_row(dsn: str, oid: str, subject_id: str) -> Optional[dict[str, Any]]:
    with _get_session(dsn) as db:
        row = db.scalars(
            select(EmLink).where(EmLink.observation_id == oid, EmLink.subject_id == subject_id)
        ).first()
        if row is None:
            return None
        return {"id": row.id, "role": row.role, "linked_by": row.linked_by}


def _tie_obs(
    cfg: Any, oid: str, subject: dict[str, Any], role: str, *, surface: str = "brainmap"
) -> dict[str, Any]:
    obs = _observation(cfg.db_dsn, oid)
    created = subj.link(
        cfg.db_dsn,
        oid,
        subject["id"],
        role=role,
        linked_by="user",
        vector=obs.get("embedding"),
    )
    if not created:
        raise BrainmapError("Already tied.", 409)
    labels.record_link_label(
        cfg, subject=subject, observation=obs, verdict="accept", surface=surface, role=role
    )
    row = _link_row(cfg.db_dsn, oid, subject["id"])
    return {"link_id": row["id"] if row else str(uuid.uuid4()), "observation": obs}


def _set_parent(cfg: Any, child_id: str, parent_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if child_id == parent_id:
        raise BrainmapError("A string needs two ends.")
    child = _subject(cfg.db_dsn, child_id)
    parent = _subject(cfg.db_dsn, parent_id)
    # No cycles: walk up from the parent.
    seen, cur = {child_id}, parent
    while cur is not None:
        if cur["id"] in seen:
            raise BrainmapError("That would file a subject under itself.")
        seen.add(cur["id"])
        cur = subj.get_subject(cfg.db_dsn, cur["parent_id"]) if cur.get("parent_id") else None
    subj.update_subject(cfg.db_dsn, child_id, parent_id=parent_id)
    subj.add_event(
        cfg.db_dsn,
        child_id,
        "parent_set",
        {"before": child.get("parent_id"), "after": parent_id, "by": "user"},
    )
    return child, parent


def tie(cfg: Any, frm: str, to: str, role: str, *, surface: str = "brainmap") -> dict[str, Any]:
    _require_enabled(cfg)
    dsn = cfg.db_dsn
    a_kind, a_ref = _resolve(dsn, frm)
    b_kind, b_ref = _resolve(dsn, to)
    if a_kind == "subject" and b_kind == "subject":
        child, parent = _set_parent(cfg, a_ref, b_ref)
        return {
            "edge": _edge(f"parent:{a_ref}", frm, to, "parent", "user", cuttable=True),
            "receipt": f"Filed “{child['title']}” under “{parent['title']}”.",
        }
    if a_kind == "obs" and b_kind == "obs":
        raise BrainmapError("Two leads make a thread: lasso them instead.")
    if role not in LINK_ROLES:
        raise BrainmapError(f"role must be one of {', '.join(LINK_ROLES)}")
    oid, sid = (a_ref, b_ref) if a_kind == "obs" else (b_ref, a_ref)
    ev_node, hub_node = (frm, to) if a_kind == "obs" else (to, frm)
    subject = _subject(dsn, sid)
    res = _tie_obs(cfg, oid, subject, role, surface=surface)
    return {
        "edge": _edge(f"link:{res['link_id']}", ev_node, hub_node, role, "user", cuttable=True),
        "receipt": f"Tied to “{subject['title']}”. Pawn will learn from it.",
    }


def cut(cfg: Any, edge_id: str) -> dict[str, Any]:
    _require_enabled(cfg)
    dsn = cfg.db_dsn
    kind, _, ref = edge_id.partition(":")
    if kind == "parent" and ref:
        child = _subject(dsn, ref)
        if not child.get("parent_id"):
            raise BrainmapError("No such string.", 404)
        subj.update_subject(dsn, ref, parent_id=None)
        subj.add_event(
            dsn, ref, "parent_set", {"before": child["parent_id"], "after": None, "by": "user"}
        )
        return {"receipt": f"Unfiled “{child['title']}”."}
    if kind != "link" or not ref:
        raise BrainmapError("Pawn wires people, tapes and goal threads itself.")
    with _get_session(dsn) as db:
        row = db.get(EmLink, ref)
        if row is None:
            raise BrainmapError("No such string.", 404)
        link = {
            "observation_id": row.observation_id,
            "subject_id": row.subject_id,
            "role": row.role,
            "linked_by": row.linked_by,
        }
        db.delete(row)
    subject = _subject(dsn, link["subject_id"])
    obs = _observations(dsn, [link["observation_id"]]).get(link["observation_id"]) or {
        "id": link["observation_id"],
        "text": "",
    }
    labels.record_link_label(
        cfg,
        subject=subject,
        observation=obs,
        verdict="reject",
        role=link["role"],
        before={"role": link["role"], "linked_by": link["linked_by"]},
    )
    subj.add_event(
        dsn,
        subject["id"],
        "unlinked",
        {"observation_id": link["observation_id"], "role": link["role"], "by": "user"},
        causes=[link["observation_id"]],
    )
    return {"receipt": "Cut. Pawn will not tie these two again."}


def _node_for_obs(dsn: str, oid: str) -> str:
    for item in itemdb.list_open_items(dsn):
        if oid in _item_obs_ids(item):
            return f"item:{item['id']}"
    return f"obs:{oid}"


def suggestion_verdict(
    cfg: Any, suggestion_id: str, verdict: str, role: str = ""
) -> dict[str, Any]:
    _require_enabled(cfg)
    dsn = cfg.db_dsn
    parts = suggestion_id.split(":")
    if len(parts) != 3 or parts[0] != "sug":
        raise BrainmapError("no such suggestion", 404)
    _, oid, sid = parts
    subject = _subject(dsn, sid)
    if verdict == "snip":
        obs = _observation(dsn, oid)
        labels.record_link_label(
            cfg, subject=subject, observation=obs, verdict="reject", role=role or "suggested"
        )
        return {"edge": None, "receipt": "Snipped. Pawn will not suggest this again."}
    if verdict != "confirm":
        raise BrainmapError("verdict must be confirm or snip")
    role = (
        role
        if role in LINK_ROLES
        else ("mentions" if subject["kind"] == "thread" else "evidence_for")
    )
    res = _tie_obs(cfg, oid, subject, role)
    return {
        "edge": _edge(
            f"link:{res['link_id']}",
            _node_for_obs(dsn, oid),
            f"subject:{sid}",
            role,
            "user",
            cuttable=True,
        ),
        "receipt": f"Tied to “{subject['title']}”. Pawn will learn from it.",
    }


def lasso(
    cfg: Any,
    node_ids: list[str],
    *,
    title: str = "",
    into: str = "",
    store: Any = None,
    surface: str = "brainmap",
) -> dict[str, Any]:
    _require_enabled(cfg)
    dsn = cfg.db_dsn
    if not node_ids:
        raise BrainmapError("Lasso at least one card.")
    if into:
        target = _subject(dsn, into)
        if target["kind"] != "thread":
            raise BrainmapError("Cards can only be filed under a thread.")
    else:
        name = " ".join((title or "").split())
        if not name:
            raise BrainmapError("Name the thread.")
        target = subj.create_subject(
            dsn,
            kind="thread",
            title=name[:200],
            origin="user",
            status="active",
            state_md="",
            fields={"from": "brainmap"},
        )
    edges = []
    for nid in node_ids:
        try:
            kind, ref = _resolve(dsn, nid)
        except BrainmapError:
            continue  # people, tapes, legacy items: nothing to file
        if kind == "subject":
            if ref == target["id"]:
                continue
            try:
                _set_parent(cfg, ref, target["id"])
            except BrainmapError:
                continue
            edges.append(
                _edge(
                    f"parent:{ref}", nid, f"subject:{target['id']}", "parent", "user", cuttable=True
                )
            )
            continue
        if _link_row(dsn, ref, target["id"]) is not None:
            continue
        try:
            res = _tie_obs(cfg, ref, target, "mentions", surface=surface)
        except BrainmapError:
            continue
        edges.append(
            _edge(
                f"link:{res['link_id']}",
                nid,
                f"subject:{target['id']}",
                "mentions",
                "user",
                cuttable=True,
            )
        )
    fresh = _subject(dsn, target["id"])
    if store is not None:
        try:
            from pawn_agent.core.emergence import vault_notes  # noqa: PLC0415

            vault_notes.render_subject(cfg, store, fresh)
        except Exception as exc:  # the board must not fail on a vault hiccup
            logger.warning("could not render thread note %s: %s", fresh["id"], exc)
    counts = _evidence_counts(dsn, [fresh["id"]])
    return {
        "subject": subject_node(fresh, evidence_count=counts.get(fresh["id"], 0)),
        "edges": edges,
    }


# ── layout ───────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_layout(dsn: str, board: str) -> dict[str, Any]:
    from pawn_core.database import _get_session as core_session  # noqa: PLC0415
    from pawn_server.core.portal_models import PortalBoard  # noqa: PLC0415

    with core_session(dsn) as db:
        row = db.scalar(select(PortalBoard).where(PortalBoard.name == board))
        if row is None:
            return {"positions": {}, "viewport": None, "updated_at": None}
        return {
            "positions": row.positions or {},
            "viewport": row.viewport,
            "updated_at": _iso(row.updated_at),
        }


def put_layout(
    dsn: str, board: str, positions: dict[str, Any], viewport: Optional[dict[str, Any]]
) -> dict[str, Any]:
    from pawn_core.database import _get_session as core_session  # noqa: PLC0415
    from pawn_server.core.portal_models import PortalBoard  # noqa: PLC0415

    clean: dict[str, Any] = {}
    for node_id, pos in (positions or {}).items():
        try:
            entry = {"x": float(pos["x"]), "y": float(pos["y"])}
        except (KeyError, TypeError, ValueError):
            continue
        if pos.get("pinned"):
            entry["pinned"] = True
        clean[str(node_id)[:300]] = entry
    with core_session(dsn) as db:
        row = db.scalar(select(PortalBoard).where(PortalBoard.name == board))
        if row is None:
            row = PortalBoard(
                id=str(uuid.uuid4()),
                name=board,
                kind="items" if board == "items" else "manual",
                title="Items" if board == "items" else board,
                created_at=_now(),
            )
            db.add(row)
        row.positions = clean
        row.viewport = viewport
        row.updated_at = _now()
        db.flush()
        return {
            "positions": row.positions,
            "viewport": row.viewport,
            "updated_at": _iso(row.updated_at),
        }
