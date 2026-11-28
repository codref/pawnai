"""Board tools for the board chat: read a board, propose changes to it.

Proposals are rows in ``portal_board_proposals`` tagged with the running
turn's ``run_id`` (``PAWN_PARENT_RUN_ID``), so the chat stream can show them
as ghosts as soon as the turn ends. Nothing here changes a board.
"""

from __future__ import annotations

import os
from typing import Any, Optional


def board_read_impl(
    cfg: Any, *, board: str, node: str = "", selection: Optional[list[str]] = None
) -> str:
    from pawn_server.core import boards  # noqa: PLC0415

    if node:
        return boards.card_detail(cfg, board, node)
    return boards.snapshot(cfg, board, selection or [])


_EVIDENCE = ("item", "obs")


def _check_string_ends(frm: str, to: str) -> None:
    """A string must be one the user could tie: evidence ↔ subject, subject ↔ subject, or a note."""
    a, b = frm.split(":", 1)[0], to.split(":", 1)[0]
    if "prop" in (a, b):
        raise ValueError("ghosts cannot be tied; propose the string to a real card")
    if {a, b} & {"person", "session"}:
        raise ValueError(
            "people and tapes are wired by Pawn from the transcript; tie a subject instead"
        )
    if a in _EVIDENCE and b in _EVIDENCE:
        raise ValueError(
            "two pieces of evidence do not take a string: propose --kind group (a new thread "
            "over them) or --kind card (a note saying what links them)"
        )


def _on_board(cfg: Any, board: str) -> set[str]:
    from pawn_server.core import boards, brainmap  # noqa: PLC0415

    graph = boards.build(cfg, board) if board != "items" else brainmap.build_board(cfg, "items")
    return {n["id"] for n in graph["nodes"]}


def board_propose_impl(
    cfg: Any,
    *,
    board: str,
    kind: str,
    frm: str = "",
    to: str = "",
    role: str = "",
    title: str = "",
    body: str = "",
    sources: Optional[list[str]] = None,
    nodes: Optional[list[str]] = None,
    reason: str = "",
) -> str:
    from pawn_server.core import boards, brainmap  # noqa: PLC0415

    run_id = (os.environ.get("PAWN_PARENT_RUN_ID") or "").strip() or None
    present = _on_board(cfg, board)
    reason = " ".join((reason or "").split())[:300]

    def _check(ids: list[str], what: str) -> None:
        missing = [i for i in ids if i not in present]
        if missing:
            raise ValueError(
                f"{what} not on board {board}: {', '.join(missing)}. "
                "Use card ids from the snapshot (board_read)."
            )

    if kind == "string":
        if not frm or not to or frm == to:
            raise ValueError("a string needs --from and --to (two different card ids)")
        _check([frm, to], "cards")
        _check_string_ends(frm, to)
        if role and role not in brainmap.LINK_ROLES:
            raise ValueError(f"--role must be one of {', '.join(brainmap.LINK_ROLES)}")
        payload: dict[str, Any] = {
            "from": frm,
            "to": to,
            "role": role or "mentions",
            "reason": reason,
        }
        what = f"string {frm} --{payload['role']}--> {to}"
    elif kind == "card":
        if not title.strip():
            raise ValueError("a note card needs --title")
        srcs = list(dict.fromkeys(sources or []))
        _check(srcs, "sources")
        payload = {
            "title": title.strip()[:200],
            "body": body.strip()[:4000],
            "sources": srcs,
            "reason": reason,
        }
        what = f"note card “{payload['title']}”"
    elif kind == "group":
        members = list(dict.fromkeys(nodes or []))
        if len(members) < 2 or not title.strip():
            raise ValueError("a group needs --title and at least two --node ids")
        _check(members, "cards")
        payload = {"title": title.strip()[:200], "nodes": members, "reason": reason}
        what = f"group “{payload['title']}” of {len(members)} cards"
    elif kind == "pin":
        ids = list(dict.fromkeys(nodes or []))
        if not ids:
            raise ValueError(
                "a pin needs at least one --node id (item:…, subject:…, obs:…, session:…)"
            )
        out = []
        for nid in ids:
            if nid in present:
                out.append(f"{nid} is already on the board")
                continue
            if not nid.split(":", 1)[0] in ("item", "subject", "obs", "session", "person"):
                raise ValueError(f"cannot pin {nid}")
            p = boards.propose(
                cfg.db_dsn, board, "pin", {"node": nid, "reason": reason}, run_id=run_id
            )
            out.append(f"proposed pin {nid} ({p['id']})")
        return "\n".join(out) + "\nThe user accepts or burns each ghost on the board."
    else:
        raise ValueError("--kind must be string, card, group or pin")
    p = boards.propose(cfg.db_dsn, board, kind, payload, run_id=run_id)
    return f"Proposed {what} ({p['id']}). The user accepts or burns it on the board; it is not done yet."
