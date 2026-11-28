"""The ``boards`` night job: work the questions the user is investigating on a board.

For every open subject with ``fields.board`` (a question board, see
:mod:`pawn_server.core.boards`):

1. **Backfill** — the nearest recent observations not yet linked to it are
   judged in one batched call; relevant ones are linked (``linked_by=board``)
   with a ``linked`` event, so they show as NEW strings on the board.
2. **Dream** on the board's own evidence (:func:`dream.dream_board`);
   hypotheses hang off the question (``parent_id``).
3. **Night note** — one call writes "While you slept: …" into the board chat.
4. **Provocation** ``board_update`` when the board moved (≥ 2 new links, or a
   candidate answer).

Links a user cut or snipped are never re-made (``labels.link_rejected``).
"""

from __future__ import annotations

import logging
from typing import Any

from pawn_agent.core.emergence import dream, labels, provocations
from pawn_agent.core.emergence import subjects as subj
from pawn_agent.core.emergence.llm import ask_json

logger = logging.getLogger(__name__)

BOARD_ROLES = ("evidence_for", "evidence_against", "updates", "resolves", "mentions")
BOARD_KINDS = ("question", "thread", "hypothesis", "decision")

_BACKFILL_PROMPT = """\
A manager is investigating this on a board:

{kind}: {title}
{state}
Already on the board (most recent):
{known}

Candidate notes from recent meetings and captures:
{candidates}

Which candidates bear on the {kind}? For each one that does, give its role:
evidence_for, evidence_against, updates, resolves (answers it), or mentions.
Skip anything only loosely related. Return JSON:
{{"links": [{{"n": 1, "role": "evidence_for", "why": "one line"}}]}}
"""

_NOTE_PROMPT = """\
You keep a manager's investigation board. Overnight, this happened on the board
about the {kind} “{title}”:

New evidence:
{evidence}

New hunches:
{hunches}

Write a short note for the morning (3-5 sentences, plain text, no headings):
what came in, what argues against, and whether the {kind} now looks answered.
Return JSON: {{"note": "...", "answered": "no"}} with answered one of no, partly, yes.
"""


def _em(cfg: Any, name: str, default: Any) -> Any:
    return getattr(getattr(cfg, "emergence", None), name, default)


def board_subjects(dsn: str) -> list[dict[str, Any]]:
    return [
        s
        for s in subj.list_subjects(dsn, kinds=BOARD_KINDS, statuses=("active", "open", "proposed"))
        if (s.get("fields") or {}).get("board")
    ]


async def backfill(cfg: Any, s: dict[str, Any], *, budget: Any = None) -> list[dict[str, Any]]:
    """Link the nearest relevant recent observations to board subject *s*."""
    dsn = cfg.db_dsn
    if s.get("centroid") is None:
        return []
    days = int(_em(cfg, "board_backfill_days", 30))
    cap = int(_em(cfg, "board_max_candidates", 24))
    max_d = float(_em(cfg, "route_ambiguous_distance", 0.45))
    known = subj.subject_observations(dsn, s["id"], limit=200)
    known_ids = {o["id"] for o in known}
    rejected = labels.rejected_link_keys(dsn)
    pool = [
        o for o in subj.recent_observations(dsn, days=days, limit=3000) if o["id"] not in known_ids
    ]
    candidates = [
        o
        for d, o in subj.nearest(s["centroid"], pool, key="embedding")
        if d <= max_d and not labels.link_rejected(rejected, s["id"], o["text"])
    ][:cap]
    if not candidates:
        return []
    data = await ask_json(
        cfg,
        _BACKFILL_PROMPT.format(
            kind=s["kind"],
            title=s["title"],
            state=(s.get("state_md") or "")[:800],
            known="\n".join(f"- {o['text'][:160]}" for o in known[-8:]) or "(nothing yet)",
            candidates="\n".join(
                f"{i}. {o['text'][:240]} ({o.get('speaker_name') or '?'}, "
                f"{o.get('source_ref') or ''}, {o['created_at']:%Y-%m-%d})"
                for i, o in enumerate(candidates, 1)
            ),
        ),
        budget=budget,
    )
    linked: list[dict[str, Any]] = []
    for entry in (data or {}).get("links", []) if isinstance(data, dict) else []:
        try:
            o = candidates[int(entry.get("n")) - 1]
        except (TypeError, ValueError, IndexError):
            continue
        role = str(entry.get("role") or "")
        if role not in BOARD_ROLES:
            continue
        if subj.link(
            dsn, o["id"], s["id"], role=role, linked_by="board", vector=o.get("embedding")
        ):
            linked.append({**o, "role": role, "why": str(entry.get("why") or "")[:200]})
    if linked:
        subj.add_event(
            dsn,
            s["id"],
            "linked",
            {"by": "board", "count": len(linked), "roles": [o["role"] for o in linked]},
            causes=[o["id"] for o in linked],
        )
    return linked


async def night_note(
    cfg: Any,
    s: dict[str, Any],
    linked: list[dict[str, Any]],
    hunches: list[dict[str, Any]],
    *,
    budget: Any = None,
) -> dict[str, Any]:
    data = await ask_json(
        cfg,
        _NOTE_PROMPT.format(
            kind=s["kind"],
            title=s["title"],
            evidence="\n".join(
                f"- ({o['role']}) {o['text'][:200]} — {o.get('speaker_name') or '?'}"
                for o in linked
            )
            or "(none)",
            hunches="\n".join(f"- {h['title']}" for h in hunches) or "(none)",
        ),
        budget=budget,
    )
    if not isinstance(data, dict) or not str(data.get("note") or "").strip():
        n = len(linked)
        return {
            "note": f"{n} new piece{'s' if n != 1 else ''} of evidence came in overnight.",
            "answered": "no",
        }
    return {"note": str(data["note"]).strip(), "answered": str(data.get("answered") or "no")}


async def work_boards(cfg: Any, store: Any, *, budget: Any = None) -> dict[str, Any]:
    """Run the board passes for every board subject; returns counts per subject."""
    out: dict[str, Any] = {}
    for s in board_subjects(cfg.db_dsn):
        if budget is not None and budget.exhausted:
            break
        board = str((s.get("fields") or {}).get("board"))
        linked = await backfill(cfg, s, budget=budget)
        hunches: list[dict[str, Any]] = []
        try:
            hunches = await dream.dream_board(cfg, store, s["id"], budget=budget)
        except Exception as exc:
            if exc.__class__.__name__ == "BudgetExhausted":
                raise
            logger.warning("board dreaming failed for %s: %s", s["id"], exc)
        resolves = [o for o in linked if o["role"] == "resolves"]
        if linked or hunches:
            note = await night_note(cfg, s, linked, hunches, budget=budget)
            _post_note(cfg, board, note["note"], linked, hunches)
            if len(linked) >= 2 or resolves or note["answered"] in ("partly", "yes"):
                answered = "looks answered" if note["answered"] == "yes" else "moved"
                provocations.add(
                    cfg,
                    kind="board_update",
                    subject_id=s["id"],
                    key=f"board:{board}:{len(linked)}:{len(hunches)}",
                    text=(
                        f"Your board **{s['title'][:120]}** {answered} overnight: "
                        f"{len(linked)} new piece{'s' if len(linked) != 1 else ''} of evidence"
                        + (f", {len(hunches)} hunch" if hunches else "")
                        + ". Take a look?"
                    ),
                    features={"novelty": 1.0, "momentum": min(1.0, len(linked) / 5)},
                )
        out[s["id"]] = {"linked": len(linked), "hunches": len(hunches), "board": board}
    return out


def _post_note(
    cfg: Any, board: str, text: str, linked: list[dict[str, Any]], hunches: list[dict[str, Any]]
) -> None:
    try:
        from pawn_server.core import boards  # noqa: PLC0415

        boards.post_night_note(
            cfg.db_dsn,
            board,
            text,
            observation_ids=[o["id"] for o in linked],
            subject_ids=[h["id"] for h in hunches],
        )
    except Exception as exc:  # the portal tables may be missing on a bare install
        logger.info("night note not posted to board %s: %s", board, exc)
