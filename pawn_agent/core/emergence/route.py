"""After a source is observed: ledger subjects, routing, the directed pass.

``after_observe`` runs right after ``observe_source`` persisted a run:

1. Decisions, commitments and predictions become ledger subjects (or update
   a near-identical open one).
2. Every live observation is routed to the nearest thread or question by
   embedding; ambiguous ones go to one batched model call; the rest stay
   unrouted for the nightly clusterer.
3. The directed pass asks each window about the open subjects nearest to
   it and stores what it finds as ``evidence`` observations with a role.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional

from pawn_agent.core.coworker.extract import locate_quote, speaker_at, start_seconds_at
from pawn_agent.core.emergence import labels
from pawn_agent.core.emergence import subjects as subj
from pawn_agent.core.emergence.llm import ask_json

logger = logging.getLogger(__name__)

LEDGER_TYPES = ("decision", "commitment", "prediction")
ROUTE_KINDS = ("thread", "question")
DIRECTED_ROLES = ("evidence_for", "evidence_against", "updates", "resolves")
_SAME_LEDGER_DISTANCE = 0.12

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}


def _em(cfg: Any, name: str, default: Any) -> Any:
    return getattr(getattr(cfg, "emergence", None), name, default)


def _month_end(year: int, month: int) -> date:
    first_next = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return first_next - timedelta(days=1)


def parse_when(text: str, *, today: Optional[date] = None) -> Optional[date]:
    """Best-effort date for 'by when' phrases ('2026-03-31', 'end of Q1', 'March')."""
    today = today or date.today()
    raw = (text or "").strip().lower()
    if not raw:
        return None
    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
    year_match = re.search(r"\b(20\d{2})\b", raw)
    year = int(year_match.group(1)) if year_match else None
    quarter = re.search(r"\bq([1-4])\b", raw)
    if quarter:
        q = int(quarter.group(1))
        y = year or (today.year if q >= (today.month - 1) // 3 + 1 else today.year + 1)
        return _month_end(y, q * 3)
    for name, month in _MONTHS.items():
        if re.search(rf"\b{name}\b", raw):
            y = year or (today.year if month >= today.month else today.year + 1)
            return _month_end(y, month)
    if "next week" in raw:
        return today + timedelta(days=7)
    if "next month" in raw:
        return today + timedelta(days=31)
    if "end of year" in raw or "year end" in raw:
        return date(year or today.year, 12, 31)
    weeks = re.search(r"(\d+|a few|few|couple of) weeks?", raw)
    if weeks:
        n = {"a few": 3, "few": 3, "couple of": 2}.get(weeks.group(1)) or int(weeks.group(1))
        return today + timedelta(weeks=n)
    return None


def _ledger_fields(obs: dict[str, Any]) -> dict[str, Any]:
    fields = dict(obs.get("fields") or {})
    fields["speaker"] = obs.get("speaker_name") or ""
    fields["quote"] = obs.get("quote") or ""
    return fields


def _review_at(cfg: Any, obs: dict[str, Any]) -> Optional[datetime]:
    fields = obs.get("fields") or {}
    if obs["type"] == "decision":
        return subj.now() + timedelta(days=int(_em(cfg, "decision_review_days", 30)))
    if obs["type"] == "prediction":
        when = parse_when(str(fields.get("by_when") or ""))
        return (
            datetime.combine(when, datetime.min.time()) if when else subj.now() + timedelta(days=90)
        )
    if obs["type"] == "commitment":
        when = parse_when(str(fields.get("due") or ""))
        return datetime.combine(when, datetime.min.time()) if when else None
    return None


def ensure_ledger(
    cfg: Any, obs: dict[str, Any], open_ledger: list[dict[str, Any]]
) -> Optional[str]:
    """Create (or update) the ledger subject for one observation; its id."""
    if obs["type"] not in LEDGER_TYPES:
        return None
    same_kind = [s for s in open_ledger if s["kind"] == obs["type"]]
    near = subj.nearest(obs.get("embedding"), same_kind)
    if near and near[0][0] < _SAME_LEDGER_DISTANCE:
        target = near[0][1]
        subj.link(
            cfg.db_dsn,
            obs["id"],
            target["id"],
            role="updates",
            score=near[0][0],
            linked_by="ledger",
            vector=obs.get("embedding"),
        )
        return str(target["id"])
    created = subj.create_subject(
        cfg.db_dsn,
        kind=obs["type"],
        title=obs["text"],
        origin="extracted",
        status="open",
        fields=_ledger_fields(obs),
        source_ref=obs.get("source_ref"),
        review_at=_review_at(cfg, obs),
        centroid=obs.get("embedding"),
    )
    subj.link(cfg.db_dsn, obs["id"], created["id"], role="origin", linked_by="ledger")
    open_ledger.append(created)
    return str(created["id"])


_ROUTE_PROMPT = """\
Attach each observation to the subject it is about, or to none.

Subjects:
{subjects}

Observations:
{observations}

Return JSON: {{"links": [{{"observation": 0, "subject": 2}}]}}
Use "subject": null when none fits. Only attach when the observation is
clearly about that subject, not merely the same general topic.
"""


async def route(
    cfg: Any,
    observations: list[dict[str, Any]],
    *,
    budget: Any = None,
) -> dict[str, str]:
    """Route observations to threads/questions. Returns ``{obs_id: subject_id}``."""
    candidates = subj.list_subjects(cfg.db_dsn, kinds=ROUTE_KINDS, statuses=subj.OPEN_STATUSES)
    candidates = [c for c in candidates if c.get("centroid") is not None]
    max_d = float(_em(cfg, "route_max_distance", 0.32))
    margin = float(_em(cfg, "route_margin", 0.04))
    ambiguous_d = float(_em(cfg, "route_ambiguous_distance", 0.45))
    routed: dict[str, str] = {}
    ambiguous: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    # Strings the user cut on the brainmap stay cut, even after a re-process.
    rejected = labels.rejected_link_keys(cfg.db_dsn) if candidates else set()
    for obs in observations:
        if not obs.get("embedding") or obs.get("status") != "live":
            continue
        ranked = [
            (d, c)
            for d, c in subj.nearest(obs["embedding"], candidates)
            if not labels.link_rejected(rejected, c["id"], obs.get("text") or "")
        ]
        if not ranked:
            continue
        best_d, best = ranked[0]
        second_d = ranked[1][0] if len(ranked) > 1 else 2.0
        if best_d <= max_d and second_d - best_d >= margin:
            subj.link(cfg.db_dsn, obs["id"], best["id"], score=best_d, vector=obs["embedding"])
            routed[obs["id"]] = best["id"]
        elif best_d <= ambiguous_d:
            ambiguous.append((obs, [c for d, c in ranked[:4] if d <= ambiguous_d]))
    if ambiguous:
        routed.update(await _route_ambiguous(cfg, ambiguous, budget=budget))
    return routed


async def _route_ambiguous(
    cfg: Any,
    ambiguous: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    budget: Any = None,
) -> dict[str, str]:
    pool: list[dict[str, Any]] = []
    for _, options in ambiguous:
        for option in options:
            if option["id"] not in {p["id"] for p in pool}:
                pool.append(option)
    subjects_txt = "\n".join(f"{i}. [{s['kind']}] {s['title']}" for i, s in enumerate(pool))
    obs_txt = "\n".join(f"{i}. [{o['type']}] {o['text']}" for i, (o, _) in enumerate(ambiguous))
    data = await ask_json(
        cfg, _ROUTE_PROMPT.format(subjects=subjects_txt, observations=obs_txt), budget=budget
    )
    out: dict[str, str] = {}
    for entry in (data or {}).get("links", []) if isinstance(data, dict) else []:
        try:
            oi, si = int(entry.get("observation")), entry.get("subject")
            if si is None:
                continue
            obs, options = ambiguous[oi]
            target = pool[int(si)]
        except (TypeError, ValueError, IndexError):
            continue
        if target["id"] not in {o["id"] for o in options}:
            continue
        subj.link(cfg.db_dsn, obs["id"], target["id"], linked_by="model", vector=obs["embedding"])
        out[obs["id"]] = target["id"]
    return out


_DIRECTED_PROMPT = """\
Below are open subjects this user is tracking, then a SOURCE.
For each subject, does the SOURCE contain evidence for it, against it, an
update to it, or its resolution (a question answered, a commitment done, a
prediction settled)?

Subjects:
{subjects}

Return JSON:
{{"evidence": [{{"subject": 0, "role": "evidence_for|evidence_against|updates|resolves",
  "assumption": null, "text": "one-line claim", "quote": "verbatim from SOURCE"}}]}}
For a decision, set "assumption" to the number of the assumption the
evidence bears on, else null. Only include clear evidence; quote it exactly.
Use an empty list when nothing bears on any subject.
{part}
SOURCE:
{source}
"""


def _subject_digest(subjects: list[dict[str, Any]]) -> str:
    lines = []
    for i, s in enumerate(subjects):
        f = s.get("fields") or {}
        line = f"{i}. [{s['kind']}] {s['title']}"
        if s["kind"] == "decision" and f.get("assumptions"):
            line += "\n   assumptions: " + "; ".join(
                f"({n}) {a}" for n, a in enumerate(f["assumptions"])
            )
        if s["kind"] == "commitment" and f.get("debtor"):
            line += f" (owed by {f['debtor']})"
        lines.append(line)
    return "\n".join(lines)


async def directed_pass(cfg: Any, result: Any, *, budget: Any = None) -> list[dict[str, Any]]:
    """One call per window about the nearest open subjects; stores evidence."""
    from pawn_agent.core.emergence import prompts  # noqa: PLC0415

    open_subjects = [
        s
        for s in subj.list_subjects(cfg.db_dsn, statuses=("active", "open", "proposed", "pending"))
        if s["kind"] in ("thread", "question", "decision", "commitment", "prediction")
        and s.get("centroid") is not None
        and s.get("source_ref") != result.source_ref
    ]
    if not open_subjects or not result.windows:
        return []
    limit = int(_em(cfg, "directed_max_subjects", 12))
    rejected = labels.rejected_link_keys(cfg.db_dsn)
    found: list[dict[str, Any]] = []
    windows = result.windows
    for w in windows:
        vector = w.embedding
        if vector is None:
            vector = (await asyncio.to_thread(labels.embed_many, cfg, [w.window.text[:6000]]))[0]
        nearby = [s for _, s in subj.nearest(vector, open_subjects)[:limit]]
        if not nearby:
            continue
        part = prompts.part_note(w.window.index, len(windows), "")
        data = await ask_json(
            cfg,
            _DIRECTED_PROMPT.format(
                subjects=_subject_digest(nearby), part=part, source=w.window.text
            ),
            budget=budget,
        )
        entries = data.get("evidence", []) if isinstance(data, dict) else []
        for entry in entries if isinstance(entries, list) else []:
            try:
                target = nearby[int(entry.get("subject"))]
            except (TypeError, ValueError, IndexError):
                continue
            role = str(entry.get("role") or "")
            quote = str(entry.get("quote") or "")
            text = str(entry.get("text") or "").strip()
            if role not in DIRECTED_ROLES or not text:
                continue
            if labels.link_rejected(rejected, target["id"], text):
                continue
            offset, score = locate_quote(w.window.text, quote)
            if score < float(_em(cfg, "quote_min_score", 0.8)):
                continue
            start = speaker = None
            if offset is not None:
                start = start_seconds_at(w.window.text, offset)
                speaker = speaker_at(w.window.text, offset)
            vec = (
                await asyncio.to_thread(
                    labels.embed_many, cfg, [labels.label_text("evidence", text, quote)]
                )
            )[0]
            oid = subj.insert_observation(
                cfg.db_dsn,
                source_kind=result.source_kind,
                source_ref=result.source_ref,
                window_id=w.id,
                type="evidence",
                text=text,
                speaker_name=speaker,
                quote=quote,
                quote_score=round(score, 3),
                start_sec=start,
                fields={
                    "subject_id": target["id"],
                    "role": role,
                    "assumption": entry.get("assumption"),
                },
                extractor="directed.v1",
                model=result.model,
                embedding=vec,
            )
            subj.link(cfg.db_dsn, oid, target["id"], role=role, linked_by="directed", vector=vec)
            found.append(
                {
                    "observation_id": oid,
                    "subject_id": target["id"],
                    "role": role,
                    "assumption": entry.get("assumption"),
                    "text": text,
                }
            )
            if role == "resolves" and target["kind"] in ("question", "commitment"):
                new_status = "resolved" if target["kind"] == "question" else "done"
                subj.set_status(
                    cfg.db_dsn,
                    target["id"],
                    new_status,
                    reason=f"resolved by {result.source_ref}",
                    causes=[oid],
                )
    return found


async def after_observe(cfg: Any, result: Any, *, store: Any = None) -> dict[str, Any]:
    """Ledger, routing and directed evidence for one persisted run (best effort).

    With a vault ``store``, Goals.md threads are synced first so a new goal
    receives evidence from the very next meeting.
    """
    stats: dict[str, Any] = {"ledger": 0, "routed": 0, "directed": 0}
    live = [dict(o, source_ref=result.source_ref) for o in result.live]
    try:
        if store is not None:
            from pawn_agent.core.emergence.threads import sync_goals  # noqa: PLC0415

            await asyncio.to_thread(sync_goals, cfg, store)
        open_ledger = subj.list_subjects(
            cfg.db_dsn, kinds=LEDGER_TYPES, statuses=subj.OPEN_STATUSES
        )
        routed = await route(cfg, live)
        stats["routed"] = len(routed)
        for obs in live:
            ledger_id = await asyncio.to_thread(ensure_ledger, cfg, obs, open_ledger)
            if ledger_id:
                stats["ledger"] += 1
                parent = routed.get(obs["id"])
                if parent:
                    subj.update_subject(cfg.db_dsn, ledger_id, parent_id=parent)
        if _em(cfg, "directed", True):
            stats["directed"] = len(await directed_pass(cfg, result))
    except Exception as exc:
        logger.warning("emergence after_observe failed: %s", exc, exc_info=True)
        stats["error"] = str(exc)
    logger.info("emergence routing %s: %s", result.source_ref, stats)
    return stats
