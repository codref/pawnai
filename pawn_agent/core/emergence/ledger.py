"""Night jobs for the ledger: assumptions, commitments, predictions.

Also the deterministic meeting brief (open commitments with attendees,
their recent concerns, shared threads, decisions whose assumptions they
own) used by meeting prep.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

from pawn_agent.core.emergence import provocations
from pawn_agent.core.emergence import subjects as subj
from pawn_agent.core.emergence import vault_notes
from pawn_agent.core.emergence.llm import ask_json
from pawn_agent.core.emergence.route import parse_when

logger = logging.getLogger(__name__)


def _em(cfg: Any, name: str, default: Any) -> Any:
    return getattr(getattr(cfg, "emergence", None), name, default)


def _aliases(cfg: Any) -> list[str]:
    return [a.lower() for a in getattr(getattr(cfg, "coworker", None), "me", []) or [] if a]


def is_mine(cfg: Any, person: str) -> bool:
    who = (person or "").lower()
    return any(a in who for a in _aliases(cfg))


def _evidence_since(cfg: Any, subject: dict[str, Any]) -> list[dict[str, Any]]:
    """Evidence on the subject plus new observations on its parent thread."""
    since = subject.get("checked_at")
    own = subj.subject_observations(cfg.db_dsn, subject["id"], since=since, limit=30)
    own = [o for o in own if o.get("role") != "origin"]
    parent = []
    if subject.get("parent_id"):
        parent = subj.subject_observations(cfg.db_dsn, subject["parent_id"], since=since, limit=20)
    seen, out = set(), []
    for o in own + parent:
        if o["id"] not in seen:
            seen.add(o["id"])
            out.append(o)
    return out


_ASSUMPTION_PROMPT = """\
A manager made this decision:
{choice}

It rests on these assumptions:
{assumptions}

New evidence since it was last checked:
{evidence}

For each assumption judge: holds, weakened or broken, using only the
evidence. Return JSON:
{{"assumptions": [{{"index": 0, "state": "holds|weakened|broken", "why": "one line citing the evidence"}}]}}
Use "holds" when the evidence says nothing about an assumption.
"""


async def watch_assumptions(cfg: Any, store: Any, *, budget: Any = None) -> int:
    changed = 0
    for d in subj.list_subjects(cfg.db_dsn, kinds=("decision",), statuses=("open",)):
        if budget is not None and budget.exhausted:
            break
        fields = dict(d.get("fields") or {})
        review = d.get("review_at")
        if (
            isinstance(review, datetime)
            and review <= subj.now()
            and not fields.get("review_raised")
        ):
            provocations.add(
                cfg,
                kind="decision_review",
                subject_id=d["id"],
                text=f"Decision **{d['title']}** is due for review. Did it work out?",
                features={"goal_thread": 0.0},
            )
            fields["review_raised"] = True
            subj.update_subject(cfg.db_dsn, d["id"], fields=fields)
        assumptions = fields.get("assumptions") or []
        if not assumptions:
            continue
        evidence = _evidence_since(cfg, d)
        if not evidence:
            continue
        data = await ask_json(
            cfg,
            _ASSUMPTION_PROMPT.format(
                choice=fields.get("choice") or d["title"],
                assumptions="\n".join(f"{i}. {a}" for i, a in enumerate(assumptions)),
                evidence="\n".join(
                    f"- {o['text']}" + (f" ({o['speaker_name']})" if o.get("speaker_name") else "")
                    for o in evidence
                ),
            ),
            budget=budget,
        )
        verdicts = data.get("assumptions", []) if isinstance(data, dict) else []
        health = dict(fields.get("assumption_health") or {})
        alerts = []
        for entry in verdicts if isinstance(verdicts, list) else []:
            try:
                index = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            state = str(entry.get("state") or "").lower()
            if index >= len(assumptions) or state not in {"holds", "weakened", "broken"}:
                continue
            previous = (health.get(str(index)) or {}).get("state")
            health[str(index)] = {
                "state": state,
                "why": str(entry.get("why") or "")[:300],
                "at": subj.now().isoformat(),
            }
            if state != "holds" and state != previous:
                alerts.append((index, state, entry.get("why") or ""))
        fields["assumption_health"] = health
        subj.update_subject(cfg.db_dsn, d["id"], fields=fields, checked_at=subj.now())
        for index, state, why in alerts:
            subj.add_event(
                cfg.db_dsn,
                d["id"],
                f"assumption_{state}",
                {"assumption": assumptions[index], "why": why},
                [o["id"] for o in evidence],
            )
            provocations.add(
                cfg,
                kind=f"assumption_{state}",
                subject_id=d["id"],
                text=(
                    f"You decided **{d['title']}** assuming “{assumptions[index]}”. "
                    f"That assumption looks {state}: {why} Revisit?"
                ),
                features={
                    "tension": 1.0 if state == "broken" else 0.6,
                    "recurrence": float(len(evidence)),
                },
            )
            changed += 1
        vault_notes.render_subject(cfg, store, subj.get_subject(cfg.db_dsn, d["id"]) or d)
    return changed


def age_commitments(cfg: Any, store: Any, *, today: Optional[date] = None) -> dict[str, int]:
    """Flag overdue commitments (both directions) and rewrite Commitments.md."""
    today = today or date.today()
    stale_days = int(_em(cfg, "commitment_days", 7))
    counts = {"overdue": 0, "open": 0}
    rows = subj.list_subjects(cfg.db_dsn, kinds=("commitment",))
    for c in rows:
        if c["status"] not in subj.OPEN_STATUSES:
            continue
        counts["open"] += 1
        fields = dict(c.get("fields") or {})
        due = parse_when(str(fields.get("due") or ""), today=today)
        created = c.get("created_at")
        age = (today - created.date()).days if isinstance(created, datetime) else 0
        overdue = (due is not None and due < today) or (due is None and age > stale_days)
        if overdue and not fields.get("overdue"):
            fields["overdue"] = True
            subj.update_subject(cfg.db_dsn, c["id"], fields=fields)
            mine = is_mine(cfg, str(fields.get("debtor") or ""))
            who = fields.get("debtor") or "Someone"
            to = f" to {fields['creditor']}" if fields.get("creditor") else ""
            when = f"due {due:%Y-%m-%d}" if due else f"{age} days ago"
            provocations.add(
                cfg,
                kind="overdue_mine" if mine else "overdue_theirs",
                subject_id=c["id"],
                text=(
                    f"You promised{to}: **{fields.get('what') or c['title']}** ({when})."
                    if mine
                    else f"{who} owes{to}: **{fields.get('what') or c['title']}** ({when}). Chase it?"
                ),
                features={"tension": 0.5, "people_weight": 0.5},
            )
            counts["overdue"] += 1
        c["fields"] = fields
    vault_notes.render_commitments(cfg, store, rows)
    return counts


_PREDICTION_PROMPT = """\
Someone predicted: {claim}
Said by: {who}. Expected by: {by_when}.

Evidence gathered since:
{evidence}

Did it come true? Return JSON:
{{"outcome": "true|false|unclear", "why": "one line citing the evidence"}}
Answer "unclear" unless the evidence settles it.
"""


async def resolve_predictions(
    cfg: Any, store: Any, *, budget: Any = None, today: Optional[date] = None
) -> int:
    today = today or date.today()
    proposed = 0
    rows = subj.list_subjects(cfg.db_dsn, kinds=("prediction",))
    candidates = subj.recent_observations(cfg.db_dsn, days=180)
    for p in rows:
        if p["status"] not in ("open",):
            continue
        if budget is not None and budget.exhausted:
            break
        review = p.get("review_at")
        if not isinstance(review, datetime) or review.date() > today:
            continue
        fields = dict(p.get("fields") or {})
        own = [
            o for o in subj.subject_observations(cfg.db_dsn, p["id"]) if o.get("role") != "origin"
        ]
        created = p.get("created_at") or subj.now()
        near = [
            o
            for d, o in subj.nearest(p.get("centroid"), candidates, key="embedding")[:15]
            if d < 0.45 and o.get("created_at") and o["created_at"] > created
        ]
        evidence = own + near
        data = await ask_json(
            cfg,
            _PREDICTION_PROMPT.format(
                claim=fields.get("claim") or p["title"],
                who=fields.get("who") or fields.get("speaker") or "?",
                by_when=fields.get("by_when") or f"{review:%Y-%m-%d}",
                evidence="\n".join(f"- {o['text']}" for o in evidence) or "(none)",
            ),
            budget=budget,
        )
        outcome = str((data or {}).get("outcome") or "").lower() if isinstance(data, dict) else ""
        if outcome not in {"true", "false"}:
            fields["checked_unclear"] = subj.now().isoformat()
            subj.update_subject(
                cfg.db_dsn, p["id"], fields=fields, review_at=subj.now() + timedelta(days=30)
            )
            continue
        fields["proposed_resolution"] = outcome
        fields["resolution_why"] = str(data.get("why") or "")[:300]
        subj.update_subject(cfg.db_dsn, p["id"], fields=fields)
        subj.set_status(cfg.db_dsn, p["id"], "pending", reason=f"proposed {outcome}")
        who = fields.get("who") or fields.get("speaker") or "Someone"
        provocations.add(
            cfg,
            kind="prediction_resolved",
            subject_id=p["id"],
            text=(
                f"{who} predicted **{p['title']}**. It looks like it "
                f"{'came true' if outcome == 'true' else 'did not happen'}: {fields['resolution_why']} "
                "Reply `useful` to confirm or `wrong` to flip it."
            ),
            features={"novelty": 0.6, "people_weight": 0.3},
        )
        proposed += 1
    vault_notes.render_predictions(
        cfg, store, subj.list_subjects(cfg.db_dsn, kinds=("prediction",))
    )
    return proposed


def meeting_brief(cfg: Any, attendees: list[str], *, limit: int = 6) -> str:
    """Markdown brief for a meeting with ``attendees`` (no model calls)."""
    names = [a.strip() for a in attendees if a and a.strip()]
    if not names:
        return ""
    lowered = [n.lower().split("@")[0].replace(".", " ") for n in names]

    def involves(text: str) -> bool:
        t = (text or "").lower()
        return any(part and part.split()[0] in t for part in lowered)

    lines: list[str] = []
    commitments = subj.list_subjects(cfg.db_dsn, kinds=("commitment",), statuses=subj.OPEN_STATUSES)
    owed_by_them = [c for c in commitments if involves(str((c["fields"] or {}).get("debtor")))]
    owed_to_them = [
        c
        for c in commitments
        if involves(str((c["fields"] or {}).get("creditor")))
        and is_mine(cfg, str((c["fields"] or {}).get("debtor")))
    ]
    if owed_to_them:
        lines.append("**You owe them**")
        lines += [
            f"- {(c['fields'] or {}).get('what') or c['title']}" for c in owed_to_them[:limit]
        ]
    if owed_by_them:
        lines.append("**They owe**")
        lines += [
            f"- {(c['fields'] or {}).get('debtor')}: {(c['fields'] or {}).get('what') or c['title']}"
            for c in owed_by_them[:limit]
        ]
    concerns = [
        o
        for o in subj.recent_observations(
            cfg.db_dsn, days=45, types=("concern",), with_embedding=False
        )
        if involves(o.get("speaker_name") or "")
    ]
    if concerns:
        lines.append("**Their recent concerns**")
        lines += [f"- {o['speaker_name']}: {o['text']}" for o in concerns[-limit:]]
    decisions = [
        d
        for d in subj.list_subjects(cfg.db_dsn, kinds=("decision",), statuses=("open",))
        if involves(str((d["fields"] or {}).get("owner")))
        or involves(str((d["fields"] or {}).get("speaker")))
    ]
    shaky = [d for d in decisions if (vault_notes._health(d) in {"weakened", "broken"})]
    if shaky:
        lines.append("**Decisions with shaky assumptions**")
        lines += [f"- {d['title']} ({vault_notes._health(d)})" for d in shaky[:limit]]
    return "\n".join(lines)
