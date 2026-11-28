"""Reflection jobs: contradictions, people digests, retrospectives.

Retrospectives are written from ``em_subject_events`` (the subject history)
plus the priority mirror: observations per goal thread versus Goals.md.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Optional

from pawn_agent.core.emergence import provocations
from pawn_agent.core.emergence import subjects as subj
from pawn_agent.core.emergence import vault_notes
from pawn_agent.core.emergence.llm import ask_json

logger = logging.getLogger(__name__)


def _em(cfg: Any, name: str, default: Any) -> Any:
    return getattr(getattr(cfg, "emergence", None), name, default)


_CONTRA_PROMPT = """\
Storyline: {title}
Current summary:
{state}

New observations:
{observations}

List statements that contradict each other or the summary (two people
describing the same thing differently, a position that changed, numbers
that do not match). Return JSON:
{{"contradictions": [{{"a": "...", "b": "...", "why": "one line"}}]}}
Use an empty list when there are none. Differences of opinion are not
contradictions unless they are about facts.
"""


async def scan_contradictions(cfg: Any, *, budget: Any = None) -> int:
    found = 0
    for t in subj.list_subjects(cfg.db_dsn, kinds=("thread",), statuses=("active",)):
        if budget is not None and budget.exhausted:
            break
        fields = dict(t.get("fields") or {})
        since = fields.get("_contra_at")
        since_dt = datetime.fromisoformat(since) if since else None
        new = subj.subject_observations(cfg.db_dsn, t["id"], since=since_dt, limit=25)
        fields["_contra_at"] = subj.now().isoformat()
        subj.update_subject(cfg.db_dsn, t["id"], fields=fields)
        if len(new) < 2:
            continue
        data = await ask_json(
            cfg,
            _CONTRA_PROMPT.format(
                title=t["title"],
                state=t.get("state_md") or "(none)",
                observations="\n".join(
                    f"- {o['text']}" + (f" ({o['speaker_name']})" if o.get("speaker_name") else "")
                    for o in new
                ),
            ),
            budget=budget,
        )
        items = data.get("contradictions", []) if isinstance(data, dict) else []
        for c in items if isinstance(items, list) else []:
            a, b = str(c.get("a") or "").strip(), str(c.get("b") or "").strip()
            if not a or not b:
                continue
            subj.add_event(
                cfg.db_dsn, t["id"], "contradiction", {"a": a, "b": b, "why": c.get("why")}
            )
            provocations.add(
                cfg,
                kind="contradiction",
                subject_id=t["id"],
                text=f"On **{t['title']}**: “{a}” vs “{b}”. {c.get('why') or ''} Which is it?",
                features={"tension": 0.8, "goal_thread": 1.0 if t["origin"] == "goal" else 0.0},
            )
            found += 1
    return found


# ── people ───────────────────────────────────────────────────────────────────


def _gallery_names(cfg: Any) -> dict[str, str]:
    """lowercased display name / alias → speaker id."""
    try:
        from sqlalchemy import select  # noqa: PLC0415

        from pawn_core.database import Speaker, _get_session  # noqa: PLC0415
    except Exception:
        return {}
    out: dict[str, str] = {}
    try:
        with _get_session(cfg.db_dsn) as db:
            for sp in db.scalars(select(Speaker).where(Speaker.active.is_(True))).all():
                out[sp.display_name.lower()] = sp.id
                for alias in sp.aliases or []:
                    out[str(alias).lower()] = sp.id
    except Exception as exc:
        logger.debug("speakers gallery unavailable: %s", exc)
    return out


_RAW_LABEL = re.compile(r"^(speaker|spk)[ _-]?\d+$", re.IGNORECASE)


def refresh_people(cfg: Any, store: Any, *, days: int = 60) -> int:
    """Write ``Pawn/People/{name}.md`` digests; People/ open loops when allowed."""
    observations = subj.recent_observations(cfg.db_dsn, days=days, with_embedding=False)
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for o in observations:
        name = (o.get("speaker_name") or "").strip()
        # Unmapped diarization labels are not people yet.
        if name and not _RAW_LABEL.match(name):
            by_person[name].append(o)
    commitments = subj.list_subjects(cfg.db_dsn, kinds=("commitment",), statuses=subj.OPEN_STATUSES)
    gallery = _gallery_names(cfg)
    allow_people_write = False
    try:
        from pawn_agent.core.coworker.autonomy import decide  # noqa: PLC0415

        allow_people_write = decide(cfg, "people_refresh").decision == "allow"
    except Exception:
        allow_people_write = False
    written = 0
    links = vault_notes.SourceLinks(cfg)
    for name, rows in sorted(by_person.items()):
        concerns = [o for o in rows if o["type"] == "concern"]
        topics = Counter(o["type"] for o in rows)
        owes = [
            c
            for c in commitments
            if name.lower() in str((c["fields"] or {}).get("debtor", "")).lower()
        ]
        owed = [
            c
            for c in commitments
            if name.lower() in str((c["fields"] or {}).get("creditor", "")).lower()
        ]
        sources = sorted({o["source_ref"] for o in rows if o.get("source_ref")})
        sid = gallery.get(name.lower())
        meta = {"pawn": "person-digest", "person": name}
        if sid:
            meta["speaker_id"] = sid
        sections = [
            (
                "Recurring concerns",
                "\n".join(vault_notes.timeline_line(o, links) for o in concerns[-12:]),
            ),
            ("Owes", "\n".join(f"- {(c['fields'] or {}).get('what') or c['title']}" for c in owes)),
            (
                "Is owed",
                "\n".join(f"- {(c['fields'] or {}).get('what') or c['title']}" for c in owed),
            ),
            (
                "Seen in",
                f"{len(sources)} sources in the last {days} days; "
                + ", ".join(f"{k} {v}" for k, v in topics.most_common()),
            ),
        ]
        if sid:
            sections.append(("Bio", f"[[{getattr(cfg.coworker, 'people_dir', 'People')}/{sid}]]"))
        key = f"{vault_notes.root(cfg)}/People/{subj.slugify(name)}.md"
        vault_notes.write_plain(cfg, store, key, meta=meta, title=name, sections=sections)
        written += 1
        if sid and allow_people_write and owes:
            try:
                from pawn_agent.core.people.notes import update_person_note  # noqa: PLC0415

                update_person_note(
                    cfg,
                    speaker_id=sid,
                    add_open_loops=[(c["fields"] or {}).get("what") or c["title"] for c in owes],
                    store=store,
                )
            except Exception as exc:
                logger.warning("people note %s not updated: %s", sid, exc)
        concern_sources = {c["source_ref"] for c in concerns}
        if len(concerns) >= 3 and len(concern_sources) >= 2:
            provocations.add(
                cfg,
                kind="recurring_concern",
                subject_id=None,
                text=(
                    f"{name} has raised {len(concerns)} concerns across "
                    f"{len(concern_sources)} conversations lately (latest: "
                    f"“{concerns[-1]['text']}”). On your radar?"
                ),
                features={"people_weight": 1.0, "recurrence": float(len(concerns))},
            )
    return written


# ── retrospectives ───────────────────────────────────────────────────────────


def period_for(kind: str, today: date) -> tuple[str, date, date]:
    """The most recently *completed* period: ``(label, start, end_exclusive)``."""
    if kind == "week":
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=7)
        iso = start.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}", start, end
    if kind == "month":
        end = today.replace(day=1)
        start = (end - timedelta(days=1)).replace(day=1)
        return f"{start:%Y-%m}", start, end
    q_start_month = ((today.month - 1) // 3) * 3 + 1
    end = today.replace(month=q_start_month, day=1)
    prev_end = end - timedelta(days=1)
    start = prev_end.replace(month=((prev_end.month - 1) // 3) * 3 + 1, day=1)
    return f"{start.year}-Q{(start.month - 1) // 3 + 1}", start, end


def priority_mirror(cfg: Any, store: Any, *, start: date, end: date) -> str:
    """Goals.md order versus where observations actually landed."""
    from pawn_core.goals import load_goals_text  # noqa: PLC0415

    goals_path = getattr(getattr(cfg, "coworker", None), "goals_path", "Goals.md")
    try:
        goals = load_goals_text(store.read(goals_path) if store.exists(goals_path) else "")
    except Exception:
        goals = load_goals_text("")
    threads = {s["slug"]: s for s in subj.list_subjects(cfg.db_dsn, kinds=("thread",))}
    counts: dict[str, int] = {}
    total = 0
    for slug, t in threads.items():
        rows = [
            o
            for o in subj.subject_observations(cfg.db_dsn, t["id"], limit=500)
            if o.get("created_at") and start <= o["created_at"].date() < end
        ]
        counts[slug] = len(rows)
        total += len(rows)
    lines = ["| Goal (your order) | Share of attention |", "|---|---|"]
    for g in goals.active:
        n = counts.get(g.slug, 0)
        lines.append(f"| {g.name} | {round(100 * n / total) if total else 0}% ({n}) |")
    others = sorted(
        (
            (t["title"], counts.get(slug, 0))
            for slug, t in threads.items()
            if t["origin"] != "goal" and counts.get(slug, 0)
        ),
        key=lambda pair: -pair[1],
    )[:5]
    for title, n in others:
        lines.append(
            f"| *(not a goal)* {title} | {round(100 * n / total) if total else 0}% ({n}) |"
        )
    return "\n".join(lines) if total else "_No routed observations in this period._"


_RETRO_PROMPT = """\
Write a {kind}ly retrospective for a manager covering {label}.

What changed in their world (subject history):
{events}

Open decisions and their assumption health:
{decisions}

Calibration (confirmed predictions): {calibration}

Return JSON:
{{"summary": "4-8 lines: the shape of the period",
  "worked": ["decisions or moves that aged well"],
  "didnt": ["ones that aged badly or stalled"],
  "watch": ["what to watch next period"],
  "worry_then": "what they were worried about at the start and whether it happened"}}
Only use the material above.
"""


async def write_retro(
    cfg: Any,
    store: Any,
    kind: str,
    *,
    today: Optional[date] = None,
    budget: Any = None,
    force: bool = False,
) -> Optional[str]:
    """Write ``Pawn/Retros/{label}.md`` for the last completed period."""
    today = today or date.today()
    label, start, end = period_for(kind, today)
    key = f"{vault_notes.root(cfg)}/Retros/{label}.md"
    if not force and store.exists(key):
        return None
    since = datetime.combine(start, datetime.min.time())
    until = datetime.combine(end, datetime.min.time())
    events = [e for e in subj.events_since(cfg.db_dsn, since) if e["at"] < until]
    if not events and not force:
        return None
    lines = []
    for e in events[:120]:
        diff = e["diff"]
        detail = (
            diff.get("after")
            if e["kind"] == "status_changed"
            else diff.get("why") or diff.get("title") or ""
        )
        lines.append(
            f"- {e['at']:%Y-%m-%d} [{e['subject_kind']}] {e['title']}: {e['kind']} {str(detail)[:160]}"
        )
    decisions = [
        f"- {d['title']} ({vault_notes._health(d) or 'unchecked'})"
        for d in subj.list_subjects(cfg.db_dsn, kinds=("decision",))[:30]
    ]
    preds = [
        p
        for p in subj.list_subjects(cfg.db_dsn, kinds=("prediction",))
        if (p.get("fields") or {}).get("confirmed") in {"true", "false"}
    ]
    hits = sum(1 for p in preds if p["fields"]["confirmed"] == "true")
    calibration = f"{hits}/{len(preds)} came true" if preds else "no confirmed predictions"
    data = await ask_json(
        cfg,
        _RETRO_PROMPT.format(
            kind=kind,
            label=label,
            events="\n".join(lines) or "(none)",
            decisions="\n".join(decisions) or "(none)",
            calibration=calibration,
        ),
        budget=budget,
    )
    data = data if isinstance(data, dict) else {}

    def bullets(value: Any) -> str:
        return "\n".join(f"- {v}" for v in value) if isinstance(value, list) else str(value or "")

    sections = [
        ("Summary", str(data.get("summary") or "")),
        ("Aged well", bullets(data.get("worked"))),
        ("Aged badly or stalled", bullets(data.get("didnt"))),
        ("Watch next", bullets(data.get("watch"))),
        ("What you worried about then", str(data.get("worry_then") or "")),
        ("Priority mirror", priority_mirror(cfg, store, start=start, end=end)),
        ("Calibration", calibration),
        ("History", "\n".join(lines)),
    ]
    vault_notes.write_plain(
        cfg,
        store,
        key,
        meta={
            "pawn": "retro",
            "period": label,
            "kind": kind,
            "from": start.isoformat(),
            "to": (end - timedelta(days=1)).isoformat(),
        },
        title=f"Retrospective {label}",
        sections=sections,
    )
    return key


async def retros_due(
    cfg: Any, store: Any, *, today: Optional[date] = None, budget: Any = None
) -> list[str]:
    """Weekly every Monday-or-later, monthly and quarterly once their period ends."""
    written = []
    for kind in ("week", "month", "quarter"):
        try:
            key = await write_retro(cfg, store, kind, today=today, budget=budget)
        except Exception as exc:
            logger.warning("retro %s failed: %s", kind, exc)
            key = None
        if key:
            written.append(key)
    return written
