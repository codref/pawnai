"""Vault notes for subjects: threads, questions, decisions, ledger, retros.

Notes are rendered from the DB. Managed ``##`` sections are replaced; any
other section the user writes is kept. The frontmatter ``status`` is the
user's control: when it differs from the DB on the next sync, the note wins
(see ``threads.sync_notes``).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Optional

from pawn_agent.core.coworker.notes import format_offset
from pawn_agent.core.emergence import subjects as subj

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"(?m)^## (.+?)\s*$")


def root(cfg: Any) -> str:
    return str(getattr(getattr(cfg, "emergence", None), "vault_root", "Pawn") or "Pawn").strip("/")


def subject_key(cfg: Any, subject: dict[str, Any]) -> str:
    kind, slug = subject["kind"], subject["slug"]
    base = root(cfg)
    if kind == "thread":
        threads_dir = getattr(getattr(cfg, "coworker", None), "threads_dir", f"{base}/Threads")
        return f"{threads_dir.strip('/')}/{slug}.md"
    if kind == "question":
        return f"{base}/Questions/{slug}.md"
    if kind == "decision":
        created = subject.get("created_at") or subj.now()
        return f"{base}/Decisions/{created:%Y-%m-%d}-{slug}.md"
    if kind == "hypothesis":
        return f"{base}/Hypotheses/{slug}.md"
    return f"{base}/{kind.title()}s/{slug}.md"


def split_sections(body: str) -> tuple[str, list[tuple[str, str]]]:
    """``(preamble, [(heading, text), …])`` for a note body."""
    matches = list(_SECTION_RE.finditer(body or ""))
    if not matches:
        return body or "", []
    preamble = body[: matches[0].start()]
    sections = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections.append((match.group(1).strip(), body[match.end() : end].strip("\n")))
    return preamble, sections


def merge_note(
    existing: str,
    *,
    meta: dict[str, Any],
    title: str,
    managed: list[tuple[str, str]],
) -> str:
    """Managed sections first (replaced), then the user's own sections."""
    from pawn_core.vault import dump_frontmatter, parse_frontmatter  # noqa: PLC0415

    old_meta: dict[str, Any] = {}
    body = ""
    if existing:
        try:
            old_meta, body = parse_frontmatter(existing)
        except Exception:
            body = existing
    merged_meta = {**old_meta, **meta}
    _, old_sections = split_sections(body)
    managed_names = {name for name, _ in managed}
    kept = [(name, text) for name, text in old_sections if name not in managed_names]
    parts = [f"# {title}", ""]
    for name, text in managed + kept:
        parts.append(f"## {name}")
        parts.append("")
        parts.append(text.strip() or "_(nothing yet)_")
        parts.append("")
    return dump_frontmatter(merged_meta, "\n".join(parts).rstrip() + "\n")


class SourceLinks:
    """Cached ``[[transcript]]`` links per source ref."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.cache: dict[str, str] = {}

    def __call__(self, kind: str, ref: str) -> str:
        if not ref:
            return ""
        if ref not in self.cache:
            link = ""
            if kind == "session":
                try:
                    from pawn_agent.core.coworker.pipeline import _source_link  # noqa: PLC0415

                    link = _source_link(self.cfg, ref)
                except Exception:
                    link = ""
            self.cache[ref] = link or f"`{ref}`"
        return self.cache[ref]


def dedupe(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per statement (the same claim is often both a concern and a fact)."""
    seen: set[tuple[str, str, int]] = set()
    out = []
    for o in observations:
        key = (
            " ".join((o.get("text") or "").lower().split()),
            o.get("source_ref") or "",
            int(o.get("start_sec") or -1),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def timeline_line(obs: dict[str, Any], links: SourceLinks) -> str:
    when = obs.get("created_at")
    day = when.strftime("%Y-%m-%d") if isinstance(when, datetime) else ""
    speaker = f"{obs['speaker_name']}: " if obs.get("speaker_name") else ""
    at = format_offset(obs.get("start_sec"))
    where = links(obs.get("source_kind") or "", obs.get("source_ref") or "")
    where = f" · {where}" + (f" at {at}" if at else "") if where else ""
    role = obs.get("role") or ""
    tag = "" if role in {"", "mentions", "origin"} else f" *({role.replace('_', ' ')})*"
    return f"- {day} · {speaker}{obs['text']}{tag}{where}"


def _write(store: Any, key: str, text: str) -> None:
    store.write(key, text)


def _read(store: Any, key: str) -> str:
    try:
        return store.read(key) if store.exists(key) else ""
    except Exception:
        return ""


def _health(subject: dict[str, Any]) -> str:
    health = (subject.get("fields") or {}).get("assumption_health") or {}
    if not health:
        return ""
    worst = "holds"
    for value in health.values():
        state = value.get("state") if isinstance(value, dict) else str(value)
        if state == "broken":
            return "broken"
        if state == "weakened":
            worst = "weakened"
    return worst


def render_subject(
    cfg: Any,
    store: Any,
    subject: dict[str, Any],
    *,
    links: Optional[SourceLinks] = None,
) -> str:
    """Write the subject's note; returns its key."""
    links = links or SourceLinks(cfg)
    key = subject.get("note_key") or subject_key(cfg, subject)
    observations = subj.subject_observations(cfg.db_dsn, subject["id"], limit=60)
    fields = subject.get("fields") or {}
    meta: dict[str, Any] = {
        "pawn": subject["kind"],
        "id": subject["id"],
        "origin": subject["origin"],
        "status": subject["status"],
    }
    if subject.get("momentum") is not None:
        meta["momentum"] = round(float(subject["momentum"]), 2)
    people = sorted({o["speaker_name"] for o in observations if o.get("speaker_name")})
    if people:
        meta["people"] = people[:12]
    state = subject.get("state_md") or ""
    managed: list[tuple[str, str]] = []
    timeline = "\n".join(timeline_line(o, links) for o in dedupe(observations)[-40:])

    if subject["kind"] == "thread":
        children = subj.list_subjects(cfg.db_dsn, kinds=("question", "decision", "commitment"))
        mine = [c for c in children if c.get("parent_id") == subject["id"]]
        questions = [
            c for c in mine if c["kind"] == "question" and c["status"] in subj.OPEN_STATUSES
        ]
        decisions = [c for c in mine if c["kind"] == "decision"]
        managed = [
            ("State", state),
            ("Open questions", "\n".join(f"- {q['title']}" for q in questions)),
            (
                "Decisions",
                "\n".join(
                    f"- {d['title']}" + (f" — assumptions {_health(d)}" if _health(d) else "")
                    for d in decisions[:20]
                ),
            ),
            ("Timeline", timeline),
        ]
    elif subject["kind"] == "question":
        for_lines = [o for o in observations if o.get("role") == "evidence_for"]
        against = [o for o in observations if o.get("role") == "evidence_against"]
        managed = [
            ("State", state),
            (
                "Evidence",
                f"{len(for_lines)} for, {len(against)} against.\n\n"
                + "\n".join(timeline_line(o, links) for o in for_lines + against),
            ),
            ("Timeline", timeline),
        ]
    elif subject["kind"] == "decision":
        health = fields.get("assumption_health") or {}
        assumptions = []
        for index, text in enumerate(fields.get("assumptions") or []):
            info = health.get(str(index)) or {}
            mark = f" — **{info.get('state')}**" if info.get("state") else ""
            why = f": {info.get('why')}" if info.get("why") else ""
            assumptions.append(f"- {text}{mark}{why}")
        review = subject.get("review_at")
        managed = [
            (
                "Decision",
                "\n".join(
                    line
                    for line in (
                        fields.get("choice") or subject["title"],
                        f"Owner: {fields.get('owner')}" if fields.get("owner") else "",
                        f"Why: {fields.get('rationale')}" if fields.get("rationale") else "",
                        f"Review: {review:%Y-%m-%d}" if isinstance(review, datetime) else "",
                        (
                            f"Council: [[{fields['council_note']}]]"
                            if fields.get("council_note")
                            else ""
                        ),
                    )
                    if line
                ),
            ),
            ("Assumptions", "\n".join(assumptions)),
            ("Timeline", timeline),
        ]
        meta["review"] = f"{review:%Y-%m-%d}" if isinstance(review, datetime) else ""
        meta["council"] = bool(fields.get("council"))
    elif subject["kind"] == "hypothesis":
        managed = [
            ("Connection", state or subject["title"]),
            (
                "Evidence",
                "\n".join(f"- {line}" for line in fields.get("evidence") or []) or timeline,
            ),
            (
                "How to use",
                "Set `status: active` to keep watching this, or `status: dismissed` to drop it.",
            ),
        ]
    else:
        managed = [("State", state), ("Timeline", timeline)]

    text = merge_note(_read(store, key), meta=meta, title=subject["title"], managed=managed)
    _write(store, key, text)
    # Remember what the note says so sync can tell a user edit from a DB change.
    stored = dict((subj.get_subject(cfg.db_dsn, subject["id"]) or subject).get("fields") or {})
    stored["_note_status"] = subject["status"]
    subj.update_subject(cfg.db_dsn, subject["id"], note_key=key, fields=stored)
    return key


def render_commitments(cfg: Any, store: Any, rows: list[dict[str, Any]]) -> str:
    """``Pawn/Commitments.md``: mine / theirs / overdue / done recently."""
    key = f"{root(cfg)}/Commitments.md"
    aliases = [a.lower() for a in getattr(getattr(cfg, "coworker", None), "me", []) or []]

    def line(row: dict[str, Any]) -> str:
        f = row.get("fields") or {}
        due = f" — due {f['due']}" if f.get("due") else ""
        to = f" → {f['creditor']}" if f.get("creditor") else ""
        age = row.get("created_at")
        since = f" (since {age:%Y-%m-%d})" if isinstance(age, datetime) else ""
        return f"- **{f.get('debtor') or '?'}**{to}: {f.get('what') or row['title']}{due}{since}"

    def is_mine(row: dict[str, Any]) -> bool:
        debtor = str((row.get("fields") or {}).get("debtor") or "").lower()
        return any(a and a in debtor for a in aliases)

    open_rows = [r for r in rows if r["status"] in subj.OPEN_STATUSES]
    overdue = [r for r in open_rows if (r.get("fields") or {}).get("overdue")]
    done = [r for r in rows if r["status"] == "done"][:20]
    managed = [
        ("Overdue", "\n".join(line(r) for r in overdue)),
        ("I owe", "\n".join(line(r) for r in open_rows if is_mine(r))),
        ("Owed to me / others", "\n".join(line(r) for r in open_rows if not is_mine(r))),
        ("Done recently", "\n".join(line(r) for r in done)),
    ]
    text = merge_note(
        _read(store, key), meta={"pawn": "commitments"}, title="Commitments", managed=managed
    )
    _write(store, key, text)
    return key


def render_predictions(cfg: Any, store: Any, rows: list[dict[str, Any]]) -> str:
    """``Pawn/Predictions.md`` with per-person hit rate on confirmed outcomes."""
    key = f"{root(cfg)}/Predictions.md"
    open_lines, resolved_lines = [], []
    tally: dict[str, list[int]] = {}
    for row in rows:
        f = row.get("fields") or {}
        who = f.get("who") or "?"
        by = f" by {f['by_when']}" if f.get("by_when") else ""
        outcome = f.get("confirmed") or ""
        if outcome in {"true", "false"}:
            hit = 1 if outcome == "true" else 0
            tally.setdefault(who, []).append(hit)
            resolved_lines.append(f"- {'✓' if hit else '✗'} **{who}**: {row['title']}{by}")
        elif f.get("proposed_resolution"):
            resolved_lines.append(
                f"- ? **{who}**: {row['title']}{by} — Pawn thinks "
                f"**{f['proposed_resolution']}** (confirm in the brief)"
            )
        else:
            open_lines.append(f"- **{who}**: {row['title']}{by}")
    calibration = "\n".join(
        f"- {who}: {sum(h)}/{len(h)} came true ({round(100 * sum(h) / len(h))}%)"
        for who, h in sorted(tally.items())
    )
    managed = [
        ("Calibration", calibration),
        ("Open", "\n".join(open_lines)),
        ("Resolved", "\n".join(resolved_lines)),
    ]
    text = merge_note(
        _read(store, key), meta={"pawn": "predictions"}, title="Predictions", managed=managed
    )
    _write(store, key, text)
    return key


def write_plain(
    cfg: Any,
    store: Any,
    key: str,
    *,
    meta: dict[str, Any],
    title: str,
    sections: list[tuple[str, str]],
) -> str:
    text = merge_note(_read(store, key), meta=meta, title=title, managed=sections)
    _write(store, key, text)
    return key
