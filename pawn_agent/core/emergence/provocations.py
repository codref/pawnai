"""Provocations: the 1-3 questions surfaced in the morning brief.

Night jobs add candidates; the briefing picks the top few, marks them shown
and writes them at the top of the daily note. Verdicts (``useful``,
``wrong``, ``not_now``) come from chat/Matrix (``useful p3a1``), the API,
or the CLI; they act on the subject (accept a proposed thread, confirm a
prediction) and train the provocation scorer.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
from sqlalchemy import select

from pawn_agent.core.emergence import subjects as subj
from pawn_agent.utils.db import EmProvocation, _get_session

logger = logging.getLogger(__name__)

KIND_WEIGHT: dict[str, float] = {
    "assumption_broken": 0.9,
    "overdue_mine": 0.75,
    "assumption_weakened": 0.7,
    "contradiction": 0.65,
    "emergent_thread": 0.6,
    "prediction_resolved": 0.55,
    "decision_review": 0.5,
    "question_answered": 0.5,
    "recurring_concern": 0.5,
    "momentum": 0.45,
    "overdue_theirs": 0.45,
    "hypothesis": 0.4,
}
FEATURES = ("goal_thread", "tension", "momentum", "novelty", "recurrence", "people_weight")
VERDICTS = ("useful", "wrong", "not_now")
_VERDICT_RE = re.compile(
    r"^\s*(useful|wrong|not[ _-]?now|later)\s+(p[0-9a-f]{3,8})\s*$", re.IGNORECASE
)
_REPEAT_DAYS = 14


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def fingerprint(kind: str, subject_id: Optional[str], text: str) -> str:
    base = f"{kind}|{subject_id or ''}|{'' if subject_id else text.lower()}"
    return hashlib.sha256(base.encode()).hexdigest()[:24]


def _short_id(db: Any) -> str:
    while True:
        candidate = "p" + secrets.token_hex(2)
        if (
            db.scalars(select(EmProvocation).where(EmProvocation.short_id == candidate)).first()
            is None
        ):
            return candidate


def _dict(row: EmProvocation) -> dict[str, Any]:
    return {
        "id": row.id,
        "short_id": row.short_id,
        "day": row.day,
        "subject_id": row.subject_id,
        "kind": row.kind,
        "text": row.text,
        "score": row.score,
        "features": dict(row.features or {}),
        "status": row.status,
        "verdict": row.verdict,
        "shown_at": row.shown_at.isoformat() if row.shown_at else None,
    }


def add(
    cfg: Any,
    *,
    kind: str,
    text: str,
    subject_id: Optional[str] = None,
    features: Optional[dict[str, float]] = None,
    day: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Add a candidate unless the same one was raised in the last two weeks."""
    fp = fingerprint(kind, subject_id, text)
    since = _now() - timedelta(days=_REPEAT_DAYS)
    with _get_session(cfg.db_dsn) as db:
        recent = db.scalars(
            select(EmProvocation).where(
                EmProvocation.fingerprint == fp, EmProvocation.created_at >= since
            )
        ).first()
        if recent is not None and recent.verdict != "not_now":
            return None
        row = EmProvocation(
            id=str(uuid.uuid4()),
            short_id=_short_id(db),
            day=day or (date.today() + timedelta(days=1)).isoformat(),
            subject_id=subject_id,
            kind=kind,
            text=text.strip(),
            features={
                k: float(v) for k, v in (features or {}).items() if isinstance(v, (int, float))
            },
            fingerprint=fp,
            status="candidate",
            created_at=_now(),
        )
        db.add(row)
        db.flush()
        return _dict(row)


def _vector(kind: str, features: dict[str, float]) -> np.ndarray:
    kinds = list(KIND_WEIGHT)
    onehot = [1.0 if kind == k else 0.0 for k in kinds]
    feats = [min(3.0, float(features.get(f, 0.0) or 0.0)) for f in FEATURES]
    return np.array(onehot + feats + [1.0])


def _learned_weights(cfg: Any) -> Optional[np.ndarray]:
    """Logistic regression on past verdicts (useful=1, wrong=0); needs 20."""
    from pawn_agent.core.emergence.ranker import fit  # noqa: PLC0415

    with _get_session(cfg.db_dsn) as db:
        rows = db.scalars(
            select(EmProvocation).where(EmProvocation.verdict.in_(("useful", "wrong")))
        ).all()
        data = [(r.kind, dict(r.features or {}), r.verdict) for r in rows]
    if len(data) < 20 or len({v for *_, v in data}) < 2:
        return None
    x = np.stack([_vector(k, f) for k, f, _ in data])
    y = np.array([1.0 if v == "useful" else 0.0 for *_, v in data])
    w, b = fit(x, y)
    return np.append(w, b)


def score(kind: str, features: dict[str, float], weights: Optional[np.ndarray] = None) -> float:
    if weights is not None:
        x = _vector(kind, features)
        z = float(x @ weights[:-1] + weights[-1])
        return round(float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))), 3)
    base = KIND_WEIGHT.get(kind, 0.3)
    bonus = (
        0.25 * float(features.get("goal_thread", 0.0))
        + 0.15 * float(features.get("tension", 0.0))
        + 0.05 * min(3.0, float(features.get("momentum", 0.0)))
        + 0.05 * min(3.0, float(features.get("recurrence", 0.0)))
        + 0.1 * float(features.get("people_weight", 0.0))
    )
    return round(min(1.5, base + bonus), 3)


def select_for_day(cfg: Any, day: Optional[str] = None) -> list[dict[str, Any]]:
    """Pick and mark today's provocations (idempotent for the same day)."""
    day = day or date.today().isoformat()
    limit = int(getattr(getattr(cfg, "emergence", None), "brief_max_per_day", 3) or 3)
    weights = _learned_weights(cfg)
    with _get_session(cfg.db_dsn) as db:
        shown = db.scalars(
            select(EmProvocation).where(EmProvocation.day == day, EmProvocation.status == "shown")
        ).all()
        if shown:
            return [_dict(r) for r in shown]
        candidates = db.scalars(
            select(EmProvocation).where(
                EmProvocation.status == "candidate", EmProvocation.day <= day
            )
        ).all()
        recent_subjects = {
            r.subject_id
            for r in db.scalars(
                select(EmProvocation).where(
                    EmProvocation.status == "shown",
                    EmProvocation.shown_at >= _now() - timedelta(days=7),
                )
            ).all()
            if r.subject_id
        }
        for row in candidates:
            s = score(row.kind, dict(row.features or {}), weights)
            if row.subject_id and row.subject_id in recent_subjects:
                s -= 0.4
            row.score = s
        ranked = sorted(candidates, key=lambda r: r.score or 0.0, reverse=True)
        picked: list[EmProvocation] = []
        seen_subjects: set[Optional[str]] = set()
        for row in ranked:
            if len(picked) >= limit:
                break
            if row.subject_id and row.subject_id in seen_subjects:
                continue
            row.status, row.day, row.shown_at = "shown", day, _now()
            seen_subjects.add(row.subject_id)
            picked.append(row)
        stale = _now() - timedelta(days=_REPEAT_DAYS)
        for row in candidates:
            if row.status == "candidate" and row.created_at and row.created_at < stale:
                row.status = "skipped"
        db.flush()
        return [_dict(r) for r in picked]


def render_section(provocations: list[dict[str, Any]]) -> list[str]:
    if not provocations:
        return []
    lines = ["## Provocations", ""]
    for p in provocations:
        lines.append(f"- {p['text']} `{p['short_id']}`")
    lines.append("")
    lines.append("_Reply `useful <id>`, `wrong <id>` or `not now <id>` in chat or Matrix._")
    lines.append("")
    return lines


def parse_verdict(text: str) -> Optional[tuple[str, str]]:
    """``useful p3a1`` → ``("useful", "p3a1")``."""
    match = _VERDICT_RE.match(text or "")
    if not match:
        return None
    raw = match.group(1).lower().replace(" ", "_").replace("-", "_")
    verdict = "not_now" if raw in {"notnow", "not_now", "later"} else raw
    return verdict, match.group(2).lower()


def apply_verdict(cfg: Any, short_id: str, verdict: str) -> str:
    """Record a verdict and act on the subject. Returns a one-line receipt."""
    verdict = verdict.strip().lower().replace(" ", "_")
    if verdict not in VERDICTS:
        return f"Unknown verdict {verdict!r}."
    with _get_session(cfg.db_dsn) as db:
        row = db.scalars(select(EmProvocation).where(EmProvocation.short_id == short_id)).first()
        if row is None:
            return f"No provocation {short_id}."
        row.verdict, row.verdict_at = verdict, _now()
        if verdict == "not_now":
            row.status = "candidate"
            row.day = (date.today() + timedelta(days=3)).isoformat()
        kind, subject_id = row.kind, row.subject_id
    effect = _act(cfg, kind, subject_id, verdict)
    return f"Noted {verdict.replace('_', ' ')} for {short_id}.{(' ' + effect) if effect else ''}"


def _act(cfg: Any, kind: str, subject_id: Optional[str], verdict: str) -> str:
    if not subject_id or verdict == "not_now":
        return ""
    s = subj.get_subject(cfg.db_dsn, subject_id)
    if s is None:
        return ""
    if kind in {"emergent_thread", "hypothesis"} and s["status"] == "proposed":
        status = "active" if verdict == "useful" else "dismissed"
        subj.set_status(cfg.db_dsn, subject_id, status, reason=f"provocation {verdict}")
        return f"{s['kind'].title()} {'now tracked' if status == 'active' else 'dropped'}."
    if kind == "prediction_resolved":
        fields = dict(s.get("fields") or {})
        proposed = str(fields.get("proposed_resolution") or "")
        if proposed in {"true", "false"}:
            confirmed = (
                proposed if verdict == "useful" else ("false" if proposed == "true" else "true")
            )
            fields["confirmed"] = confirmed
            subj.update_subject(cfg.db_dsn, subject_id, fields=fields)
            subj.set_status(cfg.db_dsn, subject_id, "resolved", reason=f"confirmed {confirmed}")
            return f"Prediction recorded as {'came true' if confirmed == 'true' else 'did not happen'}."
    if kind == "question_answered" and verdict == "wrong":
        subj.set_status(cfg.db_dsn, subject_id, "active", reason="answer rejected")
        return "Question reopened."
    if kind in {"overdue_mine", "overdue_theirs"} and verdict == "wrong":
        subj.set_status(cfg.db_dsn, subject_id, "done", reason="user: not overdue")
        return "Commitment closed."
    return ""


def list_recent(cfg: Any, *, days: int = 14) -> list[dict[str, Any]]:
    since = (date.today() - timedelta(days=days)).isoformat()
    with _get_session(cfg.db_dsn) as db:
        rows = db.scalars(
            select(EmProvocation)
            .where(EmProvocation.day >= since)
            .order_by(EmProvocation.day.desc(), EmProvocation.score.desc())
        ).all()
        return [_dict(r) for r in rows]
