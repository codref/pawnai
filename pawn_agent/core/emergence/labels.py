"""User verdicts as training rows (``em_labels``).

Labels come from single-item triage (Inbox, Matrix, notes, chat), from
compare review notes the user ticks, and from the API ("Pawn missed this").
Each row snapshots what was judged plus its embedding, so it stays usable
after observations are superseded.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from sqlalchemy import func, select

from pawn_agent.core.emergence.types import ITEM_KINDS, OBSERVATION_TYPES
from pawn_agent.utils.db import EmLabel, _get_session

logger = logging.getLogger(__name__)

# Item kind → observation type.
KIND_TYPES: dict[str, str] = {kind: type_ for type_, kind in ITEM_KINDS.items()}

# Triage action → verdict. ``later`` is not a verdict.
ACTION_VERDICTS: dict[str, str] = {
    "todo": "accept",
    "task": "accept",
    "file": "accept",
    "delete": "reject",
    "ignore": "reject",
}

# ``missed`` (the user marked a passage Pawn should have extracted) counts as
# a positive for training and few-shot.
TRAINING_VERDICTS = ("accept", "reject", "missed")
POSITIVE_VERDICTS = ("accept", "missed")
API_VERDICTS = ("accept", "reject", "missed")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", (text or "").lower()))


def label_text(type_name: str, text: str, quote: str = "") -> str:
    """The string embedded for labels and observations (same shape for both)."""
    body = f"{type_name}: {(text or '').strip()}"
    quote = " ".join((quote or "").split())
    return f"{body}\n{quote[:600]}" if quote else body


def embedder(cfg: Any) -> Optional[Callable[[str], list[float]]]:
    """Document embedder, or None when embeddings are unavailable."""
    try:
        from pawn_core.knowledge_index import _embedder  # noqa: PLC0415

        return _embedder(cfg)
    except Exception as exc:
        logger.warning("emergence embeddings unavailable: %s", exc)
        return None


def embed_many(cfg: Any, texts: list[str]) -> list[Optional[list[float]]]:
    """Embed each text; a failure yields None for that text."""
    embed = embedder(cfg)
    if embed is None:
        return [None] * len(texts)
    out: list[Optional[list[float]]] = []
    for text in texts:
        try:
            out.append(list(embed(text)) if text.strip() else None)
        except Exception as exc:
            logger.warning("emergence embed failed: %s", exc)
            out.append(None)
    return out


def upsert_label(dsn: str, **fields: Any) -> dict[str, Any]:
    """Insert or replace the label for ``(target_kind, target_id)``."""
    now = _now()
    with _get_session(dsn) as db:
        row = db.scalars(
            select(EmLabel).where(
                EmLabel.target_kind == fields["target_kind"],
                EmLabel.target_id == fields["target_id"],
            )
        ).first()
        if row is None:
            row = EmLabel(id=str(uuid.uuid4()), created_at=now)
            db.add(row)
        for key, value in fields.items():
            setattr(row, key, value)
        row.updated_at = now
        db.flush()
        return {"id": row.id, "verdict": row.verdict, "type": row.type}


def labels_enabled(cfg: Any) -> bool:
    return bool(getattr(getattr(cfg, "emergence", None), "enabled", False))


def record_item_label(
    cfg: Any, item: dict[str, Any], action: str, *, surface: str, embed: bool = True
) -> None:
    """Record the verdict implied by a triage action on a coworker item.

    No-op unless emergence is enabled. Never raises: labels must not break
    triage. ``embed=False`` skips the embedding (bulk deletes).
    """
    if not labels_enabled(cfg):
        return
    verdict = ACTION_VERDICTS.get((action or "").strip().lower())
    if verdict is None:
        return
    kind = item.get("kind") or ""
    payload = item.get("payload") or {}
    type_name = payload.get("observation_type") or KIND_TYPES.get(kind)
    if not type_name:
        return  # proposals, schedule proposals, people updates
    try:
        text, quote = item.get("text") or "", item.get("quote") or ""
        vector = embed_many(cfg, [label_text(type_name, text, quote)])[0] if embed else None
        upsert_label(
            cfg.db_dsn,
            target_kind="item",
            target_id=str(item["id"]),
            verdict=verdict,
            surface=surface,
            type=type_name,
            text=text,
            quote=quote or None,
            context=item.get("context") or None,
            source_ref=item.get("source_ref"),
            start_sec=payload.get("start_sec"),
            observation_id=payload.get("observation_id"),
            item_id=str(item["id"]),
            features={
                "quote_score": payload.get("quote_score"),
                "has_owner": bool(item.get("owner")),
                "action": action,
            },
            embedding=vector,
        )
    except Exception as exc:
        logger.warning("could not record label for item %s: %s", item.get("id"), exc)


def record_api_label(cfg: Any, body: dict[str, Any]) -> dict[str, Any]:
    """Store a label posted to ``/v1/emergence/labels``. Raises ValueError."""
    type_name = str(body.get("type") or "").strip().lower()
    type_name = KIND_TYPES.get(type_name, type_name)
    if type_name not in OBSERVATION_TYPES:
        raise ValueError(f"unknown type {body.get('type')!r}")
    verdict = str(body.get("verdict") or "missed").strip().lower()
    if verdict not in API_VERDICTS:
        raise ValueError(f"verdict must be one of {', '.join(API_VERDICTS)}")
    text = str(body.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")
    quote = str(body.get("quote") or "").strip()
    source_ref = str(body.get("source_ref") or "").strip()
    target_id = (
        str(body.get("target_id") or "").strip()
        or hashlib.sha256(f"{source_ref}|{type_name}|{_norm(text)}".encode()).hexdigest()[:32]
    )
    [vector] = embed_many(cfg, [label_text(type_name, text, quote)])
    return upsert_label(
        cfg.db_dsn,
        target_kind="api",
        target_id=target_id,
        verdict=verdict,
        surface="api",
        type=type_name,
        text=text,
        quote=quote or None,
        source_ref=source_ref or None,
        start_sec=body.get("start_sec"),
        embedding=vector,
    )


# ── links and subjects (Portal brainmap) ─────────────────────────────────────

# Labels on strings and cards. They carry no embedding, so the extraction
# ranker, few-shot retrieval and prompt search (which read embedded or
# typed rows) never mix them with "was this worth extracting" verdicts.
STRUCTURE_TARGETS = ("link", "subject")
LINK_VERDICTS = ("accept", "reject")


def link_key(subject_id: str, text: str) -> str:
    """Stable key for an (observation, subject) pair across re-processing.

    Observation ids change when a source is re-run; the normalized text does not.
    """
    digest = hashlib.sha256(_norm(text).encode()).hexdigest()[:16]
    return f"{subject_id}:{digest}"


def record_link_label(
    cfg: Any,
    *,
    subject: dict[str, Any],
    observation: dict[str, Any],
    verdict: str,
    surface: str = "brainmap",
    role: str = "",
    before: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Record that a string between *observation* and *subject* is right or wrong.

    ``reject`` also keeps routing from tying the pair again (see
    :func:`rejected_link_keys`). Never raises.
    """
    if verdict not in LINK_VERDICTS:
        raise ValueError(f"verdict must be one of {', '.join(LINK_VERDICTS)}")
    text = str(observation.get("text") or "")
    try:
        return upsert_label(
            cfg.db_dsn,
            target_kind="link",
            target_id=link_key(subject["id"], text),
            verdict=verdict,
            surface=surface,
            type=str(observation.get("type") or "evidence"),
            text=text,
            quote=observation.get("quote") or None,
            source_ref=observation.get("source_ref") or None,
            start_sec=observation.get("start_sec"),
            observation_id=observation.get("id"),
            features={
                "subject_id": subject["id"],
                "subject_kind": subject.get("kind"),
                "subject_title": subject.get("title"),
                "role": role,
            },
            before=before,
            after={"subject_id": subject["id"], "role": role} if verdict == "accept" else None,
            embedding=None,
        )
    except Exception as exc:
        logger.warning("could not record link label for %s: %s", subject.get("id"), exc)
        return None


def rejected_link_keys(dsn: str) -> set[str]:
    """Keys (``link_key``) of strings the user cut or snipped."""
    with _get_session(dsn) as db:
        rows = db.scalars(
            select(EmLabel.target_id).where(
                EmLabel.target_kind == "link", EmLabel.verdict == "reject"
            )
        ).all()
        return set(rows)


def link_rejected(rejected: set[str], subject_id: str, text: str) -> bool:
    return bool(rejected) and link_key(subject_id, text) in rejected


def record_subject_label(
    cfg: Any, subject: dict[str, Any], status: str, *, surface: str = "api"
) -> Optional[dict[str, Any]]:
    """Accepting a proposed subject is ``accept``; dismissing one is ``reject``.

    Other status changes (done, parked, resolved) are lifecycle, not verdicts,
    and are not labelled. Never raises.
    """
    verdict = {"active": "accept", "dismissed": "reject"}.get(status)
    if verdict is None:
        return None
    try:
        return upsert_label(
            cfg.db_dsn,
            target_kind="subject",
            target_id=str(subject["id"]),
            verdict=verdict,
            surface=surface,
            type=str(subject.get("kind") or "thread"),
            text=str(subject.get("title") or ""),
            context=(subject.get("state_md") or None),
            features={"origin": subject.get("origin"), "before": subject.get("status")},
            before={"status": subject.get("status")},
            after={"status": status},
            embedding=None,
        )
    except Exception as exc:
        logger.warning("could not record subject label for %s: %s", subject.get("id"), exc)
        return None


# ── review notes ─────────────────────────────────────────────────────────────

_CHECK_RE = re.compile(r"^- \[( |x|X)\] (.*)$")
_AT_RE = re.compile(r"^\[(?:(\d+):)?(\d{1,2}):(\d{2})\] ")
_KIND_RE = re.compile(r"^\*\*(\w+)\*\* (.*)$")
_RANK_SUFFIX_RE = re.compile(r" · rank [\d.]+$")
_ITEM_SUFFIX_RE = re.compile(r"( — owner: .*?)?(, due \S+)?( · \*in both\*)?$")
_REJECTED_RE = re.compile(r"^\*\*(\w+)\*\* (.*) \((\w+), match [\d.]+\)$")


def _seconds(match: re.Match[str]) -> float:
    hours = int(match.group(1) or 0)
    return hours * 3600 + int(match.group(2)) * 60 + int(match.group(3))


def parse_review_note(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Frontmatter plus one entry per checkbox line of a compare note.

    Entries: ``{section, type, text, quote, start_sec, ticked}`` where section
    is ``observations`` | ``single`` | ``other`` | ``rejected``.
    """
    from pawn_core.vault import parse_frontmatter  # noqa: PLC0415

    try:
        meta, body = parse_frontmatter(text)
    except Exception:
        meta, body = {}, text
    entries: list[dict[str, Any]] = []
    section = ""
    other_type = ""
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("## Observation items"):
            section = "observations"
        elif line.startswith("## Single-prompt items"):
            section = "single"
        elif line.startswith("## Other observations"):
            section = "other"
        elif line.startswith("## Rejected"):
            section = "rejected"
        elif line.startswith("### ") and section == "other":
            other_type = line[4:].split(" ", 1)[0].strip().lower()
        match = _CHECK_RE.match(line)
        if not match or not section:
            continue
        ticked = match.group(1).lower() == "x"
        rest = match.group(2)
        start_sec = None
        at = _AT_RE.match(rest)
        if at:
            start_sec = _seconds(at)
            rest = rest[at.end() :]
        rest = _RANK_SUFFIX_RE.sub("", rest)
        quote = ""
        if index + 1 < len(lines) and lines[index + 1].lstrip().startswith("> "):
            quote = lines[index + 1].lstrip()[2:].strip()
        if section in {"observations", "single"}:
            kind_match = _KIND_RE.match(rest)
            if not kind_match:
                continue
            type_name = KIND_TYPES.get(kind_match.group(1), kind_match.group(1))
            claim = _ITEM_SUFFIX_RE.sub("", kind_match.group(2)).strip()
        elif section == "other":
            type_name = other_type
            claim = rest.rsplit(" — ", 1)[0].strip() if " — " in rest else rest.strip()
        else:
            rejected = _REJECTED_RE.match(rest)
            if not rejected:
                continue
            type_name, claim = rejected.group(1), rejected.group(2).strip()
        if type_name not in OBSERVATION_TYPES or not claim:
            continue
        entries.append(
            {
                "section": section,
                "type": type_name,
                "text": claim,
                "quote": quote,
                "start_sec": start_sec,
                "ticked": ticked,
            }
        )
    return meta, entries


def _reviewed(meta: dict[str, Any]) -> bool:
    value = meta.get("reviewed")
    return value is True or str(value).strip().lower() in {"true", "yes", "1"}


def import_review_note(cfg: Any, key: str, text: str) -> dict[str, int]:
    """Labels from one compare note.

    A ticked line is ``accept``. An unticked line is ``reject`` only when the
    note's frontmatter says ``reviewed: true``; until then it is skipped. In
    the Rejected section a tick means the quote check was wrong (``accept``)
    and an unticked line is no label.
    """
    meta, entries = parse_review_note(text)
    reviewed = _reviewed(meta)
    source_ref = str(meta.get("source_ref") or "")
    counts = {"accept": 0, "reject": 0, "skipped": 0}
    chosen: list[tuple[dict[str, Any], str]] = []
    for entry in entries:
        if entry["ticked"]:
            verdict = "accept"
        elif reviewed and entry["section"] != "rejected":
            verdict = "reject"
        else:
            counts["skipped"] += 1
            continue
        chosen.append((entry, verdict))
    vectors = embed_many(cfg, [label_text(e["type"], e["text"], e["quote"]) for e, _ in chosen])
    for (entry, verdict), vector in zip(chosen, vectors):
        digest = hashlib.sha256(
            f"{source_ref}|{entry['type']}|{_norm(entry['text'])}".encode()
        ).hexdigest()[:32]
        upsert_label(
            cfg.db_dsn,
            target_kind="review",
            target_id=digest,
            verdict=verdict,
            surface="review",
            type=entry["type"],
            text=entry["text"],
            quote=entry["quote"] or None,
            source_ref=source_ref or None,
            start_sec=entry["start_sec"],
            extractor="single" if entry["section"] == "single" else "observations",
            features={"note": key, "section": entry["section"]},
            embedding=vector,
        )
        counts[verdict] += 1
    return counts


def import_review_notes(cfg: Any, store: Any) -> dict[str, dict[str, int]]:
    """Import every ``extract-compare-*`` note under ``coworker.reviews_dir``."""
    prefix = cfg.coworker.reviews_dir.rstrip("/") + "/"
    results: dict[str, dict[str, int]] = {}
    for key in store.list(prefix):
        if not key.rsplit("/", 1)[-1].startswith("extract-compare-"):
            continue
        try:
            results[key] = import_review_note(cfg, key, store.read(key))
        except Exception as exc:
            logger.warning("review note %s not imported: %s", key, exc)
    return results


# ── reads ────────────────────────────────────────────────────────────────────


def training_rows(dsn: str, *, include_bulk: bool = False) -> list[dict[str, Any]]:
    """Accept/reject labels that have an embedding."""
    with _get_session(dsn) as db:
        stmt = select(EmLabel).where(
            EmLabel.verdict.in_(TRAINING_VERDICTS), EmLabel.embedding.is_not(None)
        )
        if not include_bulk:
            stmt = stmt.where(EmLabel.surface != "bulk")
        rows = db.scalars(stmt).all()
        return [
            {
                "id": r.id,
                "type": r.type,
                "verdict": "accept" if r.verdict in POSITIVE_VERDICTS else "reject",
                "features": r.features or {},
                "embedding": [float(x) for x in (r.embedding if r.embedding is not None else [])],
            }
            for r in rows
        ]


def label_counts(dsn: str) -> dict[str, Any]:
    with _get_session(dsn) as db:
        rows = db.execute(
            select(EmLabel.surface, EmLabel.verdict, func.count()).group_by(
                EmLabel.surface, EmLabel.verdict
            )
        ).all()
        total = db.scalar(select(func.count()).select_from(EmLabel)) or 0
    return {"total": int(total), "by": [(s, v, int(n)) for s, v, n in rows]}


def few_shot_examples(
    dsn: str,
    type_name: str,
    vector: list[float],
    *,
    n_accept: int,
    n_reject: int,
) -> dict[str, list[dict[str, str]]]:
    """Nearest accepted and rejected labels of one type (bulk excluded)."""
    out: dict[str, list[dict[str, str]]] = {"accept": [], "reject": []}
    if not vector:
        return out
    with _get_session(dsn) as db:
        for verdict, limit in (("accept", n_accept), ("reject", n_reject)):
            if limit <= 0:
                continue
            verdicts = POSITIVE_VERDICTS if verdict == "accept" else ("reject",)
            rows = db.scalars(
                select(EmLabel)
                .where(
                    EmLabel.type == type_name,
                    EmLabel.verdict.in_(verdicts),
                    EmLabel.surface != "bulk",
                    EmLabel.embedding.is_not(None),
                )
                .order_by(EmLabel.embedding.cosine_distance(vector))
                .limit(limit)
            ).all()
            out[verdict] = [{"text": r.text, "quote": r.quote or ""} for r in rows]
    return out
