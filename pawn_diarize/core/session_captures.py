"""Recorder notes and screenshots for a diarization session.

Rows are inserted when a ``transcribe-diarize`` job is accepted, before
transcription, and keyed by ``(session_id, item_id)`` so a retry does not
duplicate them. The vault push and the analysis transcript both read them
back from here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from pawn_core.database import SessionCapture

logger = logging.getLogger(__name__)

_ITEM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_SCREENSHOTS_RE = re.compile(
    r"(?is)^##\s*Screenshots\s*\n+(.*?)(?=^##\s|\Z)",
    re.MULTILINE,
)


@dataclass
class Capture:
    """Plain snapshot of one ``session_captures`` row."""

    session_id: str
    item_id: str
    kind: str
    at: datetime
    at_offset_minutes: int
    text: Optional[str] = None
    s3_uri: Optional[str] = None
    output: Optional[str] = None
    region: Optional[dict] = None
    vault_key: Optional[str] = None
    summary: Optional[str] = None
    audio_offset_s: Optional[float] = None
    chunk_audio_start: float = 0.0
    received_at: Optional[datetime] = None


def clock_label(at: datetime, offset_minutes: Optional[int]) -> str:
    """Wall-clock ``HH:MM`` in the offset the recorder sent."""
    moment = at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)
    if offset_minutes is None:
        local = moment
    else:
        local = moment.astimezone(timezone(timedelta(minutes=int(offset_minutes))))
    return local.strftime("%H:%M")


def audio_offset_seconds(
    at: datetime,
    received_at: datetime,
    audio_start: float,
    audio_end: float,
) -> float:
    """Map a wall-clock instant onto the chunk's audio timeline.

    ``received_at`` is the chunk's wall-clock end (when the job was accepted).
    The chunk start is that instant minus the audio duration. The result is
    clamped to ``[audio_start, audio_end]``.
    """
    start = float(audio_start)
    end = float(audio_end)
    if end < start:
        end = start
    duration = max(0.0, end - start)
    moment = at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)
    accepted = (
        received_at if received_at.tzinfo is not None else received_at.replace(tzinfo=timezone.utc)
    )
    wall_start = accepted - timedelta(seconds=duration)
    raw = start + (moment - wall_start).total_seconds()
    return min(max(raw, start), end)


def _parse_at(raw: Any) -> Optional[tuple[datetime, int]]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
        return moment, 0
    offset = moment.utcoffset()
    minutes = 0 if offset is None else int(offset.total_seconds() // 60)
    return moment, minutes


def _clean_id(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    item_id = raw.strip()
    if not item_id or not _ITEM_ID_RE.match(item_id) or "--" in item_id:
        return None
    return item_id


def _as_region(raw: Any, item_id: str) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    logger.warning("session capture %s: ignoring non-object region", item_id)
    return None


def parse_payload_captures(session_id: str, params: dict) -> list[Capture]:
    """Turn optional ``annotations`` / ``screenshots`` into capture rows.

    Malformed items are logged and skipped. Duplicate ids in one payload keep
    the first item.
    """
    found: list[Capture] = []
    notes = params.get("annotations") or []
    shots = params.get("screenshots") or []
    if notes and not isinstance(notes, list):
        logger.warning("session captures: annotations is not a list; ignoring")
        notes = []
    if shots and not isinstance(shots, list):
        logger.warning("session captures: screenshots is not a list; ignoring")
        shots = []

    for kind, raw_items in (("note", notes), ("screenshot", shots)):
        for raw in raw_items:
            if not isinstance(raw, dict):
                logger.warning("session capture: skipping non-object %s item", kind)
                continue
            item_id = _clean_id(raw.get("id"))
            if item_id is None:
                logger.warning("session capture: skipping %s with missing id", kind)
                continue
            parsed = _parse_at(raw.get("at"))
            if parsed is None:
                logger.warning("session capture %s: skipping bad timestamp", item_id)
                continue
            moment, offset_minutes = parsed
            text = raw.get("text")
            if text is not None and not isinstance(text, str):
                text = str(text)
            s3_uri = raw.get("s3_uri")
            if s3_uri is not None and not isinstance(s3_uri, str):
                s3_uri = None
            output = raw.get("output")
            if output is not None and not isinstance(output, str):
                output = None
            found.append(
                Capture(
                    session_id=session_id,
                    item_id=item_id,
                    kind=kind,
                    at=moment,
                    at_offset_minutes=offset_minutes,
                    text=text.strip() if isinstance(text, str) else None,
                    s3_uri=s3_uri or None,
                    output=(output or "").strip() or None,
                    region=_as_region(raw.get("region"), item_id),
                )
            )
    return dedupe_captures(found)


def dedupe_captures(items: Sequence[Capture]) -> list[Capture]:
    """Keep the first capture for each item id."""
    seen: set[str] = set()
    kept: list[Capture] = []
    for item in items:
        if item.item_id in seen:
            continue
        seen.add(item.item_id)
        kept.append(item)
    return kept


def _row_values(
    items: Sequence[Capture],
    *,
    chunk_audio_start: float,
    received_at: datetime,
) -> list[dict]:
    rows = []
    for item in items:
        rows.append(
            {
                "session_id": item.session_id,
                "item_id": item.item_id,
                "kind": item.kind,
                "at": item.at,
                "at_offset_minutes": int(item.at_offset_minutes),
                "text": item.text,
                "s3_uri": item.s3_uri,
                "output": item.output,
                "region": item.region,
                "vault_key": None,
                "summary": None,
                "audio_offset_s": None,
                "chunk_audio_start": float(chunk_audio_start),
                "received_at": received_at,
            }
        )
    return rows


def capture_insert_statement(rows: Sequence[dict]):
    """Insert that leaves an existing ``(session_id, item_id)`` row untouched."""
    stmt = pg_insert(SessionCapture).values(list(rows))
    return stmt.on_conflict_do_nothing(index_elements=["session_id", "item_id"])


def upsert_captures(
    engine,
    session_id: str,
    items: Sequence[Capture],
    *,
    chunk_audio_start: float,
    received_at: datetime,
) -> int:
    """Insert new captures. Existing ids are left as they were."""
    rows = _row_values(
        dedupe_captures(items),
        chunk_audio_start=chunk_audio_start,
        received_at=received_at,
    )
    if not session_id or not rows:
        return 0
    stmt = capture_insert_statement(rows)
    with Session(engine) as db:
        result = db.execute(stmt)
        db.commit()
        count = getattr(result, "rowcount", 0)
    return int(count or 0)


def ingest_session_captures(
    engine,
    session_id: str,
    params: dict,
    *,
    chunk_audio_start: float,
    received_at: datetime,
) -> int:
    """Parse and store captures for one accepted job. Never raises on bad items."""
    if not (session_id or "").strip():
        if params.get("annotations") or params.get("screenshots"):
            logger.warning("session captures dropped: transcribe-diarize has no session")
        return 0
    items = parse_payload_captures(session_id.strip(), params)
    if not items:
        return 0
    return upsert_captures(
        engine,
        session_id.strip(),
        items,
        chunk_audio_start=chunk_audio_start,
        received_at=received_at,
    )


def capture_from_row(row: SessionCapture) -> Capture:
    return Capture(
        session_id=row.session_id,
        item_id=row.item_id,
        kind=row.kind,
        at=row.at,
        at_offset_minutes=int(row.at_offset_minutes or 0),
        text=row.text,
        s3_uri=row.s3_uri,
        output=row.output,
        region=row.region if isinstance(row.region, dict) else None,
        vault_key=row.vault_key,
        summary=row.summary,
        audio_offset_s=None if row.audio_offset_s is None else float(row.audio_offset_s),
        chunk_audio_start=float(row.chunk_audio_start or 0.0),
        received_at=row.received_at,
    )


def load_captures(engine, session_id: str) -> list[Capture]:
    """Return captures for *session_id* ordered by time, then id."""
    with Session(engine) as db:
        rows = db.scalars(
            select(SessionCapture)
            .where(SessionCapture.session_id == session_id)
            .order_by(SessionCapture.at, SessionCapture.item_id)
        ).all()
        return [capture_from_row(row) for row in rows]


def assign_audio_offsets(
    engine,
    session_id: str,
    *,
    audio_start: float,
    audio_end: float,
) -> int:
    """Fill ``audio_offset_s`` once, using the accept-time snapshot on each row."""
    start = float(audio_start)
    updated = 0
    with Session(engine) as db:
        rows = db.scalars(
            select(SessionCapture).where(
                SessionCapture.session_id == session_id,
                SessionCapture.audio_offset_s.is_(None),
            )
        ).all()
        for row in rows:
            if abs(float(row.chunk_audio_start or 0.0) - start) > 1e-3:
                continue
            if row.at is None or row.received_at is None:
                continue
            row.audio_offset_s = audio_offset_seconds(row.at, row.received_at, start, audio_end)
            updated += 1
        db.commit()
    return updated


def set_vault_key(engine, session_id: str, item_id: str, vault_key: str) -> None:
    with Session(engine) as db:
        row = db.get(SessionCapture, (session_id, item_id))
        if row is not None and not row.vault_key:
            row.vault_key = vault_key
        db.commit()


def set_summary(
    engine, session_id: str, item_id: str, summary: str, *, force: bool = False
) -> None:
    with Session(engine) as db:
        row = db.get(SessionCapture, (session_id, item_id))
        if row is None:
            db.commit()
            return
        if row.summary and not force:
            db.commit()
            return
        row.summary = summary
        db.commit()


def meaningful_summary(summary: Optional[str]) -> bool:
    text = (summary or "").strip()
    return bool(text) and text.lower() != "unchanged"


def note_is_visible(cap: Capture) -> bool:
    return cap.kind == "note" and bool((cap.text or "").strip())


def merge_note_annotations(body: str, captures: Sequence[Capture]) -> str:
    """Append a wrapped bullet for each note id that is not already present."""
    text = body or ""
    notes = [cap for cap in captures if note_is_visible(cap)]
    notes.sort(key=lambda cap: (cap.at, cap.item_id))
    for cap in notes:
        opener = f"<!-- pawn:note:{cap.item_id} -->"
        if opener in text:
            continue
        clock = clock_label(cap.at, cap.at_offset_minutes)
        prose = " ".join((cap.text or "").split())
        block = f"{opener}\n- {clock} — {prose}\n<!-- /pawn:note:{cap.item_id} -->\n"
        if text.strip():
            if not text.endswith("\n"):
                text += "\n"
            if not text.endswith("\n\n"):
                text += "\n"
        text += block
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def screenshot_embed(vault_key: str, session_id: str) -> str:
    """Path Obsidian can resolve without colliding on the basename."""
    name = vault_key.replace("\\", "/").rstrip("/").split("/")[-1]
    return f"screenshots/{session_id}/{name}"


def screenshot_filename(cap: Capture, used: set[str]) -> str:
    raw = ""
    if cap.s3_uri:
        raw = cap.s3_uri.split("?", 1)[0].rstrip("/").split("/")[-1]
    raw = _UNSAFE_NAME.sub("_", raw).strip("._")
    if not raw or "." not in raw:
        raw = f"{_UNSAFE_NAME.sub('_', cap.item_id) or 'shot'}.png"
    if raw in used:
        stem, dot, ext = raw.rpartition(".")
        suffix = _UNSAFE_NAME.sub("_", cap.item_id) or "shot"
        raw = f"{stem}_{suffix}.{ext}" if dot else f"{raw}_{suffix}"
    used.add(raw)
    return raw


def _shot_block(shot: Capture) -> str:
    lines = [f"<!-- pawn:shot:{shot.item_id} -->"]
    if shot.vault_key:
        lines.append(f"![[{screenshot_embed(shot.vault_key, shot.session_id)}]]")
    clock = clock_label(shot.at, shot.at_offset_minutes)
    output = (shot.output or "").strip()
    lines.append(f"{clock} · {output}" if output else clock)
    if meaningful_summary(shot.summary):
        lines.append(" ".join((shot.summary or "").split()))
    if not shot.vault_key:
        if shot.s3_uri:
            lines.append(shot.s3_uri)
        else:
            lines.append("_Screenshot was not uploaded._")
    lines.append(f"<!-- /pawn:shot:{shot.item_id} -->")
    return "\n".join(lines)


def format_screenshots_body(captures: Sequence[Capture]) -> str:
    """Managed Screenshots section body. Empty when the session has no shots."""
    shots = [cap for cap in captures if cap.kind == "screenshot"]
    shots.sort(key=lambda cap: (cap.at, cap.item_id))
    if not shots:
        return ""
    return "\n\n".join(_shot_block(shot) for shot in shots) + "\n"


def extract_screenshots(markdown: str) -> str:
    """Return the Screenshots section body, or ``\"\"`` when it is absent."""
    match = _SCREENSHOTS_RE.search(markdown or "")
    if not match:
        return ""
    return match.group(1).strip()


def captures_need_rewrite(markdown: str, captures: Sequence[Capture]) -> bool:
    """True when the note is missing a note id or its Screenshots block differs."""
    body = markdown or ""
    for cap in captures:
        if note_is_visible(cap) and f"<!-- pawn:note:{cap.item_id} -->" not in body:
            return True
    return extract_screenshots(body) != format_screenshots_body(captures).strip()


def vault_marker(cap: Capture) -> Optional[str]:
    """One managed transcript line, or None when this capture stays out of it."""
    if cap.audio_offset_s is None:
        return None
    clock = clock_label(cap.at, cap.at_offset_minutes)
    if note_is_visible(cap):
        prose = " ".join((cap.text or "").split())
        return f"_[{clock} note] {prose}_"
    if cap.kind == "screenshot" and meaningful_summary(cap.summary):
        output = (cap.output or "screen").strip() or "screen"
        prose = " ".join((cap.summary or "").split())
        return f"_[{clock} screen {output}] {prose}_"
    return None


def _format_offset(seconds: float) -> str:
    mm = int(seconds // 60)
    ss = seconds % 60
    return f"[{mm:02d}:{ss:05.2f}]"


def analysis_marker(cap: Capture) -> Optional[str]:
    """Timeline line for ``session_transcript`` / ``session_analyze``."""
    if note_is_visible(cap):
        prose = " ".join((cap.text or "").split())
        tag = f"[note] {prose}"
    elif cap.kind == "screenshot" and meaningful_summary(cap.summary):
        output = (cap.output or "screen").strip() or "screen"
        prose = " ".join((cap.summary or "").split())
        tag = f"[screen {output}] {prose}"
    else:
        return None
    if cap.audio_offset_s is None:
        return f"[{clock_label(cap.at, cap.at_offset_minutes)}] {tag}"
    return f"{_format_offset(cap.audio_offset_s)} {tag}"


def unsummarized_screenshot_count(captures: Sequence[Capture]) -> int:
    return sum(
        1 for cap in captures if cap.kind == "screenshot" and not (cap.summary or "").strip()
    )


def _split_markers(captures: Sequence[Capture], marker_fn):
    placed: list[tuple[float, str]] = []
    tail: list[str] = []
    for cap in captures:
        line = marker_fn(cap)
        if not line:
            continue
        if cap.audio_offset_s is None:
            tail.append(line)
        else:
            placed.append((float(cap.audio_offset_s), line))
    placed.sort(key=lambda item: item[0])
    return placed, tail


def render_analysis_transcript(segments: Sequence[dict], captures: Sequence[Capture]) -> str:
    """Speaker transcript with notes and screenshot changes inserted by time."""
    placed, tail = _split_markers(captures, analysis_marker)
    pending = list(placed)
    lines: list[str] = []
    current: Optional[str] = None

    def take_markers(up_to: Optional[float]) -> None:
        nonlocal current
        while pending and (up_to is None or pending[0][0] <= up_to):
            _, marker = pending.pop(0)
            if current is not None:
                lines.append("")
            lines.append(marker)
            current = None

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start_time") or 0.0)
        take_markers(start)
        display = seg.get("display") or "Speaker"
        mm = int(start // 60)
        ss = start % 60
        if display != current:
            if current is not None:
                lines.append("")
            lines.append(f"[{mm:02d}:{ss:05.2f}] {display}:")
            current = display
        lines.append(f"  {text}")

    take_markers(None)
    for marker in tail:
        if lines:
            lines.append("")
        lines.append(marker)

    pending_shots = unsummarized_screenshot_count(captures)
    if pending_shots:
        noun = "screenshot" if pending_shots == 1 else "screenshots"
        if lines:
            lines.append("")
        lines.append(f"{pending_shots} {noun}, vision not run")

    return "\n".join(lines)


def _s3_mapping(cfg: Any) -> Optional[dict]:
    if hasattr(cfg, "get_s3_config"):
        data = cfg.get_s3_config()
        return data if isinstance(data, dict) else None
    raw = getattr(cfg, "s3", None)
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        data = raw.model_dump()
        return data if data.get("bucket") else None
    if isinstance(raw, dict) and raw.get("bucket"):
        return raw
    return None


def read_s3_bytes(uri: str, cfg: Any) -> bytes:
    """Download one object from the audio bucket."""
    from pawn_diarize.core.s3 import S3Client, parse_s3_uri

    mapping = _s3_mapping(cfg)
    if not mapping:
        raise RuntimeError("s3 config missing; cannot read screenshot")
    client = S3Client.from_dict(mapping)
    bucket, key = parse_s3_uri(uri, configured_bucket=client.bucket)
    return client.get_object_bytes(key, bucket=bucket)


def _content_type(filename: str) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def copy_screenshots_to_vault(
    engine,
    store,
    cfg: Any,
    session_id: str,
    note_key: str,
    captures: Sequence[Capture],
) -> None:
    """Copy PNG bytes next to the transcript note and record ``vault_key``.

    A row that already has ``vault_key`` is not uploaded again. A failed copy
    leaves the row without a key so the note can show the ``s3_uri`` fallback.
    """
    parent = note_key.rsplit("/", 1)[0] if "/" in note_key else ""
    used: set[str] = set()
    for cap in captures:
        if cap.kind == "screenshot" and cap.vault_key:
            used.add(cap.vault_key.replace("\\", "/").rstrip("/").split("/")[-1])

    for cap in captures:
        if cap.kind != "screenshot" or cap.vault_key or not cap.s3_uri:
            continue
        filename = screenshot_filename(cap, used)
        relative = f"screenshots/{session_id}/{filename}"
        vault_key = f"{parent}/{relative}" if parent else relative
        try:
            data = read_s3_bytes(cap.s3_uri, cfg)
            store.write_bytes(vault_key, data, content_type=_content_type(filename))
        except Exception as exc:
            logger.warning(
                "screenshot copy failed for %s/%s: %s",
                session_id,
                cap.item_id,
                exc,
            )
            continue
        cap.vault_key = vault_key
        try:
            set_vault_key(engine, session_id, cap.item_id, vault_key)
        except Exception as exc:
            logger.warning(
                "screenshot vault_key save failed for %s/%s: %s",
                session_id,
                cap.item_id,
                exc,
            )
