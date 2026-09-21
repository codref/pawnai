"""Project diarization session transcripts into SiYuan (diary UI).

Postgres remains the source of truth. SiYuan holds a human-readable projection
with Speakers + Transcript (managed) and Annotations (user-owned).

Invariant: never delete+recreate an existing session document — chunked
diarize updates and daily-note block-refs depend on a stable ``doc_id``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select

from pawn_core.database import (
    SiyuanSessionDoc,
    SpeakerName,
    TranscriptionSegment,
)

from .database import SessionState, get_engine, get_session, init_db
from .siyuan import (
    DEFAULT_DAILY_PATH_TEMPLATE,
    DEFAULT_PATH_TEMPLATE,
    SiyuanClient,
    resolve_path_template,
)

logger = logging.getLogger(__name__)

# Matches ## Annotations until the next ## heading or end of document.
# Keep this tolerant of SiYuan kramdown attr lines ({: id="…"}).
_ANNOTATIONS_RE = re.compile(
    r"(?is)^##\s*Annotations\s*\n+(.*?)(?=^##\s|\Z)",
    re.MULTILINE,
)

_DEFAULT_ANNOTATIONS = "_(Add notes and tags here.)_\n"

_MANAGED_NOTICE = (
    "> Managed by Pawn. Speakers and Transcript are overwritten on sync.\n"
    "> Edit only the Annotations section (or the daily note)."
)


def _format_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _format_ts(seconds: float) -> str:
    mm = int(seconds // 60)
    ss = seconds % 60
    return f"{mm:02d}:{ss:05.2f}"


def extract_annotations(kramdown: str) -> str:
    """Return the Annotations section body, or the default stub if missing.

    Footgun: SiYuan kramdown often injects `{: id="…"}` lines. The regex keys
    off `## Annotations` headings only; attribute noise inside the body is kept
    as-is so user edits survive a round-trip.
    """
    if not kramdown:
        return _DEFAULT_ANNOTATIONS
    # Strip common SiYuan ial suffixes on the heading line for matching.
    cleaned = re.sub(r"\{:.*?\}", "", kramdown)
    m = _ANNOTATIONS_RE.search(cleaned)
    if not m:
        return _DEFAULT_ANNOTATIONS
    body = m.group(1).strip()
    return (body + "\n") if body else _DEFAULT_ANNOTATIONS


def content_hash(speakers_md: str, transcript_md: str) -> str:
    """Hash only managed sections (Annotations are excluded on purpose)."""
    payload = (speakers_md + "\n" + transcript_md).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_session_segments(
    session_id: str,
    engine,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], str]]:
    """Load ordered segments and a (audio_file, label) → display-name map."""
    with get_session(engine) as db:
        orm_segments = db.scalars(
            select(TranscriptionSegment)
            .where(TranscriptionSegment.session_id == session_id)
            .order_by(TranscriptionSegment.segment_index)
        ).all()
        if not orm_segments:
            return [], {}

        segments = [
            {
                "audio_file": s.audio_file,
                "label": s.original_speaker_label,
                "start_time": float(s.start_time),
                "end_time": float(s.end_time),
                "text": s.text or "",
            }
            for s in orm_segments
        ]

        pairs = {
            (s["audio_file"], s["label"])
            for s in segments
            if s["label"] is not None
        }
        name_lookup: Dict[Tuple[str, str], str] = {}
        if pairs:
            audio_files = list({p[0] for p in pairs})
            labels = list({p[1] for p in pairs})
            rows = db.scalars(
                select(SpeakerName).where(
                    SpeakerName.audio_file.in_(audio_files),
                    SpeakerName.local_speaker_label.in_(labels),
                )
            ).all()
            name_lookup = {
                (r.audio_file, r.local_speaker_label): r.speaker_name
                for r in rows
            }
        return segments, name_lookup


def _display_name(
    seg: Dict[str, Any],
    name_lookup: Dict[Tuple[str, str], str],
) -> str:
    label = seg.get("label")
    if label is None:
        return "Speaker"
    return name_lookup.get((seg["audio_file"], label), label)


def format_speakers_section(
    segments: List[Dict[str, Any]],
    name_lookup: Dict[Tuple[str, str], str],
    *,
    file_count: Optional[int] = None,
    time_cursor: Optional[float] = None,
) -> str:
    """Build the Speakers markdown table (talk time + turn count)."""
    talk: Dict[str, float] = defaultdict(float)
    turns: Dict[str, int] = defaultdict(int)
    for seg in segments:
        if not (seg.get("text") or "").strip():
            continue
        name = _display_name(seg, name_lookup)
        duration = float(seg["end_time"]) - float(seg["start_time"])
        talk[name] += max(0.0, duration)
        turns[name] += 1

    lines = [
        "| Speaker | Talk time | Turns |",
        "|---|---|---|",
    ]
    for name in sorted(talk.keys(), key=lambda n: (-talk[n], n.lower())):
        clock = _format_clock(talk[name])
        lines.append(f"| {name} | {clock} | {turns[name]} |")

    if not talk:
        lines.append("| _(none)_ | — | 0 |")

    meta_bits: List[str] = []
    if file_count is not None:
        meta_bits.append(f"**Files so far:** {file_count}")
    if time_cursor is not None and time_cursor > 0:
        clock = _format_clock(time_cursor)
        meta_bits.append(f"**Duration cursor:** {clock}")
    if meta_bits:
        lines.append("")
        lines.append(" · ".join(meta_bits))

    return "\n".join(lines)


def format_transcript_section(
    segments: List[Dict[str, Any]],
    name_lookup: Dict[Tuple[str, str], str],
) -> str:
    """Human-readable transcript: `**Name** · MM:SS.ss` then body lines."""
    lines: List[str] = []
    current: Optional[str] = None
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        name = _display_name(seg, name_lookup)
        if name != current:
            if current is not None:
                lines.append("")
            lines.append(f"**{name}** · {_format_ts(seg['start_time'])}")
            current = name
        lines.append(text)
    return "\n".join(lines) if lines else "_No transcript available._"


def build_document_markdown(
    session_id: str,
    speakers_md: str,
    annotations_body: str,
    transcript_md: str,
) -> str:
    """Assemble the full session document (stable section order)."""
    ann = annotations_body.strip() or _DEFAULT_ANNOTATIONS.strip()
    return "\n\n".join(
        [
            f"# {session_id}",
            _MANAGED_NOTICE,
            "## Speakers",
            speakers_md,
            "## Annotations",
            ann,
            "## Transcript",
            transcript_md,
        ]
    )


def _session_when(session_id: str, engine) -> datetime:
    """Prefer session_state.updated_at for diary dating on backfill."""
    with get_session(engine) as db:
        row = db.get(SessionState, session_id)
        if row and row.updated_at:
            ts = row.updated_at
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts
    return datetime.now(timezone.utc)


def _load_session_meta(
    session_id: str, engine
) -> Tuple[Optional[int], Optional[float]]:
    with get_session(engine) as db:
        row = db.get(SessionState, session_id)
        if not row:
            return None, None
        files = list(row.processed_files or [])
        return len(files), float(row.time_cursor or 0.0)


def _get_mapping(session_id: str, engine) -> Optional[SiyuanSessionDoc]:
    with get_session(engine) as db:
        row = db.get(SiyuanSessionDoc, session_id)
        if row is None:
            return None
        db.expunge(row)
        return row


def _upsert_mapping(
    session_id: str,
    *,
    doc_id: str,
    path: str,
    content_hash_value: str,
    daily_linked: bool,
    engine,
) -> None:
    row = SiyuanSessionDoc(
        session_id=session_id,
        doc_id=doc_id,
        path=path,
        content_hash=content_hash_value,
        daily_linked=daily_linked,
        updated_at=datetime.now(timezone.utc),
    )
    with get_session(engine) as db:
        db.merge(row)


def parse_since(value: str) -> datetime:
    """Parse ``YYYY-MM-DD`` or ISO datetime into an aware UTC datetime.

    Date-only values mean the start of that UTC day (inclusive).
    """
    raw = value.strip()
    if not raw:
        raise ValueError("empty --since value")
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        day = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return day
    # Allow trailing Z
    iso = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def list_session_ids_with_segments(
    engine,
    *,
    latest: bool = False,
    since: Optional[datetime] = None,
) -> List[str]:
    """Return session_ids that have transcription segments.

    When *latest* is true, return only the most recently updated session
    (by ``session_state.updated_at``, else max segment ``created_at``).

    When *since* is set, return sessions whose ``session_state.updated_at``
    (or, if no state row, max segment ``created_at``) is >= *since*.
    """
    if latest and since is not None:
        raise ValueError("latest and since are mutually exclusive")

    with get_session(engine) as db:
        if latest:
            # Prefer session_state.updated_at; fall back to newest segment.
            state_row = db.execute(
                select(SessionState.session_id)
                .order_by(SessionState.updated_at.desc().nullslast())
                .limit(1)
            ).first()
            if state_row:
                return [state_row[0]]
            seg_row = db.execute(
                select(TranscriptionSegment.session_id)
                .where(TranscriptionSegment.session_id.is_not(None))
                .group_by(TranscriptionSegment.session_id)
                .order_by(func.max(TranscriptionSegment.created_at).desc())
                .limit(1)
            ).first()
            return [seg_row[0]] if seg_row and seg_row[0] else []

        if since is not None:
            # Normalize naive timestamps from DB as UTC for comparison.
            since_naive = since.astimezone(timezone.utc).replace(tzinfo=None)

            from_state = db.execute(
                select(SessionState.session_id).where(
                    SessionState.updated_at >= since_naive
                )
            ).all()
            ids = {r[0] for r in from_state if r[0]}

            # Sessions with segments but no/old state: use max(created_at).
            seg_rows = db.execute(
                select(
                    TranscriptionSegment.session_id,
                    func.max(TranscriptionSegment.created_at),
                )
                .where(TranscriptionSegment.session_id.is_not(None))
                .group_by(TranscriptionSegment.session_id)
            ).all()
            for sid, max_created in seg_rows:
                if not sid or sid in ids:
                    continue
                if max_created is None:
                    continue
                cmp = max_created
                if getattr(cmp, "tzinfo", None) is not None:
                    cmp = cmp.astimezone(timezone.utc).replace(tzinfo=None)
                if cmp >= since_naive:
                    ids.add(sid)

            return sorted(ids)

        rows = db.execute(
            select(TranscriptionSegment.session_id)
            .where(TranscriptionSegment.session_id.is_not(None))
            .distinct()
            .order_by(TranscriptionSegment.session_id)
        ).all()
        return [r[0] for r in rows if r[0]]


def push_session_transcript(
    session_id: str,
    *,
    db_dsn: str,
    url: str,
    token: str,
    notebook: str,
    path_template: str = DEFAULT_PATH_TEMPLATE,
    daily_path_template: str = DEFAULT_DAILY_PATH_TEMPLATE,
    daily_note: bool = True,
    dry_run: bool = False,
) -> str:
    """Create or update the SiYuan transcript page for *session_id*.

    Returns a short human-readable status string.
    """
    if not notebook:
        return "skipped: siyuan.notebook not configured"

    engine = get_engine(db_dsn)
    init_db(engine)

    segments, name_lookup = load_session_segments(session_id, engine)
    if not segments:
        return f"skipped: no segments for session {session_id!r}"

    file_count, time_cursor = _load_session_meta(session_id, engine)
    speakers_md = format_speakers_section(
        segments,
        name_lookup,
        file_count=file_count,
        time_cursor=time_cursor,
    )
    transcript_md = format_transcript_section(segments, name_lookup)
    digest = content_hash(speakers_md, transcript_md)

    mapping = _get_mapping(session_id, engine)
    if mapping and mapping.content_hash == digest:
        return f"unchanged: {session_id} (hash match)"

    when = _session_when(session_id, engine)
    # title=session_id keeps {title} paths stable across chunks.
    doc_path = resolve_path_template(
        path_template or DEFAULT_PATH_TEMPLATE,
        session_id=session_id,
        title=session_id,
        now=when,
    )

    speaker_names = sorted(
        {
            _display_name(s, name_lookup)
            for s in segments
            if (s.get("text") or "").strip()
        }
    )
    attrs = {
        "custom-session-id": session_id,
        "custom-speaker-count": str(len(speaker_names)),
        "custom-speakers": ",".join(speaker_names),
    }

    if dry_run:
        action = "update" if mapping else "create"
        return (
            f"dry-run: would {action} {session_id} → {doc_path} "
            f"({len(segments)} segments, {len(speaker_names)} speakers)"
        )

    client = SiyuanClient(url=url, token=token, notebook_id=notebook)

    if mapping is None:
        annotations = _DEFAULT_ANNOTATIONS
        markdown = build_document_markdown(
            session_id, speakers_md, annotations, transcript_md
        )
        # Create once — do not upsert/delete; doc_id must remain stable.
        doc_id = client.create_doc(notebook, doc_path, markdown)
        if not doc_id:
            return (
                f"error: SiYuan createDoc returned empty id "
                f"for {session_id!r}"
            )
        client.set_block_attrs(doc_id, attrs)
        daily_linked = False
        if daily_note:
            try:
                daily_path = resolve_path_template(
                    daily_path_template or DEFAULT_DAILY_PATH_TEMPLATE,
                    session_id=session_id,
                    title=None,
                    now=when,
                )
                client.append_daily_note_link(
                    notebook=notebook,
                    daily_path=daily_path,
                    doc_id=doc_id,
                    title=session_id,
                )
                daily_linked = True
            except Exception as exc:
                logger.warning(
                    "siyuan transcript: daily note link failed for %r: %s",
                    session_id,
                    exc,
                )
        _upsert_mapping(
            session_id,
            doc_id=doc_id,
            path=doc_path,
            content_hash_value=digest,
            daily_linked=daily_linked,
            engine=engine,
        )
        return f"created: {session_id} → {doc_path} (doc_id={doc_id})"

    # Update path: preserve Annotations, rewrite Speakers + Transcript.
    kramdown = client.get_block_kramdown(mapping.doc_id)
    annotations = extract_annotations(kramdown)
    markdown = build_document_markdown(
        session_id, speakers_md, annotations, transcript_md
    )
    client.update_block(mapping.doc_id, markdown)
    client.set_block_attrs(mapping.doc_id, attrs)

    daily_linked = bool(mapping.daily_linked)
    if daily_note and not daily_linked:
        try:
            daily_path = resolve_path_template(
                daily_path_template or DEFAULT_DAILY_PATH_TEMPLATE,
                session_id=session_id,
                title=None,
                now=when,
            )
            client.append_daily_note_link(
                notebook=notebook,
                daily_path=daily_path,
                doc_id=mapping.doc_id,
                title=session_id,
            )
            daily_linked = True
        except Exception as exc:
            logger.warning(
                "siyuan transcript: daily note link failed for %r: %s",
                session_id,
                exc,
            )

    _upsert_mapping(
        session_id,
        doc_id=mapping.doc_id,
        path=mapping.path or doc_path,
        content_hash_value=digest,
        daily_linked=daily_linked,
        engine=engine,
    )
    return f"updated: {session_id} → {mapping.path} (doc_id={mapping.doc_id})"


def maybe_push_transcript_to_siyuan(
    session_id: str,
    cfg: Any,
    *,
    db_dsn: Optional[str] = None,
) -> None:
    """Best-effort auto-push when ``siyuan.auto_push_transcript`` is enabled.

    Never raises — diarization must not fail because SiYuan is down.
    """
    try:
        sy = getattr(cfg, "siyuan", None)
        if sy is None and hasattr(cfg, "get_siyuan_config"):
            sy_dict = cfg.get_siyuan_config() or {}
        elif sy is not None and hasattr(sy, "model_dump"):
            sy_dict = sy.model_dump()
        elif isinstance(sy, dict):
            sy_dict = sy
        else:
            sy_dict = {}

        if not sy_dict.get("auto_push_transcript"):
            return
        notebook = sy_dict.get("notebook") or ""
        if not notebook:
            logger.warning(
                "siyuan auto_push_transcript enabled but notebook "
                "is empty — skipping"
            )
            return

        resolved_dsn = (
            db_dsn or getattr(cfg, "db_dsn", None) or cfg.get("db_dsn")
        )
        path_tpl = sy_dict.get("path_template") or DEFAULT_PATH_TEMPLATE
        daily_tpl = (
            sy_dict.get("daily_note_path") or DEFAULT_DAILY_PATH_TEMPLATE
        )
        status = push_session_transcript(
            session_id,
            db_dsn=str(resolved_dsn),
            url=sy_dict.get("url") or "http://127.0.0.1:6806",
            token=sy_dict.get("token") or "",
            notebook=notebook,
            path_template=path_tpl,
            daily_path_template=daily_tpl,
            daily_note=True,
        )
        logger.info("siyuan transcript push: %s", status)
    except Exception as exc:
        logger.warning(
            "siyuan transcript push failed for %r (non-fatal): %s",
            session_id,
            exc,
        )
