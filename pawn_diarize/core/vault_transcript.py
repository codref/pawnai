"""Project diarization session transcripts into the S3 Obsidian vault.

Postgres remains the source of truth. The vault holds a human-readable
projection with Speakers + Transcript (managed) and Annotations (user-owned).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select

from pawn_core.database import SpeakerName, TranscriptionSegment
from pawn_core.vault import VaultStore, dump_frontmatter, parse_frontmatter, resolve_path_template
from pawn_core.vault_config import vault_store_from_config
from pawn_core.vault_db import get_vault_note, upsert_vault_note

from .database import SessionState, get_engine, get_session, init_db

logger = logging.getLogger(__name__)

_ANNOTATIONS_RE = re.compile(
    r"(?is)^##\s*Annotations\s*\n+(.*?)(?=^##\s|\Z)",
    re.MULTILINE,
)

_DEFAULT_ANNOTATIONS = "_(Add notes and tags here.)_\n"

_MANAGED_NOTICE = (
    "> Managed by Pawn. Speakers and Transcript are overwritten on sync.\n"
    "> Edit only the Annotations section."
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


def extract_annotations(markdown: str) -> str:
    """Return the Annotations section body, or the default stub if missing."""
    if not markdown:
        return _DEFAULT_ANNOTATIONS
    cleaned = re.sub(r"\{:.*?\}", "", markdown)
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

        pairs = {(s["audio_file"], s["label"]) for s in segments if s["label"] is not None}
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
            name_lookup = {(r.audio_file, r.local_speaker_label): r.speaker_name for r in rows}
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


def _session_duration(segments: List[Dict[str, Any]], time_cursor: Optional[float]) -> str:
    if time_cursor is not None and time_cursor > 0:
        return _format_clock(time_cursor)
    if not segments:
        return "0m 00s"
    end = max(float(s["end_time"]) for s in segments)
    return _format_clock(end)


def build_document_markdown(
    session_id: str,
    speakers_md: str,
    annotations_body: str,
    transcript_md: str,
    *,
    date_str: str,
    speaker_names: List[str],
    duration: str,
    related_analysis_wiki: Optional[str] = None,
) -> str:
    """Assemble the full session note with Obsidian YAML frontmatter."""
    ann = annotations_body.strip() or _DEFAULT_ANNOTATIONS.strip()
    meta = {
        "pawn": "transcript",
        "session_id": session_id,
        "date": date_str,
        "speakers": speaker_names,
        "duration": duration,
        "tags": ["pawn/transcript"],
    }
    body_parts = [
        f"# {session_id}",
        "",
        _MANAGED_NOTICE,
    ]
    if related_analysis_wiki:
        body_parts.extend(["", f"Related: [[{related_analysis_wiki}]]"])
    body_parts.extend(
        [
            "",
            "## Speakers",
            speakers_md,
            "",
            "## Annotations",
            ann,
            "",
            "## Transcript",
            transcript_md,
        ]
    )
    body = "\n".join(body_parts)
    if not body.endswith("\n"):
        body += "\n"
    return dump_frontmatter(meta, body)


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


def _load_session_meta(session_id: str, engine) -> Tuple[Optional[int], Optional[float]]:
    with get_session(engine) as db:
        row = db.get(SessionState, session_id)
        if not row:
            return None, None
        files = list(row.processed_files or [])
        return len(files), float(row.time_cursor or 0.0)


def parse_since(value: str) -> datetime:
    """Parse ``YYYY-MM-DD`` or ISO datetime into an aware UTC datetime."""
    raw = value.strip()
    if not raw:
        raise ValueError("empty --since value")
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        day = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return day
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
    """Return session_ids that have transcription segments."""
    if latest and since is not None:
        raise ValueError("latest and since are mutually exclusive")

    with get_session(engine) as db:
        if latest:
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
            since_naive = since.astimezone(timezone.utc).replace(tzinfo=None)

            from_state = db.execute(
                select(SessionState.session_id).where(SessionState.updated_at >= since_naive)
            ).all()
            ids = {r[0] for r in from_state if r[0]}

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


def _vault_cfg(cfg: Any) -> Any:
    vault = getattr(cfg, "vault", None)
    if vault is None:
        raise ValueError("vault configuration is missing")
    return vault


def _analysis_wiki_link(cfg: Any, session_id: str, when: datetime) -> Optional[str]:
    vault = _vault_cfg(cfg)
    template = getattr(vault, "analysis_path_template", "") or ""
    if not template:
        return None
    path = resolve_path_template(
        template,
        agent_root=getattr(vault, "agent_root", "Pawn") or "Pawn",
        session_id=session_id,
        title=session_id,
        now=when,
    )
    if path.endswith(".md"):
        path = path[:-3]
    return path


def _maybe_append_daily_link(
    store: VaultStore,
    *,
    daily_template: str,
    session_id: str,
    note_key: str,
    when: datetime,
    agent_root: str,
) -> None:
    if not daily_template:
        return
    daily_key = resolve_path_template(
        daily_template,
        agent_root=agent_root,
        session_id=session_id,
        title=session_id,
        now=when,
    )
    wiki = note_key[:-3] if note_key.endswith(".md") else note_key
    line = f"- [[{wiki}|{session_id}]]\n"
    try:
        existing = ""
        if store.exists(daily_key):
            existing = store.read(daily_key)
        if session_id in existing or f"[[{wiki}" in existing:
            return
        store.append(daily_key, line)
    except Exception as exc:
        logger.warning(
            "vault transcript: daily note link failed for %r: %s",
            session_id,
            exc,
        )


def push_session_transcript(
    session_id: str,
    *,
    db_dsn: str,
    store: VaultStore | None = None,
    cfg=None,
    dry_run: bool = False,
) -> str:
    """Create or update the vault transcript note for *session_id*."""
    if cfg is None:
        raise ValueError("cfg is required when store is not passed")
    vault = _vault_cfg(cfg)
    if not dry_run and store is None:
        if not getattr(vault, "bucket", ""):
            return "skipped: vault.s3.bucket not configured"
        store = vault_store_from_config(cfg)

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

    mapping = get_vault_note(db_dsn, session_id)
    if mapping and mapping.content_hash == digest:
        return f"unchanged: {session_id} (hash match)"

    when = _session_when(session_id, engine)
    agent_root = getattr(vault, "agent_root", "Pawn") or "Pawn"
    path_template = getattr(vault, "transcript_path_template", "") or (
        "{agent_root}/Transcripts/{date} {session_id}.md"
    )

    speaker_names = sorted(
        {_display_name(s, name_lookup) for s in segments if (s.get("text") or "").strip()}
    )
    duration = _session_duration(segments, time_cursor)
    date_str = when.strftime("%Y-%m-%d")
    related = _analysis_wiki_link(cfg, session_id, when)

    if mapping is None:
        note_key = resolve_path_template(
            path_template,
            agent_root=agent_root,
            session_id=session_id,
            title=session_id,
            now=when,
        )
        annotations = _DEFAULT_ANNOTATIONS
    else:
        note_key = mapping.key
        if dry_run:
            annotations = _DEFAULT_ANNOTATIONS
        else:
            assert store is not None
            try:
                existing = store.read(note_key)
                _, body = parse_frontmatter(existing)
                annotations = extract_annotations(body)
            except Exception:
                annotations = _DEFAULT_ANNOTATIONS

    markdown = build_document_markdown(
        session_id,
        speakers_md,
        annotations,
        transcript_md,
        date_str=date_str,
        speaker_names=speaker_names,
        duration=duration,
        related_analysis_wiki=related,
    )

    if dry_run:
        action = "update" if mapping else "create"
        return (
            f"dry-run: would {action} {session_id} → {note_key} "
            f"({len(segments)} segments, {len(speaker_names)} speakers)"
        )

    assert store is not None
    store.write(note_key, markdown)
    upsert_vault_note(
        db_dsn,
        session_id=session_id,
        key=note_key,
        content_hash=digest,
    )

    daily_tpl = getattr(vault, "daily_note_path", None)
    if daily_tpl:
        _maybe_append_daily_link(
            store,
            daily_template=daily_tpl,
            session_id=session_id,
            note_key=note_key,
            when=when,
            agent_root=agent_root,
        )

    if mapping is None:
        return f"created: {session_id} → {note_key}"
    return f"updated: {session_id} → {note_key}"


def refresh_transcript_after_relabel(
    session_id: str,
    *,
    db_dsn: str,
    cfg,
    force: bool = False,
) -> Optional[str]:
    """Refresh vault Speakers+Transcript after a DB speaker rename."""
    try:
        vault = getattr(cfg, "vault", None)
        if not getattr(vault, "bucket", ""):
            if force:
                return "skipped: vault.s3.bucket not configured"
            return None

        mapping = get_vault_note(db_dsn, session_id)
        if mapping is None and not force:
            return None

        return push_session_transcript(
            session_id,
            db_dsn=db_dsn,
            cfg=cfg,
        )
    except Exception as exc:
        logger.warning(
            "vault transcript refresh after relabel failed for %r: %s",
            session_id,
            exc,
        )
        return f"vault error: {exc}"


def maybe_push_transcript_to_vault(
    session_id: str,
    cfg: Any,
    *,
    db_dsn: Optional[str] = None,
) -> None:
    """Best-effort auto-push when ``vault.auto_push_transcript`` is enabled."""
    try:
        vault = getattr(cfg, "vault", None)
        if vault is None:
            return
        if not getattr(vault, "auto_push_transcript", False):
            return
        if not getattr(vault, "bucket", ""):
            logger.warning("vault auto_push_transcript enabled but bucket is empty — skipping")
            return

        resolved_dsn = db_dsn or getattr(cfg, "db_dsn", None)
        if not resolved_dsn and hasattr(cfg, "get"):
            resolved_dsn = cfg.get("db_dsn")
        status = push_session_transcript(
            session_id,
            db_dsn=str(resolved_dsn),
            cfg=cfg,
        )
        logger.info("vault transcript push: %s", status)
    except Exception as exc:
        logger.warning(
            "vault transcript push failed for %r (non-fatal): %s",
            session_id,
            exc,
        )
