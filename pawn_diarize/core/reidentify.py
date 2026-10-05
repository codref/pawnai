"""Re-identify and re-diarize helpers for curated Speakers gallery workflows.

``reidentify_session``
    Keeps existing turn boundaries / transcript text.  Re-extracts (or reuses)
    cluster embeddings and rematches them against the gallery, then rewrites
    ``transcription_segments.original_speaker_label`` and ``session_speaker_map``.

``rediarize_session``
    Re-fetches audio for the session, re-runs the anonymous diarization backend
    + gallery identification, and replaces segment speaker labels.  Transcription
    text is preserved when possible by midpoint merge; if audio is missing the
    call fails clearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session as OrmSession

from pawn_core.config import SpeakersConfig
from pawn_core.database import TranscriptionSegment
from pawn_diarize.core.database import (
    SessionState,
    get_engine,
    get_session,
    init_db,
    load_session_state,
    save_session_state,
)
from pawn_diarize.core.speaker_gallery import SpeakerGallery, duration_weighted_mean


@dataclass(frozen=True)
class ReidentifyResult:
    session_id: str
    segments_updated: int
    matched: Dict[str, str]
    unmatched: Tuple[str, ...]

    def summary(self) -> str:
        parts = [
            f"Reidentified session '{self.session_id}': "
            f"{self.segments_updated} segment(s) updated"
        ]
        if self.matched:
            mapped = ", ".join(f"{k}→{v}" for k, v in sorted(self.matched.items()))
            parts.append(f"matched: {mapped}")
        if self.unmatched:
            parts.append(f"unmatched: {', '.join(self.unmatched)}")
        return "; ".join(parts)


def _session_segments(engine, session_id: str) -> List[TranscriptionSegment]:
    with OrmSession(engine) as db:
        rows = list(
            db.scalars(
                select(TranscriptionSegment)
                .where(TranscriptionSegment.session_id == session_id)
                .order_by(TranscriptionSegment.segment_index)
            )
        )
        for row in rows:
            db.expunge(row)
        return rows


def _cluster_labels_from_segments(
    segments: Sequence[TranscriptionSegment],
) -> Dict[str, List[TranscriptionSegment]]:
    """Group segments by their current local/display label."""
    groups: Dict[str, List[TranscriptionSegment]] = {}
    for seg in segments:
        label = seg.original_speaker_label or "SPEAKER_00"
        groups.setdefault(label, []).append(seg)
    return groups


def reidentify_session(
    session_id: str,
    db_dsn: str,
    *,
    threshold: Optional[float] = None,
    margin: Optional[float] = None,
    device: str = "auto",
    embedding_model: Optional[str] = None,
    hf_token: Optional[str] = None,
    speakers_config: Optional[SpeakersConfig] = None,
) -> ReidentifyResult:
    """Rematch an existing session against the curated gallery.

    Uses duration-weighted means of *session_state.speaker_embeddings* when
    present; otherwise falls back to equal-weight placeholders and leaves
    unmatched labels untouched (cannot invent embeddings without audio).
    When session centroids exist, identification does not need audio files.
    """
    from pawn_diarize.core.diarization import DiarizationEngine  # noqa: PLC0415

    engine = get_engine(db_dsn)
    init_db(engine)
    cfg = speakers_config or SpeakersConfig()
    gallery = SpeakerGallery(db_dsn, config=cfg)
    thr = cfg.identify_threshold if threshold is None else threshold
    mar = cfg.identify_margin if margin is None else margin

    segments = _session_segments(engine, session_id)
    if not segments:
        raise ValueError(f"No transcription segments for session '{session_id}'")

    prior, _, _, _ = load_session_state(session_id, engine)
    # Build probe vectors from session_state centroids keyed by current labels.
    probes: Dict[str, np.ndarray] = {}
    for label, info in (prior or {}).items():
        emb = info.get("embedding") if isinstance(info, dict) else None
        if emb is None:
            continue
        vec = np.asarray(emb, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        probes[label] = vec

    if not probes:
        raise ValueError(
            f"Session '{session_id}' has no speaker_embeddings in session_state; "
            "run rediarize (needs audio) or re-process a chunk first"
        )

    # Optionally warm the extractor so model_id filtering matches enrollments.
    model_id = embedding_model
    if model_id is None:
        try:
            eng = DiarizationEngine(device=device, embedding_model=embedding_model, hf_token=hf_token)
            eng._initialize_models()
            model_id = getattr(eng._extractor, "model_id", None)
        except Exception:  # noqa: BLE001
            model_id = None

    matched: Dict[str, str] = {}
    unmatched: List[str] = []
    for label, probe in probes.items():
        hit = gallery.identify(probe, embedding_model=model_id, threshold=thr, margin=mar)
        if not hit.accepted:
            hit = gallery.identify(probe, embedding_model=None, threshold=thr, margin=mar)
        if hit.accepted and hit.display_name:
            matched[label] = hit.display_name
            gallery.upsert_session_map(
                session_id,
                label,
                speaker_id=hit.speaker_id,
                display_name=hit.display_name,
                match_score=hit.score,
                match_method="gallery",
            )
        else:
            unmatched.append(label)

    updated = 0
    if matched:
        with get_session(engine) as db:
            for from_label, to_name in matched.items():
                if from_label == to_name:
                    continue
                result = db.execute(
                    update(TranscriptionSegment)
                    .where(TranscriptionSegment.session_id == session_id)
                    .where(TranscriptionSegment.original_speaker_label == from_label)
                    .values(original_speaker_label=to_name)
                )
                updated += int(result.rowcount or 0)

        # Rename session_state keys to display names.
        new_prior: Dict[str, Any] = {}
        for label, info in (prior or {}).items():
            new_prior[matched.get(label, label)] = info
        cursor = 0.0
        processed: List[str] = []
        with OrmSession(engine) as db:
            row = db.get(SessionState, session_id)
            if row is not None:
                cursor = float(row.time_cursor or 0.0)
                processed = list(row.processed_files or [])
        save_session_state(session_id, new_prior, cursor, processed, engine)

    return ReidentifyResult(
        session_id=session_id,
        segments_updated=updated,
        matched=matched,
        unmatched=tuple(unmatched),
    )


def rediarize_session(
    session_id: str,
    db_dsn: str,
    *,
    audio_paths: Optional[List[str]] = None,
    device: str = "auto",
    threshold: float = 0.7,
    margin: float = 0.05,
    cross_file_threshold: float = 0.55,
    diarization_backend: str = "pyannote",
    diarization_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    hf_token: Optional[str] = None,
    source_map: Optional[Dict[str, str]] = None,
) -> ReidentifyResult:
    """Re-run anonymous diarization + gallery identify for a session's audio.

    Pass *audio_paths* (local, already-resolved) when segment rows store
    ``s3://`` URIs.  *source_map* should map those local temps back to the
    canonical URIs stored on segments.

    Only **named** prior centroids (e.g. ``Tom``) are reused for sticky labels.
    Anonymous ``SPEAKER_XX`` priors from a previous fragmented run are ignored
    so they cannot pin new clusters to old junk labels.
    """
    from pawn_diarize.core.diarization import (  # noqa: PLC0415
        DiarizationEngine,
        is_anonymous_speaker_label,
    )

    engine = get_engine(db_dsn)
    init_db(engine)
    segments = _session_segments(engine, session_id)
    if not segments:
        raise ValueError(f"No transcription segments for session '{session_id}'")

    if audio_paths:
        files = list(audio_paths)
    else:
        files = []
        for seg in segments:
            path = seg.audio_file
            if path and path not in files:
                files.append(path)
    if not files:
        raise ValueError(f"Session '{session_id}' has no audio_file paths on segments")

    prior, _, _, _ = load_session_state(session_id, engine)
    named_prior = {
        label: info
        for label, info in (prior or {}).items()
        if not is_anonymous_speaker_label(str(label))
    }
    if prior and not named_prior:
        print(
            "Note: session_state has only anonymous SPEAKER_XX centroids; "
            "ignoring them for cross-file naming (fresh SPEAKER_00… labels)."
        )

    diar_engine = DiarizationEngine(
        device=device,
        diarization_backend=diarization_backend,
        diarization_model=diarization_model,
        embedding_model=embedding_model,
        hf_token=hf_token,
        identify_margin=margin,
    )
    result = diar_engine.diarize(
        files if len(files) > 1 else files[0],
        db_dsn=db_dsn,
        similarity_threshold=threshold,
        store_new_speakers=False,
        cross_file_threshold=cross_file_threshold,
        prior_speaker_embeddings=named_prior or None,
        time_cursor=0.0,
        session_id=session_id,
        identify_margin=margin,
        source_map=source_map,
    )

    new_turns = result.get("segments") or []
    if not new_turns:
        raise ValueError(
            f"Rediarize of session '{session_id}' produced 0 diarization turns "
            f"from {len(files)} audio file(s). Speaker labels were left unchanged. "
            f"Check that the nemotron/pyannote backend is returning parseable output."
        )

    updated = 0
    with get_session(engine) as db:
        rows = list(
            db.scalars(
                select(TranscriptionSegment)
                .where(TranscriptionSegment.session_id == session_id)
                .order_by(TranscriptionSegment.segment_index)
            )
        )
        for row in rows:
            mid = (float(row.start_time) + float(row.end_time)) / 2.0
            speaker = None
            for turn in new_turns:
                if float(turn["start"]) <= mid < float(turn["end"]):
                    speaker = turn.get("speaker")
                    break
            if speaker and speaker != row.original_speaker_label:
                row.original_speaker_label = speaker
                updated += 1

        state = db.get(SessionState, session_id)
        now = datetime.now(timezone.utc)
        embeddings = result.get("session_speaker_embeddings") or {}
        cursor = float(result.get("new_time_cursor") or 0.0)
        if state is None:
            db.add(
                SessionState(
                    session_id=session_id,
                    processed_files=files,
                    speaker_embeddings=embeddings,
                    time_cursor=cursor,
                    updated_at=now,
                )
            )
        else:
            if embeddings:
                state.speaker_embeddings = embeddings
            state.time_cursor = cursor or float(state.time_cursor or 0.0)
            state.updated_at = now

    return ReidentifyResult(
        session_id=session_id,
        segments_updated=updated,
        matched=dict(result.get("matched_speakers") or {}),
        unmatched=tuple(result.get("new_speakers") or ()),
    )
