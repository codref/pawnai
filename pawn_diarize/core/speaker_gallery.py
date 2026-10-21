"""Curated Speakers gallery: people registry, enrollments, and identification.

Design rules (intentional, please keep them):

1. Runtime diarization NEVER writes enrollments.  Unknown speakers stay as
   ``SPEAKER_XX`` until a human runs ``speakers enroll``.
2. Matching scores a probe against each speaker's *approved* enrollments only
   (max cosine), then applies threshold + margin.  Open-set reject is the
   default when the gallery is empty or the score is ambiguous.
3. Embeddings are JSON lists of floats so model dimension changes do not
   require an Alembic rewrite — a curated gallery is small enough for Python
   cosine scoring.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session as OrmSession

from pawn_core.config import SpeakersConfig
from pawn_core.database import SessionSpeakerMap, Speaker, SpeakerEnrollment
from pawn_diarize.core.database import Embedding, get_engine, get_session, init_db
from pawn_diarize.core.voice_embeddings import cosine_similarity

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_speaker_id(display_name: str) -> str:
    """Build a stable lowercase slug from a display name."""
    slug = _SLUG_RE.sub("-", display_name.strip().lower()).strip("-")
    return slug or f"speaker-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class IdentifyHit:
    """Result of matching one anonymous cluster against the gallery."""

    speaker_id: Optional[str]
    display_name: Optional[str]
    score: float
    second_score: float
    accepted: bool


@dataclass(frozen=True)
class EnrollmentResult:
    """Receipt for a successful enrollment write."""

    enrollment_id: str
    speaker_id: str
    display_name: str
    duration: float
    quality_score: Optional[float]
    embedding_model: str
    embedding_dim: int


class SpeakerGallery:
    """Read/write API for the curated Speakers gallery.

    Instantiate once per CLI/command with a DSN.  All methods open short-lived
    SQLAlchemy sessions; nothing is cached across calls (gallery is small).
    """

    def __init__(
        self,
        db_dsn: str,
        config: Optional[SpeakersConfig] = None,
    ) -> None:
        self.db_dsn = db_dsn
        self.config = config or SpeakersConfig()
        self._engine = get_engine(db_dsn)
        init_db(self._engine)

    # ── People registry ───────────────────────────────────────────────────

    def create_speaker(
        self,
        display_name: str,
        *,
        speaker_id: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
        notes: Optional[str] = None,
    ) -> Speaker:
        """Insert a new gallery person.  Raises ``ValueError`` on id clash."""
        sid = speaker_id or slugify_speaker_id(display_name)
        now = datetime.now(timezone.utc)
        row = Speaker(
            id=sid,
            display_name=display_name.strip(),
            aliases=list(aliases or []),
            notes=notes,
            active=True,
            created_at=now,
            updated_at=now,
        )
        with get_session(self._engine) as db:
            if db.get(Speaker, sid) is not None:
                raise ValueError(f"Speaker id already exists: {sid}")
            db.add(row)
            db.flush()
            db.expunge(row)
        return row

    def get_speaker(self, speaker_id: str) -> Optional[Speaker]:
        with OrmSession(self._engine) as db:
            row = db.get(Speaker, speaker_id)
            if row is None:
                return None
            db.expunge(row)
            return row

    def find_speaker_by_name(self, name: str) -> Optional[Speaker]:
        """Resolve by id, display_name, or alias (case-insensitive)."""
        needle = name.strip().lower()
        with OrmSession(self._engine) as db:
            rows = list(db.scalars(select(Speaker).where(Speaker.active.is_(True))))
            for row in rows:
                if row.id.lower() == needle or row.display_name.lower() == needle:
                    db.expunge(row)
                    return row
                aliases = row.aliases or []
                if any(str(a).lower() == needle for a in aliases):
                    db.expunge(row)
                    return row
        return None

    def list_speakers(self, *, include_inactive: bool = False) -> List[Speaker]:
        with OrmSession(self._engine) as db:
            stmt = select(Speaker).order_by(Speaker.display_name)
            if not include_inactive:
                stmt = stmt.where(Speaker.active.is_(True))
            rows = list(db.scalars(stmt))
            for row in rows:
                db.expunge(row)
            return rows

    def rename_speaker(self, speaker_id: str, display_name: str) -> Speaker:
        with get_session(self._engine) as db:
            row = db.get(Speaker, speaker_id)
            if row is None:
                raise ValueError(f"Unknown speaker id: {speaker_id}")
            row.display_name = display_name.strip()
            row.updated_at = datetime.now(timezone.utc)
            db.flush()
            db.expunge(row)
            return row

    def update_speaker(
        self,
        speaker_id: str,
        *,
        aliases: Optional[Sequence[str]] = None,
        notes: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> Speaker:
        """Update gallery card fields (aliases / short notes / display name).

        Pass ``notes=""`` to clear the short card. ``aliases`` replaces the
        list when provided (callers merge before invoking).
        """
        with get_session(self._engine) as db:
            row = db.get(Speaker, speaker_id)
            if row is None:
                raise ValueError(f"Unknown speaker id: {speaker_id}")
            if display_name is not None:
                row.display_name = display_name.strip()
            if aliases is not None:
                row.aliases = [str(a).strip() for a in aliases if str(a).strip()]
            if notes is not None:
                row.notes = notes.strip() or None
            row.updated_at = datetime.now(timezone.utc)
            db.flush()
            db.expunge(row)
            return row

    def deactivate_speaker(self, speaker_id: str) -> None:
        with get_session(self._engine) as db:
            row = db.get(Speaker, speaker_id)
            if row is None:
                raise ValueError(f"Unknown speaker id: {speaker_id}")
            row.active = False
            row.updated_at = datetime.now(timezone.utc)

    # ── Enrollments ───────────────────────────────────────────────────────

    def list_enrollments(self, speaker_id: Optional[str] = None) -> List[SpeakerEnrollment]:
        with OrmSession(self._engine) as db:
            stmt = select(SpeakerEnrollment).order_by(SpeakerEnrollment.approved_at)
            if speaker_id:
                stmt = stmt.where(SpeakerEnrollment.speaker_id == speaker_id)
            rows = list(db.scalars(stmt))
            for row in rows:
                db.expunge(row)
            return rows

    def remove_enrollment(self, enrollment_id: str) -> None:
        with get_session(self._engine) as db:
            row = db.get(SpeakerEnrollment, enrollment_id)
            if row is None:
                raise ValueError(f"Unknown enrollment id: {enrollment_id}")
            db.delete(row)

    def enroll(
        self,
        speaker: Speaker,
        embedding: np.ndarray,
        *,
        embedding_model: str,
        source_session_id: Optional[str] = None,
        source_audio_file: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        notes: Optional[str] = None,
        force: bool = False,
    ) -> EnrollmentResult:
        """Approve one voiceprint for *speaker*.

        Quality gates (unless *force*):
        - duration ≥ ``min_enrollment_seconds`` when times are known
        - enrollment count ≤ ``max_enrollments_per_speaker``
        - pairwise cosine vs existing same-speaker enrollments (same model)
          ≥ ``min_enrollment_pairwise`` when any exist
        """
        vec = np.asarray(embedding, dtype=np.float32).flatten()
        duration = 0.0
        if start_time is not None and end_time is not None:
            duration = max(0.0, float(end_time) - float(start_time))
            if duration < self.config.min_enrollment_seconds and not force:
                raise ValueError(
                    f"Enrollment span too short ({duration:.2f}s < "
                    f"{self.config.min_enrollment_seconds}s); pass force=True to override"
                )

        existing = self.list_enrollments(speaker.id)
        same_model = [e for e in existing if e.embedding_model == embedding_model]
        if len(same_model) >= self.config.max_enrollments_per_speaker and not force:
            raise ValueError(
                f"Speaker '{speaker.display_name}' already has "
                f"{len(same_model)} enrollments for model {embedding_model} "
                f"(max {self.config.max_enrollments_per_speaker}); "
                "remove one or pass force=True"
            )

        quality: Optional[float] = None
        if same_model:
            scores = [
                cosine_similarity(vec, np.asarray(e.embedding, dtype=np.float32))
                for e in same_model
                if e.embedding_dim == len(vec)
            ]
            if scores:
                quality = float(min(scores))
                if quality < self.config.min_enrollment_pairwise and not force:
                    raise ValueError(
                        f"Enrollment pairwise similarity {quality:.3f} is below "
                        f"{self.config.min_enrollment_pairwise}; the clip may be "
                        "noisy or a different person. Pass force=True to override."
                    )

        eid = str(uuid.uuid4())
        row = SpeakerEnrollment(
            id=eid,
            speaker_id=speaker.id,
            embedding=vec.tolist(),
            embedding_model=embedding_model,
            embedding_dim=int(vec.shape[0]),
            source_session_id=source_session_id,
            source_audio_file=source_audio_file,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            quality_score=quality,
            notes=notes,
            approved_at=datetime.now(timezone.utc),
        )
        with get_session(self._engine) as db:
            db.add(row)
        return EnrollmentResult(
            enrollment_id=eid,
            speaker_id=speaker.id,
            display_name=speaker.display_name,
            duration=duration,
            quality_score=quality,
            embedding_model=embedding_model,
            embedding_dim=int(vec.shape[0]),
        )

    # ── Identification ────────────────────────────────────────────────────

    def identify(
        self,
        probe: np.ndarray,
        *,
        embedding_model: Optional[str] = None,
        threshold: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> IdentifyHit:
        """Match *probe* against active gallery enrollments (open-set).

        Scoring: for each speaker, take the **max** cosine over that speaker's
        enrollments that share the probe's dimension (and optionally model).
        Accept only when best ≥ threshold AND (best − second) ≥ margin.
        """
        thr = self.config.identify_threshold if threshold is None else threshold
        mar = self.config.identify_margin if margin is None else margin
        probe_vec = np.asarray(probe, dtype=np.float32).flatten()
        probe_dim = int(probe_vec.shape[0])

        # speaker_id → best score among its enrollments
        best_by_speaker: Dict[str, Tuple[float, str]] = {}
        with OrmSession(self._engine) as db:
            rows = list(
                db.execute(
                    select(SpeakerEnrollment, Speaker)
                    .join(Speaker, Speaker.id == SpeakerEnrollment.speaker_id)
                    .where(Speaker.active.is_(True))
                ).all()
            )
            for enrollment, speaker in rows:
                if enrollment.embedding_dim != probe_dim:
                    continue
                if embedding_model and enrollment.embedding_model != embedding_model:
                    continue
                score = cosine_similarity(
                    probe_vec, np.asarray(enrollment.embedding, dtype=np.float32)
                )
                prev = best_by_speaker.get(speaker.id)
                if prev is None or score > prev[0]:
                    best_by_speaker[speaker.id] = (score, speaker.display_name)

        if not best_by_speaker:
            return IdentifyHit(None, None, 0.0, 0.0, False)

        ranked = sorted(best_by_speaker.items(), key=lambda kv: kv[1][0], reverse=True)
        best_id, (best_score, best_name) = ranked[0]
        second_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
        accepted = best_score >= thr and (best_score - second_score) >= mar
        if not accepted:
            return IdentifyHit(None, None, best_score, second_score, False)
        return IdentifyHit(best_id, best_name, best_score, second_score, True)

    def identify_clusters(
        self,
        cluster_embeddings: Dict[str, np.ndarray],
        *,
        embedding_model: Optional[str] = None,
        threshold: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> Dict[str, IdentifyHit]:
        """Identify many anonymous labels → :class:`IdentifyHit`."""
        return {
            label: self.identify(
                emb,
                embedding_model=embedding_model,
                threshold=threshold,
                margin=margin,
            )
            for label, emb in cluster_embeddings.items()
        }

    # ── Session map helpers ───────────────────────────────────────────────

    def upsert_session_map(
        self,
        session_id: str,
        local_label: str,
        *,
        speaker_id: Optional[str],
        display_name: Optional[str],
        match_score: Optional[float],
        match_method: str,
    ) -> None:
        row = SessionSpeakerMap(
            session_id=session_id,
            local_label=local_label,
            speaker_id=speaker_id,
            display_name=display_name,
            match_score=match_score,
            match_method=match_method,
            updated_at=datetime.now(timezone.utc),
        )
        with get_session(self._engine) as db:
            db.merge(row)

    def load_session_map(self, session_id: str) -> Dict[str, SessionSpeakerMap]:
        with OrmSession(self._engine) as db:
            rows = list(
                db.scalars(
                    select(SessionSpeakerMap).where(SessionSpeakerMap.session_id == session_id)
                )
            )
            out: Dict[str, SessionSpeakerMap] = {}
            for row in rows:
                db.expunge(row)
                out[row.local_label] = row
            return out

    def count_enrollments(self, *, active_only: bool = True) -> int:
        """Number of gallery enrollment vectors (optionally active speakers only)."""
        with OrmSession(self._engine) as db:
            stmt = select(func.count()).select_from(SpeakerEnrollment)
            if active_only:
                stmt = (
                    select(func.count())
                    .select_from(SpeakerEnrollment)
                    .join(Speaker, Speaker.id == SpeakerEnrollment.speaker_id)
                    .where(Speaker.active.is_(True))
                )
            return int(db.scalar(stmt) or 0)

    def enrollment_compatibility(
        self, probe: np.ndarray, *, embedding_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Summarise why identify() might see zero compatible enrollments."""
        probe_dim = int(np.asarray(probe, dtype=np.float32).flatten().shape[0])
        total = 0
        dim_mismatch = 0
        model_mismatch = 0
        compatible = 0
        models: Dict[str, int] = {}
        dims: Dict[int, int] = {}
        with OrmSession(self._engine) as db:
            rows = list(
                db.execute(
                    select(SpeakerEnrollment, Speaker)
                    .join(Speaker, Speaker.id == SpeakerEnrollment.speaker_id)
                    .where(Speaker.active.is_(True))
                ).all()
            )
            for enrollment, _speaker in rows:
                total += 1
                mid = enrollment.embedding_model or ""
                models[mid] = models.get(mid, 0) + 1
                dims[int(enrollment.embedding_dim)] = dims.get(int(enrollment.embedding_dim), 0) + 1
                if enrollment.embedding_dim != probe_dim:
                    dim_mismatch += 1
                    continue
                if embedding_model and enrollment.embedding_model != embedding_model:
                    model_mismatch += 1
                    continue
                compatible += 1
        return {
            "probe_dim": probe_dim,
            "probe_model": embedding_model,
            "total": total,
            "compatible": compatible,
            "dim_mismatch": dim_mismatch,
            "model_mismatch": model_mismatch,
            "enrollment_models": models,
            "enrollment_dims": dims,
        }

    # ── Legacy cleanup ────────────────────────────────────────────────────

    def purge_legacy_embeddings(self) -> int:
        """Delete all rows from the old auto-accumulating ``embeddings`` table.

        The curated gallery does not read that table.  Call this once after
        migrating to Speakers so polluted vectors stop confusing tools that
        still inspect the legacy store.
        """
        with get_session(self._engine) as db:
            count = db.scalar(select(func.count()).select_from(Embedding)) or 0
            db.execute(delete(Embedding))
        return int(count)

    def legacy_embedding_count(self) -> int:
        with OrmSession(self._engine) as db:
            return int(db.scalar(select(func.count()).select_from(Embedding)) or 0)


def duration_weighted_mean(embeddings: Iterable[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Mean of segment embeddings weighted by duration (skips synthetic priors)."""
    usable = [
        e for e in embeddings if not e.get("synthetic", False) and e.get("embedding") is not None
    ]
    if not usable:
        return None
    durations = np.array(
        [max(1e-3, float(e["end"]) - float(e["start"])) for e in usable],
        dtype=np.float64,
    )
    weights = durations / durations.sum()
    stacked = np.stack([np.asarray(e["embedding"], dtype=np.float32).flatten() for e in usable])
    mean = np.average(stacked, axis=0, weights=weights)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        return mean.astype(np.float32)
    return (mean / norm).astype(np.float32)
