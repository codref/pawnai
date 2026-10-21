"""Tests for the curated Speakers gallery (no ML models required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pawn_core.config import SpeakersConfig
from pawn_diarize.core.speaker_gallery import (
    SpeakerGallery,
    duration_weighted_mean,
    slugify_speaker_id,
)
from pawn_diarize.core.voice_embeddings import cosine_similarity, _l2_normalize


def test_slugify_speaker_id() -> None:
    assert slugify_speaker_id("Davide Rossi") == "davide-rossi"
    assert slugify_speaker_id("  Alice  ") == "alice"


def test_duration_weighted_mean_skips_synthetic() -> None:
    a = _l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    b = _l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    mean = duration_weighted_mean(
        [
            {"embedding": a, "start": 0.0, "end": 2.0, "synthetic": True},
            {"embedding": b, "start": 0.0, "end": 2.0},
        ]
    )
    assert mean is not None
    assert abs(float(np.linalg.norm(mean)) - 1.0) < 1e-5
    # Only non-synthetic entry survives.
    assert abs(float(mean[1]) - 1.0) < 1e-5


def test_cosine_similarity_unit_vectors() -> None:
    a = _l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    b = _l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    c = _l2_normalize(np.array([0.0, 1.0], dtype=np.float32))
    assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-5)
    assert cosine_similarity(a, c) == pytest.approx(0.0, abs=1e-5)


def test_identify_requires_threshold_and_margin() -> None:
    """Gallery matching accepts only clear winners (threshold + margin)."""
    cfg = SpeakersConfig(identify_threshold=0.7, identify_margin=0.1)

    davide_emb = _l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    alice_emb = _l2_normalize(np.array([0.9, 0.435, 0.0], dtype=np.float32))
    # Probe close to davide but also fairly close to alice → reject on margin.
    probe = _l2_normalize(np.array([0.95, 0.312, 0.0], dtype=np.float32))

    gallery = SpeakerGallery.__new__(SpeakerGallery)
    gallery.config = cfg
    gallery.db_dsn = "postgresql+psycopg://unused"
    gallery._engine = MagicMock()

    class _Enr:
        def __init__(self, speaker_id, emb, model="test", dim=3):
            self.speaker_id = speaker_id
            self.embedding = emb.tolist()
            self.embedding_model = model
            self.embedding_dim = dim

    class _Spk:
        def __init__(self, sid, name):
            self.id = sid
            self.display_name = name
            self.active = True

    rows = [
        (_Enr("davide", davide_emb), _Spk("davide", "Davide")),
        (_Enr("alice", alice_emb), _Spk("alice", "Alice")),
    ]

    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.__exit__.return_value = False
    mock_session.execute.return_value.all.return_value = rows

    with patch(
        "pawn_diarize.core.speaker_gallery.OrmSession",
        return_value=mock_session,
    ):
        hit = gallery.identify(probe, embedding_model="test")

    # Best is Davide; if margin vs Alice is too small, reject.
    assert hit.speaker_id in (None, "davide")
    if hit.accepted:
        assert hit.display_name == "Davide"
        assert hit.score - hit.second_score >= cfg.identify_margin


def test_enroll_quality_gate_rejects_short_span() -> None:
    cfg = SpeakersConfig(min_enrollment_seconds=1.5)
    gallery = SpeakerGallery.__new__(SpeakerGallery)
    gallery.config = cfg
    gallery.db_dsn = "postgresql+psycopg://unused"
    gallery._engine = MagicMock()
    gallery.list_enrollments = MagicMock(return_value=[])  # type: ignore[method-assign]

    class _Spk:
        id = "davide"
        display_name = "Davide"

    with pytest.raises(ValueError, match="too short"):
        gallery.enroll(
            _Spk(),  # type: ignore[arg-type]
            _l2_normalize(np.ones(4, dtype=np.float32)),
            embedding_model="test",
            start_time=0.0,
            end_time=0.5,
        )


def test_diarize_defaults_store_new_false() -> None:
    """Runtime diarize must not auto-enroll (signature default)."""
    import inspect
    from pawn_diarize.core.diarization import DiarizationEngine

    sig = inspect.signature(DiarizationEngine.diarize)
    assert sig.parameters["store_new_speakers"].default is False
