"""Tests for session speaker relabel (diarize core + agent wrapper)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pawn_agent.tools.session_relabel import session_relabel_impl
from pawn_diarize.core.session_relabel import (
    RelabelResult,
    _expand_aliases,
    relabel_session_speaker,
)


def test_relabel_rejects_empty_and_identical() -> None:
    with pytest.raises(ValueError, match="session id"):
        relabel_session_speaker("dsn", "", "SPEAKER_00", "Davide")
    with pytest.raises(ValueError, match="--from"):
        relabel_session_speaker("dsn", "s1", "  ", "Davide")
    with pytest.raises(ValueError, match="--to"):
        relabel_session_speaker("dsn", "s1", "SPEAKER_00", "")
    with pytest.raises(ValueError, match="same"):
        relabel_session_speaker("dsn", "s1", "Davide", "Davide")


def test_expand_aliases_includes_display_and_raw() -> None:
    db = MagicMock()
    db.execute.side_effect = [
        # SpeakerName rows
        MagicMock(
            all=lambda: [
                ("SPEAKER_00", "WrongPerson"),
                ("SPEAKER_01", "Alice"),
            ]
        ),
        # Embedding labels matching from_label
        MagicMock(scalars=lambda: MagicMock(all=lambda: ["SPEAKER_00"])),
    ]
    aliases = _expand_aliases(db, ["/audio/a.wav"], "SPEAKER_00")
    assert aliases == {"SPEAKER_00", "WrongPerson"}


def test_relabel_session_speaker_updates_and_creates_mappings() -> None:
    seg = SimpleNamespace(
        audio_file="/audio/a.wav",
        original_speaker_label="SPEAKER_00",
        start_time=0.0,
        text="hello",
        segment_index=0,
    )
    existing_sn = SimpleNamespace(
        audio_file="/audio/a.wav",
        local_speaker_label="SPEAKER_00",
        speaker_name="WrongPerson",
    )
    state = SimpleNamespace(
        speaker_embeddings={
            "WrongPerson": {"embedding": [0.1], "total_duration": 3.0},
            "Alice": {"embedding": [0.2], "total_duration": 1.0},
        }
    )

    # execute() call order inside relabel_session_speaker:
    # 1 session_files, 2 speaker_names for expand, 3 emb labels for expand,
    # 4 affected segs, 5 affected names, 6 existing sn keys, 7 emb pairs,
    # 8 update segments, 9 update speaker_names
    def _result(rows=None, scalars_list=None, rowcount=0):
        m = MagicMock()
        m.all.return_value = rows if rows is not None else []
        m.scalars.return_value = MagicMock(all=lambda: scalars_list or [])
        m.rowcount = rowcount
        return m

    mock_db = MagicMock()
    mock_db.execute.side_effect = [
        _result(scalars_list=["/audio/a.wav"]),  # session files
        _result(rows=[("SPEAKER_00", "WrongPerson")]),  # expand SN
        _result(scalars_list=["SPEAKER_00"]),  # expand emb
        _result(scalars_list=[seg]),  # affected segs
        _result(scalars_list=[existing_sn]),  # affected names
        _result(rows=[("/audio/a.wav", "SPEAKER_00")]),  # existing keys
        # emb pairs (no missing)
        _result(rows=[("/audio/a.wav", "SPEAKER_00")]),
        _result(rowcount=2),  # update segments
        _result(rowcount=1),  # update speaker_names
    ]
    mock_db.get.return_value = state

    mock_session_cm = MagicMock()
    mock_session_cm.__enter__.return_value = mock_db
    mock_session_cm.__exit__.return_value = False

    with (
        patch(
            "pawn_diarize.core.session_relabel.get_engine",
            return_value=MagicMock(),
        ),
        patch("pawn_diarize.core.session_relabel.init_db"),
        patch(
            "pawn_diarize.core.session_relabel.OrmSession",
            return_value=mock_session_cm,
        ),
    ):
        result = relabel_session_speaker(
            "postgresql+psycopg://x/y",
            "xyz",
            "SPEAKER_00",
            "Davide",
        )

    assert isinstance(result, RelabelResult)
    assert result.segments_updated == 2
    assert result.speaker_names_updated == 1
    assert result.speaker_names_created == 0
    assert result.session_state_updated is True
    assert "Davide" in state.speaker_embeddings
    assert "WrongPerson" not in state.speaker_embeddings
    assert "Alice" in state.speaker_embeddings
    mock_db.commit.assert_called_once()
    assert "Davide" in result.summary()


def test_session_relabel_impl_delegates() -> None:
    cfg = MagicMock(db_dsn="postgresql+psycopg://x/y")
    fake = RelabelResult(
        session_id="xyz",
        from_label="SPEAKER_00",
        to_label="Davide",
        aliases=("SPEAKER_00",),
        segments_updated=4,
        speaker_names_updated=1,
        speaker_names_created=0,
        session_state_updated=True,
    )
    with patch(
        "pawn_agent.tools.session_relabel.relabel_session_speaker",
        return_value=fake,
    ) as mock_core:
        text = session_relabel_impl(
            cfg,
            session_id="xyz",
            from_speaker="SPEAKER_00",
            to_speaker="Davide",
        )
    mock_core.assert_called_once_with(
        db_dsn=cfg.db_dsn,
        session_id="xyz",
        from_label="SPEAKER_00",
        to_label="Davide",
    )
    assert "4 segment(s)" in text
