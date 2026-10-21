"""Tests for diarization backend model-id resolution (no ML loads)."""

from __future__ import annotations

from unittest.mock import patch

from pawn_diarize.core.diar_backends import (
    DEFAULT_NEMOTRON_MODEL,
    DEFAULT_PYANNOTE_MODEL,
    FALLBACK_SORTFORMER_MODEL,
    _is_rope_unsupported_error,
    _normalize_nemotron_output,
    _speaker_label,
    resolve_diarization_model_id,
    resolve_nemotron_runtime_model_id,
)


def test_nemotron_overrides_pyannote_model_id() -> None:
    assert (
        resolve_diarization_model_id(
            "nemotron", "pyannote/speaker-diarization-community-1"
        )
        == DEFAULT_NEMOTRON_MODEL
    )


def test_nemotron_keeps_explicit_nvidia_id() -> None:
    mid = "nvidia/Nemotron-3-Diarization"
    assert resolve_diarization_model_id("nemotron", mid) == mid


def test_nemotron_empty_uses_default() -> None:
    assert resolve_diarization_model_id("nemotron", None) == DEFAULT_NEMOTRON_MODEL
    assert resolve_diarization_model_id("nemotron", "") == DEFAULT_NEMOTRON_MODEL


def test_pyannote_overrides_nvidia_id() -> None:
    assert (
        resolve_diarization_model_id("pyannote", "nvidia/Nemotron-3-Diarization")
        == DEFAULT_PYANNOTE_MODEL
    )


def test_pyannote_keeps_community_id() -> None:
    mid = "pyannote/speaker-diarization-community-1"
    assert resolve_diarization_model_id("pyannote", mid) == mid


def test_nemotron_runtime_falls_back_without_rope() -> None:
    with patch(
        "pawn_diarize.core.diar_backends.nemo_supports_rope", return_value=False
    ):
        assert (
            resolve_nemotron_runtime_model_id(DEFAULT_NEMOTRON_MODEL)
            == FALLBACK_SORTFORMER_MODEL
        )


def test_nemotron_runtime_keeps_nemotron3_with_rope() -> None:
    with patch(
        "pawn_diarize.core.diar_backends.nemo_supports_rope", return_value=True
    ):
        assert (
            resolve_nemotron_runtime_model_id(DEFAULT_NEMOTRON_MODEL)
            == DEFAULT_NEMOTRON_MODEL
        )


def test_nemotron_runtime_keeps_explicit_sortformer() -> None:
    with patch(
        "pawn_diarize.core.diar_backends.nemo_supports_rope", return_value=False
    ):
        assert (
            resolve_nemotron_runtime_model_id(FALLBACK_SORTFORMER_MODEL)
            == FALLBACK_SORTFORMER_MODEL
        )


def test_rope_error_detection() -> None:
    assert _is_rope_unsupported_error(
        ValueError(
            "self_attention_model='rope' is not supported. "
            "Currently only 'abs_pos', 'rel_pos', and 'no_pos' are available."
        )
    )
    assert not _is_rope_unsupported_error(ValueError("CUDA OOM"))


def test_normalize_space_separated_speaker_lines() -> None:
    # Flat list after NeMo one-file results.extend
    raw = ["0.500 3.120 speaker_0", "3.510 7.260 speaker_1"]
    turns = _normalize_nemotron_output(raw)
    assert turns == [
        {"speaker": "SPEAKER_00", "start": 0.5, "end": 3.12},
        {"speaker": "SPEAKER_01", "start": 3.51, "end": 7.26},
    ]


def test_normalize_single_line_does_not_unwrap_to_string() -> None:
    turns = _normalize_nemotron_output(["1.000 2.000 speaker_0"])
    assert turns == [{"speaker": "SPEAKER_00", "start": 1.0, "end": 2.0}]


def test_normalize_bracket_string_and_wrapped_batch() -> None:
    raw = [["[0.08, 1.52, 0]", "[1.60, 3.20, 1]"]]
    turns = _normalize_nemotron_output(raw)
    assert turns == [
        {"speaker": "SPEAKER_00", "start": 0.08, "end": 1.52},
        {"speaker": "SPEAKER_01", "start": 1.6, "end": 3.2},
    ]


def test_normalize_numeric_triples_not_over_unwrapped() -> None:
    raw = [[[0.08, 1.52, 0], [1.6, 3.2, 1]]]
    turns = _normalize_nemotron_output(raw)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "SPEAKER_00"
    assert turns[1]["end"] == 3.2


def test_speaker_label_variants() -> None:
    assert _speaker_label("speaker_0") == "SPEAKER_00"
    assert _speaker_label("speaker0") == "SPEAKER_00"
    assert _speaker_label(3) == "SPEAKER_03"
    assert _speaker_label("SPEAKER_07") == "SPEAKER_07"
