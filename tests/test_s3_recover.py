"""Tests for S3 path recovery and session audio-path repair."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pawn_diarize.core.path_repair import PathRepairResult, repair_session_audio_paths
from pawn_diarize.core.s3 import (
    canonicalize_audio_paths,
    is_s3_path,
    recover_s3_uri,
    to_canonical_paths,
)


def test_recover_s3_uri_passthrough() -> None:
    uri = "s3://bucket/audio/file.flac"
    assert recover_s3_uri(uri, MagicMock()) == uri


def test_recover_s3_uri_from_tmp_basename() -> None:
    client = MagicMock()
    client.bucket = "recordings"
    page = {
        "Contents": [
            {"Key": "chunks/260226185101_01.flac"},
            {"Key": "chunks/other.flac"},
        ]
    }
    paginator = MagicMock()
    paginator.paginate.return_value = [page]
    client._client.get_paginator.return_value = paginator

    recovered = recover_s3_uri(
        "/tmp/pawnai_s3_abc/260226185101_01.flac",
        client,
    )
    assert recovered == "s3://recordings/chunks/260226185101_01.flac"


def test_recover_s3_uri_ambiguous_filename() -> None:
    client = MagicMock()
    client.bucket = "recordings"
    page = {
        "Contents": [
            {"Key": "a/same.flac"},
            {"Key": "b/same.flac"},
        ]
    }
    paginator = MagicMock()
    paginator.paginate.return_value = [page]
    client._client.get_paginator.return_value = paginator

    local = "/tmp/pawnai_s3_abc/same.flac"
    assert recover_s3_uri(local, client) == local


def test_canonicalize_audio_paths_remaps() -> None:
    client = MagicMock()
    client.bucket = "recordings"
    page = {"Contents": [{"Key": "x/one.flac"}]}
    paginator = MagicMock()
    paginator.paginate.return_value = [page]
    client._client.get_paginator.return_value = paginator

    paths = ["/tmp/pawnai_s3_x/one.flac", "s3://recordings/y/two.flac"]
    canonical, remaps = canonicalize_audio_paths(paths, client)
    assert is_s3_path(canonical[0])
    assert canonical[1] == "s3://recordings/y/two.flac"
    assert remaps["/tmp/pawnai_s3_x/one.flac"] == "s3://recordings/x/one.flac"


def test_canonicalize_without_client() -> None:
    paths = ["/tmp/foo.flac"]
    canonical, remaps = canonicalize_audio_paths(paths, None)
    assert canonical == paths
    assert remaps == {}


def test_to_canonical_paths_uses_path_map() -> None:
    local = "/tmp/pawn_diarize_s3_abc/chunk.flac"
    path_map = {local: "s3://bucket/audio/chunk.flac", "/local/x.wav": "/local/x.wav"}
    assert to_canonical_paths([local, "/local/x.wav"], path_map) == [
        "s3://bucket/audio/chunk.flac",
        "/local/x.wav",
    ]
    assert to_canonical_paths(["/unknown"], path_map) == ["/unknown"]


def test_repair_session_audio_paths_rewrites_db() -> None:
    """repair_session_audio_paths remaps segments, speaker_names, processed_files."""
    old = "/tmp/pawnai_s3_x/one.flac"
    new = "s3://recordings/x/one.flac"

    client = MagicMock()
    client.bucket = "recordings"
    page = {"Contents": [{"Key": "x/one.flac"}]}
    paginator = MagicMock()
    paginator.paginate.return_value = [page]
    client._client.get_paginator.return_value = paginator

    mock_engine = MagicMock()
    state = MagicMock()
    state.processed_files = [old, "s3://recordings/already.flac"]

    load_db = MagicMock()
    load_db.scalars.return_value = [old]
    load_db.get.return_value = state

    write_db = MagicMock()
    seg_result = MagicMock(rowcount=3)
    sn_result = MagicMock(rowcount=1)
    write_db.execute.side_effect = [seg_result, sn_result]
    write_state = MagicMock()
    write_state.processed_files = [old, "s3://recordings/already.flac"]
    write_db.get.return_value = write_state

    load_cm = MagicMock()
    load_cm.__enter__.return_value = load_db
    load_cm.__exit__.return_value = False

    write_cm = MagicMock()
    write_cm.__enter__.return_value = write_db
    write_cm.__exit__.return_value = False

    with (
        patch("pawn_diarize.core.path_repair.get_engine", return_value=mock_engine),
        patch("pawn_diarize.core.path_repair.init_db"),
        patch("pawn_diarize.core.path_repair.OrmSession", return_value=load_cm),
        patch("pawn_diarize.core.path_repair.get_session", return_value=write_cm),
    ):
        result = repair_session_audio_paths(
            "qwe",
            "postgresql+psycopg://unused",
            app_cfg=MagicMock(),
            s3_client=client,
        )

    assert isinstance(result, PathRepairResult)
    assert result.remaps == {old: new}
    assert result.segments_updated == 3
    assert result.speaker_names_updated == 1
    assert result.processed_files_updated is True
    assert write_state.processed_files == [new, "s3://recordings/already.flac"]


def test_repair_session_noop_when_no_remaps() -> None:
    client = MagicMock()
    client.bucket = "recordings"
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": []}]
    client._client.get_paginator.return_value = paginator

    load_db = MagicMock()
    load_db.scalars.return_value = ["s3://recordings/ok.flac"]
    load_db.get.return_value = None
    load_cm = MagicMock()
    load_cm.__enter__.return_value = load_db
    load_cm.__exit__.return_value = False

    with (
        patch("pawn_diarize.core.path_repair.get_engine", return_value=MagicMock()),
        patch("pawn_diarize.core.path_repair.init_db"),
        patch("pawn_diarize.core.path_repair.OrmSession", return_value=load_cm),
    ):
        result = repair_session_audio_paths(
            "ok",
            "postgresql+psycopg://unused",
            app_cfg={},
            s3_client=client,
        )
    assert result.remaps == {}
    assert "no paths needed" in result.summary()
