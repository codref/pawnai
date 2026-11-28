"""Sessions list, transcript, audio range and bulk reidentify."""

from __future__ import annotations

import wave
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from pawn_agent.utils.config import ApiSection
from pawn_agent.utils.db import CoworkerItem, VaultTask, get_engine
from pawn_core.config import SpeakersConfig, VaultConfig
from pawn_core.database import SessionAnalysis, SessionCapture, TranscriptionSegment
from pawn_diarize.core.database import SessionState
from pawn_server.core import api_server
from pawn_server.core import sessions_api
from pawn_server.core.sessions_api import (
    audio_signature_ok,
    merge_turns,
    parse_byte_range,
    resolve_file_offsets,
    select_samples,
)

AUTH = {"Authorization": "Bearer tok"}


@compiles(JSONB, "sqlite")
def _jsonb_as_json(element, compiler, **kw):  # noqa: ANN001
    return "JSON"


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        api=ApiSection(token="tok", stream_keepalive_seconds=0.05),
        api_token="tok",
        api_model_idle_timeout_minutes=10.0,
        db_dsn=f"sqlite:///{tmp_path / 'sessions.db'}",
        vault=VaultConfig(agent_root="Pawn", bucket=""),
        speakers=SpeakersConfig(),
        models=SimpleNamespace(embedding_model="test-embed", hf_token=None),
        s3=None,
    )


def _prepare(cfg: SimpleNamespace) -> None:
    engine = get_engine(cfg.db_dsn)
    for table in (
        TranscriptionSegment.__table__,
        SessionAnalysis.__table__,
        SessionState.__table__,
        SessionCapture.__table__,
        CoworkerItem.__table__,
        VaultTask.__table__,
    ):
        table.create(engine, checkfirst=True)


def _segment(
    session_id: str,
    index: int,
    *,
    label: str,
    start: float,
    end: float,
    text: str,
    created: datetime,
    audio: str = "clip.webm",
    words: list | None = None,
) -> TranscriptionSegment:
    return TranscriptionSegment(
        id=f"{session_id}-{index}",
        session_id=session_id,
        audio_file=audio,
        original_speaker_label=label,
        start_time=start,
        end_time=end,
        text=text,
        words=words,
        segment_index=index,
        created_at=created,
    )


def test_merge_turns_and_samples() -> None:
    turns = merge_turns(
        [
            {"id": "b", "label": "SPEAKER_00", "start": 5, "end": 8, "text": "next"},
            {"id": "a", "label": "SPEAKER_00", "start": 0, "end": 4, "text": "hello"},
            {
                "id": "c",
                "label": "Ada",
                "start": 8,
                "end": 9,
                "text": "hi",
                "words": [{"text": "hi"}],
            },
        ],
        include_words=True,
    )
    assert [turn["label"] for turn in turns] == ["SPEAKER_00", "Ada"]
    assert turns[0]["id"] == "a"
    assert [seg["id"] for seg in turns[0]["segments"]] == ["a", "b"]
    assert turns[0]["segments"][0]["words"] == []
    assert turns[1]["segments"][0]["words"] == [{"text": "hi"}]

    samples = select_samples(
        [
            {"start": 0, "end": 1, "text": "short"},
            {"start": 10, "end": 40, "text": "longest"},
            {"start": 2, "end": 2.2, "text": "   "},
            {"start": 4, "end": 12, "text": "mid"},
            {"start": 50, "end": 70, "text": "second"},
        ]
    )
    assert [item["text"] for item in samples] == ["mid", "longest", "second"]


def test_byte_range_and_signature_expiry() -> None:
    assert parse_byte_range(None, 20) is None
    assert parse_byte_range("bytes=0-10", 20) == (0, 10)
    assert parse_byte_range("bytes=15-", 20) == (15, 19)
    with pytest.raises(sessions_api.SessionsApiError) as exc:
        parse_byte_range("bytes=30-40", 20)
    assert exc.value.status_code == 416
    assert exc.value.headers["Content-Range"] == "bytes */20"

    assert audio_signature_ok("tok", "meet", 0, 100, "nope", now=50) is False
    good = sessions_api.audio_signature("tok", "meet", 0, 100)
    assert audio_signature_ok("tok", "meet", 0, 100, good, now=100) is True
    assert audio_signature_ok("tok", "meet", 0, 100, good, now=101) is False


def test_offset_fallback_accumulates_probed_durations() -> None:
    stored = [{"path": "a.wav", "start": 0.0, "end": 10.0}]
    known, probed = resolve_file_offsets(["a.wav"], stored, lambda _path: 99)
    assert probed is False
    assert known == [{"path": "a.wav", "offset": 0.0, "duration": 10.0}]

    rows, probed = resolve_file_offsets(
        ["a.wav", "b.wav"],
        stored,
        lambda path: 5.0 if path == "b.wav" else 0.0,
    )
    assert probed is True
    assert rows[0]["offset"] == 0.0
    assert rows[1] == {"path": "b.wav", "offset": 10.0, "duration": 5.0}


def test_list_paging_facets_and_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    _prepare(cfg)
    day = datetime(2026, 11, 27, 10, 0, tzinfo=timezone.utc)
    older = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)
    engine = get_engine(cfg.db_dsn)
    with Session(engine) as db:
        db.add_all(
            [
                _segment("alpha", 0, label="Ada", start=0, end=30, text="plan", created=day),
                _segment("beta", 0, label="SPEAKER_00", start=0, end=5, text="who", created=older),
                _segment(
                    "gamma",
                    0,
                    label="Bob",
                    start=0,
                    end=2,
                    text="short",
                    created=datetime(2026, 11, 26, 8, 0, tzinfo=timezone.utc),
                ),
            ]
        )
        db.add(
            SessionAnalysis(
                id="an-1",
                session_id="alpha",
                source="session:alpha",
                model="test",
                title="Weekly",
                summary="Roadmap",
                key_topics="budget",
                tags=["weekly"],
                sentiment="calm",
                analyzed_at=day,
            )
        )
        db.add(
            CoworkerItem(
                id="item-1",
                short_id="i1",
                source_kind="session",
                source_ref="alpha",
                kind="todo",
                text="Send notes",
                fingerprint="fp-1",
                status="new",
                created_at=day,
            )
        )
        db.commit()

    monkeypatch.setattr(
        sessions_api,
        "_events_by_session",
        lambda _cfg, ids: {"alpha": {"id": "ev-1", "title": "Weekly"}} if "alpha" in ids else {},
    )
    page = sessions_api.list_sessions_page(cfg, limit=1, offset=0, sort="recent")
    assert page["facets"]["total"] == 3
    assert page["facets"]["needs_speakers"] == 1
    assert page["facets"]["tags"][0] == {"name": "weekly", "count": 1}
    assert page["data"][0]["session_id"] == "alpha"
    assert page["data"][0]["items_open"] == 1
    assert page["data"][0]["event"]["title"] == "Weekly"

    older_page = sessions_api.list_sessions_page(cfg, limit=10, sort="oldest")
    assert [row["session_id"] for row in older_page["data"]] == ["beta", "gamma", "alpha"]

    window = sessions_api.list_sessions_page(cfg, date_from="2026-11-26", date_to="2026-11-27")
    assert {row["session_id"] for row in window["data"]} == {"alpha", "gamma"}

    analyzed = sessions_api.list_sessions_page(cfg, has_analysis=True)
    assert [row["session_id"] for row in analyzed["data"]] == ["alpha"]

    needy = sessions_api.list_sessions_page(cfg, needs_speakers=True)
    assert [row["session_id"] for row in needy["data"]] == ["beta"]

    second = sessions_api.list_sessions_page(cfg, limit=1, offset=1, sort="duration")
    assert second["facets"]["total"] == 3
    assert len(second["data"]) == 1


def test_transcript_merges_turns_and_captures(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _prepare(cfg)
    created = datetime(2026, 11, 27, tzinfo=timezone.utc)
    engine = get_engine(cfg.db_dsn)
    with Session(engine) as db:
        db.add_all(
            [
                _segment(
                    "meet",
                    0,
                    label="SPEAKER_00",
                    start=0,
                    end=4,
                    text="hello",
                    created=created,
                    audio="a.webm",
                    words=[{"word": "hello", "start": 0.1, "end": 0.4}],
                ),
                _segment(
                    "meet",
                    1,
                    label="SPEAKER_00",
                    start=4,
                    end=6,
                    text="there",
                    created=created,
                    audio="a.webm",
                ),
                _segment(
                    "meet",
                    2,
                    label="Ada",
                    start=6,
                    end=9,
                    text="ok",
                    created=created,
                    audio="b.webm",
                ),
            ]
        )
        db.add(
            SessionCapture(
                session_id="meet",
                item_id="shot-1",
                kind="screenshot",
                at=created,
                at_offset_minutes=0,
                text="board",
                vault_key="Pawn/Captures/board.png",
                audio_offset_s=5.0,
                chunk_audio_start=0.0,
                received_at=created,
            )
        )
        db.commit()

    body = sessions_api.session_transcript(cfg, "meet", words=True)
    assert [turn["label"] for turn in body["turns"]] == ["SPEAKER_00", "Ada"]
    assert body["turns"][0]["segments"][0]["words"][0]["text"] == "hello"
    assert body["turns"][0]["segments"][1]["words"] == []
    assert body["captures"][0]["t"] == 5.0
    assert body["captures"][0]["image_key"] == "Pawn/Captures/board.png"
    assert [item["audio_file"] for item in body["files"]] == ["a.webm", "b.webm"]
    plain = sessions_api.session_transcript(cfg, "meet", words=False)
    assert "words" not in plain["turns"][0]["segments"][0]


def _silent_wav(path: Path, frames: int = 800) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * frames)


def test_audio_range_signature_and_codec(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _prepare(cfg)
    wav_path = tmp_path / "talk.wav"
    _silent_wav(wav_path)
    raw_path = tmp_path / "talk.pcm"
    raw_path.write_bytes(b"\x00" * 20)
    created = datetime(2026, 11, 27, tzinfo=timezone.utc)
    engine = get_engine(cfg.db_dsn)
    with Session(engine) as db:
        db.add(
            _segment(
                "meet",
                0,
                label="Ada",
                start=0,
                end=0.1,
                text="hi",
                created=created,
                audio=str(wav_path),
            )
        )
        db.add(
            _segment(
                "raw",
                0,
                label="Ada",
                start=0,
                end=0.1,
                text="hi",
                created=created,
                audio=str(raw_path),
            )
        )
        db.commit()

    with TestClient(api_server.create_app(cfg)) as client:
        issued = client.post("/v1/sessions/meet/audio-url", headers=AUTH)
        assert issued.status_code == 200, issued.text
        signed = urlparse(issued.json()["files"][0]["url"])
        partial = client.get(f"{signed.path}?{signed.query}", headers={"Range": "bytes=0-10"})
        assert partial.status_code == 206, partial.text
        assert partial.headers["content-range"].startswith("bytes 0-10/")
        assert len(partial.content) == 11
        assert partial.headers["content-type"].startswith("audio/wav")

        too_far = client.get(
            f"{signed.path}?{signed.query}", headers={"Range": "bytes=999999-999999"}
        )
        assert too_far.status_code == 416

        missing = client.get(signed.path)
        assert missing.status_code == 401

        bearer = client.get(
            "/v1/sessions/meet/audio/0",
            headers={**AUTH, "Range": "bytes=0-3"},
        )
        assert bearer.status_code == 206
        assert len(bearer.content) == 4

        rejected = client.get("/v1/sessions/raw/audio/0", headers=AUTH)
        assert rejected.status_code == 415


def test_bulk_reidentify_returns_a_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    _prepare(cfg)
    seen: list[str] = []

    def fake_reidentify(cfg, session_id):  # noqa: ANN001
        seen.append(session_id)
        return {"session_id": session_id}

    monkeypatch.setattr(sessions_api, "reidentify_session_api", fake_reidentify)
    with TestClient(api_server.create_app(cfg)) as client:
        resp = client.post(
            "/v1/sessions/reidentify",
            headers=AUTH,
            json={"session_ids": ["a", "b", "c"], "limit": 2},
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["session_ids"] == ["a", "b"]
        assert body["queued"] is True
    assert seen == ["a", "b"]
    engine = get_engine(cfg.db_dsn)
    with Session(engine) as db:
        row = db.get(VaultTask, body["job_id"])
        assert row is not None
        assert row.status == "done"


def _flac_header(rate: int = 16000, total: int = 16000 * 300) -> bytes:
    """``fLaC`` + a STREAMINFO block (last-metadata flag set)."""
    packed = (rate << 44) | (0 << 41) | (15 << 36) | total
    streaminfo = (4096).to_bytes(2, "big") * 2 + b"\x00" * 6 + packed.to_bytes(8, "big")
    streaminfo += b"\x00" * 16  # MD5
    return b"fLaC" + bytes([0x80, 0, 0, 34]) + streaminfo


def test_header_durations_and_sniffing(tmp_path: Path) -> None:
    head = _flac_header()
    assert sessions_api.flac_duration(head) == pytest.approx(300.0)
    assert sessions_api.duration_from_header(head) == pytest.approx(300.0)
    wav_path = tmp_path / "x.wav"
    _silent_wav(wav_path, frames=8000 * 3)
    raw = wav_path.read_bytes()
    assert sessions_api.wav_duration(raw[:44], len(raw)) == pytest.approx(3.0)
    assert sessions_api.sniff_audio_mime(head) == "audio/flac"
    assert sessions_api.sniff_audio_mime(raw[:16]) == "audio/wav"
    assert sessions_api.sniff_audio_mime(b"OggS\x00\x02") == "audio/ogg"
    assert sessions_api.sniff_audio_mime(b"\x1a\x45\xdf\xa3\x01") == "audio/webm"
    assert sessions_api.sniff_audio_mime(b"\x00\x00\x00\x20ftypM4A ") == "audio/mp4"
    assert sessions_api.sniff_audio_mime(b"\x00" * 20) is None
    assert sessions_api.audio_mime("s3://b/x_01.flac") == "audio/flac"


def test_offsets_estimated_from_segments_when_probe_fails() -> None:
    files = ["a.flac", "b.flac", "c.flac"]
    bounds = {"a.flac": (0.03, 299.9), "b.flac": (300.03, 599.9), "c.flac": (600.6, 790.0)}
    rows, probed = resolve_file_offsets(files, [], lambda _p: 0.0, bounds)
    assert probed is True
    assert [round(r["offset"], 2) for r in rows] == [0.0, 300.03, 600.6]
    assert round(rows[-1]["duration"], 2) == round(790.0 - 600.6, 2)
    assert all(r.get("estimated") for r in rows)


def test_flac_served_and_expired_link_is_403(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _prepare(cfg)
    flac_path = tmp_path / "talk_01.flac"
    flac_path.write_bytes(_flac_header(total=16000 * 2) + b"\x00" * 200)
    noext = tmp_path / "talk_02"
    noext.write_bytes(_flac_header(total=16000) + b"\x00" * 50)
    created = datetime(2026, 11, 27, tzinfo=timezone.utc)
    engine = get_engine(cfg.db_dsn)
    with Session(engine) as db:
        db.add(_segment("f", 0, label="Ada", start=0, end=1.5, text="hi", created=created, audio=str(flac_path)))
        db.add(_segment("f", 1, label="Ada", start=2.1, end=2.8, text="yo", created=created, audio=str(noext)))
        db.commit()
    files = sessions_api.session_files(cfg, "f")
    assert [(f["offset"], f["duration"]) for f in files] == [(0.0, 2.0), (2.0, 1.0)]
    assert files[0]["estimated"] is False

    with TestClient(api_server.create_app(cfg)) as client:
        flac = client.get("/v1/sessions/f/audio/0", headers={**AUTH, "Range": "bytes=0-3"})
        assert flac.status_code == 206
        assert flac.headers["content-type"].startswith("audio/flac")
        sniffed = client.get("/v1/sessions/f/audio/1", headers={**AUTH, "Range": "bytes=0-3"})
        assert sniffed.status_code == 206
        assert sniffed.headers["content-type"].startswith("audio/flac")
        expired = client.get("/v1/sessions/f/audio/0?sig=deadbeef&exp=1")
        assert expired.status_code == 403


def test_session_files_cache_and_invalidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    calls: list[str] = []

    def fake_load(_cfg: object, sid: str) -> list[dict]:
        calls.append(sid)
        return [{"index": 0, "audio_file": "a.flac", "offset": 0.0, "duration": 1.0, "mime": "audio/flac"}]

    monkeypatch.setattr(sessions_api, "_load_session_files", fake_load)
    sessions_api.session_files(cfg, "c")
    sessions_api.session_files(cfg, "c")
    assert calls == ["c"]
    sessions_api.invalidate_session_files("c")
    sessions_api.session_files(cfg, "c")
    assert calls == ["c", "c"]
