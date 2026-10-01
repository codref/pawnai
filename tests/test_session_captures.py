"""Notes and screenshots attached to a diarization session."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.dialects import postgresql

from pawn_agent.core.screenshot_vision import normalize_summary
from pawn_diarize.core.session_captures import (
    Capture,
    _row_values,
    audio_offset_seconds,
    capture_insert_statement,
    clock_label,
    dedupe_captures,
    format_screenshots_body,
    merge_note_annotations,
    parse_payload_captures,
    render_analysis_transcript,
)


def _at() -> datetime:
    return datetime.fromisoformat("2026-10-01T22:10:00+02:00")


def _note(**overrides) -> Capture:
    data = dict(
        session_id="sess",
        item_id="ab12",
        kind="note",
        at=_at(),
        at_offset_minutes=120,
        text="decision: ship v2",
    )
    data.update(overrides)
    return Capture(**data)


def _shot(**overrides) -> Capture:
    data = dict(
        session_id="sess",
        item_id="cd34",
        kind="screenshot",
        at=_at() + timedelta(seconds=5),
        at_offset_minutes=120,
        s3_uri="s3://bucket/session/session_shot_01.png",
        output="DP-1",
    )
    data.update(overrides)
    return Capture(**data)


def test_parse_keeps_first_id_and_skips_bad_items():
    params = {
        "annotations": [
            {"id": "ab12", "at": "2026-10-01T22:10:00+02:00", "text": "first"},
            {"id": "ab12", "at": "2026-10-01T22:11:00+02:00", "text": "second"},
            {"id": "", "at": "2026-10-01T22:11:00+02:00", "text": "no id"},
            {"id": "bad time", "at": "not-a-date", "text": "bad"},
            "nope",
        ],
        "screenshots": [
            {
                "id": "cd34",
                "at": "2026-10-01T22:10:05+02:00",
                "s3_uri": "s3://bucket/session/session_shot_01.png",
                "output": "DP-1",
                "region": None,
            }
        ],
    }
    items = parse_payload_captures("sess", params)
    assert [item.item_id for item in items] == ["ab12", "cd34"]
    note = items[0]
    assert note.kind == "note"
    assert note.text == "first"
    assert note.at_offset_minutes == 120
    assert clock_label(note.at, note.at_offset_minutes) == "22:10"


def test_dedupe_keeps_the_first_text():
    kept = dedupe_captures(
        [
            _note(text="first"),
            _note(text="second"),
        ]
    )
    assert len(kept) == 1
    assert kept[0].text == "first"


def test_insert_leaves_existing_ids_alone():
    received = datetime(2026, 10, 1, 20, 15, tzinfo=timezone.utc)
    rows = _row_values([_note()], chunk_audio_start=0.0, received_at=received)
    stmt = capture_insert_statement(rows)
    sql = str(stmt.compile(dialect=postgresql.dialect())).upper()
    assert "ON CONFLICT" in sql
    assert "DO NOTHING" in sql


def test_audio_offset_clamps_to_the_chunk():
    received = datetime(2026, 10, 1, 20, 15, tzinfo=timezone.utc)
    start_wall = received - timedelta(seconds=300)
    inside = audio_offset_seconds(start_wall + timedelta(seconds=120), received, 100.0, 400.0)
    before = audio_offset_seconds(start_wall - timedelta(minutes=30), received, 100.0, 400.0)
    after = audio_offset_seconds(received + timedelta(minutes=30), received, 100.0, 400.0)
    assert inside == 220.0
    assert before == 100.0
    assert after == 400.0


def test_merge_annotations_appends_and_skips_known_ids():
    body = "USER ANNOTATION LINE\n"
    once = merge_note_annotations(body, [_note()])
    assert "USER ANNOTATION LINE" in once
    assert "<!-- pawn:note:ab12 -->" in once
    assert "- 22:10 — decision: ship v2" in once
    twice = merge_note_annotations(once, [_note(text="changed")])
    assert twice.count("<!-- pawn:note:ab12 -->") == 1
    assert "changed" not in twice


def test_screenshot_block_falls_back_to_s3_uri():
    body = format_screenshots_body([_shot()])
    assert "<!-- pawn:shot:cd34 -->" in body
    assert "![[" not in body
    assert "s3://bucket/session/session_shot_01.png" in body
    assert "22:10 · DP-1" in body

    copied = _shot(vault_key="Pawn/Transcripts/screenshots/sess/session_shot_01.png")
    copied.summary = "arrow added between the two boxes"
    embedded = format_screenshots_body([copied])
    assert "![[screenshots/sess/session_shot_01.png]]" in embedded
    assert "arrow added between the two boxes" in embedded
    assert "s3://" not in embedded


def test_missing_upload_is_explicit():
    body = format_screenshots_body([_shot(s3_uri=None)])
    assert "_Screenshot was not uploaded._" in body


def test_analysis_transcript_interleaves_notes_and_skips_unchanged():
    segments = [{"display": "Alice", "start_time": 12.0, "text": "hello"}]
    note = _note(audio_offset_s=10.0)
    change = _shot(audio_offset_s=22.0, summary="arrow added between the two boxes")
    same = _shot(
        item_id="ef56",
        audio_offset_s=30.0,
        summary="unchanged",
        at=_at() + timedelta(seconds=20),
    )
    pending = _shot(
        item_id="gh78",
        audio_offset_s=40.0,
        summary=None,
        at=_at() + timedelta(seconds=30),
    )
    text = render_analysis_transcript(segments, [note, change, same, pending])
    assert "[00:10.00] [note] decision: ship v2" in text
    assert "[screen DP-1] arrow added between the two boxes" in text
    assert "unchanged" not in text
    assert "1 screenshot, vision not run" in text
    assert text.index("[note]") < text.index("Alice")
    assert text.index("Alice") < text.index("[screen DP-1]")


def test_normalize_summary_collapses_no_change():
    assert normalize_summary("Unchanged.") == "unchanged"
    assert normalize_summary("arrow added") == "arrow added"
