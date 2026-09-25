"""Unit tests for vault transcript helpers (no live S3)."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import patch

from botocore.exceptions import ClientError

from pawn_core.vault import VaultStore
from pawn_diarize.core.vault_transcript import (
    build_document_markdown,
    content_hash,
    extract_annotations,
    format_speakers_section,
    format_transcript_section,
    parse_since,
    push_session_transcript,
)


def _segs():
    return [
        {
            "audio_file": "a.wav",
            "label": "SPEAKER_00",
            "start_time": 0.0,
            "end_time": 12.5,
            "text": "Hello there",
        },
        {
            "audio_file": "a.wav",
            "label": "SPEAKER_00",
            "start_time": 12.5,
            "end_time": 20.0,
            "text": "How are you?",
        },
        {
            "audio_file": "a.wav",
            "label": "SPEAKER_01",
            "start_time": 20.0,
            "end_time": 35.0,
            "text": "Doing well",
        },
    ]


def test_format_speakers_section_talk_time_and_turns():
    name_lookup = {
        ("a.wav", "SPEAKER_00"): "Alice",
        ("a.wav", "SPEAKER_01"): "Bob",
    }
    md = format_speakers_section(_segs(), name_lookup, file_count=2, time_cursor=35.0)
    assert "| Alice |" in md
    assert "| Bob |" in md
    assert "2" in md
    assert "Files so far:** 2" in md
    assert "Duration cursor:" in md


def test_format_transcript_section_groups_speakers():
    name_lookup = {("a.wav", "SPEAKER_00"): "Alice"}
    md = format_transcript_section(_segs(), name_lookup)
    assert "**Alice** · 00:00.00" in md
    assert "Hello there" in md
    assert "How are you?" in md
    assert "**SPEAKER_01** · 00:20.00" in md


def test_extract_annotations_preserves_body():
    kramdown = """# sess

## Speakers

| A | B |

## Annotations

My note about **Alice**.

- follow up

## Transcript

hello
"""
    body = extract_annotations(kramdown)
    assert "My note about **Alice**." in body
    assert "follow up" in body


def test_extract_annotations_default_when_missing():
    body = extract_annotations("# only title\n")
    assert "Add notes" in body


def test_content_hash_stable_and_ignores_annotations_via_inputs():
    a = content_hash("speakers", "transcript")
    b = content_hash("speakers", "transcript")
    c = content_hash("speakers", "transcript changed")
    assert a == b
    assert a != c


def test_build_document_markdown_section_order_and_frontmatter():
    md = build_document_markdown(
        "my-session",
        "| Speaker | Talk time | Turns |\n|---|---|---|",
        "remember this",
        "**Alice** · 00:00.00\nhi",
        date_str="2026-09-25",
        speaker_names=["Alice"],
        duration="12m 03s",
    )
    assert md.startswith("---\n")
    assert "pawn: transcript" in md
    assert "session_id: my-session" in md
    assert md.index("## Speakers") < md.index("## Annotations")
    assert md.index("## Annotations") < md.index("## Transcript")
    assert "remember this" in md
    assert "Managed by Pawn" in md


def test_parse_since_date_and_iso():
    from datetime import timezone

    day = parse_since("2026-09-01")
    assert day.year == 2026 and day.month == 9 and day.day == 1
    assert day.tzinfo == timezone.utc
    assert day.hour == 0

    iso = parse_since("2026-09-01T15:30:00Z")
    assert iso.hour == 15 and iso.minute == 30


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        data = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        self.objects[Key] = data
        return {"ETag": '"etag1"'}

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        body = self.objects[Key]
        return {"ETag": '"etag1"', "ContentLength": len(body)}

    def list_objects_v2(self, **kwargs):
        return {"Contents": [], "IsTruncated": False}


class _Cfg:
    class vault:
        bucket = "test-bucket"
        prefix = ""
        agent_root = "Pawn"
        transcript_path_template = "Pawn/Transcripts/{session_id}.md"
        analysis_path_template = "Pawn/Analyses/{session_id}.md"
        daily_note_path = None


def test_push_preserves_annotations_on_update():
    fake = _FakeS3Client()
    store = VaultStore(bucket="test-bucket", client=fake)
    segs = _segs()
    name_lookup = {
        ("a.wav", "SPEAKER_00"): "Alice",
        ("a.wav", "SPEAKER_01"): "Bob",
    }
    speakers_v1 = format_speakers_section(segs, name_lookup)
    transcript_v1 = format_transcript_section(segs, name_lookup)

    mapping_state: dict = {}

    def fake_get_note(dsn, session_id):
        return mapping_state.get(session_id)

    def fake_upsert(dsn, *, session_id, key, content_hash):
        from types import SimpleNamespace

        mapping_state[session_id] = SimpleNamespace(
            session_id=session_id, key=key, content_hash=content_hash
        )

    fixed_when = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    with (
        patch(
            "pawn_diarize.core.vault_transcript.load_session_segments",
            return_value=(segs, name_lookup),
        ),
        patch(
            "pawn_diarize.core.vault_transcript._load_session_meta",
            return_value=(1, 35.0),
        ),
        patch(
            "pawn_diarize.core.vault_transcript._session_when",
            return_value=fixed_when,
        ),
        patch(
            "pawn_diarize.core.vault_transcript.get_engine",
        ),
        patch(
            "pawn_diarize.core.vault_transcript.init_db",
        ),
        patch(
            "pawn_diarize.core.vault_transcript.get_vault_note",
            side_effect=fake_get_note,
        ),
        patch(
            "pawn_diarize.core.vault_transcript.upsert_vault_note",
            side_effect=fake_upsert,
        ),
    ):
        status1 = push_session_transcript(
            "sess-1",
            db_dsn="postgresql://local/test",
            store=store,
            cfg=_Cfg(),
        )
        assert status1.startswith("created:")

        key = "Pawn/Transcripts/sess-1.md"
        body = store.read(key)
        assert "Add notes" in body

        custom = body.replace(
            "_(Add notes and tags here.)_",
            "USER ANNOTATION LINE",
        )
        fake.objects[key] = custom.encode("utf-8")

        segs2 = list(segs)
        segs2[0] = {**segs2[0], "text": "Hello there (edited)"}
        speakers_v2 = format_speakers_section(segs2, name_lookup)
        assert speakers_v2 != speakers_v1 or transcript_v1

        with patch(
            "pawn_diarize.core.vault_transcript.load_session_segments",
            return_value=(segs2, name_lookup),
        ):
            status2 = push_session_transcript(
                "sess-1",
                db_dsn="postgresql://local/test",
                store=store,
                cfg=_Cfg(),
            )
        assert status2.startswith("updated:")
        updated = store.read(key)
        assert "USER ANNOTATION LINE" in updated
        assert "Hello there (edited)" in updated
