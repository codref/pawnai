"""Unit tests for SiYuan transcript diary helpers (no live SiYuan)."""

from __future__ import annotations

from pawn_diarize.core.siyuan_transcript import (
    build_document_markdown,
    content_hash,
    extract_annotations,
    format_speakers_section,
    format_transcript_section,
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
    md = format_speakers_section(
        _segs(), name_lookup, file_count=2, time_cursor=35.0
    )
    assert "| Alice |" in md
    assert "| Bob |" in md
    assert "2" in md  # Alice turns
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


def test_extract_annotations_strips_siyuan_ial_noise():
    kramdown = (
        '## Speakers\n'
        "table\n"
        '## Annotations {: id="20260101000000-abcdefg"}\n'
        "\n"
        "user wrote this\n"
        "\n"
        "## Transcript\n"
        "x\n"
    )
    body = extract_annotations(kramdown)
    assert "user wrote this" in body


def test_extract_annotations_default_when_missing():
    body = extract_annotations("# only title\n")
    assert "Add notes" in body


def test_content_hash_stable_and_ignores_annotations_via_inputs():
    a = content_hash("speakers", "transcript")
    b = content_hash("speakers", "transcript")
    c = content_hash("speakers", "transcript changed")
    assert a == b
    assert a != c


def test_build_document_markdown_section_order():
    md = build_document_markdown(
        "my-session",
        "| Speaker | Talk time | Turns |\n|---|---|---|",
        "remember this",
        "**Alice** · 00:00.00\nhi",
    )
    assert md.startswith("# my-session")
    assert md.index("## Speakers") < md.index("## Annotations")
    assert md.index("## Annotations") < md.index("## Transcript")
    assert "remember this" in md
    assert "Managed by Pawn" in md


def test_parse_since_date_and_iso():
    from datetime import timezone

    from pawn_diarize.core.siyuan_transcript import parse_since

    day = parse_since("2026-09-01")
    assert day.year == 2026 and day.month == 9 and day.day == 1
    assert day.tzinfo == timezone.utc
    assert day.hour == 0

    iso = parse_since("2026-09-01T15:30:00Z")
    assert iso.hour == 15 and iso.minute == 30
