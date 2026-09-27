"""Markdown chunking and change detection for the knowledge index."""

from pawn_core.knowledge_index import chunk_markdown, chunk_transcript, substantial_change


def test_chunk_markdown_by_heading():
    text = "# Title\n\nIntro line.\n\n## Next\n\nMore.\n"
    chunks = chunk_markdown(text)
    headings = [heading for heading, _body in chunks]
    assert "Title" in headings
    assert "Next" in headings


def test_chunk_transcript_windows():
    chunks = chunk_transcript("SPEAKER_00: hello " * 50, max_chars=80)
    assert len(chunks) > 1


def test_substantial_change():
    assert substantial_change("", "a real note")
    assert not substantial_change("", "   ")
    assert substantial_change("short", "short" + ("x" * 50))
    assert substantial_change("# A\n\nbody", "# A\n\nbody\n\n## New\n\nextra")
