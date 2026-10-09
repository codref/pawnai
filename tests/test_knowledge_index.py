"""Markdown chunking and change detection for the knowledge index."""

from types import SimpleNamespace

from pawn_core.knowledge_index import (
    chunk_markdown,
    chunk_transcript,
    compose_search_query,
    substantial_change,
)


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


def test_compose_search_query_uses_instruct_template():
    cfg = SimpleNamespace(
        sallm=SimpleNamespace(
            embedding_model="ollama/qwen3-embedding:0.6b",
            embedding_api_base="http://localhost:11434",
        ),
        coworker=SimpleNamespace(embed_dim=1024),
    )
    composed = compose_search_query(cfg, "waiting for Miguel")
    assert "waiting for Miguel" in composed
    assert composed != "waiting for Miguel"
    assert "Instruct:" in composed or "Query:" in composed
