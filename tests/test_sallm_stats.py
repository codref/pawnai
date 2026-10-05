"""Tests for sallm session stats formatting."""

from __future__ import annotations

from pawn_agent.core.sallm_session import format_session_stats


def test_format_session_stats_without_metrics() -> None:
    text = format_session_stats(
        {
            "conversation_id": "matrix:!room:hs",
            "model": "openai/gpt-4o",
            "active_skill": "sessions",
            "goal": "",
            "stack": [{"skill": "sessions", "depth": 0, "note": ""}],
            "message_count": 4,
            "chunk_count": 2,
            "pending_extracts": 0,
            "max_steps": 8,
            "last_metrics": {},
        }
    )
    assert "matrix:!room:hs" in text
    assert "sessions" in text
    assert "messages**: 4" in text
    assert "No turn metrics yet" in text


def test_format_session_stats_with_last_turn() -> None:
    text = format_session_stats(
        {
            "conversation_id": "cli",
            "model": "openai/gemma",
            "active_skill": "converse",
            "goal": "summarize",
            "stack": [
                {"skill": "converse", "depth": 0, "note": ""},
                {"skill": "sessions", "depth": 1, "note": ""},
            ],
            "message_count": 10,
            "chunk_count": 5,
            "pending_extracts": 1,
            "max_steps": 8,
            "last_metrics": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "context_messages": 10,
                "prompt_messages": 8,
                "elapsed_ms": 1234.5,
                "reasoning_tokens": 20,
            },
        }
    )
    assert "goal**: summarize" in text
    assert "converse → sessions" in text
    assert "pending extracts**: 1" in text
    assert "in 100 / out 50 / total 150" in text
    assert "reasoning tokens**: 20" in text
