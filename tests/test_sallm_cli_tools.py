"""Tests for pawn CliTool entrypoints."""

from __future__ import annotations

from unittest.mock import patch

from pawn_agent.core.sallm_skills import build_pawn_skills
from pawn_agent.core.sallm_tools import build_pawn_clitools
from pawn_agent.tools.cli import session_transcript, sessions_list


def test_build_pawn_clitools_names() -> None:
    tools = build_pawn_clitools()
    assert set(tools) == {
        "sessions_list",
        "session_transcript",
        "session_analyze",
        "siyuan_save",
        "schedule_propose",
        "queue_push",
    }


def test_build_pawn_skills_includes_sessions() -> None:
    registry = build_pawn_skills()
    assert "converse" in registry.names()
    assert "sessions" in registry.names()
    sessions = registry.get("sessions")
    assert sessions.tools is not None
    assert "sessions_list" in sessions.tools


def test_converse_exposes_all_tools_including_sessions_list() -> None:
    """Root skill must not hide CliTools — controller often keeps converse."""
    registry = build_pawn_skills()
    available = build_pawn_clitools()
    visible = registry.resolve_tools("converse", available)
    assert "sessions_list" in visible
    assert set(visible) == set(available)


def test_sessions_list_help() -> None:
    try:
        sessions_list.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_session_transcript_help() -> None:
    try:
        session_transcript.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_sessions_list_calls_impl() -> None:
    with (
        patch(
            "pawn_agent.tools.cli.sessions_list.load_agent_config",
            return_value=object(),
        ),
        patch(
            "pawn_agent.tools.cli.sessions_list.list_sessions_impl",
            return_value="session-a | 3 segments",
        ) as mock_impl,
    ):
        code = sessions_list.main(["--limit", "3"])
    assert code == 0
    mock_impl.assert_called_once()
    assert mock_impl.call_args.kwargs["limit"] == 3


def test_session_transcript_requires_session_id() -> None:
    try:
        session_transcript.main([])
        raised = False
    except SystemExit as exc:
        raised = True
        assert exc.code != 0
    assert raised


def test_siyuan_save_from_analysis_calls_impl() -> None:
    from pawn_agent.tools.cli import siyuan_save

    with (
        patch(
            "pawn_agent.tools.cli.siyuan_save.load_agent_config",
            return_value=object(),
        ),
        patch(
            "pawn_agent.tools.cli.siyuan_save.save_analysis_to_siyuan_impl",
            return_value="Saved to SiYuan: ok",
        ) as mock_impl,
    ):
        code = siyuan_save.main(
            ["--session-id", "daniel-20260630", "--from-analysis"]
        )
    assert code == 0
    mock_impl.assert_called_once()
    assert mock_impl.call_args.args[1] == "daniel-20260630"


def test_format_analysis_markdown() -> None:
    from types import SimpleNamespace

    from pawn_agent.tools.save_to_siyuan import format_analysis_markdown

    md = format_analysis_markdown(
        SimpleNamespace(
            title="Exit Interview",
            summary="Left for Open Loop.",
            key_topics="security",
            speaker_highlights=None,
            sentiment="mixed",
            sentiment_tags=["concern"],
            tags=["exit"],
        )
    )
    assert "# Exit Interview" in md
    assert "## Summary" in md
    assert "Open Loop" in md
