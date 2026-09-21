"""Tests for pawn CliTool entrypoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pawn_agent.core.sallm_skills import build_pawn_skills
from pawn_agent.core.sallm_tools import build_pawn_clitools
from pawn_agent.tools.cli import session_delete, session_transcript, sessions_list
from pawn_agent.tools.delete_session import delete_session_impl


def test_build_pawn_clitools_names() -> None:
    tools = build_pawn_clitools()
    assert set(tools) == {
        "sessions_list",
        "session_transcript",
        "session_analyze",
        "session_delete",
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
    assert "session_delete" in sessions.tools


def test_converse_exposes_all_tools_including_sessions_list() -> None:
    """Root skill must not hide CliTools — controller often keeps converse."""
    registry = build_pawn_skills()
    available = build_pawn_clitools()
    visible = registry.resolve_tools("converse", available)
    assert "sessions_list" in visible
    assert "session_delete" in visible
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


def test_session_delete_help() -> None:
    try:
        session_delete.main(["--help"])
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
    assert session_transcript.main([]) != 0


def test_session_transcript_accepts_positional_id() -> None:
    with (
        patch(
            "pawn_agent.tools.cli.session_transcript.load_agent_config",
            return_value=object(),
        ),
        patch(
            "pawn_agent.tools.cli.session_transcript.query_conversation_impl",
            return_value="transcript ok",
        ) as mock_impl,
    ):
        code = session_transcript.main(["daniel-20260630"])
    assert code == 0
    mock_impl.assert_called_once()
    assert mock_impl.call_args.args[1] == "daniel-20260630"


def test_session_delete_requires_flags() -> None:
    with pytest.raises(SystemExit) as missing_both:
        session_delete.main([])
    assert missing_both.value.code != 0
    with pytest.raises(SystemExit) as missing_confirm:
        session_delete.main(["--session-id", "meeting-1"])
    assert missing_confirm.value.code != 0


def test_session_delete_confirm_mismatch_skips_impl() -> None:
    with (
        patch(
            "pawn_agent.tools.cli.session_delete.load_agent_config",
            return_value=MagicMock(db_dsn="postgresql+psycopg://x/y"),
        ),
        patch(
            "pawn_agent.tools.cli.session_delete.delete_session_impl",
            side_effect=ValueError("confirmation mismatch"),
        ) as mock_impl,
    ):
        code = session_delete.main(
            ["--session-id", "meeting-1", "--confirm", "other"]
        )
    assert code != 0
    mock_impl.assert_called_once()


def test_session_delete_matching_flags_call_impl() -> None:
    with (
        patch(
            "pawn_agent.tools.cli.session_delete.load_agent_config",
            return_value=object(),
        ),
        patch(
            "pawn_agent.tools.cli.session_delete.delete_session_impl",
            return_value="Deleted session 'meeting-1': 3 segment(s).",
        ) as mock_impl,
    ):
        code = session_delete.main(
            ["--session-id", "meeting-1", "--confirm", "meeting-1"]
        )
    assert code == 0
    mock_impl.assert_called_once()
    assert mock_impl.call_args.kwargs["session_id"] == "meeting-1"
    assert mock_impl.call_args.kwargs["confirm"] == "meeting-1"


def test_delete_session_impl_rejects_confirm_mismatch() -> None:
    cfg = MagicMock(db_dsn="postgresql+psycopg://x/y")
    with pytest.raises(ValueError, match="confirmation mismatch"):
        delete_session_impl(cfg, session_id="meeting-1", confirm="other")


def test_delete_session_impl_rejects_empty_id() -> None:
    cfg = MagicMock(db_dsn="postgresql+psycopg://x/y")
    with pytest.raises(ValueError, match="empty"):
        delete_session_impl(cfg, session_id="  ", confirm="  ")


def test_delete_session_impl_deletes_and_returns_receipt() -> None:
    cfg = MagicMock(db_dsn="postgresql+psycopg://x/y")
    mock_db = MagicMock()
    results = [
        MagicMock(rowcount=3),
        MagicMock(rowcount=1),
        MagicMock(rowcount=1),
        MagicMock(rowcount=2),
    ]
    mock_db.execute.side_effect = results
    mock_session_cm = MagicMock()
    mock_session_cm.__enter__.return_value = mock_db
    mock_session_cm.__exit__.return_value = False

    with (
        patch("pawn_agent.tools.delete_session.get_engine") as mock_engine_fn,
        patch(
            "pawn_agent.tools.delete_session.Session",
            return_value=mock_session_cm,
        ),
    ):
        engine = MagicMock()
        mock_engine_fn.return_value = engine
        receipt = delete_session_impl(
            cfg, session_id="meeting-1", confirm="meeting-1"
        )

    assert "meeting-1" in receipt
    assert "3 segment(s)" in receipt
    assert "1 analysis row(s)" in receipt
    assert "1 session_state row(s)" in receipt
    assert "2 graph triple(s)" in receipt
    mock_db.commit.assert_called_once()
    assert mock_db.execute.call_count == 4
    mock_engine_fn.assert_called_once_with(cfg.db_dsn)
    engine.dispose.assert_not_called()


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


def test_siyuan_save_content_file_calls_impl(tmp_path) -> None:
    from pawn_agent.tools.cli import siyuan_save

    note = tmp_path / "note.md"
    note.write_text("# Tom's notes\n\nLine two.\n", encoding="utf-8")

    with (
        patch(
            "pawn_agent.tools.cli.siyuan_save.load_agent_config",
            return_value=object(),
        ),
        patch(
            "pawn_agent.tools.cli.siyuan_save.save_to_siyuan_impl",
            return_value="siyuan://blocks/abc",
        ) as mock_impl,
    ):
        code = siyuan_save.main(
            [
                "--session-id",
                "daniel-20260630",
                "--content-file",
                str(note),
                "--title",
                "Notes",
            ]
        )
    assert code == 0
    mock_impl.assert_called_once()
    kwargs = mock_impl.call_args.kwargs
    assert kwargs["session_id"] == "daniel-20260630"
    assert "Tom's notes" in kwargs["content"]
    assert kwargs["title"] == "Notes"


def test_siyuan_save_rejects_multiline_content() -> None:
    from pawn_agent.tools.cli import siyuan_save

    with patch(
        "pawn_agent.tools.cli.siyuan_save.load_agent_config",
        return_value=object(),
    ):
        code = siyuan_save.main(
            [
                "--session-id",
                "s1",
                "--content",
                "line1\nline2",
            ]
        )
    assert code != 0


def test_siyuan_save_rejects_long_content() -> None:
    from pawn_agent.tools.cli import siyuan_save

    with patch(
        "pawn_agent.tools.cli.siyuan_save.load_agent_config",
        return_value=object(),
    ):
        code = siyuan_save.main(
            [
                "--session-id",
                "s1",
                "--content",
                "x" * 241,
            ]
        )
    assert code != 0


def test_prepare_commands_file_ref_for_siyuan_style() -> None:
    """sallm materializes @note so pawn can pass --content-file safely."""
    from pathlib import Path

    from sallm.tools import prepare_commands

    text = """```run
siyuan_save --session-id s1 --content-file @note
```
```file note
# Deep analysis

Tom's summary with "quotes"
```"""
    prepared = prepare_commands(text)
    try:
        assert prepared.error is None
        assert prepared.commands[0][0] == "siyuan_save"
        path = prepared.commands[0][-1]
        body = Path(path).read_text(encoding="utf-8")
        assert "Tom's summary" in body
        assert 'with "quotes"' in body
    finally:
        prepared.cleanup()


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
