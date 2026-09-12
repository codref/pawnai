"""Unit tests for Matrix bot helpers and config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pawn_agent.utils.config import MatrixBotConfig, load_config
from pawn_server.core.matrix_bot import (
    chunk_text,
    conversation_id,
    extract_prompt,
    is_direct_room,
    matrix_reply_body,
    normalize_body,
    verification_allowed,
)


def test_conversation_id() -> None:
    assert conversation_id("!abc:example.com") == "matrix:!abc:example.com"


def test_is_direct_room() -> None:
    assert is_direct_room(1) is True
    assert is_direct_room(2) is True
    assert is_direct_room(3) is False


def test_normalize_body_strips_reply_fallback() -> None:
    raw = "> <@alice:hs> hi\n\nplease summarize"
    assert normalize_body(raw) == "please summarize"


def test_normalize_body_edit() -> None:
    assert (
        normalize_body("* old", is_edit=True, new_body="fresh text")
        == "fresh text"
    )
    assert normalize_body("* patched", is_edit=True) == "patched"


def test_extract_prompt_prefix_and_dm() -> None:
    assert (
        extract_prompt("!pawn hello", command_prefix="!pawn", is_dm=False)
        == "hello"
    )
    assert (
        extract_prompt("hello", command_prefix="!pawn", is_dm=True) == "hello"
    )
    assert extract_prompt("hello", command_prefix="!pawn", is_dm=False) is None
    assert extract_prompt("!pawn", command_prefix="!pawn", is_dm=False) is None


def test_chunk_text() -> None:
    assert chunk_text("short") == ["short"]
    parts = chunk_text("abcdefghij", limit=4)
    assert parts == ["abcd", "efgh", "ij"]


def test_verification_allowed() -> None:
    bot = "@bot:hs"
    assert verification_allowed(bot, bot_user_id=bot, inviters=["@a:hs"]) is True
    assert verification_allowed("@a:hs", bot_user_id=bot, inviters=["@a:hs"]) is True
    assert verification_allowed("@x:hs", bot_user_id=bot, inviters=["@a:hs"]) is False
    assert verification_allowed("@x:hs", bot_user_id=bot, inviters=[]) is True


def test_matrix_reply_body_strips_tool_trail() -> None:
    raw = (
        "[tool] sessions_list → Found 10 session(s): daniel | segments: 327\n"
        "[tool] session_analyze → ok\n\n"
        "Here are your sessions."
    )
    assert matrix_reply_body(raw) == "Here are your sessions."
    assert matrix_reply_body("Just an answer.") == "Just an answer."


def test_matrix_bot_config_defaults() -> None:
    mb = MatrixBotConfig()
    assert mb.enabled is False
    assert mb.command_prefix == "!pawn"
    assert mb.inviters == []


def test_load_config_parses_matrix_bot(tmp_path: Path) -> None:
    path = tmp_path / "pawnai.yaml"
    path.write_text(
        yaml.dump(
            {
                "matrix_bot": {
                    "enabled": True,
                    "homeserver_url": "https://matrix.example.com",
                    "user_id": "@bot:example.com",
                    "user_token": "syt_token",
                    "command_prefix": "!x",
                    "inviters": ["@you:example.com"],
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(str(path))
    assert cfg.matrix_bot.enabled is True
    assert cfg.matrix_bot.user_id == "@bot:example.com"
    assert cfg.matrix_bot.command_prefix == "!x"
    assert cfg.matrix_bot.inviters == ["@you:example.com"]


def test_require_nio_message() -> None:
    from pawn_server.core import matrix_bot as mb

    # Smoke: validation helpers raise clear errors without a live homeserver.
    with pytest.raises(RuntimeError, match="homeserver_url"):
        mb._validate_cfg(MatrixBotConfig(enabled=True))
    with pytest.raises(RuntimeError, match="user_token or user_password"):
        mb._validate_cfg(
            MatrixBotConfig(
                enabled=True,
                homeserver_url="https://hs",
                user_id="@b:hs",
            )
        )
