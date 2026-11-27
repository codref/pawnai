"""Tests for ``pawn-server portal import`` (sallm chats → portal mirror)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sallm.state.repository import SessionRepository
from typer.testing import CliRunner

from pawn_core.database import Base, _get_session
from pawn_server.core.portal_import import import_sallm_chats, matches, split_prompt
from pawn_server.core.portal_models import (
    PORTAL_TABLES,
    PortalChat,
    PortalChatFolder,
    PortalMessage,
)


@pytest.fixture
def env(tmp_path: Path) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'portal.db'}"
    engine = sa.create_engine(dsn)
    Base.metadata.create_all(engine, tables=PORTAL_TABLES)
    engine.dispose()
    state_dir = tmp_path / "sallm"
    repo = SessionRepository(state_dir / "state.db")
    repo.ensure_session("chat:abc")
    repo.append_message(
        "chat:abc",
        role="user",
        content=(
            "what is on my plate?\n\n---\nContext from the user's Obsidian vault "
            "(reference material; the message above is the request):\n\n"
            "Active note: [[Today]] (vault path `Pawn/Today.md`)"
        ),
    )
    repo.append_message(
        "chat:abc", role="assistant", content="```run\nplanner_day\n```", kind="tool"
    )
    repo.append_message("chat:abc", role="user", content="Tool results: ...", kind="tool")
    repo.append_message("chat:abc", role="assistant", content="Three meetings.")
    repo.ensure_session("matrix:!room:hs")
    repo.append_message("matrix:!room:hs", role="user", content="hey")
    repo.append_message("matrix:!room:hs", role="assistant", content="Hello!")
    repo.ensure_session("note:Pawn/Tasks/x.md")
    repo.append_message("note:Pawn/Tasks/x.md", role="user", content="task body")
    repo.ensure_session("chat:empty")
    repo.close()
    cfg = SimpleNamespace(db_dsn=dsn, sallm=SimpleNamespace(state_dir=str(state_dir)))
    return SimpleNamespace(cfg=cfg, dsn=dsn)


def test_split_prompt_separates_injected_context() -> None:
    text, ctx = split_prompt(
        "hi --- Context from the user's Obsidian vault (reference material; "
        "the message above is the request): Active note: [[A]] (vault path `A.md`) body"
    )
    assert text == "hi"
    assert ctx["active_note"] == "A.md"
    assert ctx["prompt_context"].startswith("Active note:")
    assert split_prompt("plain") == ("plain", {"imported": True})


def test_matches_prefixes() -> None:
    assert matches("chat:1", ("chat:",))
    assert not matches("note:x", ("chat:", "matrix:"))
    assert matches("anything", ("*",))


def test_import_mirrors_chat_turns_and_is_idempotent(env: SimpleNamespace) -> None:
    report = import_sallm_chats(env.cfg)
    assert sorted(k for k, _, _ in report.imported) == ["chat:abc", "matrix:!room:hs"]
    assert report.skipped_empty == ["chat:empty"]

    with _get_session(env.dsn) as db:
        chat = db.scalar(sa.select(PortalChat).where(PortalChat.conversation_key == "chat:abc"))
        assert chat is not None
        assert chat.title == "what is on my plate?"
        folder = db.get(PortalChatFolder, chat.folder_id)
        assert folder is not None and folder.name == "Imported"
        msgs = db.scalars(
            sa.select(PortalMessage)
            .where(PortalMessage.chat_id == chat.id)
            .order_by(PortalMessage.seq)
        ).all()
        assert [(m.role, m.content) for m in msgs] == [
            ("user", "what is on my plate?"),
            ("assistant", "Three meetings."),
        ]
        assert msgs[0].context["active_note"] == "Pawn/Today.md"
        assert (
            db.scalar(sa.select(sa.func.count()).where(PortalChat.conversation_key.like("note:%")))
            == 0
        )

    again = import_sallm_chats(env.cfg)
    assert again.imported == []
    assert sorted(again.skipped_existing) == ["chat:abc", "matrix:!room:hs"]
    with _get_session(env.dsn) as db:
        assert db.scalar(sa.select(sa.func.count()).select_from(PortalChatFolder)) == 1


def test_import_dry_run_writes_nothing(env: SimpleNamespace) -> None:
    report = import_sallm_chats(env.cfg, prefixes=("*",), dry_run=True)
    assert len(report.imported) == 3
    with _get_session(env.dsn) as db:
        assert db.scalar(sa.select(sa.func.count()).select_from(PortalChat)) == 0


def test_import_without_folder_with_tags(env: SimpleNamespace) -> None:
    import_sallm_chats(env.cfg, prefixes=("matrix:",), folder=None, tags=["#matrix"])
    with _get_session(env.dsn) as db:
        chat = db.scalar(sa.select(PortalChat))
        assert chat.folder_id is None
        assert chat.tags == ["matrix"]
        assert chat.tags_auto is False


def test_cli_dry_run(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    from pawn_server.cli import commands

    monkeypatch.setattr("pawn_agent.utils.config.load_config", lambda _path=None: env.cfg)
    result = CliRunner().invoke(commands.app, ["portal", "import", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Would import 2 chat(s)" in result.output
