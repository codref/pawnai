"""Vault scanner debounce decisions, expressed through note-state transitions."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from pawn_core.knowledge_index import content_hash, substantial_change


class _Stat:
    def __init__(self, etag: str) -> None:
        self.etag = etag


class _Store:
    def __init__(self) -> None:
        self.files = {"Ideas/One.md": "---\ntags: [idea]\n---\n# One\n\nHello idea\n"}

    def list(self, prefix: str, suffix: str = ".md"):
        return [key for key in self.files if key.startswith(prefix.strip("/"))]

    def stat(self, key: str):
        return _Stat("etag-1") if key in self.files else None

    def read(self, key: str) -> str:
        return self.files[key]


def test_watch_folder_match():
    from pawn_server.core.vault_scanner import _watched

    assert _watched("Ideas/One.md", ["Ideas/"])
    assert not _watched("Pawn/Notes/x.md", ["Ideas/"])


def test_substantial_change_skips_whitespace():
    assert not substantial_change("hello", "hello")
    assert substantial_change("hello", "hello\n\n## Added\n\nbody")


def test_quiet_window_math():
    seen = datetime.now(timezone.utc) - timedelta(seconds=10)
    quiet = timedelta(seconds=120)
    assert datetime.now(timezone.utc) - seen < quiet


def test_idea_note_is_not_processed(monkeypatch) -> None:
    from pawn_server.core import vault_scanner

    text = "---\ntags: [idea]\nstatus: inbox\n---\n# One\n\nHello idea\n"
    store = _Store()
    store.files["Ideas/One.md"] = text
    digest = content_hash(text)
    state = SimpleNamespace(
        etag="etag-1",
        content_hash=digest,
        last_seen_at=datetime.now(timezone.utc) - timedelta(hours=1),
        last_processed_hash="",
    )
    monkeypatch.setattr(vault_scanner.itemdb, "get_note_state", lambda _dsn, _key: state)
    monkeypatch.setattr(vault_scanner.itemdb, "upsert_note_state", lambda *_a, **_k: None)
    called: list[str] = []

    async def _process(*_args, **_kwargs):
        called.append("process")

    monkeypatch.setattr("pawn_agent.core.coworker.pipeline.process_note", _process)
    cfg = SimpleNamespace(
        db_dsn="dsn",
        coworker=SimpleNamespace(watch_folders=["Ideas/"], watch_tags=[], note_quiet_seconds=0),
        vault=SimpleNamespace(agent_root="Pawn"),
    )

    async def _run() -> bool:
        return await vault_scanner._consider_note(
            cfg,
            store,
            "Ideas/One.md",
            datetime.now(timezone.utc),
            timedelta(0),
            require_watch=True,
        )

    assert asyncio.run(_run()) is False
    assert called == []
