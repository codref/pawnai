"""Vault scanner debounce decisions, expressed through note-state transitions."""

from datetime import datetime, timedelta, timezone

from pawn_core.knowledge_index import substantial_change


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
