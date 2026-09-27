"""Tests for TLS path resolution used by ``pawn-server serve``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pawn_server.cli.commands import resolve_ssl_files


def test_resolve_ssl_none() -> None:
    assert resolve_ssl_files(None, None) == (None, None)
    assert resolve_ssl_files("", "  ") == (None, None)


def test_resolve_ssl_requires_both(tmp_path: Path) -> None:
    cert = tmp_path / "cert.pem"
    cert.write_text("x")
    with pytest.raises(ValueError, match="both"):
        resolve_ssl_files(str(cert), None)
    with pytest.raises(ValueError, match="both"):
        resolve_ssl_files(None, str(cert))


def test_resolve_ssl_missing_file(tmp_path: Path) -> None:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("c")
    with pytest.raises(ValueError, match="ssl_keyfile not found"):
        resolve_ssl_files(str(cert), str(key))


def test_resolve_ssl_ok(tmp_path: Path) -> None:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("c")
    key.write_text("k")
    assert resolve_ssl_files(str(cert), str(key)) == (str(cert), str(key))
