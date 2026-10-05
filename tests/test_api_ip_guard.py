"""API docs toggle and IP blacklist / brute-force guard."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from pawn_agent.utils.config import ApiSection
from pawn_agent.utils.db import (
    ApiIpBlacklist,
    add_ip_blacklist,
    clear_ip_blacklist,
    is_ip_blacklisted,
    list_ip_blacklist,
    remove_ip_blacklist,
)
from pawn_core.database import Base
from pawn_server.core import api_server, ip_guard


@pytest.fixture
def dsn(tmp_path: Path) -> str:
    path = tmp_path / "guard.db"
    url = f"sqlite:///{path}"
    engine = sa.create_engine(url)
    Base.metadata.create_all(engine, tables=[ApiIpBlacklist.__table__])
    engine.dispose()
    return url


def _cfg(dsn: str, **api_kwargs: Any) -> SimpleNamespace:
    api = ApiSection(token="tok", **api_kwargs)
    return SimpleNamespace(
        api=api,
        api_token=api.token,
        api_model_idle_timeout_minutes=10.0,
        db_dsn=dsn,
    )


@pytest.fixture
def client(dsn: str) -> Iterator[TestClient]:
    # Lower thresholds so tests stay fast; leave whitelist empty so TestClient
    # peer ("testclient") is subject to heuristics.
    cfg = _cfg(
        dsn,
        enable_docs=False,
        whitelist_ips=[],
        auth_fail_threshold=3,
        not_found_threshold=3,
        bruteforce_window_seconds=60,
    )
    with TestClient(api_server.create_app(cfg)) as c:
        yield c


def test_docs_disabled(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_docs_enabled(dsn: str) -> None:
    app = api_server.create_app(_cfg(dsn, enable_docs=True, whitelist_ips=[]))
    with TestClient(app) as c:
        assert c.get("/openapi.json").status_code == 200
        assert c.get("/docs").status_code == 200


def test_health_never_counted(client: TestClient, dsn: str) -> None:
    for _ in range(10):
        assert client.get("/health").status_code == 200
    assert list_ip_blacklist(dsn) == []


def test_auth_fail_blacklists_ip(client: TestClient, dsn: str) -> None:
    for _ in range(3):
        assert client.get("/v1/models").status_code == 401
    assert is_ip_blacklisted(dsn, "testclient")
    # Further requests blocked before auth
    authed = {"Authorization": "Bearer tok"}
    assert client.get("/v1/models", headers=authed).status_code == 403


def test_not_found_scan_blacklists_ip(client: TestClient, dsn: str) -> None:
    for i in range(3):
        assert client.get(f"/no-such-path-{i}").status_code == 404
    assert is_ip_blacklisted(dsn, "testclient")
    authed = {"Authorization": "Bearer tok"}
    assert client.get("/v1/models", headers=authed).status_code == 403


def test_whitelist_skips_bruteforce(dsn: str) -> None:
    cfg = _cfg(
        dsn,
        whitelist_ips=["testclient"],
        auth_fail_threshold=2,
        not_found_threshold=2,
    )
    with TestClient(api_server.create_app(cfg)) as c:
        for _ in range(5):
            assert c.get("/v1/models").status_code == 401
        assert list_ip_blacklist(dsn) == []
        authed = {"Authorization": "Bearer tok"}
        assert c.get("/v1/models", headers=authed).status_code == 200


def test_bruteforce_disabled(dsn: str) -> None:
    cfg = _cfg(
        dsn,
        whitelist_ips=[],
        bruteforce_enabled=False,
        auth_fail_threshold=1,
        not_found_threshold=1,
    )
    with TestClient(api_server.create_app(cfg)) as c:
        for _ in range(5):
            assert c.get("/v1/models").status_code == 401
        assert list_ip_blacklist(dsn) == []


def test_manual_blacklist_crud(dsn: str) -> None:
    add_ip_blacklist(dsn, "203.0.113.9", reason="manual")
    rows = list_ip_blacklist(dsn)
    assert len(rows) == 1
    assert rows[0].ip == "203.0.113.9"
    assert is_ip_blacklisted(dsn, "203.0.113.9")
    assert remove_ip_blacklist(dsn, "203.0.113.9")
    assert not is_ip_blacklisted(dsn, "203.0.113.9")
    add_ip_blacklist(dsn, "203.0.113.1", reason="a")
    add_ip_blacklist(dsn, "203.0.113.2", reason="b")
    assert clear_ip_blacklist(dsn) == 2


def test_normalize_ip() -> None:
    assert ip_guard.normalize_ip("::ffff:127.0.0.1") == "127.0.0.1"
    assert ip_guard.is_whitelisted("::ffff:127.0.0.1", ["127.0.0.1"])


def _request(
    peer: str,
    headers: Optional[dict[str, str]] = None,
) -> Any:
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [
            (k.lower().encode(), v.encode())
            for k, v in (headers or {}).items()
        ],
        "client": (peer, 12345),
        "server": ("127.0.0.1", 8000),
    }
    return Request(scope)


def test_resolve_client_ip_ignores_headers_by_default() -> None:
    req = _request(
        "203.0.113.1",
        {"x-forwarded-for": "198.51.100.9", "x-real-ip": "198.51.100.9"},
    )
    assert ip_guard.resolve_client_ip(req) == "203.0.113.1"
    assert (
        ip_guard.resolve_client_ip(
            req, trust_proxy=True, trusted_proxies=["127.0.0.1"]
        )
        == "203.0.113.1"
    )


def test_resolve_client_ip_trusts_xff_from_proxy() -> None:
    req = _request(
        "127.0.0.1",
        {"x-forwarded-for": "198.51.100.9, 127.0.0.1"},
    )
    assert (
        ip_guard.resolve_client_ip(
            req, trust_proxy=True, trusted_proxies=["127.0.0.1", "::1"]
        )
        == "198.51.100.9"
    )


def test_resolve_client_ip_prefers_x_real_ip() -> None:
    req = _request(
        "127.0.0.1",
        {
            "x-real-ip": "198.51.100.7",
            "x-forwarded-for": "198.51.100.9",
        },
    )
    assert (
        ip_guard.resolve_client_ip(
            req, trust_proxy=True, trusted_proxies=["127.0.0.1"]
        )
        == "198.51.100.7"
    )


def test_proxy_ban_uses_forwarded_client(dsn: str) -> None:
    """Blacklist the real client IP, not the reverse-proxy peer."""
    cfg = _cfg(
        dsn,
        whitelist_ips=[],
        auth_fail_threshold=2,
        not_found_threshold=99,
        trust_proxy=True,
        trusted_proxies=["testclient"],
    )
    headers = {"X-Forwarded-For": "203.0.113.50"}
    with TestClient(api_server.create_app(cfg)) as c:
        for _ in range(2):
            assert c.get("/v1/models", headers=headers).status_code == 401
        assert is_ip_blacklisted(dsn, "203.0.113.50")
        assert not is_ip_blacklisted(dsn, "testclient")
        assert (
            c.get(
                "/v1/models",
                headers={**headers, "Authorization": "Bearer tok"},
            ).status_code
            == 403
        )


def test_expired_blacklist_is_purged(dsn: str) -> None:
    from datetime import datetime, timedelta, timezone

    add_ip_blacklist(dsn, "198.51.100.1", reason="temp", ttl_seconds=1)
    # Force expiry in DB
    engine = sa.create_engine(dsn)
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE api_ip_blacklist"
                " SET expires_at = :exp WHERE ip = :ip"
            ),
            {"exp": past.isoformat(), "ip": "198.51.100.1"},
        )
    engine.dispose()
    assert not is_ip_blacklisted(dsn, "198.51.100.1")
    assert list_ip_blacklist(dsn) == []
