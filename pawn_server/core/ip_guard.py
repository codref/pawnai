"""API IP guard — whitelist, brute-force / scan heuristics, DB blacklist.

Tracks per-IP 401 (auth fail) and 404 (path scan) counts inside a sliding
window.  When a threshold is crossed the IP is persisted to
``api_ip_blacklist`` and subsequent requests are rejected with 403.

Whitelisted IPs (config ``api.whitelist_ips``) skip all checks.  Counters are
process-local; the durable ban list lives in PostgreSQL.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# path → never count toward scan / auth heuristics
_SKIP_PATHS = frozenset({"/health"})

# In-memory sliding windows: ip → deque of event timestamps
_auth_fails: Dict[str, Deque[float]] = defaultdict(deque)
_not_founds: Dict[str, Deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def reset_counters() -> None:
    """Clear process-local event windows (tests / create_app)."""
    with _lock:
        _auth_fails.clear()
        _not_founds.clear()


def normalize_ip(ip: str) -> str:
    """Strip IPv4-mapped IPv6 so ``::ffff:127.0.0.1`` matches ``127.0.0.1``."""
    if ip.startswith("::ffff:"):
        return ip[7:]
    return ip


def _peer_ip(request: Request) -> str:
    if request.client is None:
        return "unknown"
    return normalize_ip(request.client.host)


def _parse_forwarded_chain(header_value: str) -> list[str]:
    return [
        normalize_ip(part.strip())
        for part in header_value.split(",")
        if part.strip()
    ]


def resolve_client_ip(
    request: Request,
    *,
    trust_proxy: bool = False,
    trusted_proxies: Optional[Iterable[str]] = None,
) -> str:
    """Return the client IP for blacklist / whitelist decisions.

    When ``trust_proxy`` is false (default), always uses the TCP peer.
    When true, ``X-Real-IP`` / ``X-Forwarded-For`` are honoured only if the
    peer is in ``trusted_proxies`` (so random clients cannot spoof headers).
    For XFF chains, walks right-to-left and skips trusted proxy hops.
    """
    peer = _peer_ip(request)
    if not trust_proxy:
        return peer

    trusted = {normalize_ip(p) for p in (trusted_proxies or []) if p}
    if peer not in trusted:
        return peer

    real = request.headers.get("x-real-ip")
    if real:
        candidate = normalize_ip(real.strip())
        if candidate:
            return candidate

    xff = request.headers.get("x-forwarded-for")
    if not xff:
        return peer

    chain = _parse_forwarded_chain(xff)
    if not chain:
        return peer
    for candidate in reversed(chain):
        if candidate not in trusted:
            return candidate
    return chain[0]


def client_ip(
    request: Request,
    *,
    trust_proxy: bool = False,
    trusted_proxies: Optional[Iterable[str]] = None,
) -> str:
    """Compatibility wrapper around :func:`resolve_client_ip`."""
    return resolve_client_ip(
        request,
        trust_proxy=trust_proxy,
        trusted_proxies=trusted_proxies,
    )


def is_whitelisted(ip: str, whitelist: Iterable[str]) -> bool:
    """Exact-match check against the configured whitelist."""
    needle = normalize_ip(ip)
    return any(normalize_ip(entry) == needle for entry in whitelist)


def _prune(window: Deque[float], now: float, window_seconds: float) -> None:
    cutoff = now - window_seconds
    while window and window[0] < cutoff:
        window.popleft()


def record_event(
    ip: str,
    kind: str,
    *,
    window_seconds: int,
    auth_fail_threshold: int,
    not_found_threshold: int,
) -> Optional[str]:
    """Record a 401/404 for *ip*.

    Returns a ban reason string when a threshold is crossed, else None.
    """
    now = time.monotonic()
    with _lock:
        if kind == "auth_fail":
            bucket = _auth_fails[ip]
            threshold = auth_fail_threshold
            reason = f"auth_fail:{auth_fail_threshold} in {window_seconds}s"
        elif kind == "not_found":
            bucket = _not_founds[ip]
            threshold = not_found_threshold
            reason = f"not_found:{not_found_threshold} in {window_seconds}s"
        else:
            return None
        bucket.append(now)
        _prune(bucket, now, float(window_seconds))
        if len(bucket) >= threshold:
            bucket.clear()
            return reason
    return None


def maybe_blacklist(
    dsn: str,
    ip: str,
    reason: str,
    *,
    ttl_seconds: Optional[int],
    hit_count: int = 0,
) -> None:
    """Persist *ip* to the DB blacklist; log on failure without raising."""
    try:
        from pawn_agent.utils.db import add_ip_blacklist  # noqa: PLC0415

        add_ip_blacklist(
            dsn,
            ip,
            reason=reason,
            ttl_seconds=ttl_seconds,
            hit_count=hit_count,
        )
        logger.warning("API IP blacklisted: %s (%s)", ip, reason)
    except Exception:
        logger.exception("Failed to blacklist IP %s", ip)


def check_blacklisted(dsn: str, ip: str) -> bool:
    """Return True if *ip* is actively blacklisted. Fail-open on DB errors."""
    try:
        from pawn_agent.utils.db import is_ip_blacklisted  # noqa: PLC0415

        return is_ip_blacklisted(dsn, ip)
    except Exception:
        logger.exception(
            "Blacklist lookup failed for %s — allowing request", ip
        )
        return False


class IpGuardMiddleware(BaseHTTPMiddleware):
    """Reject blacklisted IPs; auto-ban after repeated 401/404 responses."""

    def __init__(self, app: ASGIApp, get_cfg: Any) -> None:
        super().__init__(app)
        self._get_cfg = get_cfg

    async def dispatch(
        self, request: Request, call_next
    ):  # type: ignore[no-untyped-def]
        cfg = self._get_cfg()
        if cfg is None:
            return await call_next(request)

        api = getattr(cfg, "api", None)
        if api is None or not bool(getattr(api, "bruteforce_enabled", True)):
            return await call_next(request)

        ip = resolve_client_ip(
            request,
            trust_proxy=bool(getattr(api, "trust_proxy", False)),
            trusted_proxies=list(getattr(api, "trusted_proxies", None) or []),
        )
        whitelist = list(getattr(api, "whitelist_ips", None) or [])
        if is_whitelisted(ip, whitelist):
            return await call_next(request)

        dsn = getattr(cfg, "db_dsn", None)
        if dsn and check_blacklisted(dsn, ip):
            return JSONResponse({"detail": "Forbidden"}, status_code=403)

        response = await call_next(request)

        path = request.url.path
        if path in _SKIP_PATHS or not dsn:
            return response

        kind: Optional[str] = None
        if response.status_code == 401:
            kind = "auth_fail"
        elif response.status_code == 404:
            kind = "not_found"
        if kind is None:
            return response

        reason = record_event(
            ip,
            kind,
            window_seconds=int(getattr(api, "bruteforce_window_seconds", 300)),
            auth_fail_threshold=int(getattr(api, "auth_fail_threshold", 10)),
            not_found_threshold=int(getattr(api, "not_found_threshold", 40)),
        )
        if reason:
            maybe_blacklist(
                dsn,
                ip,
                reason,
                ttl_seconds=getattr(api, "blacklist_ttl_seconds", None),
                hit_count=int(
                    getattr(api, "auth_fail_threshold", 10)
                    if kind == "auth_fail"
                    else getattr(api, "not_found_threshold", 40)
                ),
            )
        return response
