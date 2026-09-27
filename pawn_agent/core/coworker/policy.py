"""Pure attention policy: dedupe, ignore list, quiet hours, daily cap."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

_WS_RE = re.compile(r"\s+")


def fingerprint(text: str, thread: str = "") -> str:
    """Stable id for 'this item, on this thread'."""
    normalized = _WS_RE.sub(" ", (text or "").strip().lower())
    raw = f"{normalized}|{(thread or '').strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _parse_hhmm(value: str) -> Optional[time]:
    value = (value or "").strip()
    match = re.match(r"^(\d{1,2}):(\d{2})$", value)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def in_quiet_hours(now: datetime, quiet: Optional[str], timezone_name: str) -> bool:
    """True when *now* falls inside ``HH:MM-HH:MM`` (wraps past midnight)."""
    if not quiet or "-" not in quiet:
        return False
    start_s, end_s = quiet.split("-", 1)
    start = _parse_hhmm(start_s)
    end = _parse_hhmm(end_s)
    if start is None or end is None:
        return False
    try:
        local = now.astimezone(ZoneInfo(timezone_name or "UTC"))
    except Exception:
        local = now
    current = local.time()
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def matches_ignore(text: str, ignore: list[str]) -> bool:
    haystack = (text or "").lower()
    for entry in ignore:
        needle = (entry or "").strip().lower()
        if needle and needle in haystack:
            return True
    return False


@dataclass
class PolicyContext:
    """Inputs the policy needs besides the item itself."""

    now: datetime
    timezone_name: str = "UTC"
    quiet_hours: Optional[str] = None
    max_per_day: int = 5
    notified_today: int = 0
    ignore: list[str] | None = None
    seen_fingerprints: set[str] | None = None
    suppressed_fingerprints: set[str] | None = None


@dataclass
class PolicyDecision:
    interrupt: bool
    reason: str


def apply_policy(
    *,
    text: str,
    thread: str,
    interrupt: bool,
    fingerprint_value: str,
    ctx: PolicyContext,
) -> PolicyDecision:
    """Return whether this item may interrupt. Filing still happens either way."""
    if not interrupt:
        return PolicyDecision(False, "score did not request an interrupt")
    suppressed = ctx.suppressed_fingerprints or set()
    seen = ctx.seen_fingerprints or set()
    if fingerprint_value in suppressed:
        return PolicyDecision(False, "suppressed")
    if fingerprint_value in seen:
        return PolicyDecision(False, "already seen")
    if matches_ignore(text, list(ctx.ignore or [])):
        return PolicyDecision(False, "ignore list")
    if in_quiet_hours(ctx.now, ctx.quiet_hours, ctx.timezone_name):
        return PolicyDecision(False, "quiet hours")
    if ctx.notified_today >= ctx.max_per_day:
        return PolicyDecision(False, "daily cap")
    if not (thread or "").strip():
        return PolicyDecision(False, "no active thread")
    return PolicyDecision(True, "matches an active thread")
